import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable

from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model
from apex import amp

from ..datasets.loader.dataset import BaseDataset
from ..datasets.loader.gtav_dataset import GTAVDataset
from ..datasets.loader.cityscapes_dataset import CityscapesDataset
from ..datasets.loader.synthia_dataset import SYNTHIADataset
from ..datasets.loader.bdd_dataset import BDDDataset
from ..datasets.metrics.miou import mean_iou, get_hist, intersectionAndUnionGPU
from ..datasets.metrics.acc import acc, acc_with_hist

from ..models.losses.ranger import Ranger
from ..models.losses.cos_annealing_with_restart import CosineAnnealingLR_with_Restart
from ..models.registry import DATASET

import os
import logging
import time
import numpy as np
import random
import pickle
import pdb
import sys

def seed_everything(seed=888):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

def train_net(net,
              cfg,
              gpu,
              ):
    # init seed
    seed_everything(cfg.RANDOM_SEED)
    
    # init logger
    dir_cp = cfg.WORK_DIR
    if not os.path.exists(dir_cp):
        try:
            os.makedirs(dir_cp)
        except Exception:
            pass
    logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
                    filename=os.path.join(cfg.WORK_DIR,'train.log'),
                    filemode='a',
                    level=logging.INFO)
    logger = logging.getLogger("sseg.trainer")
    sh = logging.StreamHandler(sys.stdout)
    logger.addHandler(sh)

    # init net
    dist.init_process_group(
        backend='nccl', 
        init_method='tcp://127.0.0.1:6789', 
        world_size=cfg.TRAIN.N_PROC_PER_NODE,
        rank=gpu
        )
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:{}".format(gpu))
    
    if cfg.TRAIN.N_PROC_PER_NODE > 1:
        net = convert_syncbn_model(net) # to syncbn
    net.to(device)

    # train dataset
    early_stopping = cfg.TRAIN.EARLY_STOPPING
    anns = cfg.DATASET.ANNS
    image_dir = cfg.DATASET.IMAGEDIR
    val_resize_size = cfg.DATASET.VAL.RESIZE_SIZE
    use_aug = cfg.DATASET.USE_AUG


    # train dataset
    train = DATASET[cfg.DATASET.TYPE](anns, image_dir, use_aug=use_aug)
    train_sampler = DistributedSampler(train, num_replicas=cfg.TRAIN.N_PROC_PER_NODE, rank=gpu)
    train_data = DataLoader(train, cfg.TRAIN.BATCHSIZE, sampler=train_sampler, num_workers=cfg.DATASET.NUM_WORKER, drop_last=True, pin_memory=True)
    source_iter = iter(train_data)

    # target dataset
    if cfg.DATASET.TARGET.ANNS != '':
        t_anns = cfg.DATASET.TARGET.ANNS
        t_image_dir = cfg.DATASET.TARGET.IMAGEDIR
        t_train = DATASET[cfg.DATASET.TARGET.TYPE](t_anns, t_image_dir, use_aug=use_aug)
        t_train_sampler = DistributedSampler(t_train, num_replicas=cfg.TRAIN.N_PROC_PER_NODE, rank=gpu)
        t_train_data = DataLoader(t_train, cfg.TRAIN.BATCHSIZE, sampler=t_train_sampler, num_workers=cfg.DATASET.NUM_WORKER, drop_last=True, pin_memory=True)
        target_iter = iter(t_train_data)
    
    # val dataset  
    val_anns = cfg.DATASET.VAL.ANNS
    val_image_dir = cfg.DATASET.VAL.IMAGEDIR
    val = DATASET[cfg.DATASET.VAL.TYPE](val_anns, val_image_dir)
    val_sampler = DistributedSampler(val, num_replicas=cfg.TRAIN.N_PROC_PER_NODE, rank=gpu)
    val_data = DataLoader(val, cfg.TEST.BATCH_SIZE, sampler=val_sampler, num_workers=cfg.DATASET.NUM_WORKER, pin_memory=True)

    n_train = len(train_data) * cfg.TRAIN.BATCHSIZE
    expect_iter = n_train * cfg.TRAIN.EPOCHES //cfg.TRAIN.BATCHSIZE

    # optimizer 
    optimizer, D_optimizer_dict = build_optimizer(net, cfg)
    optimizers = [optimizer, ]
    for name, optim in D_optimizer_dict.items():
        optimizers.append(optim)

    schedulers = []
    if cfg.TRAIN.SCHEDULER == "CosineAnnealingLR_with_Restart":
        scheduler = CosineAnnealingLR_with_Restart(
            optimizer, 
            T_max=cfg.TRAIN.COSINEANNEALINGLR.T_MAX*expect_iter//cfg.TRAIN.EPOCHES, 
            T_mult=cfg.TRAIN.COSINEANNEALINGLR.T_MULT,
            eta_min=cfg.TRAIN.LR*0.001
            )
        schedulers.append(scheduler)
        for optim in D_optimizer_dict.values():
            scheduler = CosineAnnealingLR_with_Restart(
                optim, 
                T_max=cfg.TRAIN.COSINEANNEALINGLR.T_MAX*expect_iter//cfg.TRAIN.EPOCHES, 
                T_mult=cfg.TRAIN.COSINEANNEALINGLR.T_MULT,
                eta_min=cfg.TRAIN.LR*0.001
                )
            schedulers.append(scheduler)
    elif cfg.TRAIN.SCHEDULER == "LambdaLR":
        lr_lambda = lambda iter: (1-(iter  / expect_iter))**0.9
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        schedulers.append(scheduler)
        for optim in D_optimizer_dict.values():
            scheduler = LambdaLR(optim, lr_lambda=lr_lambda)
            schedulers.append(scheduler)
        
    
    # resume optimizer state
    if os.path.exists(os.path.join(dir_cp, 'state.pth')):
        state = torch.load(os.path.join(dir_cp, 'state.pth'), device)
        resume_epoch = state['epoch']
        resume_iter = state['iter']
        result = state['result']
        for i, opt in enumerate(optimizers):
            opt.load_state_dict(state['optimizers'][i])
        for i, sch in enumerate(schedulers):
            sch.load_state_dict(state['schedulers'][i])
    else:
        resume_epoch = 0
        resume_iter = 0
        result = []
    
    # apex
    net, optimizers = amp.initialize(net, optimizers, opt_level=cfg.TRAIN.APEX_OPT)
    for i, (name, optim) in enumerate(D_optimizer_dict.items()):
        D_optimizer_dict[name] = optimizers[i+1]
    net = DDP(net, delay_allreduce=True)

    if gpu == 0:
        logger.info(cfg)
        logger.info("resume from epoch {} iter {}".format(resume_epoch, resume_iter) )
        logger.info("Start training!")

    max_metrics = 0
    max_metrics_epoch = 0
    metrics_decay_count = 0
    epoch = resume_epoch
    iter_cnt = resume_iter
    log_total_loss = {}
    log_total_loss['loss'] = 0
    iter_report_start = time.time()
    while iter_cnt < cfg.TRAIN.EPOCHES * len(train_data):
        if iter_cnt % cfg.TRAIN.ITER_REPORT == 0:
            log_total_loss = {}
            log_total_loss['loss'] = 0
            iter_report_start = time.time()

        net.train()

        # source data
        try:
            s = next(source_iter)
        except StopIteration:
            source_iter = iter(train_data)
            s = next(source_iter)
            epoch += 1
        images = Variable(s[0].cuda())
        labels = Variable(s[1].type(torch.LongTensor).cuda())
        images_names = s[2]

        # uda target dataset
        if cfg.DATASET.TARGET.ANNS != '':
            try:
                t = next(target_iter)
            except StopIteration:
                target_iter = iter(t_train_data)
                t = next(target_iter)
            t_images = Variable(t[0].cuda())
            t_images_names = t[2]

        if cfg.MODEL.TYPE == "UDA_Segmentor":
            # pdb.set_trace()
            loss_dict = net(
                source=images,
                target=t_images,
                source_label=labels,
                )
        else:
            loss_dict = net(images, labels) 

        if len(loss_dict) > 1:
            for loss_name, loss_value in loss_dict.items():
                log_loss = reduce_tensor(loss_value.clone().detach(), cfg.TRAIN.N_PROC_PER_NODE).item()
                log_total_loss[loss_name] = log_loss if loss_name not in log_total_loss else log_total_loss[loss_name] + log_loss

        loss = sum(torch.mean(loss) for name, loss in loss_dict.items() if "D_" not in name)
            
        log_total_loss['loss'] += reduce_tensor(loss.clone().detach(), cfg.TRAIN.N_PROC_PER_NODE).item()
        
        optimizer.zero_grad()
        with amp.scale_loss(sum(loss) if loss.size() else loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        for name, optim in D_optimizer_dict.items():
            name = 'D_' + name + '_loss'
            if iter_cnt % cfg.MODEL.DISCRIMINATOR.UPDATE_T == 0:
                optim.zero_grad()
                for loss_name in loss_dict.keys():
                    if name in loss_name:
                        with amp.scale_loss(sum(loss_dict[loss_name]) if loss_dict[loss_name].size() else loss_dict[loss_name], optim) as scaled_loss:
                            scaled_loss.backward()
                optim.step()
                
        if cfg.TRAIN.SCHEDULER == "CosineAnnealingLR_with_Restart" or cfg.TRAIN.SCHEDULER == "LambdaLR":
            for scheduler in schedulers:
                scheduler.step()

        iter_cnt += 1
        # print loss
        if gpu == 0 and iter_cnt%cfg.TRAIN.ITER_REPORT == 0:
            # eta 
            iter_report_end = time.time()
            iter_report_time = iter_report_end-iter_report_start
            eta = itv2time(iter_report_time * (expect_iter - iter_cnt) / cfg.TRAIN.ITER_REPORT)
            
            if cfg.MODEL.TYPE == "UDA_Segmentor":
                logger.info('eta: {}, epoch: {}, iter: {} , time: {:.3f} s/iter, lr: {:.2e}, D_lr: {:.2e}'
                        .format(eta, epoch + 1, iter_cnt, iter_report_time/cfg.TRAIN.ITER_REPORT, optimizer.param_groups[-1]['lr'], list(D_optimizer_dict.values())[0].param_groups[-1]['lr'] if D_optimizer_dict else 0 )  + print_loss_dict(log_total_loss, cfg.TRAIN.ITER_REPORT))
            else:
                logger.info('eta: {}, epoch: {}, iter: {}, time: {:.3f} s/iter, lr: {:.2e}'.format(eta, epoch + 1, iter_cnt, iter_report_time/cfg.TRAIN.ITER_REPORT, optimizer.param_groups[-1]['lr']) + print_loss_dict(log_total_loss, cfg.TRAIN.ITER_REPORT))
        
        # val
        if cfg.DATASET.VAL.ANNS != '' and iter_cnt % cfg.TRAIN.ITER_VAL == 0:
            with torch.no_grad():
                net.eval()
                n_class = cfg.MODEL.PREDICTOR.NUM_CLASSES
                intersection_sum = 0
                union_sum = 0
                for i, b in enumerate(val_data):
                    images = b[0].cuda(non_blocking=True)
                    labels = b[1].type(torch.LongTensor).cuda(non_blocking=True)

                    tmp_images = F.interpolate(images, val_resize_size[::-1], mode='bilinear', align_corners=True)
                    logits = net(tmp_images)
                    logits = F.interpolate(logits, labels.size()[1:], mode='bilinear', align_corners=True)

                    label_pred = logits.max(dim=1)[1]
                    
                    intersection, union = intersectionAndUnionGPU(label_pred, labels, n_class)
                    intersection_sum += intersection
                    union_sum += union
                
                dist.all_reduce(intersection_sum), dist.all_reduce(union_sum)
                intersection_sum = intersection_sum.cpu().numpy()
                union_sum = union_sum.cpu().numpy()

                if gpu == 0:
                    iu = intersection_sum / (union_sum + 1e-10)
                    mean_iu = np.mean(iu)

                    result_item = {'epoch': epoch+1}
                    result_item.update({'iou': mean_iu})
                    result_item.update(result_list2dict(iu,'iou'))
                    result.append(result_item)
                    logger.info('epoch: {}, val_miou: {:.4f}({:.4f})'.format(epoch + 1, mean_iu, print_top(result, 'iou')) + print_iou_list(iu))
                    
                    # early stopping
                    if mean_iu >= max_metrics:
                        max_metrics = mean_iu
                        max_metrics_iter_cnt = "epoch{}_{}".format(epoch+1, iter_cnt)
                        metrics_decay_count = 0
                    else:
                        metrics_decay_count += 1
                    if metrics_decay_count > early_stopping and early_stopping >= 0:
                        logger.info('early stopping! epoch{} max metrics: {:.4f}'.format(max_metrics_iter_cnt, max_metrics))
                        break
        
            # save
            if gpu == 0:    
                # save model
                if cfg.TRAIN.SAVE_ALL:
                    torch.save(net.state_dict(), os.path.join(dir_cp, 'CP{}_{}.pth'.format(epoch + 1, iter_cnt)))      
                torch.save(net.state_dict(), os.path.join(dir_cp, 'last_iter.pth'))
                torch.save(net.state_dict(), os.path.join(dir_cp, 'epoch_{}.pth'.format(epoch + 1)))
                if max_metrics_iter_cnt == "epoch{}_{}".format(epoch+1, iter_cnt):
                    torch.save(net.state_dict(), os.path.join(dir_cp, 'best_iter.pth'))
                # save state
                state = {
                    'epoch': epoch,
                    'iter': iter_cnt,
                    'result': result,
                    'optimizers': [opt.state_dict() for opt in optimizers],
                    'schedulers': [sch.state_dict() for sch in schedulers]
                }
                torch.save(state, os.path.join(dir_cp, 'state.pth'))

    if gpu == 0:
        logger.info('End! epoch{} max metrics: {:.4f}'.format(max_metrics_epoch, max_metrics))
        
def build_optimizer(model, cfg):
    optimizer = cfg.TRAIN.OPTIMIZER
    lr = cfg.TRAIN.LR
    param = [
        {'params': model.backbone.parameters(), "lr": lr*0.1},
        {'params': model.decoder.parameters(), "lr": lr},
        {'params': model.predictor.parameters(), "lr": lr}
        ]
    if(optimizer == 'SGD'):
        optimizer = optim.SGD(param, momentum=0.9, weight_decay=0.0005)
    elif(optimizer == 'Adam'):
        optimizer = optim.Adam(param, betas=(0.9, 0.999), weight_decay=0.0005)
    elif(optimizer == 'Ranger'):
        optimizer = Ranger(param, weight_decay=0.0005)
    else:
        optimizer = optim.SGD(param, momentum=0.9, weight_decay=0.0005)

    D_optimizer_dict = {}
    if len(cfg.MODEL.DISCRIMINATOR.TYPE)>0:
        d_params = {}
        for d_name, D in model.discriminators.named_children():
            d_params[d_name] = D.parameters()

        for i, d_name in enumerate(cfg.MODEL.DISCRIMINATOR.TYPE):
            D_optimizer_dict[d_name] = optim.Adam(d_params[d_name], lr=cfg.MODEL.DISCRIMINATOR.LR[i], betas=(0.9, 0.999))
        
    return optimizer, D_optimizer_dict


def print_loss_dict(loss_dict, iter_cnt):
    res = ''
    for loss_name, loss_value in loss_dict.items():
        res += ', {}: {:.6f}'.format(loss_name, loss_value/iter_cnt)
    return res

def print_iou_list(iou_list):
    res = ''
    for i, iou in enumerate(iou_list):
        res += ', {}: {:.4f}'.format(i, iou)
    return res

def print_top(result, metrics, top=0.1):
    res = np.array([x[metrics] for x in result])
    res = np.sort(res)
    # top = int(len(res) * 0.1) + 1
    top = 1
    return res[-top:].mean()

def result_list2dict(iou_list, metrics):
    res = {}
    for i, iou in enumerate(iou_list):
        res[metrics+str(i)] = iou
    return res

def itv2time(iItv):
    h = int(iItv//3600)
    sUp_h = iItv-3600*h
    m = int(sUp_h//60)
    sUp_m = sUp_h-60*m
    s = int(sUp_m)
    return "{}h {:0>2d}min".format(h,m,s)

def reduce_tensor(tensor, world_size=1):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor