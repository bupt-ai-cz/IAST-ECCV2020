import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR
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
import tqdm
import math
import glob
import sys
from PIL import Image


def seed_everything(seed=888):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

def self_train_net(net, net_pseudo, cfg, gpu):
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
    net.to(device)
    net_pseudo.to(device)

    # train dataset
    result = []
    early_stopping = cfg.TRAIN.EARLY_STOPPING
    anns = cfg.DATASET.ANNS
    image_dir = cfg.DATASET.IMAGEDIR
    train_resize_size = cfg.DATASET.RESIZE_SIZE
    val_resize_size = cfg.DATASET.VAL.RESIZE_SIZE
    pseudo_resize_size = cfg.DATASET.TARGET.PSEUDO_SIZE
    origin_size = cfg.DATASET.TARGET.ORIGIN_SIZE
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
        
        # pseudo-label
        t_train_sl = DATASET[cfg.DATASET.TARGET.TYPE](t_anns, t_image_dir, resize_size=pseudo_resize_size,)
        t_train_sampler_sl = DistributedSampler(t_train_sl, num_replicas=cfg.TRAIN.N_PROC_PER_NODE, rank=gpu)
        t_train_data_sl = DataLoader(t_train_sl, cfg.DATASET.TARGET.PSEUDO_BATCH_SIZE, sampler=t_train_sampler_sl, num_workers=cfg.DATASET.NUM_WORKER, pin_memory=True)


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

    # generate pseudo lable
    with torch.no_grad():
        net_pseudo.eval()
        net.eval()
        n_class = cfg.MODEL.PREDICTOR.NUM_CLASSES
        pseudo_save_dir = os.path.join(cfg.WORK_DIR, "pseudo/label")
        if not cfg.DATASET.TARGET.PSEUDO_SAVE_DIR == '':
            pseudo_save_dir = cfg.DATASET.TARGET.PSEUDO_SAVE_DIR
        if not os.path.exists(pseudo_save_dir):
            try:
                os.makedirs(pseudo_save_dir)
            except Exception:
                pass

        # gen pseudo-label
        if cfg.DATASET.TARGET.PSEUDO_PL == "IAST":
            logits_npy_files = []
            cls_thresh = np.ones(n_class)*0.9
            if gpu == 0:
                pbar = tqdm.tqdm(total=len(t_train_data_sl))
            for i, b in enumerate(t_train_data_sl):
                if gpu == 0 :
                    pbar.update(1)
                # pseudo-label files exist
                if cfg.DATASET.TARGET.SKIP_GEN_PSEUDO:
                    break

                images = Variable(b[0].cuda())
                names = b[2]
                logits = nn.Softmax(dim=1)(net_pseudo(images))
                # originsize
                # logits = F.interpolate(logits, size=origin_size[::-1], mode="bilinear", align_corners=True)

                max_items = logits.max(dim=1)
                label_pred = max_items[1].data.cpu().numpy()
                logits_pred = max_items[0].data.cpu().numpy()

                logits_cls_dict = {c: [cls_thresh[c]] for c in range(n_class)}
                for cls in range(n_class):
                    logits_cls_dict[cls].extend(logits_pred[label_pred == cls].astype(np.float16))
            
                # instance adaptive selector
                tmp_cls_thresh = ias_thresh(logits_cls_dict, alpha=cfg.DATASET.TARGET.PSEUDO_PL_ALPHA,  cfg=cfg, w=cls_thresh, gamma=cfg.DATASET.TARGET.PSEUDO_PL_GAMMA)
                beta = cfg.DATASET.TARGET.PSEUDO_PL_BETA
                cls_thresh = beta*cls_thresh + (1-beta)*tmp_cls_thresh 
                cls_thresh[cls_thresh>=1] = 0.999

                np_logits = logits.data.cpu().numpy()
                for _i, name in enumerate(names):
                    name = os.path.splitext(os.path.basename(name))[0] 
                    # save pseudo label
                    logit = np_logits[_i].transpose(1,2,0)
                    label = np.argmax(logit, axis=2)
                    logit_amax = np.amax(logit, axis=2)
                    label_cls_thresh = np.apply_along_axis(lambda x: [cls_thresh[e] for e in x], 1, label)
                    ignore_index = logit_amax < label_cls_thresh
                    label[ignore_index] = 255
                    pseudo_label_name = name + '_pseudo_label.png'
                    pseudo_color_label_name = name + '_pseudo_color_label.png'
                    pseudo_label_path = os.path.join(pseudo_save_dir, pseudo_label_name)
                    pseudo_color_label_path = os.path.join(pseudo_save_dir, pseudo_color_label_name)
                    Image.fromarray(label.astype(np.uint8)).convert('P').save(pseudo_label_path)
                    # colorize_mask(label).save(pseudo_color_label_path)
            if gpu==0:
                pbar.close()

    # target dataset with pseudo-label
        t_train = DATASET[cfg.DATASET.TARGET.TYPE](t_anns, t_image_dir, use_aug=use_aug, pseudo_dir=pseudo_save_dir)
        t_train_sampler = DistributedSampler(t_train, num_replicas=cfg.TRAIN.N_PROC_PER_NODE, rank=gpu)
        t_train_data = DataLoader(t_train, cfg.TRAIN.BATCHSIZE, sampler=t_train_sampler, num_workers=cfg.DATASET.NUM_WORKER, drop_last=True, pin_memory=True)
        target_iter = iter(t_train_data)    

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
            t_labels = Variable(t[1].type(torch.LongTensor).cuda())
            t_images_names = t[2]

        loss_dict = net(
                    source=images,
                    target=t_images,
                    source_label=labels,
                    target_label=t_labels,
                    )

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

def ias_thresh(conf_dict, cfg, alpha, w=None, gamma=1.0):
    if w is None:
        w = np.ones(cfg.MODEL.PREDICTOR.NUM_CLASSES)
    # threshold 
    cls_thresh = np.ones(cfg.MODEL.PREDICTOR.NUM_CLASSES,dtype = np.float32)
    for idx_cls in np.arange(0, cfg.MODEL.PREDICTOR.NUM_CLASSES):
        if conf_dict[idx_cls] != None:
            arr = np.array(conf_dict[idx_cls])
            cls_thresh[idx_cls] = np.percentile(arr, 100 * (1 - alpha * w[idx_cls] ** gamma))
    return cls_thresh

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

def colorize_mask(mask):
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def reduce_tensor(tensor, world_size=1):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor