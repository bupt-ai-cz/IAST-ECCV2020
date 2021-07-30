import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler, RandomSampler
from torch.autograd import Variable

from ..datasets.loader.dataset import BaseDataset
from ..datasets.loader.gtav_dataset import GTAVDataset
from ..datasets.loader.cityscapes_dataset import CityscapesDataset
from ..datasets.loader.bdd_dataset import BDDDataset

from ..datasets.metrics.miou import intersectionAndUnionGPU
from ..datasets.metrics.acc import acc, acc_with_hist
from ..models.losses.ranger import Ranger
from ..models.losses.cos_annealing_with_restart import CosineAnnealingLR_with_Restart
from ..models.registry import DATASET

import os
import time
import numpy as np
import tqdm
import pdb

def eval_net(net, cfg, gpu):
    
    # train dataset
    result = []
    early_stopping = cfg.TRAIN.EARLY_STOPPING
    anns = cfg.DATASET.ANNS
    image_dir = cfg.DATASET.IMAGEDIR
    use_aug = cfg.DATASET.USE_AUG
    scales = cfg.TEST.RESIZE_SIZE
    bs = cfg.TEST.BATCH_SIZE
    num_work = cfg.TEST.NUM_WORKER
    use_flip = cfg.TEST.USE_FLIP

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
    
    # val dataset  
    val_anns = cfg.DATASET.VAL.ANNS
    val_image_dir = cfg.DATASET.VAL.IMAGEDIR
    val = DATASET[cfg.DATASET.VAL.TYPE](val_anns, val_image_dir)
    val_sampler = DistributedSampler(val, num_replicas=cfg.TEST.N_PROC_PER_NODE, rank=gpu)
    val_data = DataLoader(val, bs, num_workers=num_work, sampler=val_sampler)

    if gpu == 0:
        print('Eval Size: {}'.format(scales))
        print('Use Flip: {}'.format(use_flip))
    
    with torch.no_grad():
        net.eval()
        n_class = cfg.MODEL.PREDICTOR.NUM_CLASSES
        intersection_sum = 0
        union_sum = 0

        if gpu == 0:
            pbar = tqdm.tqdm(total=len(val_data))

        for i, b in enumerate(val_data):
            if gpu == 0 :
                pbar.update(1)
            images = b[0].cuda(non_blocking=True)
            labels = b[1].type(torch.LongTensor).cuda(non_blocking=True)

            pred_result = []
            for scale in scales:
                tmp_images = F.interpolate(images, scale[::-1], mode='bilinear', align_corners=True)
                logits = F.softmax(net(tmp_images), dim=1)

                if use_flip:
                    flip_logits = F.softmax(net(torch.flip(tmp_images, dims=[3])), dim=1)
                    logits += torch.flip(flip_logits, dims=[3])

                logits = F.interpolate(logits, labels.size()[1:], mode='bilinear', align_corners=True)
                pred_result.append(logits)
            result = sum(pred_result)

            label_pred = result.max(dim=1)[1]

            intersection, union = intersectionAndUnionGPU(label_pred, labels, n_class)
            intersection_sum += intersection
            union_sum += union
        
        if gpu == 0:
            pbar.close()
            

        dist.all_reduce(intersection_sum), dist.all_reduce(union_sum)
        intersection_sum = intersection_sum.cpu().numpy()
        union_sum = union_sum.cpu().numpy()

        if gpu == 0:
            iu = intersection_sum / (union_sum + 1e-10)
            mean_iu = np.mean(iu)
            print('val_miou: {:.4f}'.format(mean_iu) + print_iou_list(iu))


def print_iou_list(iou_list):
    res = ''
    for i, iou in enumerate(iou_list):
        res += ', {}: {:.4f}'.format(i, iou)
    return res

