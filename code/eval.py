import torch
import argparse
import os
import pdb
import torch.multiprocessing as mp


from sseg.models.segmentors.generalized_segmentor import GeneralizedSegmentor
from sseg.models.segmentors.uda_segmentor import UDASegmentor
from sseg.models.default import cfg
from sseg.models.backbones import resnet, efficientnet
from sseg.models.decoder import deeplabv2_decoder
from sseg.models.predictor import base_predictor
from sseg.models.losses import mse_loss, bce_loss
from sseg.models.discriminator import base_discriminator
from sseg.workflow.trainer import train_net

from sseg.workflow.eval import eval_net


def main_worker(proc_idx, cfg):
    if cfg.MODEL.TYPE== "Generalized_Segmentor":
        net = GeneralizedSegmentor(cfg)
    elif cfg.MODEL.TYPE== "UDA_Segmentor" or cfg.MODEL.TYPE== "SL_UDA_Segmentor":
        net = UDASegmentor(cfg)
    else:
        raise Exception('error MODEL.TYPE {} !'.format(cfg.MODEL.TYPE))

    # resume main net
    state_dict = None
    last_cp = os.path.join(cfg.WORK_DIR, 'last_epoch.pth')
    resume_cp_path = None
    if cfg.TRAIN.RESUME_FROM != "":
        resume_cp_path = cfg.TRAIN.RESUME_FROM
        state_dict = torch.load(resume_cp_path, map_location=torch.device('cpu'))
    
    if state_dict:
        model_dict = net.state_dict()
        if "module." in list(state_dict.keys())[0]:
            pretrained_dict = {k[7:]: v for k, v in state_dict.items() if k[7:] in model_dict}
        else:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    if proc_idx == 0:
        print('Resume from: {}'.format(resume_cp_path))
    eval_net(net=net, cfg=cfg, gpu=proc_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch sseg")
    parser.add_argument(
        "--config_file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--resume_from")
    parser.add_argument("--gpu_num", type=int, default=1)

    args = parser.parse_args()
    cfg.TRAIN.RESUME_FROM = args.resume_from

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    mp.spawn(main_worker, nprocs=args.gpu_num, args=(cfg,))
    
    

