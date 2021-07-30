from yacs.config import CfgNode as CN
 
_C = CN()

_C.WORK_DIR = "" # save model chechkpoint and traning log to work_dir

# Random seed
_C.RANDOM_SEED = 888

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'Generalized_Segmentor' # "Generalized_Segmentor", "UDA_Segmentor"

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.TYPE = "R-50-C1-C5" # backbone: ResNet or EfficientNet
_C.MODEL.BACKBONE.PRETRAINED = True
_C.MODEL.BACKBONE.WITH_IBN = False # IBN: only ResNet

_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.TYPE = "DeepLabV2Dedoder"

_C.MODEL.PREDICTOR = CN()
_C.MODEL.PREDICTOR.TYPE = "BasePredictor"
_C.MODEL.PREDICTOR.LOSS = "CrossEntropy"
_C.MODEL.PREDICTOR.NUM_CLASSES = 19 # Cityscapes 19

_C.MODEL.DISCRIMINATOR = CN()
_C.MODEL.DISCRIMINATOR.LOSS = "MSELoss"
_C.MODEL.DISCRIMINATOR.TYPE = []
_C.MODEL.DISCRIMINATOR.WEIGHT = []
_C.MODEL.DISCRIMINATOR.LR = []
_C.MODEL.DISCRIMINATOR.UPDATE_T = 1.0
_C.MODEL.DISCRIMINATOR.LAMBDA_ENTROPY_WEIGHT = .0
_C.MODEL.DISCRIMINATOR.LAMBDA_KLDREG_WEIGHT = .0

# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.ITER_VAL = 2000
_C.TRAIN.EPOCHES = 10
_C.TRAIN.OPTIMIZER = "SGD"
_C.TRAIN.BATCHSIZE = 2
_C.TRAIN.ITER_REPORT = 50
_C.TRAIN.LR = 0.001
_C.TRAIN.N_PROC_PER_NODE = 1

_C.TRAIN.APEX_OPT = 'O1' # Apex option 'O0'/'O1'/'O2'/'O3'
_C.TRAIN.EARLY_STOPPING = -1
_C.TRAIN.SAVE_ALL = False
_C.TRAIN.RESUME_FROM = '' # for continuing training
_C.TRAIN.PSEUDO_RESUME_FROM = '' # for generating pseudo-labels

_C.TRAIN.SCHEDULER = "" # "CosineAnnealingLR_with_Restart" / "LambdaLR" / ""

# CosineAnnealingLR_with_Restart
_C.TRAIN.COSINEANNEALINGLR = CN()
_C.TRAIN.COSINEANNEALINGLR.T_MAX = 1
_C.TRAIN.COSINEANNEALINGLR.T_MULT = 1.0





# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = ''
_C.DATASET.ANNS = ''
_C.DATASET.IMAGEDIR = ''
_C.DATASET.RESIZE_SIZE = [] # W H
_C.DATASET.USE_AUG = False
_C.DATASET.NUM_WORKER = 2

_C.DATASET.VAL = CN()
_C.DATASET.VAL.TYPE = ''
_C.DATASET.VAL.ANNS = ''
_C.DATASET.VAL.IMAGEDIR = ''
_C.DATASET.VAL.RESIZE_SIZE = [] # W H
_C.DATASET.VAL.ORIGIN_SIZE = [2048, 1024] # W H

_C.DATASET.TARGET = CN()
_C.DATASET.TARGET.TYPE = ''
_C.DATASET.TARGET.ANNS = ''
_C.DATASET.TARGET.IMAGEDIR = ''
_C.DATASET.TARGET.PSEUDO_LOSS_WEIGHT = 1.0
_C.DATASET.TARGET.SOURCE_LOSS_WEIGHT = 1.0
_C.DATASET.TARGET.SKIP_GEN_PSEUDO = False
_C.DATASET.TARGET.PSEUDO_SAVE_DIR = ''

_C.DATASET.TARGET.PSEUDO_PL = "IAST"
_C.DATASET.TARGET.PSEUDO_PL_GAMMA = 1.0
_C.DATASET.TARGET.PSEUDO_PL_BETA = 0.9
_C.DATASET.TARGET.PSEUDO_PL_ALPHA = 0.2
_C.DATASET.TARGET.PSEUDO_BATCH_SIZE = 2


_C.DATASET.TARGET.PSEUDO_SIZE = [1280, 640] # WH
_C.DATASET.TARGET.ORIGIN_SIZE = [2048, 1024] # WH

# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.RESIZE_SIZE = [] # multi scale test eg. [[1024, 512], [1536, 768], [2048, 1024]]
_C.TEST.USE_FLIP = False
_C.TEST.BATCH_SIZE = 2
_C.TEST.NUM_WORKER = 2
_C.TEST.N_PROC_PER_NODE = 1

cfg = _C

# test
if __name__ == "__main__":
    cfg.merge_from_file("config/test.yaml")
    cfg.freeze()
 
    cfg2 = cfg.clone()
    cfg2.defrost()
    cfg2.MODEL.PREDICTOR.NUM_CLASSES = 8
    cfg2.freeze()
 
    print("cfg:")
    print(cfg)
    print("cfg2:")
    print(cfg2)
