# IAST: Instance Adaptive Self-training for Unsupervised Domain Adaptation (ECCV 2020)
This repo is the official implementation of our paper ["Instance Adaptive Self-training for Unsupervised Domain Adaptation"](https://arxiv.org/abs/2008.12197)

## Introduction

### Abstract
The divergence between labeled training data and unlabeled testing data is a significant challenge for recent deep learning models. Unsupervised domain adaptation (UDA) attempts to solve such a problem. Recent works show that self-training is a powerful approach to UDA. However, existing methods have difficulty in balancing scalability and performance. In this paper, we propose an instance adaptive self-training framework for UDA on the task of semantic segmentation. To effectively improve the quality of pseudo-labels, we develop a novel pseudo-label generation strategy with an instance adaptive selector. Besides, we propose the region-guided regularization to smooth the pseudo-label region and sharpen the non-pseudo-label region. Our method is so concise and efficient that it is easy to be generalized to other unsupervised domain adaptation methods. Experiments on 'GTA5 to Cityscapes' and 'SYNTHIA to Cityscapes' demonstrate the superior performance of our approach compared with the state-of-the-art methods.

### IAST Overview
![](figs/fig_overview.png)

### Result
| source  | target     | device                | GPU memory | mIoU-19 | mIoU-16 | mIoU-13 | model |
|---------|------------|-----------------------|------------|---------|---------|---------|-------|
| GTA5    | Cityscapes | Tesla V100-32GB       | 18.5 GB    | 51.88   | -       | -       |   [download](https://drive.google.com/file/d/1y_juW7C2HRKUMasXUsDLc3SEtB4pGzDf/view?usp=sharing)    |
| GTA5    | Cityscapes | Tesla T4              | 6.3 GB     | 51.20   | -       | -       |   [download](https://drive.google.com/file/d/1Tl8eMRsYLeTP4OQS9vAEqLpKwfrakOyi/view?usp=sharing)    |
| SYNTHIA | Cityscapes | Tesla V100-32GB       | 18.5 GB    | -       | 51.54   | 57.81   |   [download](https://drive.google.com/file/d/1IkElfEynRJWfJLssA0dM38NVRMufp1fa/view?usp=sharing)    |
| SYNTHIA | Cityscapes | Tesla T4              | 9.8 GB     | -       | 51.24   | 57.70   |   [download](https://drive.google.com/file/d/1A_3Sgo0-CUNrCIledzvhoC74eiCB9NRA/view?usp=sharing)    |


## Setup

### 1) Envs
- Pytorch >= 1.0
- Python >= 3.6
- cuda >= 9.0
 
Install python packages
```
$ pip install -r  requirements.txt
```

`apex` :  Tools for easy mixed precision and distributed training in Pytorch
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### 2) Download Dataset
Please download the datasets from these links:

- [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) 
- [SYNTHIA](https://synthia-dataset.net/)
- [Cityscapes](https://www.cityscapes-dataset.com/)

Dataset directory should have this structure:

```
${ROOT_DIR}/data/GTA5/
${ROOT_DIR}/data/GTA5/images
${ROOT_DIR}/data/GTA5/labels

${ROOT_DIR}/data/SYNTHIA_RAND_CITYSCAPES/RAND_CITYSCAPES
${ROOT_DIR}/data/SYNTHIA_RAND_CITYSCAPES/RAND_CITYSCAPES/RGB
${ROOT_DIR}/data/SYNTHIA_RAND_CITYSCAPES/RAND_CITYSCAPES/GT

${ROOT_DIR}/data/cityscapes
${ROOT_DIR}/data/cityscapes/leftImg8bit
${ROOT_DIR}/data/cityscapes/gtFine
```

### 3) Download Pretrained Models

We provide pre-trained models. We recommend that you download them and put them in `pretrained_models/`, which will save a lot of time for training and ensure consistent results.

V100 models
- GTA5 to Cityscapes: [G_gtav_at_warmup_v100.pth](https://drive.google.com/file/d/17Ajhp73mJ7XYDNnmxgIPSYR-LChuC9vY/view?usp=sharing) and [M_gtav_at_warmup_v100.pth](https://drive.google.com/file/d/1MmruHl_vzu6D7keSJl6pT4y15slZX-ev/view?usp=sharing)
- SYNTHIA to Cityscapes: [G_syn_at_warmup_v100.pth](https://drive.google.com/file/d/1xhwGXUP9sMhh03OY2LVE4jX6t6zje8VI/view?usp=sharing) and [M_syn_at_warmup_v100.pth](https://drive.google.com/file/d/1f-nNpL1Z0sMdCnH-DF159HxNlfhOnAZS/view?usp=sharing)

T4 models
- GTA5 to Cityscapes: [G_gtav_at_warmup_t4.pth](https://drive.google.com/file/d/1J6TbdDaD5gkh68kN_5qDUd1hJ_JhhWTb/view?usp=sharing) and [M_gtav_at_warmup_t4.pth](https://drive.google.com/file/d/1MpgMGQVPM9hdpgeFoBXTg1Ltc5pJHmsS/view?usp=sharing)
- SYNTHIA to Cityscapes: [G_syn_at_warmup_t4.pth](https://drive.google.com/file/d/1-6vsPNOGukg-mxoJLKYQUwoFkFFi8kx4/view?usp=sharing) and [M_syn_at_warmup_t4.pth](https://drive.google.com/file/d/1sB8v1udK3PqSEta9pt9wEHPXy-gxyjDY/view?usp=sharing)

(Optional) Of course, if you have plenty of time, you can skip this step and start training from scratch. We also provide these scripts.

## Training
Our original experiments are all carried out on Tesla-V100, and there will be a large number of GPU memory usage (`batch_size=8`). For low GPU memory devices, we also trained on Tesla-T4 to ensure that most people can reproduce the results (`batch_size=2`).


Start self-training (download the pre-trained models first)

```
cd code

# GTA5 to Cityscapes (V100)
sh ../scripts/self_training_only/run_gtav2cityscapes_self_traing_only_v100.sh
# GTA5 to Cityscapes (T4)
sh ../scripts/self_training_only/run_gtav2cityscapes_self_traing_only_t4.sh
```

```
# SYNTHIA to Cityscapes (V100)
sh ../scripts/self_training_only/run_syn2cityscapes_self_traing_only_v100.sh
# SYNTHIA to Cityscapes (T4)
sh ../scripts/self_training_only/run_syn2cityscapes_self_traing_only_t4.sh
```


(Optional) Training from scratch
```
cd code

# GTA5 to Cityscapes (V100)
sh ../scripts/from_scratch/run_gtav2cityscapes_self_traing_v100.sh
# GTA5 to Cityscapes (T4)
sh ../scripts/from_scratch/run_gtav2cityscapes_self_traing_t4.sh
```

```
# SYNTHIA to Cityscapes (V100)
sh ../scripts/from_scratch/run_syn2cityscapes_self_traing_v100.sh
# SYNTHIA to Cityscapes (T4)
sh ../scripts/from_scratch/run_syn2cityscapes_self_traing_t4.sh
```

## Evaluation

```
cd code
python eval.py --config_file <path_to_config_file> --resume_from <path_to_*.pth_file>
```

Support multi-scale testing and flip testing.
```
# Modify the following parameters in the config file

TEST:
  RESIZE_SIZE: [[1024, 512], [1280, 640], [1536, 768], [1800, 900], [2048, 1024]] 
  USE_FLIP: False 
```

## Citation
Please cite this paper in your publications if it helps your research:

```
@article{mei2020instance,
  title={Instance Adaptive Self-Training for Unsupervised Domain Adaptation},
  author={Mei, Ke and Zhu, Chuang and Zou, Jiaqi and Zhang, Shanghang},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

## Author
Ke Mei
- email: raykoo@bupt.edu.cn
- wechat: meikekekeke

If you have any questions, you can contact me directly.
