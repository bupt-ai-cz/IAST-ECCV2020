import numpy as np
import torch
from PIL import Image
import cv2
import pdb
import traceback

from torch.utils.data import Dataset
from torchvision import transforms
from .utils import *
from ..aug.segaug import seg_aug, crop_aug
from ...models.registry import DATASET

@DATASET.register("BaseDataset")
class BaseDataset(Dataset):
    '''
    Arguments:
        anns(String): annotation file path
        image_dir(String): images dir
        scale(float): scale factor of the resized images
        resize_size(list): resize images to [H, W], *scale will be invalid
        center_crop(float): `new_H = H / center_crop`  `new_W = W / center_crop`
    '''
    def __init__(self, anns, image_dir, scale=1, resize_size=None, center_crop=1, use_aug=False, num_classes=19, pseudo_dir=None):
        self.image_list, self.label_list = read_anns(anns, image_dir)
        self.transforms = trans
        self.scale = scale
        self.normalize = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }          
        self.center_crop = center_crop
        self.resize_size = resize_size
        self.usd_aug = use_aug
        self.num_classes = num_classes
        self.pseudo_dir = pseudo_dir
        if not (len(self.image_list) == len(self.label_list)):
            print('Warning: images and labels wrong')
        
    def __getitem__(self, idx):
        im = self.image_list[idx]
        label = self.label_list[idx]

        # pseodo_label
        if self.pseudo_dir:
            pseodo_label_name = os.path.splitext(os.path.basename(im))[0] + '_pseudo_label.png'
            pseudo_label_path = os.path.join(self.pseudo_dir, pseodo_label_name)
            label = pseudo_label_path
        try:
            im = Image.open(im)
            label = Image.open(label)
            if self.scale != 1 or self.resize_size or self.center_crop!=1:
                im = np.array(
                    resize_img(
                        img_pil=crop_img(im, self.center_crop), 
                        scale=self.scale, 
                        type='image', 
                        resize_size=self.resize_size
                        ), 
                    dtype=np.uint8)
                    # )  else np.array(im, dtype=np.uint8)
                label = self.transform_mask(
                    np.array(
                        resize_img(
                            img_pil=crop_img(label, self.center_crop), 
                            scale=self.scale, 
                            type='label',  
                            resize_size=self.resize_size
                            ), 
                        dtype=np.uint8
                        ))
            else:
                size = im.size
                im = np.array(im, dtype=np.uint8)
                label = self.transform_mask(np.array(label.resize(size, Image.NEAREST), dtype=np.uint8))
                # ) if self.scale != 1 or self.resize_size or self.center_crop!=1 else np.array(label, dtype=np.uint8)
            if self.usd_aug:
                augmented = self.aug(im, label)
                im = augmented['image']
                label = augmented['mask']
            im, label = self.transforms(im, label, self.normalize)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print('---------------------')
            print(self.image_list[idx])
            print('---------------------')
            # traceback.print_exc()
            idx = idx - 1 if idx > 0 else idx + 1 
            return self.__getitem__(idx)
        
        return im, label, self.image_list[idx]
        
    def transform_mask(self, label, threshold=128):
        '''
        input(numpy.narray): shape [H, W]
            if class number > 2, value means the class id
            if class number = 2, 0: background, 1: target
        threshold(int): < threshold background, > threshold target
        '''
        res = label[:,:,0] if len(label.shape) == 3 else label
        if label.max()<threshold:
            return res
        else:
            res[res <= threshold] = 0
            res[res > threshold] = 1
        return res

    def aug(self, image, label):
        return seg_aug(image, label)
    
    def __len__(self):
        return len(self.image_list)


