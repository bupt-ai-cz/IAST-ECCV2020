import numpy as np
import imageio

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
from .dataset import BaseDataset


@DATASET.register("SYNTHIADataset")
class SYNTHIADataset(BaseDataset):
    # overwrite
    def transform_mask(self, label):
        id_map = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                              15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                              8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in id_map.items():
            label_copy[label == k] = v
        return label_copy

    # overwrite
    def aug(self, image, label):
        return crop_aug(image, label, 512, 1024, min_max_height=(341, 640), w2h_ratio=2)

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
            label = np.asarray(imageio.imread(label, format='PNG-FI'))[:,:,0]
            label = Image.fromarray(label)
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
            print('---------------------')
            print(self.image_list[idx])
            print('---------------------')
            traceback.print_exc()
            idx = idx - 1 if idx > 0 else idx + 1 
            return self.__getitem__(idx)
        
        return im, label, self.image_list[idx]

@DATASET.register("MsSYNTHIADataset")
class MsSYNTHIADataset(BaseDataset):
    # overwrite
    def transform_mask(self, label):
        id_map = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                              15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                              8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in id_map.items():
            label_copy[label == k] = v
        return label_copy

    # overwrite
    def aug(self, image, label):
        return crop_aug(image, label, 512, 1024, min_max_height=(341, 640), w2h_ratio=2)

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
            label = np.asarray(imageio.imread(label, format='PNG-FI'))[:,:,0]
            label = Image.fromarray(label)
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
            print('---------------------')
            print(self.image_list[idx])
            print('---------------------')
            traceback.print_exc()
            idx = idx - 1 if idx > 0 else idx + 1 
            return self.__getitem__(idx)
        
        return im, label, self.image_list[idx]

