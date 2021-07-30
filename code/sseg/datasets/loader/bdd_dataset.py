import numpy as np
from .dataset import BaseDataset
from ..aug.segaug import crop_aug
from ...models.registry import DATASET

@DATASET.register("BDDDataset")
class BDDDataset(BaseDataset):
    # overwrite
    def transform_mask(self, label):
        return label
    
    # overwrite
    def aug(self, image, label):
        return crop_aug(image, label, 512, 1024, min_max_height=(620, 640), w2h_ratio=2)


@DATASET.register("MsBDDDataset")
class MsBDDDataset(BaseDataset):
    # overwrite
    def transform_mask(self, label):
        return label
    
    # overwrite
    def aug(self, image, label):
        return crop_aug(image, label, 512, 1024, min_max_height=(341, 640), w2h_ratio=2)

