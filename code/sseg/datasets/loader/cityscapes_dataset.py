import numpy as np
from .dataset import BaseDataset
from ..aug.segaug import crop_aug
from ...models.registry import DATASET

@DATASET.register("CityscapesDataset")
class CityscapesDataset(BaseDataset):
    # overwrite
    def transform_mask(self, label):
        return label
    
    # overwrite
    def aug(self, image, label):
        return crop_aug(image, label, 512, 1024, min_max_height=(900, 1000), w2h_ratio=2)


@DATASET.register("MsCityscapesDataset")
class MsCityscapesDataset(BaseDataset):
    # overwrite
    def transform_mask(self, label):
        return label
    
    # overwrite
    def aug(self, image, label):
        return crop_aug(image, label, 512, 1024, min_max_height=(341, 1000), w2h_ratio=2)
