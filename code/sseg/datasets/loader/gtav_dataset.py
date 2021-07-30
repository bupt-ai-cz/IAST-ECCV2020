import numpy as np
from .dataset import BaseDataset
from ..aug.segaug import crop_aug
from ...models.registry import DATASET

@DATASET.register("GTAVDataset")
class GTAVDataset(BaseDataset):
    # overwrite
    def transform_mask(self, label):
        id_map = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in id_map.items():
            label_copy[label == k] = v
        return label_copy

    # overwrite
    def aug(self, image, label):
        return crop_aug(image, label, 512, 1024, min_max_height=(800, 900), w2h_ratio=2)

@DATASET.register("MsGTAVDataset")
class MsGTAVDataset(BaseDataset):
    # overwrite
    def transform_mask(self, label):
        id_map = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in id_map.items():
            label_copy[label == k] = v
        return label_copy

    # overwrite
    def aug(self, image, label):
        return crop_aug(image, label, 512, 1024, min_max_height=(341, 950), w2h_ratio=2)

