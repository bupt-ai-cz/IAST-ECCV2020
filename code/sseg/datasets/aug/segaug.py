from albumentations import (
    HorizontalFlip,
    RandomSizedCrop,
    RandomBrightnessContrast,   
    Compose, 
)


def seg_aug(image, mask):
    aug = Compose([
              HorizontalFlip(p=0.5),
              RandomBrightnessContrast(p=0.3),
              ])

    augmented = aug(image=image, mask=mask)
    return augmented


def crop_aug(image, mask, h, w, min_max_height, w2h_ratio=2):
    aug = Compose([
              HorizontalFlip(p=0.5),
              RandomBrightnessContrast(p=0.3),              
              RandomSizedCrop(height=h, width=w, min_max_height=min_max_height, w2h_ratio=2),
              ])

    augmented = aug(image=image, mask=mask)
    return augmented