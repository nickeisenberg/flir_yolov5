import os
import cv2
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image

file_name = os.path.expanduser(
    "~/Datasets/flir/images_thermal_train/data/video-24ysbPEGoEKKDvRt6-frame-000000-4C4FHWxwNaMyohLZt.jpg"
)

image = np.array(Image.open(file_name).convert("L"))

scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(640 * scale)),
        A.PadIfNeeded(
            min_height=int(512 * scale),
            min_width=int(640 * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=640, height=512),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.Normalize(mean=[0], std=[1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="coco", min_visibility=0.4, label_fields=["labels"],
    ),
)

val_transforms = A.Compose(
    [
        A.Normalize(mean=[0], std=[1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="coco", min_visibility=0.4, label_fields=["labels"],
    ),
)

t = train_transforms(
    image=image, 
    bboxes=[
        [1, 1, 10 , 10], 
        [10, 11, 321, 11]
    ], 
    labels=[0, 1]
)

t["image"]

t["labels"]

t["bboxes"]
