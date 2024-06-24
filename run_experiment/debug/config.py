import os
import json

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import DataLoader

from src.yolo_utils.dataset import YoloDataset2, yolo_unpacker
from src.yolo_utils.utils import make_yolo_anchors
from src.yolo_utils.coco_transformer import coco_transformer
from src.yolov5.yolov5 import YOLOv5
from src.yolov5.train_module import TrainModule


def config_coco():
    flir_root = os.path.expanduser("~/datasets/flir")

    path = os.path.join(flir_root, "images_thermal_train", "coco.json")
    with open(path, "r") as oj:
        coco = json.load(oj)

    instructions = {}
    for cat in coco["categories"]:
        name = cat["name"]
        if name not in [
            'car', 
            'truck', 
        ]:
            instructions[name] = "ignore"
    
    tcoco = coco_transformer(
        coco, instructions, (25, 640), (25, 512), (0, 640), (0, 512)
    )
    return tcoco


def config_train_module_inputs(coco):
    save_root = os.path.relpath(__file__)
    save_root = save_root.split(os.path.basename(save_root))[0]
    loss_log_root = os.path.join(save_root, "loss_logs")
    state_dict_root = os.path.join(save_root, "state_dicts")
    if not os.path.isdir(loss_log_root):
        os.makedirs(loss_log_root)
    if not os.path.isdir(state_dict_root):
        os.makedirs(state_dict_root)
    in_channels = 1
    num_classes = 79 + 1
    img_width = 640
    img_height = 512
    anchors = make_yolo_anchors(coco, img_width, img_height, 9)
    scales = [32, 16, 8]
    return [in_channels, num_classes, img_width, img_height, anchors, scales,
            loss_log_root, state_dict_root]


def config_datasets(tcoco, anchors, scales):
    return_shape = (
        (3, 16, 20, 6),
        (3, 32, 40, 6),
        (3, 64, 80, 6),
    )

    train_transform = A.Compose(
        [
            A.Normalize(mean=[0], std=[1], max_pixel_value=255,),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco", min_visibility=0.4, label_fields=["labels"],
        ),
    )
    img_root = os.path.expanduser("~/datasets/flir/images_thermal_train/")
    tdataset = YoloDataset2(
        coco=tcoco, 
        img_root=img_root,
        return_shape=return_shape,
        normalized_anchors=anchors,
        scales=scales,
        transform=train_transform
    )

    tdataset.data = {idx: tdataset.data[idx] for idx in range(10)}

    return tdataset


def config_trainer():
    tcoco = config_coco()

    (
        in_channels, 
        num_classes, 
        img_width, 
        img_height, 
        anchors, 
        scales,
        loss_log_root,
        state_dict_root
    ) = config_train_module_inputs(tcoco)

    t_dataset = config_datasets(tcoco, anchors, scales)

    train_loader = DataLoader(t_dataset, 1)

    device = 0

    train_module = TrainModule(
        yolo=YOLOv5(in_channels, num_classes),
        device=device,
        img_width=img_width, 
        img_height=img_height, 
        normalized_anchors=anchors, 
        scales=scales,
        loss_log_root=loss_log_root,
        state_dict_root=state_dict_root
    )

    num_epochs = 200

    config = {
        "train_module": train_module,
        "device": device,
        "train_loader": train_loader,
        "val_loader": train_loader,
        "num_epochs": num_epochs,
        "unpacker": yolo_unpacker
    }

    return config
