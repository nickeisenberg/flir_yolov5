import os
from sys import path
path.append(os.path.expanduser("~/GitRepos/flir_yolov5"))

from torch.utils.data import DataLoader

from src.yolov5.yolov5 import YOLOv5
from src.yolo_utils.dataset import yolo_unpacker
from src.yolov5.train_module import TrainModule
from run_experiment.with_aug_bbox10_shuf.config import (
    config_coco,
    config_datasets,
    config_train_module_inputs,
)

tcoco, vcoco = config_coco()

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

yolo = YOLOv5(in_channels, num_classes)
train_module = TrainModule(
    yolo=yolo,
    device="cuda",
    img_width=img_width, 
    img_height=img_height, 
    normalized_anchors=anchors, 
    scales=scales,
)

sd_path = os.path.expanduser(
    "~/GitRepos/flir_yolov5/run_experiment/with_aug_bbox10_shuf/state_dicts/val_ckp.pth"
)
train_module.load_checkpoint(sd_path)

tdataset, vdataset = config_datasets(tcoco, vcoco, anchors, scales)

tdataloader = DataLoader(tdataset, 64)
vdataloader = DataLoader(vdataset, 64)

map = train_module.map_evaluate(
    vdataloader,
    yolo_unpacker,
    min_box_dim=(10, 10)
)
