import os
from sys import path
path.append(os.path.expanduser("~/GitRepos/flir_yolov5"))
import torch
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from src.yolo_utils.targets import decode_yolo_output
from src.yolo_utils.utils import nms, iou
from src.yolo_utils.box_viewers import (
    view_boxes_actual,
)
from src.yolov5.yolov5 import YOLOv5

from run_experiment.car_and_others.config import (
    config_coco,
    config_datasets,
    config_train_module_inputs
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

tdataset, vdataset = config_datasets(tcoco, vcoco, anchors, scales)

yolov5 = YOLOv5(in_channels, num_classes)

sd = torch.load(
    os.path.join(state_dict_root +'.2', "val_ckp.pth"),
    map_location="cpu"
)

yolov5.load_state_dict(sd["MODEL_STATE"])

img, target = vdataset[225]
img = img.unsqueeze(0)
prediction = yolov5(img)
pil_img: Image.Image = transforms.ToPILImage()(img[0])

decoded_prediction = decode_yolo_output(
    prediction, img_width, img_height, .95, anchors, scales, True
)
actual = decode_yolo_output(
    tuple(target), img_width, img_height, .95, anchors, scales, False
)

pred_box_idxs = nms(decoded_prediction["bboxes"], .3)
pred_bboxes = decoded_prediction["bboxes"][pred_box_idxs]
pred_category_ids = decoded_prediction["category_ids"][pred_box_idxs]

actual_bboxes = actual["bboxes"]
actual_category_ids = actual["category_ids"]

view_boxes_actual(pil_img, pred_bboxes, actual["bboxes"])


