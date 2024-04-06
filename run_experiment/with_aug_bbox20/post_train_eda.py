import os
from sys import path
path.append(os.path.expanduser("~/GitRepos/flir_yolov5"))
import torch
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from src.yolo_utils.targets import decode_yolo_tuple
from src.yolo_utils.box_viewers import (
    view_pred_vs_actual,
)
from src.yolov5.yolov5 import YOLOv5

from run_experiment.with_aug_bbox20.config import (
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
    os.path.join(state_dict_root, "val_ckp.pth"),
    map_location="cpu"
)

yolov5.load_state_dict(sd["MODEL_STATE"])


 #img, target = vdataset[225]
# img, target = vdataset[45]
# img, target = vdataset[105]
# img, target = vdataset[118]
img, target = vdataset[600]
# img, target = vdataset[701]
img = img.unsqueeze(0)
prediction = yolov5(img)

decoded_prediction = decode_yolo_tuple(
    yolo_tuple=prediction, 
    img_width=img_width, 
    img_height=img_height, 
    normalized_anchors=anchors, 
    scales=scales, 
    score_thresh=.95,
    iou_thresh=.3,
    is_pred=True
)

actual = decode_yolo_tuple(
    yolo_tuple=tuple(target), 
    img_width=img_width, 
    img_height=img_height, 
    normalized_anchors=anchors, 
    scales=scales, 
    is_pred=False
)

pil_img: Image.Image = transforms.ToPILImage()(img[0])

view_pred_vs_actual(
    pil_img, 
    boxes=decoded_prediction["boxes"], 
    scores=decoded_prediction["scores"], 
    labels=decoded_prediction["labels"], 
    boxes_actual=actual["boxes"], 
    labels_actual=actual["labels"]
)


loss_df = pd.read_csv(os.path.join(loss_log_root, "train_log.csv"))
loss_df["batch"] = loss_df.index // (272 * 3)
loss_df = loss_df.groupby("batch").mean()
loss_df.plot()
plt.show()

loss_df = pd.read_csv(os.path.join(loss_log_root, "val_log.csv"))
loss_df["batch"] = loss_df.index // (30 * 3)
loss_df = loss_df.groupby("batch").mean()
loss_df.plot()
plt.show()
