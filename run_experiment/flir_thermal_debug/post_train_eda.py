import os
from sys import path
path.append(os.path.expanduser("~/GitRepos/flir_yolov5"))
import torch
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from src.yolo_utils.targets import decode_yolo_output
from src.yolo_utils.box_viewers import (
    view_boxes,
    view_boxes_from_coco_image_id
)
from src.yolov5.yolov5 import YOLOv5

from run_experiment.flir_thermal_debug.config import (
    config_coco,
    config_datasets,
    config_train_module_inputs
)

coco = config_coco()

(
    in_channels, 
    num_classes, 
    img_width, 
    img_height, 
    anchors, 
    scales,
    loss_log_root,
    state_dict_root
) = config_train_module_inputs(coco)

dataset = config_datasets(coco, anchors, scales)

yolov5 = YOLOv5(in_channels, num_classes)

sd = torch.load(
    os.path.join(state_dict_root, "train_ckp.pth"),
    map_location="cpu"
)

sd["EPOCHS_RUN"]

yolov5.load_state_dict(sd["MODEL_STATE"])

img, target = dataset[3]
img = img.unsqueeze(0)
prediction = yolov5(img)
decoded_prediction = decode_yolo_output(
    prediction, img_width, img_height, .7, anchors, scales, True
)
pil_img: Image.Image = transforms.ToPILImage()(img[0])
view_boxes(pil_img, decoded_prediction["bboxes"])

actual = decode_yolo_output(
    tuple(target), img_width, img_height, .97, anchors, scales, False
)
view_boxes(pil_img, actual["bboxes"])

loss_df = pd.read_csv(os.path.join(loss_log_root, "train_log.csv"))
loss_df["batch"] = loss_df.index // 30
loss_df = loss_df.groupby("batch").mean()
loss_df = loss_df.drop("batch", axis=1)
loss_df.plot()
plt.show()
