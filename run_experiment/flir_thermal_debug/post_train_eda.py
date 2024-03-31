import os
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
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
    config_some_hyperparams
)

exp_root = os.path.expanduser(
    "~/GitRepos/flir_yolov5/run_experiment/flir_thermal_debug"
)

coco = config_coco()

(
    in_channels, 
    num_classes, 
    img_width, 
    img_height, 
    anchors, 
    scales
) = config_some_hyperparams(coco)

dataset = config_datasets(coco, anchors, scales)

yolov5 = YOLOv5(in_channels, num_classes)

sd = torch.load(
    os.path.join(exp_root, "state_dicts", "val_ep_22.pth"),
    map_location="cpu"
)

yolov5.load_state_dict(sd)


img, target = dataset[3]
img = img.unsqueeze(0)

prediction = yolov5(img)
decoded_prediction = decode_yolo_output(
    prediction, img_width, img_height, .99, anchors, scales, True
)
pil_img: Image.Image = transforms.ToPILImage()(img[0])
view_boxes(pil_img, decoded_prediction["bboxes"])

actual = decode_yolo_output(
    tuple(target), img_width, img_height, .97, anchors, scales, False
)
view_boxes(pil_img, actual["bboxes"])
