import os
from sys import path

from torch.utils.data import DataLoader
path.append(os.path.expanduser("~/GitRepos/flir_yolov5"))

from PIL import Image
import torchvision.transforms as transforms

from src.yolov5.yolov5 import YOLOv5
from src.yolo_utils.targets import decode_yolo_tuple
from src.yolo_utils.box_viewers import view_pred_vs_actual 
from src.yolo_utils.dataset import yolo_unpacker
from src.yolov5.train_module import TrainModule
from run_experiment.with_aug_bbox20.config import (
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

_, vdataset = config_datasets(tcoco, vcoco, anchors, scales)


#--------------------------------------------------

yolo = YOLOv5(in_channels, num_classes)
train_module = TrainModule(
    yolo=yolo,
    device="cpu",
    img_width=img_width, 
    img_height=img_height, 
    normalized_anchors=anchors, 
    scales=scales,
)

sd_path = os.path.expanduser(
    "~/GitRepos/flir_yolov5/run_experiment/with_aug_bbox20/state_dicts/val_ckp.pth"
)
train_module.load_checkpoint(sd_path)

img, target = vdataset[701]
img = img.unsqueeze(0)
target = tuple([t.unsqueeze(0) for t in target])
prediction = train_module.model(img)

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
    yolo_tuple=target, 
    img_width=img_width, 
    img_height=img_height, 
    normalized_anchors=anchors, 
    scales=scales, 
    is_pred=False
)
pil_img: Image.Image = transforms.ToPILImage()(img[0])
view_pred_vs_actual(
    pil_img, 
    boxes=decoded_prediction[0]["boxes"], 
    scores=decoded_prediction[0]["scores"], 
    labels=decoded_prediction[0]["labels"], 
    boxes_actual=actual[0]["boxes"], 
    labels_actual=actual[0]["labels"]
)
