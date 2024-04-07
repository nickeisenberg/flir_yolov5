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
vdataloader = DataLoader(vdataset, 1)

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

batch = next(iter(vdataloader))
images, targets = yolo_unpacker(batch, "cpu")

prediction = train_module.model(images)
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
    yolo_tuple=targets, 
    img_width=img_width, 
    img_height=img_height, 
    normalized_anchors=anchors, 
    scales=scales, 
    is_pred=False
)

idx = 0
pil_img: Image.Image = transforms.ToPILImage()(images[idx])
view_pred_vs_actual(
    pil_img, 
    boxes=decoded_prediction[idx]["boxes"], 
    scores=decoded_prediction[idx]["scores"], 
    labels=decoded_prediction[idx]["labels"], 
    boxes_actual=actual[idx]["boxes"], 
    labels_actual=actual[idx]["labels"]
)
