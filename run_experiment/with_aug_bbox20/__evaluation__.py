import os
from sys import path
import torch
from copy import deepcopy

from torch.utils.data import DataLoader
path.append(os.path.expanduser("~/GitRepos/flir_yolov5"))

from PIL import Image
import torchvision.transforms as transforms

from src.yolov5.yolov5 import YOLOv5
from src.yolo_utils.targets import decode_yolo_tuple
from src.yolo_utils.box_viewers import view_pred_vs_actual 
from src.yolo_utils.dataset import yolo_unpacker
from src.yolov5.train_module import TrainModule, TrainModule2
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
sd_path = os.path.expanduser(
    "~/GitRepos/flir_yolov5/run_experiment/with_aug_bbox20/state_dicts/val_ckp.pth"
)
sd = torch.load(sd_path, map_location="cpu")["MODEL_STATE"]

sd_path_cp = os.path.expanduser(
    "~/GitRepos/flir_yolov5/run_experiment/with_aug_bbox20/state_dicts/val_ckp_cp.pth"
)
sd_cp = torch.load(sd_path, map_location="cpu")["MODEL_STATE"]

for key0, key1 in zip(sd.keys(), sd_cp.keys()):
    same = (sd[key0] != sd[key1]).sum()
    if not same == 0:
        print(same)
#--------------------------------------------------

#--------------------------------------------------
yolo_cp = YOLOv5(in_channels, num_classes)
yolo_cp.load_state_dict(sd_cp)

yolo = YOLOv5(in_channels, num_classes)
yolo.load_state_dict(sd)

for key0, key1 in zip(yolo_cp.state_dict().keys(), yolo.state_dict().keys()):
    same = (yolo_cp.state_dict()[key0] != yolo.state_dict()[key1]).sum()
    if not same == 0:
        print(same)

train_module = TrainModule(
    yolo=yolo,
    device="cpu",
    img_width=img_width, 
    img_height=img_height, 
    normalized_anchors=anchors, 
    scales=scales,
)

for key0, key1 in zip(yolo_cp.state_dict().keys(), train_module.model.state_dict().keys()):
    same = (yolo_cp.state_dict()[key0] != yolo.state_dict()[key1]).sum()
    if not same == 0:
        print(same)
#--------------------------------------------------


img, target = vdataset[701]
img = img.unsqueeze(0)
target = tuple([t.unsqueeze(0) for t in target])

with torch.no_grad():
    prediction_tr = train_module.model(img)

with torch.no_grad():
    prediction = yolo(img)

with torch.no_grad():
    prediction_cp = yolo_cp(img)

prediction_ = tuple([deepcopy(x) for x in prediction])

for x, y in zip(prediction_, prediction_tr):
    print((x!=y).sum())

for x, y, z in zip(prediction_tr, prediction, prediction_cp):
    print((x!=y).sum())
    print((x!=z).sum())
    print((y!=z).sum())

decoded_prediction = decode_yolo_tuple(
    yolo_tuple=prediction_, 
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
