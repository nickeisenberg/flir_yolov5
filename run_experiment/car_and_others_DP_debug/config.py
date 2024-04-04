import os
import json

from torch.utils.data import DataLoader

from src.yolo_utils.dataset import YoloDataset, yolo_unpacker
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
        if name not in ['car', 'bus', 'motor', 'other vehicle', 'truck']:
            instructions[name] = "ignore"
        elif name in ['bus', 'motor', 'other vehicle', 'truck']:
            instructions[name] = "vehicle_not_car"
            
    tcoco = coco_transformer(
        coco, instructions, (30, 640), (30, 512), (10, 630), (10, 502)
    )

    path = os.path.join(flir_root, "images_thermal_val", "coco.json")
    with open(path, "r") as oj:
        coco = json.load(oj)

    instructions = {}
    for cat in coco["categories"]:
        name = cat["name"]
        if name not in ['car', 'bus', 'motor', 'other vehicle', 'truck']:
            instructions[name] = "ignore"
        elif name in ['bus', 'motor', 'other vehicle', 'truck']:
            instructions[name] = "vehicle_not_car"
            
    vcoco = coco_transformer(
        coco, instructions, (30, 640), (30, 512), (10, 630), (10, 502)
    )
    return tcoco, vcoco


def config_save_roots():
    save_root = os.path.relpath(__file__)
    save_root = save_root.split(os.path.basename(save_root))[0]
    loss_log_root = os.path.join(save_root, "loss_logs")
    state_dict_root = os.path.join(save_root, "state_dicts")
    if not os.path.isdir(loss_log_root):
        os.makedirs(loss_log_root)
    if not os.path.isdir(state_dict_root):
        os.makedirs(state_dict_root)
    return loss_log_root, state_dict_root


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
    num_classes = 81 + 1
    img_width = 640
    img_height = 512
    anchors = make_yolo_anchors(coco, img_width, img_height, 9)
    scales = [32, 16, 8]
    return [in_channels, num_classes, img_width, img_height, anchors, scales,
            loss_log_root, state_dict_root]


def config_datasets(tcoco, vcoco, anchors, scales):
    return_shape = (
        (3, 16, 20, 6),
        (3, 32, 40, 6),
        (3, 64, 80, 6),
    )

    img_root = os.path.expanduser("~/datasets/flir/images_thermal_train/")
    tdataset = YoloDataset(tcoco, img_root, return_shape, anchors, scales)
    tdataset.data = {idx: tdataset.data[idx] for idx in range(400)}

    img_root = os.path.expanduser("~/datasets/flir/images_thermal_val/")
    vdataset = YoloDataset(vcoco, img_root, return_shape, anchors, scales)
    tdataset.data = {idx: tdataset.data[idx] for idx in range(400)}

    return tdataset, vdataset


def config_trainer():
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

    t_dataset, v_dataset = config_datasets(tcoco, vcoco, anchors, scales)

    train_loader = DataLoader(t_dataset, 32)
    val_loader = DataLoader(v_dataset, 32)

    device = [0, 1]

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

    num_epochs = 100

    config = {
        "train_module": train_module,
        "device": device,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "num_epochs": num_epochs,
        "unpacker": yolo_unpacker
    }

    return config


# coco = config_coco()
# 
# id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
# cat_counts = {cat["name"]: 0 for cat in coco["categories"]}
# 
# for ann in coco["annotations"]:
#     cat = id_to_name[ann["category_id"]]
#     cat_counts[cat] += 1
# 
# for tup in sorted(cat_counts.items(), key=lambda x: x[1])[:: -1]:
#     if tup[1] > 0:
#         print(tup)
# 
# _, num, _, _, ancs, scales, _, _ = config_train_module_inputs(coco)
# 
# dataset = config_datasets(coco, ancs, scales)
# 
# ids = []
# for idx in range(len(dataset.data)):
#     for id in dataset.data[idx]["category_ids"]:
#         if id not in ids:
#             ids.append(id)
# 
# ids
