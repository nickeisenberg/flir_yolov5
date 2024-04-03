import os
import json

from torch.utils.data import DataLoader

from src.yolo_utils.dataset import YoloDataset, yolo_unpacker
from src.yolo_utils.utils import make_yolo_anchors
from src.yolo_utils.coco_transformer import coco_transformer
from src.yolov5.train_module import TrainModule


def config_coco():
    path = os.path.expanduser("~/Datasets/flir/images_thermal_train/coco.json")
    with open(path, "r") as oj:
        coco = json.load(oj)

    instructions = {}
    for cat in coco["categories"]:
        name = cat["name"]
        if name not in ["truck", "motor", "car"]:
            instructions[name] = "ignore"

    tcoco = coco_transformer(
        coco, instructions, (15, 640), (10, 512), (10, 630), (10, 502)
    )
    return tcoco


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
    num_classes = 4 + 1
    img_width = 640
    img_height = 512
    anchors = make_yolo_anchors(coco, img_width, img_height, 9)
    scales = [32, 16, 8]
    return [in_channels, num_classes, img_width, img_height, anchors, scales,
            loss_log_root, state_dict_root]


def config_datasets(coco, anchors, scales):
    return_shape = (
        (3, 16, 20, 6),
        (3, 32, 40, 6),
        (3, 64, 80, 6),
    )

    img_root = os.path.expanduser("~/Datasets/flir/images_thermal_train/")
    dataset = YoloDataset(coco, img_root, return_shape, anchors, scales)
    dataset.data = {idx: dataset.data[idx] for idx in range(10)}

    return dataset


def config_trainer():
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

    train_loader = DataLoader(dataset, 1)

    train_module = TrainModule(
        in_channels, 
        num_classes, 
        img_width, 
        img_height, 
        anchors, 
        scales,
        loss_log_root,
        state_dict_root
    )

    device = "cuda:0"
    num_epochs = 100

    config = {
        "train_module": train_module,
        "device": device,
        "train_loader": train_loader,
        "num_epochs": num_epochs,
        "unpacker": yolo_unpacker
    }

    return config


# coco = config_coco()
# 
# _, num, _, _, ancs, scales = config_some_hyperparams(coco)
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
