import os
import json
from typing import cast

from torch.utils.data import DataLoader
from torch.nn import Module
from torch import float32, no_grad, Tensor, tensor, vstack
from torch.optim import Adam

from src.trainer.logger import CSVLogger
from src.yolo_utils.dataset import YoloDataset, yolo_unpacker
from src.yolo_utils.utils import make_yolo_anchors
from src.yolo_utils.coco_transformer import coco_transformer
from src.yolov5.yolov5 import YOLOv5
from src.yolo_utils.loss import YOLOLoss


def config_logger():
    save_root = os.path.relpath(__file__)
    save_root = save_root.split(os.path.basename(save_root))[0]
    logger = CSVLogger(
        os.path.join(save_root, "loss_logs"),
        os.path.join(save_root, "state_dicts")
    )
    return logger


def config_coco():
    path = os.path.expanduser("~/Datasets/flir/images_thermal_train/coco.json")
    with open(path, "r") as oj:
        coco = json.load(oj)

    instructions = {}
    for cat in coco["categories"]:
        name = cat["name"]
        if name in ["bike", "bus", "motor", "other vehicle", "truck"]:
            instructions[name] = "vehicle_not_car"
        elif name != "car":
            instructions[name] = "ignore"


    tcoco = coco_transformer(
        coco, instructions, (30, 640), (30, 512), (1, 639), (1, 511)
    )

    path = os.path.expanduser("~/Datasets/flir/images_thermal_val/coco.json")
    with open(path, "r") as oj:
        coco = json.load(oj)

    instructions = {}
    for cat in coco["categories"]:
        name = cat["name"]
        if name in ["bike", "bus", "motor", "other vehicle", "truck"]:
            instructions[name] = "vehicle_not_car"
        elif name != "car":
            instructions[name] = "ignore"

    vcoco = coco_transformer(
        coco, instructions, (30, 640), (30, 512), (1, 639), (1, 511)
    )
    return tcoco, vcoco


class TrainModule(Module):
    def __init__(self, 
                 in_channels: int, 
                 num_classes: int, 
                 img_width: int,
                 img_height: int,
                 normalized_anchors: Tensor,
                 scales: list[int]):
        super().__init__()

        self.model = YOLOv5(in_channels, num_classes)
        self.loss_fn = YOLOLoss()
        self.optimizer = Adam(self.model.parameters(), lr=.0001)
        
        self.img_width, self.img_height = img_width, img_height
        self.normalized_anchors = normalized_anchors
        self.scales = scales 
        
        self.scaled_anchors = []
        for scale_id, scale in enumerate(self.scales):
            scaled_anchors = self.normalized_anchors[3 * scale_id: 3 * (scale_id + 1)]
            scaled_anchors *= tensor(
                [self.img_width / scale ,self.img_height / scale]
            )
            self.scaled_anchors.append(scaled_anchors)

        self.scaled_anchors = vstack(self.scaled_anchors).to(0).to(float32)

        self.logger = config_logger()


    def forward(self, x):
        return self.model(x)


    def train_batch_pass(self, *args):
        self.model.train()

        inputs, targets = args
        device = inputs.device.type

        assert type(inputs) == Tensor
        targets = cast(tuple[Tensor, ...], targets)

        self.optimizer.zero_grad()

        outputs = self.model(inputs)

        total_loss = tensor(0.0, requires_grad=True).to(device)
        for scale_id, (output, target) in enumerate(zip(outputs, targets)):
            scaled_anchors = self.scaled_anchors[3 * scale_id: 3 * (scale_id + 1)]
            loss, batch_history = self.loss_fn(output, target, scaled_anchors)
            total_loss += loss
            self.logger.log_batch(batch_history)

        total_loss.backward()

        self.optimizer.step()


    def val_batch_pass(self, *args):
        self.model.eval()

        inputs, targets = args

        assert type(inputs) == Tensor
        targets = cast(tuple[Tensor, ...], targets)
        
        with no_grad():
            outputs = self.model(inputs)
        
        for scale_id, (output, target) in enumerate(zip(outputs, targets)):
            scaled_anchors = self.scaled_anchors[3 * scale_id: 3 * (scale_id + 1)]
            _, batch_history = self.loss_fn(output, target, scaled_anchors)
            self.logger.log_batch(batch_history)


def config_some_hyperparams(coco):
    in_channels = 1
    num_classes = 4 + 1
    img_width = 640
    img_height = 512
    anchors = make_yolo_anchors(coco, img_width, img_height, 9)
    scales = [32, 16, 8]
    return in_channels, num_classes, img_width, img_height, anchors, scales


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
        scales
    ) = config_some_hyperparams(coco)

    dataset = config_datasets(coco, anchors, scales)

    train_loader = DataLoader(dataset, 1)

    train_module = TrainModule(
        in_channels, 
        num_classes, 
        img_width, 
        img_height, 
        anchors,
        scales
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


if __name__ == "__main__":
    pass

import numpy as np

tcoco, vcoco = config_coco()

counts = {
    cat["name"]: 0 
    for cat in tcoco["categories"]
}
id_to_name = {cat["id"]: cat["name"] for cat in tcoco["categories"]}
widths_heights = {
        cat["name"]: {"height": [], "width": []} 
    for cat in tcoco["categories"]
}
id_to_name = {cat["id"]: cat["name"] for cat in tcoco["categories"]}

for ann in tcoco["annotations"]:
    cat = id_to_name[ann["category_id"]]
    bbox = ann["bbox"]
    counts[cat] += 1
    widths_heights[cat]["width"].append(bbox[2])
    widths_heights[cat]["height"].append(bbox[3])

for tup in sorted(counts.items(), key=lambda x: x[1])[::-1]:
    if tup[1] > 600:
        print(tup)
        q = [.1, .3, .5, .7, .9]
        print(np.quantile(widths_heights[tup[0]]["width"], q))
        print(np.quantile(widths_heights[tup[0]]["height"], q))


counts = {
    cat["name"]: 0 
    for cat in vcoco["categories"]
}
id_to_name = {cat["id"]: cat["name"] for cat in vcoco["categories"]}
widths_heights = {
        cat["name"]: {"height": [], "width": []} 
    for cat in vcoco["categories"]
}
id_to_name = {cat["id"]: cat["name"] for cat in vcoco["categories"]}

for ann in vcoco["annotations"]:
    cat = id_to_name[ann["category_id"]]
    bbox = ann["bbox"]
    counts[cat] += 1
    widths_heights[cat]["width"].append(bbox[2])
    widths_heights[cat]["height"].append(bbox[3])

for tup in sorted(counts.items(), key=lambda x: x[1])[::-1]:
    if tup[1] > 0:
        print(tup)
        q = [.1, .3, .5, .7, .9]
        print(np.quantile(widths_heights[tup[0]]["width"], q))
        print(np.quantile(widths_heights[tup[0]]["height"], q))
