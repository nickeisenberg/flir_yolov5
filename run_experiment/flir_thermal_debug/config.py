import os
import json
from torch.optim import Adam
from torch.utils.data import DataLoader

from trainer_module import TrainModule

from src.yolov5.yolov5 import YOLOv5
from src.trainer.logger import CSVLogger
from src.yolo_utils.loss import YOLOLoss
from src.yolo_utils.dataset import YoloDataset, yolo_unpacker
from src.yolo_utils.utils import make_yolo_anchors
from src.yolo_utils.coco_transformer import coco_transformer

path = os.path.expanduser("~/Datasets/flir/images_thermal_train/coco.json")
with open(path, "r") as oj:
    coco = json.load(oj)

in_channels = 1
num_classes = 16
model = YOLOv5(in_channels, num_classes)

device = "cuda"
loss_fn = YOLOLoss(device)
optimizer = Adam(model.parameters(), lr=.001)
img_width = 640
img_height = 512
anchors = make_yolo_anchors(coco, img_width, img_height, 9)
scales = [32, 16, 8]

save_root = os.path.relpath(__file__)
save_root = save_root.split(os.path.basename(save_root))[0]
logger = CSVLogger(
    os.path.join(save_root, "loss_logs"),
    os.path.join(save_root, "state_dicts")
)

module = TrainModule(
    in_channels, num_classes, device, img_width, img_height, 
    anchors, scales, logger
)

instructions = {}
for cat in coco["categories"]:
    name = cat["name"]
    if name not in ["truck", "motor", "car"]:
        instructions[name] = "ignore"

tcoco = coco_transformer(
    coco, instructions, (15, 640), (10, 512), (10, 630), (10, 502)
)

return_shape = (
    (3, 16, 20, 6),
    (3, 32, 40, 6),
    (3, 64, 80, 6),
)
img_root = os.path.expanduser("~/Datasets/flir/images_thermal_train/")
dataset = YoloDataset(tcoco, img_root, return_shape, anchors, scales)
dataset.data = {idx: dataset.data[idx] for idx in range(10)}

train_dataloader = DataLoader(dataset, 2)
val_dataloader = DataLoader(dataset, 2)

num_epochs = 3

config = {
    "model": module,
    "loss_fn": loss_fn,
    "optimizer": optimizer,
    "device": device,
    "save_root": save_root,
    "train_loader": train_dataloader,
    "val_loader": val_dataloader,
    "num_epochs": num_epochs,
    "unpacker": yolo_unpacker
}
