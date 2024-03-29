import os
from torch.optim import Adam
from src.yolov5.yolov5 import YOLOv5
from src.yolov5.loss import YOLOLoss

in_channels = 1
num_classes = 1
model = YOLOv5(in_channels, num_classes)

device = "cuda"

loss_fn = YOLOLoss(device)

optimizer = Adam(model.parameters(), lr=.001)

config = {
    "model": model,
    "loss_fn": loss_fn,
    "optimizer": optimizer,
    "device": device
}
