import os
import json
from typing import cast

from torch.utils.data import DataLoader
from torch.nn import Module
from torch import float32, no_grad, Tensor, tensor, vstack, save, load
from torch.optim import Adam

from src.trainer.logger import CSVLogger
from src.yolo_utils.dataset import YoloDataset, yolo_unpacker
from src.yolo_utils.utils import make_yolo_anchors
from src.yolo_utils.coco_transformer import coco_transformer
from src.yolov5.yolov5 import YOLOv5
from src.yolo_utils.loss import YOLOLoss


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

        self.loss_log_root, self.state_dict_root = config_save_roots() 
        self.logger = CSVLogger(self.loss_log_root)

        self.epochs_run = 0
        if os.path.isfile(os.path.join(self.state_dict_root, "train_ckp.pth")):
            self.load_checkpoint()


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


    def save_checkpoint(self, which, epoch, save_to: str | None = None):
        checkpoint = {}
        if save_to is None:
            save_to = os.path.join(
                self.state_dict_root, f"{which}_ckp.pth"
            )
        checkpoint["MODEL_STATE"] = self.model.state_dict()
        checkpoint["OPTIMIZER_STATE"] = self.optimizer.state_dict()
        checkpoint["EPOCHS_RUN"] = epoch
        save(checkpoint, save_to)
        print(f"EPOCH {epoch} checkpoint saved at {save_to}")


    def load_checkpoint(self, which="train", load_from: str | None = None):
        if load_from is None:
            load_from = os.path.join(
                self.state_dict_root, f"{which}_ckp.pth"
            )
        checkpoint = load(load_from)
        self.model.load_state_dict(checkpoint["MODEL_STATE"],)
        self.optimizer.load_state_dict(checkpoint["OPTIMIZER_STATE"])
        self.epochs_run = checkpoint["EPOCHS_RUN"]


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
    num_epochs = 60

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
