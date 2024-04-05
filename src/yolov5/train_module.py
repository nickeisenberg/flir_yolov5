import os
from typing import cast
from torch.nn import Module
from torch import float32, no_grad, Tensor, tensor, vstack, save, load
from torch.optim import Adam
from torch.nn import DataParallel

from src.trainer.logger import CSVLogger
from src.yolov5.yolov5 import YOLOv5
from src.yolo_utils.loss import YOLOLoss


class TrainModule(Module):
    def __init__(self,
                 yolo: YOLOv5,
                 device: list[int] | int | str,
                 img_width: int,
                 img_height: int,
                 normalized_anchors: Tensor,
                 scales: list[int],
                 loss_log_root: str,
                 state_dict_root: str):
        super().__init__()

        if isinstance(device, str):
            self.device = device
            self.model = yolo.to(device)
        elif isinstance(device, int):
            self.device = device
            self.model = yolo.to(device)
        elif isinstance(device, list):
            self.device = device[0]
            self.device_ids = device
            self.model = DataParallel(yolo, self.device_ids)
            self.model.to(self.device)
        else:
            raise Exception("wrong model initialization")


        self.loss_fn = YOLOLoss()
        self.optimizer = Adam(self.model.parameters(), lr=.0001)
        
        self.img_width, self.img_height = img_width, img_height
        self.normalized_anchors = normalized_anchors
        self.scales = scales 
        
        _scaled_anchors = []
        for scale_id, scale in enumerate(self.scales):
            scaled_anchors = self.normalized_anchors[3 * scale_id: 3 * (scale_id + 1)]
            scaled_anchors *= tensor(
                [self.img_width / scale ,self.img_height / scale]
            )
            _scaled_anchors.append(scaled_anchors)

        self.scaled_anchors = vstack(_scaled_anchors).to(float32)

        self.loss_log_root, self.state_dict_root = loss_log_root, state_dict_root
        self.logger = CSVLogger(self.loss_log_root)

        self.epochs_run = 0
        if os.path.isfile(os.path.join(self.state_dict_root, "train_ckp.pth")):
            self.load_checkpoint()


    def forward(self, x):
        return self.model(x)


    def train_batch_pass(self, *args):
        self.model.train()

        inputs, targets = args

        _device = inputs.device.type

        self.scaled_anchors = self.scaled_anchors.to(_device)

        assert type(inputs) == Tensor
        targets = cast(tuple[Tensor, ...], targets)

        self.optimizer.zero_grad()

        outputs = self.model(inputs)

        total_loss = tensor(0.0, requires_grad=True).to(_device)
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
        _device = inputs.device.type

        assert type(inputs) == Tensor
        targets = cast(tuple[Tensor, ...], targets)
        
        self.scaled_anchors = self.scaled_anchors.to(_device)
        
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

        if isinstance(self.model, DataParallel):
            checkpoint["MODEL_STATE"] = self.model.module.state_dict()
            checkpoint["OPTIMIZER_STATE"] = self.optimizer.state_dict()
            checkpoint["EPOCHS_RUN"] = epoch
        else:
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

        for state in checkpoint["OPTIMIZER_STATE"]["state"].values():
            for k, v in state.items():
                if isinstance(v, Tensor):
                    state[k] = v.to(self.device)

        if isinstance(self.model, DataParallel):
            self.model.module.load_state_dict(checkpoint["MODEL_STATE"])
            self.optimizer.load_state_dict(checkpoint["OPTIMIZER_STATE"])
            self.epochs_run = checkpoint["EPOCHS_RUN"]
        else:
            self.model.load_state_dict(checkpoint["MODEL_STATE"])
            self.optimizer.load_state_dict(checkpoint["OPTIMIZER_STATE"])
            self.epochs_run = checkpoint["EPOCHS_RUN"]
