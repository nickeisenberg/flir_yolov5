import os
from torch.nn import Module
from torch import no_grad
from torch.optim import Adam
from src.yolov5.yolov5 import YOLOv5
from src.yolo_utils.loss import YOLOLoss
from src.trainer.logger import CSVLogger 


class TrainModule(Module):
    def __init__(self, in_channels, num_classes, device, normalized_anchors):
        super().__init__()

        self.model = YOLOv5(in_channels, num_classes)
        self.loss_fn = YOLOLoss(device)
        self.optimizer = Adam(self.model.parameters(), lr=.001)
        
        save_root = ""
        self.logger = CSVLogger(
            os.path.join(save_root, "loss_logs"),
            os.path.join(save_root, "state_dicts"),
        )

        self.normalized_anchors = normalized_anchors


    def forward(self, x):
        return self.model(x)


    def train_batch_pass(self, *args):
        self.model.train()

        inputs, targets = args

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss, batch_history = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

        self.logger.log_batch(batch_history)


    def val_batch_pass(self, *args):
        self.model.eval()

        inputs, targets = args

        with no_grad():
            outputs = self.model(inputs)
            _, batch_history = self.loss_fn(outputs, targets)
            self.logger.log_batch(batch_history)
