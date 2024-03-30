import os
from torch.nn import Module
from torch import no_grad, Tensor, tensor
from torch.optim import Adam
from src.yolov5.yolov5 import YOLOv5
from src.yolo_utils.loss import YOLOLoss
from src.trainer.logger import CSVLogger 


class TrainModule(Module):
    def __init__(self, 
                 in_channels: int, 
                 num_classes: int, 
                 device: str, 
                 img_width: int,
                 img_height: int,
                 normalized_anchors: Tensor,
                 scales: list[int],
                 logger: CSVLogger):
        super().__init__()

        self.device = device
        self.model = YOLOv5(in_channels, num_classes)
        self.loss_fn = YOLOLoss(device)
        self.optimizer = Adam(self.model.parameters(), lr=.001)
        
        self.img_width, self.img_height = img_width, img_height
        self.normalized_anchors = normalized_anchors
        self.scales = scales 
        
        self.logger = logger


    def forward(self, x):
        return self.model(x)


    def train_batch_pass(self, *args):
        self.model.train()

        inputs, targets = args

        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        
        total_loss = tensor(0, requires_grad=True).to(self.device)
        for scale_id, (output, target) in enumerate(zip(outputs, targets)):
            scale = self.scales[scale_id]
            scaled_anchors = self.normalized_anchors[3 * scale_id: 3 * (scale_id + 1)]
            scaled_anchors *= tensor(
                [self.img_width / scale ,self.img_height / scale]
            )
            loss, batch_history = self.loss_fn(output, target, scaled_anchors)
            total_loss += loss
            self.logger.log_batch(batch_history)

        total_loss.backward()

        self.optimizer.step()


    def val_batch_pass(self, *args):
        self.model.eval()

        inputs, targets = args
        
        with no_grad():
            outputs = self.model(inputs)
        
        for scale_id, (output, target) in enumerate(zip(outputs, targets)):
            scale = self.scales[scale_id]
            scaled_anchors = self.normalized_anchors[3 * scale_id: 3 * (scale_id + 1)]
            scaled_anchors *= tensor(
                [self.img_width / scale ,self.img_height / scale]
            )
            _, batch_history = self.loss_fn(output, target, scaled_anchors)
            self.logger.log_batch(batch_history)
