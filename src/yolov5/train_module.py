import os
from typing import Any, Callable, cast
from torch.nn import Module
from torch import float32, no_grad, Tensor, tensor, vstack, save, load
from torch.optim import Adam
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.metric import Metric
from tqdm import tqdm

from src.trainer.logger import CSVLogger
from src.yolov5.yolov5 import YOLOv5
from src.yolo_utils.loss import YOLOLoss
from src.yolo_utils.targets import decode_yolo_tuple


class TrainModule(Module):
    def __init__(self,
                 yolo: YOLOv5,
                 device: list[int] | int | str,
                 img_width: int,
                 img_height: int,
                 normalized_anchors: Tensor,
                 scales: list[int],
                 state_dict_root: str | None = None,
                 loss_log_root: str | None = None):
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
    
        if state_dict_root is not None:
            self.state_dict_root = state_dict_root
            if os.path.isfile(os.path.join(self.state_dict_root, "train_ckp.pth")):
                self.load_checkpoint()

        if loss_log_root is not None:
            self.logger = CSVLogger(self.loss_log_root)

        self.epochs_run = 0


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


    def evaluate(self,
                 loader: DataLoader, 
                 unpacker: Callable):

        self.model.eval()

        map = MeanAveragePrecision(
            box_format='xywh', warn_on_many_detections=False
        )
        map.warn_on_many_detections = False

        pbar = tqdm(loader)
        for data in pbar:
            inputs, targets = unpacker(data, self.device)

            assert type(inputs) == Tensor
            targets = cast(tuple[Tensor, ...], targets)
            
            with no_grad():
                predictions = self.model(inputs)
                decoded_predictions = decode_yolo_tuple(
                    yolo_tuple=predictions, 
                    img_width=self.img_width, 
                    img_height=self.img_height, 
                    normalized_anchors=self.normalized_anchors, 
                    scales=self.scales, 
                    score_thresh=.95,
                    iou_thresh=.3,
                    is_pred=True
                )
                decoded_targets = decode_yolo_tuple(
                    yolo_tuple=targets, 
                    img_width=self.img_width, 
                    img_height=self.img_height, 
                    normalized_anchors=self.anchors, 
                    scales=self.scales, 
                    is_pred=False
                )
                map.update(preds=decoded_predictions, target=decoded_targets)

            computes = map.compute()
            pbar.set_postfix(
                    map=computes["map"],
                    map_50=computes["map_50"],
                    map_75=computes["map_75"]
            )

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


    def load_checkpoint(self, load_from: str | None = None):
        if load_from is None:
            load_from = os.path.join(
                self.state_dict_root, f"train_ckp.pth"
            )
        train_checkpoint = load(load_from)

        for state in train_checkpoint["OPTIMIZER_STATE"]["state"].values():
            for k, v in state.items():
                if isinstance(v, Tensor):
                    state[k] = v.to(self.device)

        if isinstance(self.model, DataParallel):
            self.model.module.load_state_dict(train_checkpoint["MODEL_STATE"])
            self.optimizer.load_state_dict(train_checkpoint["OPTIMIZER_STATE"])
            self.epochs_run = train_checkpoint["EPOCHS_RUN"]
        else:
            self.model.load_state_dict(train_checkpoint["MODEL_STATE"])
            self.optimizer.load_state_dict(train_checkpoint["OPTIMIZER_STATE"])
            self.epochs_run = train_checkpoint["EPOCHS_RUN"]
