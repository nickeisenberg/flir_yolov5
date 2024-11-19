import torch
import torch.nn as nn
from typing import Tuple

from .utils import iou


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean') 
        self.bce = nn.BCEWithLogitsLoss(reduction='mean') 
        self.cross_entropy = nn.CrossEntropyLoss() 
        self.sigmoid = nn.Sigmoid() 
        self.lambda_class = 1.
        self.lambda_noobj = 10.
        self.lambda_obj = 1.
        self.lambda_box = 10.


    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor, 
                scaled_anchors) -> Tuple[torch.Tensor, dict]:
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        no_object_loss = self.bce( 
            (pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]), 
        )

        scaled_anchors = scaled_anchors.reshape((1, 3, 1, 1, 2))

        if obj.sum() > 0:

            box_preds = torch.cat(
                [
                    self.sigmoid(pred[..., 1: 3]), 
                    torch.exp(pred[..., 3: 5]) * scaled_anchors
                ],
                dim=-1
            ) 

            ious = iou(box_preds[obj], target[..., 1: 5][obj]).detach() 
            
            object_loss = self.mse(
                self.sigmoid(pred[..., 0: 1][obj]), 
                ious * target[..., 0: 1][obj]
            ) 

            # Calculating box coordinate loss
            pred[..., 1: 3] = self.sigmoid(pred[..., 1: 3])
            target[..., 3: 5] = torch.log(1e-6 + target[..., 3: 5] / scaled_anchors) 

            box_loss = self.mse(
                pred[..., 1: 5][obj], 
                target[..., 1: 5][obj]
            )

            # Claculating class loss 
            class_loss = self.cross_entropy(
                pred[..., 5:][obj], 
                target[..., 5][obj].long()
            )

        else:
            device = pred.device.type
            box_loss = torch.tensor([0]).to(device)
            object_loss = torch.tensor([0]).to(device)
            class_loss = torch.tensor([0]).to(device)

        total_loss = self.lambda_box * box_loss
        total_loss += self.lambda_obj * object_loss
        total_loss += self.lambda_class * class_loss 
        total_loss += self.lambda_noobj * no_object_loss
        
        history = {}
        history["box_loss"] = box_loss.item() * self.lambda_box
        history["object_loss"] = object_loss.item() * self.lambda_obj
        history["no_object_loss"] = no_object_loss.item() * self.lambda_noobj
        history["class_loss"] = class_loss.item() * self.lambda_class
        history["total_loss"] = total_loss.item()

        return total_loss, history
