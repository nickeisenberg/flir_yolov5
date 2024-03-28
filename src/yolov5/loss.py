import torch
import torch.nn as nn
from typing import Tuple

from ..utils import iou

class YOLOLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean') 
        self.bce = nn.BCEWithLogitsLoss(reduction='mean') 
        self.cross_entropy = nn.CrossEntropyLoss() 
        self.sigmoid = nn.Sigmoid() 
        self.device = device
        self.lambda_class = 1.
        self.lambda_noobj = 10.
        self.lambda_obj = 1.
        self.lambda_box = 10.


    def forward(self, pred, target, scaled_anchors) -> Tuple[torch.Tensor, dict]:
        """
        Recall that the pred and target is a tuple of 3 tensors. As of now,
        This forward only handles each piece separately, ie, pred[0] and 
        target[0] etc. I may generalize to just accept the whole tuple later.

        pred: torch.Tensor, shape=(batch, 3, num_rows, num_cols, 1, 5 + num_classes)
            pred[..., :] = (x, y, w, h, prob, class_probabilities)
        target: torch.Tensor, shape=(batch, 3, num_rows, num_cols, 1, 6)
            target[..., :] = (x, y, w, h, prob, class_ID)

        (x_off, y_off, w/ s, h /s, prop, ....) ---> (x, y, w, h, prob, class)

        (t1, t2, t3) tk.shape = (batch, 3, row, col, ____)
        """

        obj = target[..., 4] == 1
        no_obj = target[..., 4] == 0

        no_object_loss = self.bce( 
            (pred[..., 4:5][no_obj]), (target[..., 4:5][no_obj]), 
        )

        scaled_anchors = scaled_anchors.reshape((1, 3, 1, 1, 2))

        if obj.sum() > 0:

            box_preds = torch.cat(
                [
                    self.sigmoid(pred[..., 0: 2]), 
                    torch.exp(pred[..., 2: 4]) * scaled_anchors
                ],
                dim=-1
            ) 

            ious = iou(box_preds[obj], target[..., 0: 4][obj]).detach() 
            
            object_loss = self.mse(
                self.sigmoid(pred[..., 4: 5][obj]), 
                ious * target[..., 4: 5][obj]
            ) 

            # Calculating box coordinate loss
            pred[..., 0: 2] = self.sigmoid(pred[..., 0: 2])
            target[..., 2: 4] = torch.log(1e-6 + target[..., 2: 4] / scaled_anchors) 

            box_loss = self.mse(
                pred[..., 0: 4][obj], 
                target[..., 0: 4][obj]
            )

            # Claculating class loss 
            class_loss = self.cross_entropy(
                pred[..., 5:][obj], 
                target[..., 5][obj].long()
            )
        else:
            box_loss = torch.tensor([0]).to(self.device)
            object_loss = torch.tensor([0]).to(self.device)
            class_loss = torch.tensor([0]).to(self.device)

        total_loss = self.lambda_box * box_loss
        total_loss += self.lambda_obj * object_loss
        total_loss += self.lambda_noobj * no_object_loss
        total_loss += self.lambda_class * class_loss 
        
        history = {}
        history["box_loss"] = box_loss.item()
        history["object_loss"] = object_loss.item()
        history["no_object_loss"] = no_object_loss.item()
        history["class_loss"] = class_loss.item()
        history["total_loss"] = total_loss.item()

        return total_loss, history

