import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Tuple

from .base import TrainerModule


class Trainer(TrainerModule):
    def __init__(self,
                 model: nn.Module,
                 loss_fn: Callable[[Tensor, Tensor], Tuple[Tensor, dict]],
                 optimizer: torch.optim.Optimizer,
                 save_root: str):

        self.loss_log_root = save_root
        self.state_dict_root = save_root
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer


    def train_batch_pass(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss, batch_history = self.loss_fn(outputs, targets)
        self.logger.log_batch(batch_history)
        loss.backward()
        self.optimizer.state_dict()


    def val_batch_pass(self, inputs, targets):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            _, batch_history = self.loss_fn(outputs, targets)
            self.logger.log_batch(batch_history)
