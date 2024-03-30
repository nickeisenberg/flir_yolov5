import torch
import os
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Callable, Tuple
from tqdm import tqdm

from .logger import CSVLogger


def default_unpacker(data, device):
    data = [d.to(device) for d in data]
    return data


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 loss_fn: Callable[[Tensor, Tensor], Tuple[Tensor, dict]],
                 optimizer: torch.optim.Optimizer,
                 save_root: str):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.unpacker = default_unpacker

        self.save_root = save_root

        self.logger = CSVLogger(save_root)

    def fit(self, 
            train_loader: DataLoader,
            num_epochs: int,
            device: str,
            val_loader: DataLoader | None = None, 
            test_loader: DataLoader | None = None): 
    
        self.model = self.model.to(device)

        self._config_log_roots(self.logger)

        for epoch in range(1, num_epochs + 1):
            self.epoch_pass("train", train_loader, device)
            self.logger.log_epoch("train", epoch)
            # self.logger.save_checkpoint("train", epoch, self.model)

            if val_loader is not None:
                self.epoch_pass("val", val_loader, device)
                self.logger.log_epoch("val", epoch)
                # self.logger.save_checkpoint("val", epoch, self.model)

            if test_loader is not None:
                self.epoch_pass("test", test_loader, device)
                self.logger.log_epoch("test", epoch)


    def epoch_pass(self, which: str, loader: DataLoader, device: str):
        pbar = tqdm(loader)
        for batch_idx, data in enumerate(pbar):
            inputs, targets = self.unpacker(data, device)

            if which == "train":
                self.train_batch_pass(inputs, targets)
                pbar.set_postfix(None, True, **self.logger._avg_epoch_history)

            elif which == "val":
                self.val_batch_pass(inputs, targets)
                pbar.set_postfix(None, True, **self.logger._avg_epoch_history)


    def train_batch_pass(self, inputs, targets):
        self.model.train()

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss, batch_history = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.state_dict()

        self.logger.log_batch(batch_history)


    def val_batch_pass(self, inputs, targets):
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(inputs)
            _, batch_history = self.loss_fn(outputs, targets)
            self.logger.log_batch(batch_history)


    @staticmethod
    def _config_log_roots(logger: CSVLogger):
        if not os.path.isdir(logger.loss_log_root):
            os.makedirs(logger.loss_log_root)
        print(f"Loss logs will be saved to {logger.loss_log_root}")
        if not os.path.isdir(logger.state_dict_root):
            os.makedirs(logger.state_dict_root)
        print(f"State dicts will be saved to {logger.state_dict_root}")
        return None
