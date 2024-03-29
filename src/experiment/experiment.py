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
                 optimizer: torch.optim.Optimizer):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.logger = self._config_log()

        self.unpacker = default_unpacker


    def fit(self, 
            train_loader: DataLoader,
            num_epochs: int,
            device: str,
            val_loader: DataLoader | None = None, 
            test_loader: DataLoader | None = None): 

        self._config_log_roots(self.logger)

        for epoch in range(1, num_epochs + 1):
            self.epoch_pass("train", train_loader, device)
            self.logger.log_epoch("train", epoch)

            if val_loader is not None:
                self.epoch_pass("val", val_loader, device)
                self.logger.log_epoch("val", epoch)

            if test_loader is not None:
                self.epoch_pass("test", test_loader, device)
                self.logger.log_epoch("test", epoch)


    def epoch_pass(self, which: str, loader: DataLoader, device: str):
        pbar = tqdm(loader)
        for batch_idx, data in enumerate(pbar):
            data = self.unpacker(data, device)

            if which == "train":
                batch_history = self.train_batch_pass(*data)
                pbar.set_postfix(None, True, **batch_history)

            elif which == "val":
                batch_history = self.val_batch_pass(*data)
                pbar.set_postfix(None, True, **batch_history)


    def train_batch_pass(self, *args):
        inputs, targets = args
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss, batch_history = self.loss_fn(outputs, targets)
        self.logger.log_batch(batch_history)
        loss.backward()
        self.optimizer.state_dict()
        return batch_history


    def val_batch_pass(self, *args):
        inputs, targets = args
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            _, batch_history = self.loss_fn(outputs, targets)
            self.logger.log_batch(batch_history)
        return batch_history


    @staticmethod
    def _config_log():
        root = os.getcwd()
        loss_log_root = os.path.join(root, "loss_logs")
        state_dict_root = os.path.join(root, "state_dicts")
        logger = CSVLogger(loss_log_root, state_dict_root)
        return logger


    @staticmethod
    def _config_log_roots(logger: CSVLogger):
        if not os.path.isdir(logger.loss_log_root):
            os.makedirs(logger.loss_log_root)
        print(f"Loss logs will be saved to {logger.loss_log_root}")
        if not os.path.isdir(logger.state_dict_root):
            os.makedirs(logger.state_dict_root)
        print(f"State dicts will be saved to {logger.state_dict_root}")
        return None
