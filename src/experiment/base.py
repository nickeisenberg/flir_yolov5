import os
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Callable, Tuple
from tqdm import tqdm

from .logger import CSVLogger


class TrainerModule(ABC):
    def __init__(self):
        self.logger = self._config_log()
        self._config_log_roots(self.logger)


    @abstractmethod
    def train_batch_pass(self, inputs, targets):
        pass
    
    def val_batch_pass(self, inputs, targets):
        pass

    def test_batch_pass(self, inputs, targets):
        pass

    def epoch_pass(self, which: str, loader: DataLoader, device: str):
        pbar = tqdm(loader)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            if which == "train":
                self.train_batch_pass(inputs, targets)

            elif which == "val":
                self.val_batch_pass(inputs, targets)


    def fit(self, 
            train_loader: DataLoader,
            num_epochs: int,
            device: str,
            val_loader: DataLoader | None = None, 
            test_loader: DataLoader | None = None): 

        for epoch in range(1, num_epochs + 1):
            self.epoch_pass("train", train_loader, device)
            self.logger.log_epoch("train", epoch)

            if val_loader is not None:
                self.epoch_pass("val", val_loader, device)
                self.logger.log_epoch("val", epoch)

            if test_loader is not None:
                self.epoch_pass("test", test_loader, device)
                self.logger.log_epoch("test", epoch)


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
