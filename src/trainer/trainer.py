import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm


def default_unpacker(data, device):
    data = [d.to(device) for d in data]
    return data


class Trainer:
    def __init__(self, train_module: nn.Module):
        
        self.train_module = train_module 

    def fit(self, 
            train_loader: DataLoader,
            num_epochs: int,
            device: str | int,
            unpacker: Callable | None = None,
            val_loader: DataLoader | None = None):

        if not unpacker:
            unpacker = default_unpacker
        else:
            unpacker = unpacker

        epochs_run = self.train_module.epochs_run
        for epoch in range(epochs_run + 1, num_epochs + 1):
            self.epoch_pass("train", epoch, train_loader, device, unpacker)
            self.train_module.logger.log_epoch("train")
             
            if self.train_module.logger.save_checkpoint_flag("train"):
                self.train_module.save_checkpoint("train", epoch)

            if val_loader is not None:
                self.epoch_pass("val", epoch, val_loader, device, unpacker)
                self.train_module.logger.log_epoch("val")

                if self.train_module.logger.save_checkpoint_flag("val"):
                    self.train_module.save_checkpoint("val", epoch)


    def epoch_pass(self, 
                   which: str,
                   epoch: int,
                   loader: DataLoader, 
                   device: str | int, 
                   unpacker: Callable):

        pbar = tqdm(loader)

        for batch_idx, data in enumerate(pbar):
            data = unpacker(data, device)

            if which == "train":
                self.train_module.train_batch_pass(*data)
                pbar.set_postfix(
                    None, 
                    True,
                    EPOCH=epoch,
                    **self.train_module.logger._avg_epoch_history
                )

            elif which == "val":
                self.train_module.val_batch_pass(*data)
                pbar.set_postfix(
                    None, 
                    True, 
                    EPOCH=epoch,
                    **self.train_module.logger._avg_epoch_history
                )
