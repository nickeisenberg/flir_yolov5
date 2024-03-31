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
            device: str,
            unpacker: Callable | None = None,
            val_loader: DataLoader | None = None):

        if not unpacker:
            unpacker = default_unpacker
        else:
            unpacker = unpacker
    
        self.train_module.model = self.train_module.model.to(device)

        for epoch in range(1, num_epochs + 1):
            self.epoch_pass("train", train_loader, device, unpacker)
            self.train_module.logger.log_epoch("train")
            self.train_module.logger.save_checkpoint(
                "train", epoch, self.train_module.model
            )

            if val_loader is not None:
                self.epoch_pass("val", val_loader, device, unpacker)
                self.train_module.logger.log_epoch("val")
                self.train_module.logger.save_checkpoint(
                    "val", epoch, self.train_module.model
                )


    def epoch_pass(self, 
                   which: str, 
                   loader: DataLoader, 
                   device: str, 
                   unpacker: Callable):

        pbar = tqdm(loader)

        for batch_idx, data in enumerate(pbar):
            data = unpacker(data, device)

            if which == "train":
                self.train_module.train_batch_pass(*data)
                pbar.set_postfix(
                    None, True, **self.train_module.logger._avg_epoch_history
                )

            elif which == "val":
                self.train_module.val_batch_pass(*data)
                pbar.set_postfix(
                    None, True, **self.train_module.logger._avg_epoch_history
                )
