import os
import pandas as pd
from torch import save
from torch.nn import Module
from collections import defaultdict


class CSVLogger:
    def __init__(self,
                 loss_log_root: str, 
                 state_dict_root: str,
                 best_key='total_loss'):
        
        self.loss_log_root = loss_log_root
        self.state_dict_root = state_dict_root

        self.best_key = best_key

        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)

        self._epoch_history = defaultdict(list)
        self._avg_epoch_history = defaultdict(float)
        
        self.best_val_loss = 1e6
        self.best_train_loss = 1e6


    def log_batch(self, loss_dict: dict) -> None:
        """
        Log loss after each batch. Update the CSV or whatever file.
        """

        for key in loss_dict:
            self._epoch_history[key].append(loss_dict[key])

            self._avg_epoch_history[key] = sum(
                self._epoch_history[key]
            ) / len(self._epoch_history[key])
        

    def log_epoch(self, which) -> None:
        file_name = f"{which}_log.csv"
        loss_log_file_path  = os.path.join(self.loss_log_root, file_name)

        df = pd.DataFrame(self._epoch_history)

        if not os.path.isfile(loss_log_file_path):
            df.to_csv(loss_log_file_path, index=False)

        else:
            df.to_csv(loss_log_file_path, mode='a', header=False, index=False)

        for k in self._avg_epoch_history:
            if which == "train":
                self.train_history[k].append(self._avg_epoch_history[k])
            elif which == "val":
                self.val_history[k].append(self._avg_epoch_history[k])
        
        self._epoch_history = defaultdict(list)
        self._avg_epoch_history = defaultdict(float)

    
    def save_checkpoint(self, which, epoch, model: Module):
        save_ckp = False

        if which == "train":
            loss = self.train_history[self.best_key][-1]
            if loss < self.best_train_loss:
                save_ckp = True
                self.best_train_loss = loss

        elif which == "val":
            loss = self.val_history[self.best_key][-1]
            if loss < self.best_val_loss:
                save_ckp = True
                self.best_val_loss = loss
        
        if save_ckp:
            save_to = os.path.join(self.state_dict_root, f"{which}_ep_{epoch}.pth")
            save(model.state_dict(), save_to)
            print(f"Train checkpoint saved to {save_to}")


def config_log_roots(logger: CSVLogger):
    if not os.path.isdir(logger.loss_log_root):
        os.makedirs(logger.loss_log_root)
    print(f"Loss logs will be saved to {logger.loss_log_root}")
    if not os.path.isdir(logger.state_dict_root):
        os.makedirs(logger.state_dict_root)
    print(f"State dicts will be saved to {logger.state_dict_root}")
    return None
