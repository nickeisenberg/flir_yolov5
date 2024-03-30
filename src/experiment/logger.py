import os
import numpy as np
import pandas as pd
from torch import save
from torch.nn import Module
from collections import defaultdict


class CSVLogger:
    def __init__(self,
                 save_root: str,
                 best_key='total_loss'):
        
        self.loss_log_root = os.path.join(save_root, "loss_logs")
        self.state_dict_root = os.path.join(save_root, "state_dicts")

        self.best_key = best_key
        
        self.train_history = {}
        self.val_history = {}

        self._epoch_history = defaultdict(list)
        
        self.best_val_loss = 1e6
        self.best_train_loss = 1e6


    def log_batch(self, loss_dict: dict, reduce: str | None ="mean") -> None:
        """
        Log loss after each batch. Update the CSV or whatever file.
        """

        for key in loss_dict:
            if reduce == "mean":
                self._epoch_history[key].append(np.mean(loss_dict[key]))
            elif reduce == "none" or reduce is None:
                self._epoch_history[key].append(loss_dict[key])
            else:
                raise Exception("reduce must be mean or none")
        

    def log_epoch(self, which, epoch) -> None:
        file_name = f"{which}_log.csv"
        loss_log_file_path  = os.path.join(self.loss_log_root, file_name)

        df = pd.DataFrame(self._epoch_history)

        if not os.path.isfile(loss_log_file_path):
            df.to_csv(loss_log_file_path, index=False)

        else:
            df.to_csv(loss_log_file_path, mode='a', header=False, index=False)
        
        if which == "train":
            self.train_history[epoch] = self._epoch_history
        elif which == "val":
            self.val_history[epoch] = self._epoch_history

        self._epoch_history = defaultdict(list)

    
    def save_checkpoint(self, which, epoch, model: Module):

        if which == "train":
            epoch_history = self.train_history[epoch]
            avg_total_loss = np.mean(epoch_history[self.best_key])

            if avg_total_loss < self.best_train_loss:
                self.best_train_loss = avg_total_loss
                state_dict = model.state_dict()

                save_to = os.path.join(self.state_dict_root, f"train_ep_{epoch}.pth")
                save(state_dict, save_to)
                print(f"Train checkpoint saved to {save_to}")


        elif which == "val":
            epoch_history = self.val_history[epoch]
            avg_total_loss = np.mean(epoch_history[self.best_key])

            if avg_total_loss < self.best_val_loss:
                self.best_train_loss = avg_total_loss
                state_dict = model.state_dict()

                save_to = os.path.join(self.state_dict_root, f"val_ep_{epoch}.pth")
                save(state_dict, save_to)
                print(f"Validation checkpoint saved to {save_to}")

        else:
            raise Exception("which must be train or val")
