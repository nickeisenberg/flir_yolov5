import os
import pandas as pd
from collections import defaultdict

from torchvision.datasets.mnist import Callable


class CSVLogger:
    def __init__(self,
                 loss_log_root: str,
                 best_train_val: float = 1e6,
                 train_check: Callable = lambda cur, prev: cur < prev,
                 best_validation_val: float = 1e6,
                 validation_check: Callable = lambda cur, prev: cur < prev):
        
        self.loss_log_root = loss_log_root

        self.train_history = defaultdict(list)
        self.validation_history = defaultdict(list)

        self._epoch_history = defaultdict(list)
        self._avg_epoch_history = defaultdict(float)

        self.best_train_val = best_train_val
        self.train_check = train_check

        self.best_validation_val = best_validation_val
        self.validation_check = validation_check


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
                self.validation_history[k].append(self._avg_epoch_history[k])
        
        self._epoch_history = defaultdict(list)
        self._avg_epoch_history = defaultdict(float)


    def save_checkpoint_flag(self, which, check_key="total_loss"):
        save_ckp = False

        if which == "train":
            value = self.train_history[check_key][-1]
            if self.train_check(value, self.best_train_val):
                save_ckp = True
                self.best_train_val = value

        elif which == "val":
            value = self.validation_history[check_key][-1]
            if self.validation_check(value, self.best_validation_val):
                save_ckp = True
                self.best_validation_val = value
        
        return save_ckp
