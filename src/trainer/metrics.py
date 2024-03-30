from collections import defaultdict
from torch import Tensor, tensor, hstack
from numpy import round

class Accuracy:
    def __init__(self):
        self.metric = {
            "accuracy": 0.0
        }

        self._running_total = tensor([])

    def log(self, predictions: Tensor, targets: Tensor):
        predictions, targets = predictions.cpu(), targets.cpu()
        acc = ((predictions == targets) * 1).float()
        self._running_total = hstack([self._running_total, acc])
        self.metric["accuracy"] = round(
             self._running_total.mean().item() * 100, 2
         )

    def reset_on_epoch(self):
        self.__init__()
