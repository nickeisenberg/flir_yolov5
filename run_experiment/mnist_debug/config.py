import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from src.trainer.logger import CSVLogger
from src.trainer.metrics import Accuracy 


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(30976, 256)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.linear1(self.flatten(x))
        x = self.linear2(x)
        return x


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        loss = self.loss(predictions, targets)
        return loss, {"total_loss": loss.item()}


def config_logger():
    save_root = os.path.relpath(__file__)
    save_root = save_root.split(os.path.basename(save_root))[0]
    logger = CSVLogger(
        os.path.join(save_root, "loss_logs"), 
        os.path.join(save_root, "state_dicts"), 
    )
    return logger


class TrainerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.loss_fn = Loss()
        self.optimizer = Adam(self.model.parameters(), .0001)

        self.logger = config_logger()
        self.metrics = [Accuracy()]


    def forward(self, x):
        return self.model(x)


    def train_batch_pass(self, *args):
        self.model.train()

        inputs, targets = args

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss, batch_history = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

        self.logger.log_batch(batch_history)
        
        if self.metrics:
            for metric in self.metrics:
                metric.log(torch.argmax(outputs, -1), targets)
                self.logger.log_batch(metric.metric)


    def val_batch_pass(self, *args):
        self.model.eval()

        inputs, targets = args

        with torch.no_grad():
            outputs = self.model(inputs)
            _, batch_history = self.loss_fn(outputs, targets)
            self.logger.log_batch(batch_history)

        if self.metrics:
            for metric in self.metrics:
                metric.log(torch.argmax(outputs, -1), targets)
                self.logger.log_batch(metric.metric)


def config_loaders():
    mnist = MNIST(
        os.path.expanduser("~/Datasets/mnist/"), True, transform=ToTensor()
    )
    train_dataset = Subset(mnist, range(50000))
    val_dataset = Subset(mnist, range(50000, 60000))
    train_loader = DataLoader(train_dataset, 32, shuffle=True)
    val_loader = DataLoader(val_dataset, 32)
    return train_loader, val_loader


def config_trainer():
    train_module = TrainerModule()
    train_loader, val_loader = config_loaders()
    num_epochs = 1
    device = "cuda:0"
    config = {
        "train_module": train_module,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "num_epochs": num_epochs,
        "device": device
    }
    return config
