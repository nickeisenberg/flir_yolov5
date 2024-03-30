import os
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

mnist = MNIST(
    os.path.expanduser("~/Datasets/mnist/"), True, transform=ToTensor()
)
train_dataset = Subset(mnist, range(50000))
val_dataset = Subset(mnist, range(50000, 60000))

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


model = Model()

save_root = os.path.relpath(__file__)
save_root = save_root.split(os.path.basename(save_root))[0]

model = Model()
loss_fn = Loss()
optimizer = Adam(model.parameters(), .0001)

train_loader = DataLoader(train_dataset, 32, shuffle=True)
val_loader = DataLoader(val_dataset, 32)

num_epochs = 1

device = "cuda:0"

config = {
    "save_root": save_root,
    "model": model,
    "loss_fn": loss_fn,
    "optimizer": optimizer,
    "train_loader": train_loader,
    "val_loader": val_loader,
    "test_loader": test_loader,
    "num_epochs": num_epochs,
    "device": device
}

from tqdm import tqdm
import torch

model = model.to(device)

pbar = tqdm(train_loader)
running = 0
running_correct = 0
total = 0
for i, (inputs, targets) in enumerate(pbar):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    preds = model(inputs)
    loss, _ = loss_fn(preds, targets)
    loss.backward()
    optimizer.step()
    running += loss.item()
    avg = running / (i + 1)
    running_correct += (torch.argmax(preds, dim=-1) == targets).sum().item()
    total += len(targets) 
    pbar.set_postfix(loss=avg, acc=(running_correct/total))


pbar = tqdm(val_loader)
running = 0
running_correct = 0
total = 0
model.eval()
for i, (inputs, targets) in enumerate(pbar):
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
        preds = model(inputs)
        loss, _ = loss_fn(preds, targets)
        running += loss.item()
        avg = running / (i + 1)
        running_correct += (torch.argmax(preds, dim=-1) == targets).sum().item()
        total += len(targets) 
        pbar.set_postfix(loss=avg, acc=(running_correct/total))
