from sys import path

from torch.nn import DataParallel
path.append(__file__.split("run_experiment")[0])

from config import config_trainer
from src.trainer.trainer import Trainer

config = config_trainer()

if __name__ == "__main__":
    trainer = Trainer(
        train_module=config["train_module"],
    )
    
    device = config["device"]
    if isinstance(device, str) or isinstance(device, int):
        device = device
    elif isinstance(device, list):
        device = device[0]
    else:
        raise Exception("device error")
    
    assert type(device) == str or type(device) == int

    trainer.fit(
        train_loader=config["train_loader"],
        num_epochs=config["num_epochs"],
        device=device,
        unpacker=config["unpacker"]
    )
