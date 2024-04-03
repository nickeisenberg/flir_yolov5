from sys import path
path.append(__file__.split("run_experiment")[0])

from config import config_trainer
from src.trainer.trainer import Trainer

config = config_trainer()

if __name__ == "__main__":
    trainer = Trainer(
        train_module=config["train_module"],
    )


    trainer.fit(
        train_loader=config["train_loader"],
        num_epochs=config["num_epochs"],
        device=config["device"],
        unpacker=config["unpacker"]
    )
