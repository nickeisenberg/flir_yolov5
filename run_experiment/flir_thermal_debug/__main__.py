from sys import path
path.append(__file__.split("run_experiment")[0])

from config import config_trainer
from src.trainer.trainer import Trainer
from src.trainer.logger import config_log_roots 

config = config_trainer()

trainer = Trainer(
    train_module=config["train_module"],
)

if __name__ == "__main__":
    config_log_roots(trainer.train_module.logger)

    trainer.fit(
        train_loader=config["train_loader"],
        num_epochs=config["num_epochs"],
        device=config["device"],
        val_loader=config["val_loader"],
        unpacker=config["unpacker"]
    )
