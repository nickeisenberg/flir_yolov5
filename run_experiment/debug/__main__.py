from sys import path
path.append(__file__.split("run_experiment")[0])

from config import config
from src.experiment.experiment import Trainer


trainer = Trainer(
    model=config["model"], 
    loss_fn=config["loss_fn"], 
    optimizer=config["optimizer"],
    save_root=config["save_root"]
)

trainer.fit(
    train_loader=config["train_loader"], 
    num_epochs=config["num_epochs"], 
    device=config["device"],
    val_loader=config["val_loader"],
    test_loader=config["test_loader"]
)
