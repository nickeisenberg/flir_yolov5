from config import config
from src.trainer.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(
        config["model"], 
        config["loss_fn"], 
        config["optimizer"],
        config["save_root"]
    )
    trainer.fit(
        train_loader=config["train_loader"],
        num_epochs=config["num_epochs"],
        device=config["device"],
        val_loader=config["val_loader"]
    )
