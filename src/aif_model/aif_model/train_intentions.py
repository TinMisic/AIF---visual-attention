import os
from typing import Any

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as tt

from aif_model.networks import FullyConnected
import aif_model.utils as utils
import config as c


class Trainee(pl.LightningModule):

    def __init__(self, model, config, log_dir):
        super().__init__()
        self.model = model
        self.config = config
        self.history = {
            "train_loss": [],
            "train_grad_norm": [],
            "val_loss": [],
            "learning_rate": [],
        }
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), self.config["lr"], weight_decay=self.config["reg"],
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, total_iters=3000), # learning rate warm-up),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Compute the forward pass and the loss.
        x, y = batch
        total_loss = self.model.loss(x,y)
        self.history["train_loss"].append(total_loss.item())

        # Log the current learning rate.
        lr = self.lr_schedulers().get_last_lr()[0]
        self.history["learning_rate"].append(lr)

        return total_loss

    def on_before_optimizer_step(self, optimizer):
        total_grad_norm = torch.norm(torch.stack([
            torch.norm(p.grad) for p in self.model.parameters()
        ]))
        self.history["train_grad_norm"].append(total_grad_norm.item())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_loss = self.model.loss(x,y)
        self.history["val_loss"].append(val_loss.item())

    def on_train_epoch_end(self):
        pass

    def on_train_end(self):
        # Save the model.
        torch.save(self.model.cpu(), "intentions.pt")

        # Plot the training losses.
        n_steps = len(self.history["train_loss"])
        n_epochs = len(self.history["val_loss"])
        xs = np.linspace(0, n_epochs, n_steps)
        fig, ax = plt.subplots()
        ax.set_title("Loss value during training")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.plot(xs, self.history["train_loss"], lw=0.6, label="Total Train Loss")
        ax.plot(np.arange(n_epochs), self.history["val_loss"], lw=3., label="Validation Loss")
        ax.legend()
        fig.savefig(os.path.join(self.log_dir, "training_loss.png"))
        plt.close(fig)

        # Plot the gradient norm.
        fig, ax = plt.subplots()
        ax.set_title("Gradient Norm during training")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gradient Norm")
        ax.plot(self.history["train_grad_norm"], lw=0.6)
        fig.savefig(os.path.join(self.log_dir, "grad_norm.png"))
        plt.close(fig)

        # Plot the learning rate schedule.
        fig, ax = plt.subplots()
        ax.set_title("Learning rate schedule")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Learning rate")
        ax.plot(self.history["learning_rate"])
        fig.savefig(os.path.join(self.log_dir, "learning_rate.png"))
        plt.close(fig)

def load_dataset(args):
    data_path = 'datasets/'+args.ds+'/translation.csv'
    ds = utils.IntentionDataset(data_path)
    train_gen, test_gen = utils.split_dataset(ds, percent=0.75)
    return (train_gen,test_gen)

def main(args):
    pl.seed_everything(args.seed)
    plt.style.use("ggplot")

    # Load the dataset
    train_loader, test_loader = load_dataset(args)

    # Initialize the network
    net = FullyConnected(c.latent_size, 2 * c.num_intentions)

    # Train using Pytorch Lightning.
    trainee = Trainee(
        model=net,
        config={"lr": args.lr, "reg": args.reg},
        log_dir="logs",
    )
    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        deterministic=True,
        max_epochs=args.epochs,
        enable_progress_bar=True,
        gradient_clip_val=args.clip_grad,
        # logger=None,
    )
    trainer.fit(
        model=trainee, train_dataloaders=train_loader, val_dataloaders=test_loader
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=c.learning_rate, type=float)
    parser.add_argument("--reg", default=1e-4, type=float)
    parser.add_argument("--epochs", default=c.n_epochs, type=int)
    parser.add_argument("--batch_size", default=c.n_batch, type=int)
    parser.add_argument("--latent_dim", default=c.latent_size, type=int)
    parser.add_argument("--clip_grad", default=0, type=float)
    parser.add_argument("--ds", default="32closer",type=str)
    args = parser.parse_args()

    main(args)

