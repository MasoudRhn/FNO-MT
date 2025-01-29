import torch
import json
import time
import wandb
import importlib
from dataset import OneToOnedataset
from dataset import Inferencedataset
from torch.utils.data import DataLoader
from Loss_Functions import h1_loss
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from math import sqrt
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger


class HITDataModule(pl.LightningDataModule):
    def __init__(self, config_path="config.json"):
        super().__init__()
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def train_dataloader(self):

        json_train_file = "dataset/OneToOne/timestep_data_train.json"
        train_dataset = OneToOnedataset.OneToOneDataset(json_train_file)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["mgpu_batch_size"],
            num_workers=15,
            shuffle=False,
        )

        return train_loader

    def val_dataloader(self):
        json_val_file = "dataset/OneToOne/timestep_data_validation.json"
        val_dataset = OneToOnedataset.OneToOneDataset(json_val_file)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["mgpu_batch_size"],
            num_workers=15,
            shuffle=False,
        )

        return val_loader


class HITFNO_Lightning(pl.LightningModule):

    def __init__(self, config_path="config.json"):
        super().__init__()
        with open(config_path, "r") as f:
            self.config = json.load(f)
        model_module = importlib.import_module(f"models.{self.config['model_name']}")
        model_class = getattr(model_module, "Net3d")
        self.model = model_class(
            self.config["mode1"],
            self.config["mode2"],
            self.config["mode3"],
            self.config["width"],
        )
        self.h1_loss = h1_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config["scheduler_step"],
            gamma=self.config["scheduler_gamma"],
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        input = batch[:, 0]
        true_output = batch[:, 1]
        pred = self.model(input)
        loss = self.h1_loss(pred, true_output, alpha=0.01)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input = batch[:, 0]
        true_output = batch[:, 1]
        pred = self.model(input)
        loss = self.h1_loss(pred, true_output, alpha=0.01)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


if __name__ == "__main__":

    pl.seed_everything(32, workers=True)
    torch.manual_seed(32)
    torch.cuda.manual_seed_all(32)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    data_loader = HITDataModule()
    model = HITFNO_Lightning()

    csv_logger = CSVLogger(save_dir="logs", name="masoudtest1_multigpu")

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        logger=csv_logger,
        log_every_n_steps=100,
    )

    trainer.fit(model, data_loader)
