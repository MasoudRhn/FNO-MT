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
import random
import numpy as np
from neuralop.models import FNO


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(32)


class Trainer:
    def __init__(
        self,
        gpu_id,
        config_path="config.json",
    ):

        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)

        self.gpu_id = gpu_id
        self.world_size = torch.cuda.device_count()
        wandb.init(
            project="January", name=f"{self.config['experiment']}", config=self.config
        )
        self.config = wandb.config

        model_module = importlib.import_module(f"models.{self.config.model_name}")
        model_class = getattr(model_module, "Net3d")
        self.model = model_class(
            self.config.mode1, self.config.mode2, self.config.mode3, self.config.width
        )

        self.model = self.model.to(self.gpu_id)
        self.model = DDP(
            self.model, device_ids=[self.gpu_id], find_unused_parameters=False
        )

        if gpu_id == 0:
            wandb.watch(self.model, log="all")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate * self.world_size,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.scheduler_step,
            gamma=self.config.scheduler_gamma,
        )

        json_train_file = "dataset/OneToOne/timestep_data_train.json"
        json_val_file = "dataset/OneToOne/timestep_data_validation.json"
        json_test_file = "dataset/OneToOne/timestep_data_inference.json"

        train_dataset = OneToOnedataset.OneToOneDataset(json_train_file)

        if self.gpu_id == 0:
            val_dataset = OneToOnedataset.OneToOneDataset(json_val_file)

        # inference_dataset = Inferencedataset.InferenceDataSet(json_test_file)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.mgpu_batch,
            num_workers=4,
            shuffle=False,
            drop_last=True,
            sampler=DistributedSampler(train_dataset, shuffle=False),
        )
        if self.gpu_id == 0:
            self.val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False
            )

        # self.inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

    def validate_model(self):
        if self.gpu_id == 0:
            self.model.eval()
            total_loss = 0
            with torch.no_grad():
                for x in self.val_loader:
                    x = x.to(self.gpu_id, dtype=torch.float32)

                    current_input = x[:, 0]
                    pred = self.model(current_input)
                    true_output = x[:, 1]
                    h1_Loss = h1_loss(pred, true_output)
                    total_loss += h1_Loss.item()

            # print(f"i am validation loss in mgpu = {total_loss} with trainloader size of = {len(self.val_loader)}")

            avg_loss = total_loss / (len(self.val_loader))

        return avg_loss

    def train(self):
        if self.gpu_id == 0:
            train_losses = []
            val_losses = []
            best_val_loss = float("inf")
            patience_counter = 0
            patience = 5
        for epoch in range(self.config.epochs):

            self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            total_loss = 0

            start_time = time.time()

            # print(f"Before update at {self.gpu_id}: {self.model.module.conv1.conv0.weights1[0,0,0,0,0]}")

            for batch_idx, x in enumerate(self.train_loader):

                # print(f"shape of x = {x.shape}")
                x = x.to(self.gpu_id, dtype=torch.float32)
                # print(f"i am gpu {self.gpu_id} my train loader size is {len(self.train_loader)}")
                # print(f"i am gpu {self.gpu_id} I am recieving x with shape of {x.shape}")
                self.optimizer.zero_grad()
                input = x[:, 0]
                true_output = x[:, 1]
                pred = self.model(input)
                h1_Loss = h1_loss(pred, true_output, 0.01)
                Loss_cum = h1_Loss
                Loss_cum.backward()
                self.optimizer.step()
                # print(f"after update at {self.gpu_id}: {self.model.module.conv1.conv0.weights1[0,0,0,0,0]}")

                total_loss += Loss_cum.item()
                # if self.gpu_id==0:
                # print(f"batch = {batch_idx}i am gpu {self.gpu_id} my gpu total_loss is  {total_loss}")

            self.scheduler.step()
            epoch_time = time.time() - start_time
            print(f"Time on GPU {self.gpu_id} for epoch {epoch + 1}: {epoch_time:.2f}s")
            print(f"i am gpu {self.gpu_id} my gpu total_loss is  {total_loss}")

            if self.gpu_id == 0:
                avg_train_loss = total_loss / (len(self.train_loader))
                train_losses.append(avg_train_loss)
                print(f"loss of = {train_losses}")
                avg_val_loss = self.validate_model()
                val_losses.append(avg_val_loss)

                wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss})
                epoch_time = time.time() - start_time
                print(
                    f"Epoch [{epoch + 1}/{self.config.epochs}], "
                    f"Train Loss: {avg_train_loss:.7f}, "
                    f"Val Loss: {avg_val_loss:.7f}, "
                    f"Time: {epoch_time:.2f}s, "
                    f"LearningRate: {self.optimizer.param_groups[0]['lr']}"
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(
                        f"Early stopping patience counter: {patience_counter}/{patience}"
                    )

                if patience_counter >= patience or (epoch + 1) == self.config.epochs:
                    final_model_path = f"saved_models/{self.config['model_name']}_{self.config['experiment']}_final.pth"
                    # torch.save(self.model.state_dict(), final_model_path)
                    torch.save(self.model.module.state_dict(), final_model_path)
                    print(f"Model saved to {final_model_path} at epoch {epoch + 1}")
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break

        # return

    def inference(self):
        saved_model_path = f"saved_models/{self.config['model_name']}_{self.config['experiment']}_final.pth"
        self.model.load_state_dict(
            torch.load(saved_model_path, map_location=self.device)
        )
        self.model.eval()
        print(f"Loaded model from {saved_model_path}")

        inference_loss = []

        input_data = None

        for i, x in enumerate(self.inference_loader):
            x = x.to(self.device, dtype=torch.float32)

            if input_data is None:

                input_data = x
                continue

            with torch.no_grad():

                prediction = self.model(input_data)
                h1_Loss = h1_loss(prediction, x, 0.01)
                inference_loss.append(h1_Loss.item())

                wandb.log({"inference_loss": h1_Loss})
                input_data = prediction

        print(f"Inference completed")

        return inference_loss


def ddp_setup(rank: int, world_size: int):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank, world_size):

    ddp_setup(rank, world_size)
    trainer = Trainer(gpu_id=rank)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    # world_size =2
    print(f"world size is = {world_size}")
    mp.spawn(main, args=(world_size,), nprocs=world_size)
