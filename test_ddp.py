import os
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import Callback
import torch.distributed as dist
import json

# Initialize the process group
if not dist.is_initialized():
    dist.init_process_group(backend='nccl')

# Check number of GPUs using PyTorch
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'Not set')}")
print(f"SLURM_STEP_GPUS: {os.environ.get('SLURM_STEP_GPUS', 'Not set')}")
print(f"SLURM_JOB_GPUS: {os.environ.get('SLURM_JOB_GPUS', 'Not set')}")

print(f"Torch CUDA device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(50, 10)  # 50 samples, 10 features
        self.labels = torch.randint(0, 2, (50,))  # 50 labels (binary classification)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Dummy model
class DummyModel(pl.LightningModule):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.layer = torch.nn.Linear(10, 2)  # Simple linear layer

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        output = self(data)
        loss = torch.nn.functional.cross_entropy(output, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        output = self(data)
        loss = torch.nn.functional.cross_entropy(output, labels)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Custom callback to log distributed training initialization
class DistributedTrainingLogger(Callback):
    def on_train_start(self, trainer, pl_module):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Initializing distributed: GLOBAL_RANK: {rank}, WORLD_SIZE: {world_size}")

# Custom callback to count gradient steps and store data
class GradientStepCounter(Callback):
    def __init__(self):
        self.total_steps = 0
        self.step_data = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.total_steps += 1
        self.step_data.append({
            'batch_idx': batch_idx,
        })

    def on_train_epoch_end(self, trainer, pl_module):
        # Gather the total steps from all processes
        total_steps_tensor = torch.tensor(self.total_steps, device=pl_module.device)
        dist.all_reduce(total_steps_tensor, op=dist.ReduceOp.SUM)

        # Save local step data
        local_file_path = f"step_data_rank_{trainer.global_rank}_epoch_{trainer.current_epoch}.json"
        with open(local_file_path, 'w') as f:
            json.dump(self.step_data, f)

        # Gather step data from all ranks
        all_step_data = [None] * dist.get_world_size()
        dist.all_gather_object(all_step_data, self.step_data)

        if trainer.is_global_zero:
            # Combine all step data and save to a final file
            combined_step_data = [step for rank_data in all_step_data for step in rank_data]
            combined_file_path = f"combined_step_data_epoch_{trainer.current_epoch}.json"
            with open(combined_file_path, 'w') as f:
                json.dump(combined_step_data, f)
            print(f"Total steps taken in epoch {trainer.current_epoch}: {total_steps_tensor.item()}")
            print(f"Combined step data saved to '{combined_file_path}'")
        
        # Reset step data for the next epoch
        self.step_data = []

# Initialize dataset and dataloader with DistributedSampler
train_dataset = DummyDataset()
# train_sampler = DistributedSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# val_dataset = DummyDataset()
# val_sampler = DistributedSampler(val_dataset)
# val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, sampler=val_sampler)

# Initialize model
model = DummyModel()

# Initialize the custom callbacks
distributed_training_logger = DistributedTrainingLogger()
gradient_step_counter = GradientStepCounter()

# Initialize trainer
trainer = Trainer(
    accelerator='gpu',
    devices=num_gpus,
    strategy=DDPStrategy(find_unused_parameters=True),
    max_epochs=10,
    callbacks=[distributed_training_logger, gradient_step_counter]
)

# Run training
trainer.fit(model, train_dataloaders=train_dataloader)
