import os
import sys
import glob
import json
import torch
import h5py
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Callable, List, Dict, Tuple, Any
from torch.utils.data import DataLoader, Dataset

# Custom imports from project structure
from fixed_dataset import FixedDataset
from seg_recon_vit3d import SegRecon_ViT_3D
from transforms import * 
from transforms.factory import transform_factory
from utils.read_yaml import read_yaml
from utils.model_select import model_select
from utils.configwandb import config2wandb
from loss import loss_select
from utils.fixed_save_checkpoint import fixed_save_checkpoint
from utils.save_checkpoint_model import save_checkpoint
from torchmetrics.image import StructuralSimilarityIndexMeasure
from metrics.sam import SAMScore
from transforms.ifft_transform import InverseFourierSpectralTransform


# Argument parser
parser = argparse.ArgumentParser(description="Train 3D ViT model using YAML config.")
parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")
args = parser.parse_args()

# Clean GPU memory
torch.cuda.empty_cache()

# Load YAML config
config = read_yaml(args.config)
config_file = args.config.split('/')[-1]


# --------------------- CONFIG -----------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESSING = config['preprocessing']  # or None or "downsampled"
BATCH_SIZE = config['train']['batch_size']
NUM_EPOCHS = config['train']['epochs']
LEARNING_RATE = config['train']['lr']
TRANSFORM = config['train']['transform_index']
MODEL_NAME = config['model']
fixed_checkpoint_name = config["fixed_checkpoint_name"]
SAVE_STEPS = int(config['save_steps'])
need_ifft = bool(config['ifft'])

try:
    CODE_PATH = os.environ.get("CODE_PATH")
    BASE_PATH = os.environ.get(config['base_path'])
    SAVE_PATH =  os.path.join(CODE_PATH, config['save_checkpoint_path'])
    print('SAVE_PATH: ' + SAVE_PATH)

except:
    print("Oops.. Path error, be sure to set a CODE_PATH and DATA_PATH system variables") 

print('saving to :' + SAVE_PATH + fixed_checkpoint_name)


if need_ifft: 
    if 'shift' in TRANSFORM: 
        shift_fft = True
    else: 
        shift_fft = False
    if 'continuous' in TRANSFORM:
        stack_type = 'continuous'
    else: 
        stack_type = 'alternate'

    ifft_function  = InverseFourierSpectralTransform(
                    norm = 'minmax',
                    channel_first = True, 
                    device = 'cuda', 
                    shift = shift_fft, 
                    stack_type = stack_type)


# ------------------ DATASET + LOADER -----------------------------------------
train_dataset = FixedDataset(
    mode="train",
    base_path=BASE_PATH,
    transform=transform_factory(TRANSFORM),  # You can define transforms here
    preprocessing=PREPROCESSING
)

val_dataset = FixedDataset(
    mode="val",
    base_path=BASE_PATH,
    transform=transform_factory(TRANSFORM),
    preprocessing=PREPROCESSING
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ------------------ MODEL AND OPTIMIZER -------------------------------------------------

model = model_select(config).to(DEVICE) # Select model and loss function
criterion = loss_select(config)


optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Optional scheduler
scheduler = None
if config.get("scheduler", False):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


# ------------------ INITIALIZING WANDB ---------------------------------------------------

name = os.path.basename(config_file).replace(".yaml", '')
run = wandb.init(project="SegRecViT",
                            reinit=True,
                            config = config2wandb(config), 
                            entity= 'rainbow-ai', 
                            notes="Running experiment",
                            name=name)


# ------------------ TRAIN LOOP -----------------------------------------------------------
trn_history = []
val_history = []
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for x, y, meta in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
        
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        output = model(x)

        if need_ifft:
            # Compute per-sample min/max
            meta['fft_ymin'] = torch.amin(output, dim=(1, 2, 3), keepdim=True).detach()  # (B, 1, 1, 1)
            meta['fft_ymax'] = torch.amax(output, dim=(1, 2, 3), keepdim=True).detach()  # (B, 1, 1, 1)
            #print(f'output shape: {output.shape}')
            #print(f"meta shape: {meta['fft_ymin'].shape}, {meta['fft_ymax'].shape}")
            output = ifft_function(output, meta)

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()


    avg_train_loss = train_loss / len(train_loader)
    trn_history.append(avg_train_loss)

    if config['wandb']:
        wandb.log({f"Train Avg Loss {config['loss']}": avg_train_loss})
        wandb.log({"epoch": epoch})

    # ------------------ EVAL LOOP --------------------------------------------------------
    model.eval()
    val_loss = 0.0
    i = 0
    ssim_function = StructuralSimilarityIndexMeasure().to('cpu')
    sam_function = SAMScore().to('cpu')
    ssim = 0
    sam = 0

    with torch.no_grad():
        for x, y, meta in tqdm(val_loader, desc="Validation"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            output = model(x)
            if need_ifft: 
                meta['fft_ymin'] = torch.amin(output, dim=(1, 2, 3), keepdim=True).detach()  # (B, 1, 1, 1)
                meta['fft_ymax'] = torch.amax(output, dim=(1, 2, 3), keepdim=True).detach()  # (B, 1, 1, 1)
                output = ifft_function(output,meta)
                
            loss = criterion(output, y)
            val_loss += loss.item()
            i += 1
            ssim += ssim_function(output.cpu(),y.cpu())
            sam += sam_function(output.cpu(),y.cpu())

    avg_val_loss = val_loss / len(val_loader)
    val_history.append(avg_val_loss)
    
    avg_ssim = ssim/len(val_loader)
    avg_sam = sam/len(val_loader)

    if config['wandb']:
        wandb.log({f"Validation Avg Loss {config['loss']}": avg_val_loss})
        wandb.log({"val_epoch": epoch})
        wandb.log({"SSIM": avg_ssim})
        wandb.log({"SAM": avg_sam})

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    if avg_val_loss == np.array(val_history).min(): 

        save_checkpoint(model,SAVE_PATH + fixed_checkpoint_name + '.pth', epoch, i, optimizer, avg_val_loss)
        print(f"Best model saved to {SAVE_PATH + fixed_checkpoint_name}")
    
    if epoch%SAVE_STEPS ==0: 
        save_checkpoint(model,SAVE_PATH + fixed_checkpoint_name + '.pth' , epoch, i, optimizer, avg_val_loss)
        print(f"Step model saved to {SAVE_PATH + fixed_checkpoint_name}")
    
     # Step scheduler
    if scheduler:
        scheduler.step()


# ------------------ SAVE MODEL -------------------------------------------------------------------------------
save_checkpoint(model,SAVE_PATH + fixed_checkpoint_name + '.pth', epoch, i, optimizer, avg_val_loss)
print(f"Final Model saved to {SAVE_PATH}")



