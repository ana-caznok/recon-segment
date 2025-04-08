'''
For those that don't have a lot of space free, saves best model with the same name
'''
import torch
import os


def save_checkpoint(model, save_path_name, epoch, iteration, optimizer, best_loss, disable=False, wandb_id=None):
    # Prepare checkpoint state
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': best_loss
    }

    # Include wandb ID if provided
    if wandb_id is not None:
        state['wandb_run_id'] = wandb_id

    # Save checkpoint unless disabled
    if disable:
        print("WARNING: Not saving checkpoint")    
    else:
        torch.save(state, save_path_name)