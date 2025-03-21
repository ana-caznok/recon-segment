'''
For those that don't have a lot of space free, saves best model with the same name
'''
import torch
import os


def fixed_save_checkpoint(model_path, epoch, iteration, model, optimizer, best_loss, fixed_checkpoint_name, disable=False):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': best_loss
    }
    if disable:
        print("WARNING: Not saving checkpoint")    
    else:
        torch.save(state, os.path.join(model_path, f'{fixed_checkpoint_name}.pth'))