'''
For those that don't have a lot of space free, saves best model with the same name
'''
import torch
import os


def save_checkpoint(model, save_path_name, epoch, iteration, optimizer, best_loss, disable=False):
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
        torch.save(state, save_path_name)