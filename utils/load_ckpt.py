import os
import glob
import torch
from torch import nn
from typing import Dict, Any


def load_best_ckpt(model: nn.Module, configs: Dict[str, Any], exact_only=False, exact_file_name=None):
    # First attempt to load exact config.yaml name
    if exact_file_name is None:
        latest_chp = os.path.join(configs['save_checkpoint_path'], f"{configs['fixed_checkpoint_name']}.pth")
        print(f"Looking for checkpoint in {latest_chp}. Exact path only: {exact_only}.")
    else:
        latest_chp = exact_file_name
        print(f"Loading from checkpoint in {exact_file_name} passed through argument exact_file_name in load_best_ckpt.")

    if not os.path.isfile(latest_chp):
        if not exact_only:
            print(f"Failed to load with exact strategy, searching for checkpoint.")
            if not os.path.exists(configs['save_checkpoint_path']):
                tmp = configs['save_checkpoint_path']
                print(f'Directory {tmp} does not exist!')

            if(len(os.listdir(configs['save_checkpoint_path']))):
                print('There is no checkpoint in checkpoint path folder!')

            if(len(glob.glob(configs['save_checkpoint_path']+'/best*')) > 0 ):
                list_of_chps = glob.glob(os.path.join(configs['save_checkpoint_path'],'best*.pth')) 
            else:
                list_of_chps = glob.glob(os.path.join(configs['save_checkpoint_path'],'*.pth')) 
                print("Warning: There is no checkpoint with best in the name...")

            tmp = configs['save_checkpoint_path']
            print(f'In {tmp} there are :{list_of_chps}')
            latest_chp = max(list_of_chps, key=os.path.getctime)
        else:
            raise ValueError(f"Could't find checkpoint {latest_chp}! :(")
        
    # Loading model
    resume_file = latest_chp
    if resume_file is not None:
        if os.path.isfile(resume_file):
            print("=> Loading model from checkpoint: '{}'".format(resume_file))
            checkpoint = torch.load(resume_file, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            try:
                best_loss = checkpoint['best_loss']
            except:
                print('Forcing best_loss = 1000 due to code version mismatch')
                best_loss = 1000

    print(f"Loaded checkpoint from epoch: {checkpoint['epoch']}")

    return model, resume_file, best_loss