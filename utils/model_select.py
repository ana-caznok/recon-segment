from torch import nn
from typing import Dict, Any
from seg_recon_vit3d import *
from Restormer import Restormer
from seg_recon_vit3d_stable import *
from utils import *
import wandb
from utils.load_ckpt import load_best_ckpt


def model_select(configs: Dict[str, Any]) -> nn.Module:
    '''
    Returns the appropriate initialized model given "model" config in .yaml file
    and possibly other hyperparameters included in config file

    '''
    model_str = configs.get("model")
    parc_str = model_str.split('_')
     
    
    if 'seg-rec' in parc_str: 
       
        fromto = parc_str[1].split('to')
        start = int(fromto[0])
        tot_channels = int(fromto[1])

        if 'ifft' in model_str: 
            need_ifft = True
        else: 
            need_ifft = False
        
        if 'attn' in model_str: 
            num_heads = int(model_str.split('attn')[1])
        else: 
            num_heads = 12
        
        if 'patch' in model_str: 
            patch_size = int(model_str.split('patch')[1])
        else: 
            patch_size = 32

        
        if 'stable' in parc_str: 
            if 'stabmin' in parc_str: 
                xminmax = True 
            else: 
                xminmax = False
                
            model = SegRecon_ViT_3D_StableTrain(C_input=start, total_channels=tot_channels, patch_size=patch_size, xminmax=xminmax, num_heads=num_heads, ifft=need_ifft)
        
        else: 
            model = SegRecon_ViT_3D(C_input=start, total_channels=tot_channels, patch_size=patch_size ,num_heads=num_heads, ifft=need_ifft)

    elif model_str=='restormer': 
         model = Restormer(inp_channels=4,out_channels=61)
    
    try:
        model, resume_file, best_loss = load_best_ckpt(model, configs, exact_only=True)
        checkpoint = torch.load(resume_file, map_location='cpu')
        last_epoch = checkpoint['epoch']
        configs['start_epoch'] = last_epoch + 1
        total_epochs = configs['train']['epochs']
        configs['train']['epochs'] = max(total_epochs, last_epoch + 1)

        print(f"Resuming from epoch {last_epoch + 1}/{configs['train']['epochs']}")

        # If W&B run ID is in checkpoint, resume it
        wandb_id = checkpoint.get('wandb_run_id', None)
        #wandb_id = configs['resume_wandb_id']
        configs['wandb_id'] = wandb_id
        if wandb_id:
            print(f"Resuming W&B run ID: {wandb_id}")
            wandb.init(id=wandb_id, 
                       resume="allow", 
                       config=configs, 
                       project="SegRecViT", 
                       entity='rainbow-ai')
        else:
            print("No W&B ID found in checkpoint — starting new run.")

    except Exception as e:
        print("No checkpoint found or error loading — starting from scratch.")
        print(e)
        configs['start_epoch'] = 0  # start fresh

      
    
    
    print(f"Selected model {model_str}: {model.__class__.__name__}")

    return model, configs