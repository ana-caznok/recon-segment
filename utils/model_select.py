from torch import nn
from typing import Dict, Any
from seg_recon_vit3d import *
from Restormer import Restormer
from seg_recon_vit3d_stable import *
from seg_recon_vit3d_restormer import *
from seg_recon_vit3d_dualtask import * 
from utils import *
import wandb
from utils.load_ckpt import load_best_ckpt
from seg_recon_vit3d_overlap import *
from HSCNN_Plus import *
from Restormer_Seg import *
from monai.networks.nets import UNet


def model_select(configs: Dict[str, Any]) -> nn.Module:
    '''
    Returns the appropriate initialized model given "model" config in .yaml file
    and possibly other hyperparameters included in config file

    '''
    model_str = configs.get("model")
    parc_str = model_str.split('_')

    if 'to' in model_str:
        fromto = parc_str[1].split('to')
        start = int(fromto[0])
        tot_channels = int(fromto[1])
    else: 
        start = 4
        tot_channels = 61
    
    if 'seg-rec' in parc_str: 

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

        if 'nconv' in model_str: 
            nconv = int(model_str.split('nconv')[1])
        else: 
            nconv=1

        if 'feedfoward' in model_str: 
            ff = True
        else: 
            ff=False

        if 'stable' in parc_str: 
            if 'stabmin' in parc_str: 
                xminmax = True 
            else: 
                xminmax = False
                
            model = SegRecon_ViT_3D_StableTrain(C_input=start, total_channels=tot_channels, patch_size=patch_size, xminmax=xminmax, num_heads=num_heads, ifft=need_ifft)
        
        elif 'rest' in parc_str: 
            model = SegRecon_ViT_3D_Rest(C_input=start, total_channels=tot_channels, patch_size=16, num_heads=12, ifft=need_ifft)
        
        elif 'dual-task' in parc_str: 
            model = SegRecon_ViT_3D_DualTask(C_input=start, total_channels=tot_channels, patch_size=16, num_heads=12, ifft=need_ifft)

        elif 'overlap' in parc_str: 
            model = SegRecon_ViT_3D_Overlap(C_input=start, total_channels=tot_channels, patch_size=32, num_heads=12, ifft=need_ifft,overlap=True, nconv=nconv)

        else: 
            model = SegRecon_ViT_3D(C_input=start, total_channels=tot_channels, patch_size=patch_size ,num_heads=num_heads, ifft=need_ifft, feedfoward=ff)

    elif 'restormer' in model_str: 
        if 'seg' in model_str: 
            dec = 'complex' if 'complex' in model_str else 'simple'
            model = Restormer_Seg(inp_channels=start,out_channels=tot_channels,my_seg_decoder=dec)
        else:
            model = Restormer(inp_channels=start,out_channels=tot_channels)

    elif 'hscnn' in model_str: 
        model = HSCNN_Plus(in_channels=start, out_channels=tot_channels)

    elif 'monai-unet' in model_str:

        # Extract UNet-specific configs with defaults
        spatial_dims = configs.get("spatial_dims", 2)
        channels = configs.get("channels", [16, 32, 64, 128])
        strides = configs.get("strides", [2, 2, 2])
        kernel_size = configs.get("kernel_size", 3)
        up_kernel_size = configs.get("up_kernel_size", 3)
        num_res_units = configs.get("num_res_units", 2)
        act = configs.get("act", "PRELU")
        norm = configs.get("norm", "INSTANCE")
        dropout = configs.get("dropout", 0.0)
        bias = configs.get("bias", True)
        adn_ordering = configs.get("adn_ordering", "NDA")

        # Instantiate MONAI UNet
        model = UNet(
            spatial_dims=spatial_dims,
            in_channels=start,
            out_channels=tot_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering
        )
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