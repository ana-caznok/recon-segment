from torch import nn
from typing import Dict, Any
from seg_recon_vit3d import *
from Restormer import Restormer


def model_select(configs: Dict[str, Any]) -> nn.Module:
    '''
    Returns the appropriate initialized model given "model" config in .yaml file
    and possibly other hyperparameters included in config file

    '''
    model_str = configs.get("model")
    parc_str = model_str.split('_')
    
    if 'seg-rec' in parc_str: 
        if len(parc_str)>1: 
            tot_channels = int(parc_str[1])
        else: 
            tot_channels = 61

        model = SegRecon_ViT_3D(C_input=31, total_channels=tot_channels)

    elif model_str=='restormer': 
         model = Restormer(inp_channels=4,out_channels=61)      
    
    
    print(f"Selected model {model_str}: {model.__class__.__name__}")

    return model