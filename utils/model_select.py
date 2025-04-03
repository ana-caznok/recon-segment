from torch import nn
from typing import Dict, Any
from seg_recon_vit3d import *
from Restormer import Restormer
from seg_recon_vit3d_stable import *


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
        
        if 'stable' in parc_str: 
            model = SegRecon_ViT_3D_StableTrain(C_input=start, total_channels=tot_channels, ifft=need_ifft)
        
        else: 
            model = SegRecon_ViT_3D(C_input=start, total_channels=tot_channels, ifft=need_ifft)

    elif model_str=='restormer': 
         model = Restormer(inp_channels=4,out_channels=61)

      
    
    
    print(f"Selected model {model_str}: {model.__class__.__name__}")

    return model