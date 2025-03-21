from torch import nn
from typing import Dict, Any

# Models
#from HyperSkinUtils.baseline.MST_Plus_Plus import MST_Plus_Plus
#from unet import UNet
#from mstplusplusplusunet import MSTPlusPlusPlusUNet
#from vitmstppunet import VITMSTPPUNet
#from vitmstpp import VITMSTPP
#from HSCNN_Plus import *
#from hrnet import *
#from Restormer import *
#from vitmstpp_pad_new import *
#from vitmstpp_optimized import *


def model_select(configs: Dict[str, Any]) -> nn.Module:
    '''
    Returns the appropriate initialized model given "model" config in .yaml file
    and possibly other hyperparameters included in config file

    BIG CHANCE 13/11 62 -> 61
    '''
    model_str = configs.get("model")
    if model_str == "baseline_rgb":
        model = MST_Plus_Plus(in_channels=3, out_channels=31, n_feat=31, stage=3)
    elif model_str == "baseline_msi":
        model = MST_Plus_Plus(in_channels=4, out_channels=31, n_feat=31, stage=3)
    elif model_str == "baseline":
        model = MST_Plus_Plus(in_channels=4, out_channels=61, n_feat=61, stage=3)
    elif model_str == "large_baseline":
        model = MST_Plus_Plus(in_channels=4, out_channels=61, n_feat=61, stage=6)
    elif model_str == "unet":
        model = UNet(n_channels=4, n_classes=61, norm=True, dim='2d', init_channel=64)
    elif model_str == "unet_pseudo_3d":
        model = UNet(n_channels=1, n_classes=1, norm="instance", dim='3d', init_channel=16, do_pseudo_3d=True)
    elif model_str == "mstplusplusplusunet":
        model = MSTPlusPlusPlusUNet(upsample=True, size="large")
    elif model_str == "mstplusplusplusunet3d":
        model = MSTPlusPlusPlusUNet(upsample=False, size="large")
    elif model_str == "mstplusplusplusunet3dsmall":
        model = MSTPlusPlusPlusUNet(upsample=False, size="small")
    elif model_str == "vitmstppunet":
        raise DeprecationWarning
        model = VITMSTPPUNet()
    elif model_str == "vitmstpp":
        model = VITMSTPP()
    elif model_str == "hscnn":
        model = HSCNN_Plus(in_channels=4, out_channels=61, num_blocks=30)
    elif model_str=='hrnet': 
        model = SGN(in_channels=4, out_channels=61, start_channels=64)
    elif model_str=='restormer': 
        model = Restormer(inp_channels=4,out_channels=61)
    elif model_str=='vitmstpp_pad': 
        model = VITMSTPP_Pad(mst_size=6,C_input=4,total_channels=61)
    elif model_str =='vitmstpp_optimized': 
        model = VITMSTPP_Optimal(mst_size=6,C_input=4,total_channels=61)        
    
    
    print(f"Selected model {model_str}: {model.__class__.__name__}")

    return model