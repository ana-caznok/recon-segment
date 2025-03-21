from typing import Dict, Any

from torch.nn import L1Loss, MSELoss
from metrics.sam import SAMScore
from torchmetrics.image import SpectralAngleMapper
from loss.mraessimsamloss import MRAESSIMSAMLoss
from loss.msesam import MSESAMLoss
from loss.channel_loss import ChannelLoss


def loss_select(configs: Dict[str, Any]) -> object:
    index = configs.get("loss", None)
    
    print(f"Selecting loss: {index}")
    
    if index is None:
        return None
    elif index == "L1Loss":
        return L1Loss()
    elif index == "MSE":
        return MSELoss()
    elif index == "MSESAMLoss":
        return MSESAMLoss()
    elif index == "SAM":
        return SAMScore()
    elif index == "torchmetrics_SAM":
        return SpectralAngleMapper(compute_with_cache=False)
    elif index == "MRAESSIMSAMLoss":
        return MRAESSIMSAMLoss()
    elif index == "SSIMSAMLoss":
        return MRAESSIMSAMLoss(enable_mrae=False)
    elif index == "MRAESAMLoss":
        return MRAESSIMSAMLoss(enable_ssim=False)
    elif index == "MRAESSIMLoss":
        return MRAESSIMSAMLoss(enable_sam=False)
    elif index == "ChannelLoss":
        channel_weights = configs.get("channel_w", None)
        return ChannelLoss(channel_weights, enable_sam=False, dynamic_weight=False)
    elif index == "ChannelLossDynamic":
        channel_weights = configs.get("channel_w", None)
        return ChannelLoss(channel_weights, enable_sam=False, dynamic_weight=True)
    elif index == "ChannelLossSam":
        channel_weights = configs.get("channel_w", None)
        return ChannelLoss(channel_weights, enable_sam=True, dynamic_weight=False)
    elif index == "ChannelLossSamDynamic":
        channel_weights = configs.get("channel_w", None)
        return ChannelLoss(channel_weights, enable_sam=True, dynamic_weight=True)
    else:
        raise ValueError(f"Could not find loss index {index}")
    