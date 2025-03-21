import torch
from torch import nn
from torchmetrics.image import SpectralAngleMapper


class ChannelLoss(nn.Module):
    def __init__(self, channel_weights, enable_sam=False, dynamic_weight=False):
        super().__init__()
        self.enable_sam = enable_sam
        self.dynamic_weight = dynamic_weight
        self.l1 = nn.L1Loss()
        self.sam = SpectralAngleMapper(compute_with_cache=False)
        
        self.w = torch.tensor(channel_weights)
        
        self.w_sum = self.w.sum()
        if self.dynamic_weight:
            self.w = nn.Parameter(self.w)

    def forward(self, pred, target):
        if pred.device != self.w.device:
            print(f"Putting weights in {pred.device}")
            self.w = self.w.to(pred.device)
            self.w_sum = self.w_sum.to(pred.device)
    
        l1s_channel = torch.stack([self.l1(pred[:,channel,:,:],target[:,channel,:,:]) for channel in range(61)])
        
        l1s_channel = l1s_channel*self.w
        l1_loss = l1s_channel.mean()

        # Weight regularization
        if self.dynamic_weight:
            reg = torch.abs(self.w.sum() - self.w_sum)
        else:
            reg = 0

        if self.enable_sam:
            sam = self.sam(pred + 1e-5, target + 1e-5)
        else:
            sam = 0

        return l1_loss + sam + reg 
    
    def reset(self):
        self.sam.reset()

    def __str__(self):
        return "AnaChannelLoss"
