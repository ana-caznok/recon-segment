import torch
from torch import nn
from loss.useful_losses import Loss_MRAE, Loss_PSNR, Loss_RMSE
from torchmetrics.image import SpectralAngleMapper, StructuralSimilarityIndexMeasure


class MRAESSIMSAMLoss(nn.Module):
    def __init__(self, enable_mrae=True, enable_sam=True, enable_ssim=True, weighted_sam = False):
        super().__init__()
        self.enable_mrae = enable_mrae
        self.enable_sam = enable_sam
        self.enable_ssim = enable_ssim
        self.weighted_sam = weighted_sam
        
        assert int(enable_mrae) + int(enable_sam) + int(enable_ssim) > 0, "Please enable some loss"
        
        self.mrae = Loss_MRAE()
        self.sam = SpectralAngleMapper(compute_with_cache=False)
        self.ssim = StructuralSimilarityIndexMeasure(compute_with_cache=False)

    def forward(self, pred, target):
        loss = 0
        pred += 1e-5
        target += 1e-5

        if self.enable_mrae:
            loss += self.mrae(pred, target)
        
        if self.enable_sam:
            if self.weighted_sam: 
                w = 2
            else: 
                w = 1
            loss += w*self.sam(pred, target)
        
        if self.enable_ssim:
            loss += (1 - self.ssim(pred, target))
        
        return loss
    
    def reset(self):
        self.sam.reset()
        self.ssim.reset()

    def __str__(self):
        return f"MRAESSIMSAMLoss - MRAE: {self.enable_mrae}, SAM: {self.enable_sam}, SSIM: {self.enable_ssim}"
