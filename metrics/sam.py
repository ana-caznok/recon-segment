import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt 
from typing import List

def visualize_sam_map(sam_map, saved_path):
    sam_map = np.array(sam_map).squeeze()
    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(20, 15))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(sam_map[i, :, :], vmin=0, vmax=1, cmap='gray')
        ax.axis('off')
        ax.set_title(f"SAM: {sam_map[i, :, :].mean():.3f}")
    fig.colorbar(im, ax=axes.ravel().tolist(), pad = 0.01)
    plt.savefig(f'{saved_path}.png')

def sam_fn(pred, target):
    '''
    pred, target: [c, w, h]
    '''
    pred, target = pred.squeeze(), target.squeeze()
    up = torch.sum((target*pred), dim = 0)   # [w, h]
    down1 = torch.sum((target**2), dim = 0).sqrt()
    down2 = torch.sum((pred**2), dim = 0).sqrt()

    map = torch.arccos(up / (down1 * down2))
    score = torch.mean(map[~torch.isnan(map)])
    map[torch.isnan(map)] = 0
    return score, map


class SAMScore(nn.Module):
    '''
    Returns the score value from Challenge owners sam_fn
    '''
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert len(target.shape) == 4 and len(pred.shape) == 4, "SAMScore accepts a 4D batch as an input"

        sam_scores: List[torch.Tensor] = []
        for p, t in zip(pred, target):
            sam_scores.append(sam_fn(p, t)[0])
        
        return torch.stack(sam_scores).mean()
    
    def reset(self):
        pass
