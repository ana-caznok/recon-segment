import torch
import numpy as np
from typing import Union, Tuple, Dict, Any


class Downsample():
    '''
    Simple downsampling by a factor by Ana
    '''
    def __init__(self, factor: int):
        self.factor = factor

    def __call__(self, 
                 x: np.ndarray, 
                 y: np.ndarray,
                 m: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        
        # If using downsampling also donwsample metadata mask!
        if torch.is_tensor(m["mask"]) or isinstance(m["mask"], np.ndarray):
            m["mask"] = m["mask"][:, ::self.factor, ::self.factor]

        return x[:, ::self.factor, ::self.factor], y[:, ::self.factor, ::self.factor], m
    
    def __str__(self) -> str:
        return f"Downsample by a factor of {self.factor}"
    

class DownsampleInput():
    '''
    Simple downsampling by a factor, only in input!
    '''
    def __init__(self, factor: int):
        self.factor = factor

    def __call__(self, 
                 x: np.ndarray, 
                 y: np.ndarray,
                 m: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        return x[:, ::self.factor, ::self.factor], y, m
    
    def __str__(self) -> str:
        return f"Downsample only input by a factor of {self.factor}"
