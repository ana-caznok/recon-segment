import torch
import numpy as np
from utils.preprocessing_utils import random_crop
from typing import Union, Tuple, Dict, Any


def crop_channels(cube, channels, m=None):
    '''
        It assumes img and cube have the channel in first dim
            cube: [c2, x,y]

    '''
    assert cube.shape[0] >= channels
   
    cube = cube[0:channels, :, :]
   
    return cube


class ChannelCrop():
    '''
    Real time random cropping
    '''
    def __init__(self, 
                 channels: int 
                 ):
        self.channels = channels
   

    def __call__(self, 
                 x: np.ndarray, 
                 y: np.ndarray,
                 m: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        y = crop_channels( y, self.channels)
        return x, y, m

    def __str__(self) -> str:
        return f"Crop Channels until : {self.channels}"
    