import torch
import numpy as np
from utils.preprocessing_utils import random_crop
from typing import Union, Tuple, Dict, Any


def random_crop_channel_first(img, cube, width, height, m=None):
    '''
        It assumes img and cube have the channel in first dim

            img: [c1, x,y]
            cube: [c2, x,y]

    '''
    assert img.shape[1] >= height
    assert img.shape[2] >= width
    assert img.shape[1] == cube.shape[1]
    assert img.shape[2] == cube.shape[2]
    x = np.random.randint(0, img.shape[2] - width)
    y = np.random.randint(0, img.shape[1] - height)
    img = img[:, y:y+height, x:x+width]
    cube = cube[:, y:y+height, x:x+width]
    if torch.is_tensor(m) or isinstance(m, np.ndarray):
        m = m[:, y:y+height, x:x+width]
        return img, cube, m
    else:
        return img, cube


class RandomCrop():
    '''
    Real time random cropping
    '''
    def __init__(self, 
                 width: int, 
                 height: int):
        self.width = width
        self.height = height

    def __call__(self, 
                 x: np.ndarray, 
                 y: np.ndarray,
                 m: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        x, y, m["mask"] = random_crop_channel_first(x, y, self.width, self.height, m=m["mask"])
        return x, y, m

    def __str__(self) -> str:
        return f"RandomCrop with width: {self.width} and height {self.height}"
    