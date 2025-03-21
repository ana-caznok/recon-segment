import torch
import numpy as np
from utils.preprocessing_utils import random_crop
from typing import Union, Tuple, Dict, Any
import scipy
import scipy.io



def rgb2hyp(rgb,gain_function,norm = False):
  
  pseudo_hyp = np.zeros([gain_function.shape[0], rgb.shape[1], rgb.shape[2]])
  for c in range(gain_function.shape[0]): 
    pseudo_hyp[c,:,:] = rgb[0,:,:]*gain_function[c,0] + rgb[1,:,:]*gain_function[c,1] + rgb[2,:,:]*gain_function[c,2]
  
  if norm: 
     pseudo_hyp = (pseudo_hyp - pseudo_hyp.min())/(pseudo_hyp.max() - pseudo_hyp.min())
  
  return pseudo_hyp


class RGB2Pseudo_Hyp():
    '''
    Real time random cropping
    '''
    def __init__(self, 
                 base_path: str,
                 camera: str, 
                 norm: bool=False):
        self.base_path = base_path
        self.camera = camera
        self.norm = norm

    def __call__(self, 
                 x: np.ndarray, 
                 y: np.ndarray,
                 m: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        
        if self.camera =='D40': 
           gain_path = 'example_D40_camera_w_gain.mat'
        
        else: 
           gain_path = 'cie_1964_w_gain.mat'


        gain_function = scipy.io.loadmat(self.base_path + 'transforms/' + gain_path)
        x = rgb2hyp(x,gain_function['filters'])

        return x, y, m

    def __str__(self) -> str:
        return f"RGB2Pseudo_Hyp with: {self.base_path} and camera {self.camera}"
    