import torch
import numpy as np
from utils.preprocessing_utils import random_crop
from typing import Union, Tuple, Dict, Any
import scipy
import scipy.io

def rgb2hyp(
    rgb: Union[np.ndarray, torch.Tensor], gain_function: Union[np.ndarray, torch.Tensor],
    norm: bool = False,return_torch: bool = True) -> Union[np.ndarray, torch.Tensor]:
    """
    Inputs: 
    - rgb: (3, H, W) RGB image, gain_function: (C, 3) gain matrix, norm: Whether to normalize the output
    - return_torch: If True, use torch for all operations and return tensor
    Returns:
    - Pseudo hyperspectral image: shape (C, H, W)
    """

    # Select device if using torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)

    if return_torch:
        # Convert to torch tensors and move to device
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb).float().to(device)
        if isinstance(gain_function, np.ndarray):
            gain_function = torch.from_numpy(gain_function).float().to(device)
        
        rgb = rgb[:3]
        # Perform weighted sum using einsum (C, 3) x (3, H, W) -> (C, H, W)
        pseudo_hyp = torch.einsum('ci,ihw->chw', gain_function, rgb)

        # Optional normalization
        if norm:
            min_val = pseudo_hyp.min()
            max_val = pseudo_hyp.max()
            pseudo_hyp = (pseudo_hyp - min_val + 0.001) / (max_val - min_val)

        return pseudo_hyp

    else:
        # Standard NumPy computation
        pseudo_hyp = np.zeros([gain_function.shape[0], rgb.shape[1], rgb.shape[2]])
        for c in range(gain_function.shape[0]):
            pseudo_hyp[c, :, :] = (rgb[0, :, :] * gain_function[c, 0] +
                                   rgb[1, :, :] * gain_function[c, 1] +
                                   rgb[2, :, :] * gain_function[c, 2])

        if norm:
            pseudo_hyp = (pseudo_hyp - pseudo_hyp.min() + 0.001) / (pseudo_hyp.max() - pseudo_hyp.min())

        return pseudo_hyp


class RGB2Pseudo_Hyp:
    """
    Callable class for converting RGB to pseudo hyperspectral images.
    """

    def __init__(self, base_path: str,
                       camera: str,
                       norm: bool = False,
                       return_torch: bool = True):
        
        self.base_path = base_path
        self.camera = camera
        self.norm = norm
        self.return_torch = return_torch

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        m: Dict[str, Any]
         ) -> Tuple[Union[np.ndarray, torch.Tensor], np.ndarray, Dict[str, Any]]:
        """
        Transforms the input image using RGB -> pseudo hyperspectral conversion.

        Parameters:
        - x: (3, H, W) RGB image
        - y: target data (unchanged)
        - m: metadata (unchanged)

        Returns:
        - Transformed x, y, m
        """
        # Select gain file based on camera type
        gain_file = 'example_D40_camera_w_gain.mat' if self.camera == 'D40' else 'cie_1964_w_gain.mat'
        gain_path = self.base_path + 'transforms/' + gain_file

        # Load gain matrix from .mat file
        gain_function = scipy.io.loadmat(gain_path)['filters']

        # Apply the RGB -> hyperspectral conversion
        x = rgb2hyp(x, gain_function, norm=self.norm, return_torch=self.return_torch)

        return x, y, m

    def __str__(self) -> str:
        return f"RGB2Pseudo_Hyp(base_path='{self.base_path}', camera='{self.camera}', norm={self.norm}, return_torch={self.return_torch})"

    