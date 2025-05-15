import torch
import numpy as np
from utils.preprocessing_utils import random_crop
from typing import Union, Tuple, Dict, Any
import scipy
import scipy.io
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union


def interpolate_channels(
    image: Union[np.ndarray, torch.Tensor],
    return_torch: bool = True, 
    final_channels = 31
    ) -> Union[np.ndarray, torch.Tensor]:
    """
    Interpolates a (C, H, W) image with C < 31 to (31, H, W) using trilinear interpolation
    along the spectral (channel) axis (treated as depth).

    Args:
        image: Input tensor or ndarray of shape (C, H, W), with C < 31
        return_torch: Whether to return a torch.Tensor or a NumPy array

    Returns:
        Interpolated output of shape (31, H, W)
    """
    # Convert to tensor if needed
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()

    assert image.ndim == 3, "Input must have shape (C, H, W)"
    C, H, W = image.shape
    assert C < 31, f"Expected fewer than 31 channels, got {C}"
    #print(image.shape)
    # Reshape to 5D tensor: (N, C, D, H, W) → treat channels as depth
    # So we move channel to "depth" axis: (C, H, W) → (1, 1, C, H, W)
    image = image.unsqueeze(0).unsqueeze(0)  # now (1, 1, C, H, W)

    # Interpolate depth dimension (C → final_channels)
    interp = F.interpolate(image, size=(final_channels, H, W), mode='trilinear', align_corners=True)

    # Remove batch and channel dimensions, result is (1, 1, final_channels, H, W)
    # Rearrange back to (final_channels, H, W)
    interp = interp.squeeze(0).squeeze(0)  # final shape: (final_channels, H, W)
    #print(interp.shape)

    return interp if return_torch else interp.cpu().numpy()


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

    Modes:
    - If interpolate_to_31 = True: use interpolation to generate 31 channels.
    - Else: use standard rgb2hyp + gain matrix.
    """

    def __init__(self,
                 base_path: str,
                 camera: str,
                 norm: bool = False,
                 return_torch: bool = True,
                 ):
        self.base_path = base_path
        self.camera = camera
        self.norm = norm
        self.return_torch = return_torch

        if 'interp' in self.camera:
            self.interpolate = True
            self.final_channels = int(self.camera.split('terp')[1])
        else: 
            self.interpolate = False
            self.final_channels = 31

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        m: Dict[str, Any]
    ) -> Tuple[Union[np.ndarray, torch.Tensor], np.ndarray, Dict[str, Any]]:
        """
        Applies RGB or MSI to pseudo-HSI transformation via either:
        - Interpolation to 31 channels
        - RGB gain matrix projection

        Returns:
            Tuple (x_transformed, y, m)
        """
        if self.interpolate:
            # Use direct RGB interpolation
            x = interpolate_channels(x, return_torch=self.return_torch, final_channels=self.final_channels)
        else:
            # Use gain matrix projection
            gain_file = 'example_D40_camera_w_gain.mat' if self.camera == 'D40' else 'cie_1964_w_gain.mat'
            gain_path = self.base_path + 'transforms/' + gain_file

            # Load gain matrix from .mat file
            gain_function = scipy.io.loadmat(gain_path)['filters']

            x = rgb2hyp(x, gain_function, norm=self.norm, return_torch=self.return_torch)

        return x, y, m

    def __str__(self) -> str:
        return (f"RGB2Pseudo_Hyp(base_path='{self.base_path}', camera='{self.camera}', "
                f"norm={self.norm}, return_torch={self.return_torch}, interpolate={self.interpolate}, final_channels={self.final_channels})")
