import numpy as np
import torch
from typing import Tuple, Dict, Any


def fourier_transform_spectral(img: np.ndarray, norm: str = 'abs', device: str = 'cpu', channel_first = True) -> torch.Tensor:
    """
    Apply a 1D Fourier Transform along the spectral (channel) dimension of a hyperspectral image.
    
    Args:
        img (np.ndarray): Input hyperspectral image of shape (H, W, C).
        norm (str): Normalization method: 'abs', 'minmax', or 'none'.
        device (str): Device to run the FFT on ('cpu' or 'cuda').
    
    Returns:
        torch.Tensor: FFT-transformed image of shape (H, W, C), complex or magnitude depending on `norm`.
    """
    if channel_first: 
        dim = 0 
    else: 
        dim = -1
    img_tensor = torch.tensor(img, dtype=torch.float32, device=device)  # [H, W, C]
    fft_result = torch.fft.fft(img_tensor, dim=dim)  # Apply FFT over spectral dim (C)
    
    if norm == 'abs':
        fft_result = torch.abs(fft_result)
    elif norm == 'minmax':
        min_vals = torch.amin(torch.abs(fft_result), dim=dim, keepdim=True)
        max_vals = torch.amax(torch.abs(fft_result), dim=dim, keepdim=True)
        fft_result = (torch.abs(fft_result) - min_vals) / (max_vals - min_vals + 0.01)  # Avoid div by zero
        fft_result = fft_result*(-1) + 1
    
    if device =='cuda': 
        fft_result = fft_result.cpu().numpy()
    else: 
        fft_result = fft_result.numpy()

    return fft_result


class FourierSpectralTransform:
    """
    Transform class to apply 1D FFT along the spectral dimension of a hyperspectral image.
    """
    def __init__(self, 
                 norm: str = 'abs',
                 transf_cube: bool = False,
                 channel_first: bool = True, 
                 device: str = 'cpu' 
                 ):
        self.channel_first = channel_first
        self.norm = norm
        self.transf_cube = transf_cube
        self.device = device

    def __call__(self,
                 x: np.ndarray,
                 cube: np.ndarray,
                 meta: Dict[str, Any]
                 ) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
        """
        Apply the spectral FFT transform.

        Args:
            x (np.ndarray): Hyperspectral image [C, W, H C]
            cube (np.ndarray): Extra cube (unchanged)
            meta (dict): Metadata (unchanged)
        
        Returns:
            Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]: Transformed image, original cube, metadata
        """
        x_transformed = fourier_transform_spectral(x, self.norm, self.device, self.channel_first)

        if self.transf_cube: 
            cube = fourier_transform_spectral(cube, self.norm, self.device, self.channel_first)

        return x_transformed, cube, meta

    def __str__(self):
        return f"FourierSpectralTransform(norm={self.norm}, transform cube={self.transf_cube}, device={self.device})"
