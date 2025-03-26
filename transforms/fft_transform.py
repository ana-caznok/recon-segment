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

def fft_real_imag_normalized(img: np.ndarray, norm: str = 'realimag', device: str = 'cpu', channel_first: bool = True) -> np.ndarray:
    """
    Apply FFT along the spectral dimension and return a normalized, interleaved real+imag output.
    Output shape will have 2x channels. Even = real, Odd = imag. Values in [0.0001, 1].

    Args:
        img (np.ndarray): Input hyperspectral image, either (C, H, W) if channel_first=True, or (H, W, C)
        device (str): 'cpu' or 'cuda' - controls where the FFT is computed
        channel_first (bool): If True, assumes channels are the first dimension

    Returns:
        np.ndarray: Output array with shape (2C, H, W) or (H, W, 2C),
                    where even-indexed channels are real part, odd-indexed are imaginary part.
    """
    # Determine which axis corresponds to the spectral (channel) dimension
    dim = 0 if channel_first else -1

    # Convert the input image to a PyTorch tensor on the specified device
    img_tensor = torch.tensor(img, dtype=torch.float32, device=device)

    # Perform 1D FFT along the spectral/channel axis
    fft_result = torch.fft.fft(img_tensor, dim=dim)

    # Extract the real and imaginary parts of the complex FFT result
    real = fft_result.real
    imag = fft_result.imag

    # Define a function to normalize values to the range [0.0001, 1]
    def normalize(t):
        # Compute per-spectrum minimum and maximum along the spectral axis
        t_min = t.amin(dim=dim, keepdim=True)
        t_max = t.amax(dim=dim, keepdim=True)
        # Normalize to [0, 1], avoiding divide-by-zero with small epsilon
        normed = (t - t_min) / (t_max - t_min + 1e-8)
        # Rescale to [0.0001, 1]
        return normed * (1 - 0.0001) + 0.0001

    # Normalize real and imaginary parts separately
    real = normalize(real)
    imag = normalize(imag)

    # Stack real and imaginary tensors along a new intermediate dimension
    # This results in a shape like [2, C, H, W] or [H, W, 2, C], depending on layout
    stacked = torch.stack([real, imag], dim=dim + 1 if not channel_first else dim + 1)

    # Flatten the new 2-element stack into the spectral axis to interleave:
    # Even indices become real, odd indices become imaginary
    # Final shape: (2C, H, W) or (H, W, 2C)
    interleaved = stacked.flatten(start_dim=dim, end_dim=dim + 1)

    if device == 'cuda': 
        interleaved = interleaved.cpu().numpy()
    else: 
        interleaved = interleaved.numpy()

    # Return the result as a NumPy array on CPU
    return interleaved


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

        # Set the appropriate transformation function based on the normalization mode
        if self.norm == 'realimag':
            self.trans_function = fft_real_imag_normalized
        else:
            self.trans_function = fourier_transform_spectral

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
        x_transformed = self.trans_function(x, self.norm, self.device, self.channel_first)

        if self.transf_cube: 
            cube = self.trans_function(cube, self.norm, self.device, self.channel_first)

        return x_transformed, cube, meta

    def __str__(self):
        return f"FourierSpectralTransform(norm={self.norm}, transform cube={self.transf_cube}, device={self.device})"
