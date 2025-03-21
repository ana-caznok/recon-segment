import numpy as np
from typing import Tuple, Dict, Any
import torch
import torch.nn.functional as F
import scipy.signal
import numpy as np


def create_4d_spectrogram_torch(hyperspectral_data, bandwidth=2, window=5, mfft=32, norm='abs', device='cpu'):
    """
    Creates a 4D spectrogram of a hyperspectral image in the channel domain, fully vectorized, using PyTorch.
    
    Args:
        hyperspectral_data: A 3D numpy array representing the hyperspectral image (rows, cols, bands).
        window_size: Size of the STFT window.
        hop_length: Hop length for the STFT.
        
    Returns:
        A 4D tensor representing the spectrogram (rows, cols, bands, frequencies).
    """
    # Convert input hyperspectral data to PyTorch tensor and move to the appropriate device
    hyperspectral_data = torch.tensor(hyperspectral_data, dtype=torch.float32, device=device)
    
    rows, cols, bands = hyperspectral_data.shape
    spectral_profiles = hyperspectral_data.view(-1, bands)  # Reshape to (rows*cols, bands)
    
    # Vectorized computation of the STFT for all spectral profiles (treat each pixel as a transient)
    spgram_all, freq_spect, t_spect = get_normalized_spectrogram_torch(
        spectral_profiles, bandwidth, np.array([window]), mfft, 1, norm, device
    )
    
    # Reshape the result to match the 4D output format (rows, cols, mfft, mfft)
    spectrogram_4d = spgram_all.view(rows, cols, mfft, mfft)
    
    return spectrogram_4d

def get_normalized_spectrogram_torch(
    fids, bandwidth, window, mfft, hop, norm, device
):
    """
    Get normalized spectrogram of fids, fully vectorized, using PyTorch.
    
    fids: Spectral data (each pixel as a "transient") of shape (qntty, bands).
    """
    qntty = fids.shape[0]  # This is now the number of pixels in the hyperspectral image (rows * cols)
    
    # Compute the STFT for all spectral profiles in one step (fully vectorized)
    spgram_all = []
    for i in range(qntty):
        spgram = get_stft(fids[i], bandwidth, window, mfft, hop, device)
        spgram_all.append(spgram)
    
    spgram_all = torch.stack(spgram_all, dim=0)
    
    # Normalize using the specified method (fully vectorized)
    if norm == "minmax":
        spgram_all = normalize_complex_vector_min_max(spgram_all)
    else:
        spgram_all = normalize_complex_vector_abs(spgram_all)

    # Calculate frequency and time arrays (vectorized)
    freq_spect = torch.flip(torch.linspace(0, bandwidth // 2, mfft, device=device), dims=[0])
    t_spect = torch.linspace(0, 1, spgram_all.shape[2], device=device)

    return spgram_all, freq_spect, t_spect

def get_stft(signal, bandwidth, window, mfft, hop, device):
    """
    Get the Short-Time Fourier Transform (STFT) of a single signal (spectral profile).
    """
    # This function assumes `scipy.signal.stft` can be replaced with a PyTorch implementation
    # We'll use a simple implementation here for demonstration, but this should be optimized.
    n_samples = signal.shape[0]
    
    # Create window (this is just a simple Hamming window)
    win = torch.hamming_window(window[0], device=device)
    
    
    # Apply the STFT calculation (using torch.fft for FFT)
    stft_result = torch.stft(signal, n_fft=mfft, hop_length=hop, win_length=window[0], 
                             window=win, return_complex=True)
   

    stft_result = pad_spectrogram(stft_result, padding_mode='reflect')
    
    
    return stft_result

def normalize_complex_vector_abs(spgram_all):
    """
    Normalize the spectrogram using the absolute value (magnitude) normalization.
    """
    return torch.abs(spgram_all) / torch.max(torch.abs(spgram_all), dim=-1, keepdim=True)[0]

def normalize_complex_vector_min_max(spgram_all):
    """
    Normalize the spectrogram using min-max normalization.
    """
    min_vals = torch.min(spgram_all, dim=-1, keepdim=True)[0]
    max_vals = torch.max(spgram_all, dim=-1, keepdim=True)[0]
    return (spgram_all - min_vals) / (max_vals - min_vals)

def pad_spectrogram(spectrogram, padding_mode='reflect'):
    """
    Pad the spectrogram to the target shape using torchvision.transforms with reflect or mirror padding.
    
    Args:
        spectrogram: Tensor representing the spectrogram (shape: [channels, height, width])
        target_shape: The desired target shape (height, width) for the spectrogram.
        padding_mode: Padding mode for reflection, can be 'reflect' or 'mirror'.
        
    Returns:
        Padded spectrogram with the desired target shape.
    """
    padding_size = 32 - spectrogram.shape[0] 
    #print(padding_size)
    
    # Apply the padding using torch.nn.functional.pad
    padding = (padding_size, 0) #left, right, top, bottom, padding.
    # If you need padding on other dimensions, adjust `padding` accordingly.
    padded_spectrogram = F.pad(spectrogram.T, padding, mode=padding_mode) 
    #print(padded_spectrogram.shape)
    return padded_spectrogram.T




class Spectrogram4D():
    '''
    Real time random cropping
    '''
    def __init__(self, 
                 bandwidth: int,
                 window:int, 
                 mfft: int, 
                 norm:str, 
                 device:str
                 ):
        
        self.bandwidth = bandwidth
        self.window = window
        self.mfft = mfft
        self.norm = norm
        self.device= device

    def __call__(self, 
                 img: np.ndarray, 
                 cube: np.ndarray,
                 m: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        
        img = create_4d_spectrogram_torch(img, self.bandwidth, self.window, self.mfft, self.norm, self.device)
        
        return img, cube, m

    def __str__(self) -> str:
        return f"RandomCrop with width: {self.width} and height {self.height}"
    