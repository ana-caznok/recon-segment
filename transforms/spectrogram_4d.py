import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any


def create_4d_spectrogram_torch_channel_first(hyper_data: np.ndarray,
                                              bandwidth: int = 2,
                                              window: int = 5,
                                              mfft: int = 32,
                                              norm: str = 'abs',
                                              device: str = 'cpu') -> torch.Tensor:
    """
    Create 4D spectrogram from channel-first hyperspectral data.

    Args:
        hyper_data (np.ndarray): Input image of shape (C, H, W)
        bandwidth (int): Bandwidth for STFT
        window (int): Window length for STFT
        mfft (int): Number of FFT bins
        norm (str): Normalization method: 'abs', 'minmax', or 'none'
        device (str): 'cpu' or 'cuda'

    Returns:
        torch.Tensor: 4D spectrogram of shape (F, T, H, W)
    """
    hyper_tensor = torch.tensor(hyper_data, dtype=torch.float32, device=device)  # [C, H, W]
    C, H, W = hyper_tensor.shape
    pixels = hyper_tensor.permute(1, 2, 0).reshape(-1, C)  # (H*W, C)

    spgram_all, _, _ = get_normalized_spectrogram_torch(
        pixels, bandwidth, np.array([window]), mfft, 1, norm, device
    )  # (H*W, F, T)

    spgram_all = spgram_all.view(H, W, mfft, -1).permute(2, 3, 0, 1)  # → (F, T, H, W)
    if device == 'cuda': 
        spgram_all = spgram_all.cpu().numpy()
    else: 
        spgram_all = spgram_all.cpu()
    return spgram_all


def get_normalized_spectrogram_torch(
    fids, bandwidth, window, mfft, hop, norm, device
):
    qntty = fids.shape[0]
    spgram_all = []

    for i in range(qntty):
        spgram = get_stft(fids[i], bandwidth, window, mfft, hop, device)
        spgram_all.append(spgram)

    spgram_all = torch.stack(spgram_all, dim=0)  # (N, F, T)

    if norm == "minmax":
        spgram_all = normalize_complex_vector_min_max(spgram_all)
    elif norm == "abs":
        spgram_all = normalize_complex_vector_abs(spgram_all)
    # else: keep complex

    freq_spect = torch.flip(torch.linspace(0, bandwidth // 2, mfft, device=device), dims=[0])
    t_spect = torch.linspace(0, 1, spgram_all.shape[2], device=device)

    return spgram_all, freq_spect, t_spect


def get_stft(signal, bandwidth, window, mfft, hop, device):
    win = torch.hamming_window(window[0], device=device)
    stft_result = torch.stft(signal, n_fft=mfft, hop_length=hop,
                             win_length=window[0], window=win,
                             return_complex=True)
    return pad_spectrogram(stft_result, padding_mode='reflect')


def normalize_complex_vector_abs(spgram_all):
    return torch.abs(spgram_all) / (torch.amax(torch.abs(spgram_all), dim=-1, keepdim=True) + 1e-8)


def normalize_complex_vector_min_max(spgram_all):
    mag = torch.abs(spgram_all)
    min_vals = torch.amin(mag, dim=-1, keepdim=True)
    max_vals = torch.amax(mag, dim=-1, keepdim=True)
    return (mag - min_vals) / (max_vals - min_vals + 1e-8)


def pad_spectrogram(spectrogram, padding_mode='reflect'):
    padding_size = 32 - spectrogram.shape[0]
    padding = (padding_size, 0)
    return F.pad(spectrogram.T, padding, mode=padding_mode).T


class Spectrogram4D():
    """
    Applies 4D spectrogram transform to channel-first hyperspectral images: (C, H, W) → (F, T, H, W)
    """
    def __init__(self,
                 bandwidth: int,
                 window: int,
                 mfft: int,
                 norm: str = 'abs',
                 device: str = 'cpu'):
        self.bandwidth = bandwidth
        self.window = window
        self.mfft = mfft
        self.norm = norm
        self.device = device

    def __call__(self,
                 img: np.ndarray,
                 cube: np.ndarray,
                 meta: Dict[str, Any]) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:

        img_transformed = create_4d_spectrogram_torch_channel_first(
            img, self.bandwidth, self.window, self.mfft, self.norm, self.device
        )
        return img_transformed, cube, meta

    def __str__(self):
        return f"Spectrogram4D_ChannelFirst(F={self.mfft}, T=?, norm={self.norm})"
