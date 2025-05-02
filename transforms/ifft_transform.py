import torch

def inverse_fft_from_real_imag(
    y_transformed: torch.Tensor,
    meta: dict,
    norm: str = 'minmax',
    device: str = 'cpu',
    channel_first: bool = True,
    shift: bool = True,
    stack_type: str = 'alternate', 
    e=0) -> torch.Tensor:
    """
    Vectorized inverse FFT with per-sample normalization support.
    """
    x_tensor = y_transformed.to(device)  # (B, 2C, H, W)
    x_tensor = x_tensor - e

    if norm == 'minmax':
        if meta['fft_ymin'].dim() < 4:
            # Get the batch size from the 0th dimension
            B = meta['fft_ymin'].shape[0] 
            # Reshape both tensors to shape [B, 1, 1, 1]
            meta['fft_ymin'] = meta['fft_ymin'].view(B, 1, 1, 1)
            meta['fft_ymax'] = meta['fft_ymax'].view(B, 1, 1, 1)

        y_min = meta['fft_ymin'].to(device)  # shape (B, 1, 1, 1)
        y_max = meta['fft_ymax'].to(device)  # shape (B, 1, 1, 1)

        # Ensure broadcast shape matches x_tensor: (B, 2C, H, W)
        x_tensor = (x_tensor - 0.0001) / (1 - 0.0001)
        x_tensor = x_tensor * (y_max - y_min + 1e-8) + y_min
    
    elif norm =='minmax-byc': 
        y_min = meta['fft_ymin'].to(device)
        y_max = meta['fft_ymax'].to(device)  

        # Ensure broadcast shape matches x_tensor: (B, 2C, H, W)
        x_tensor = (x_tensor - 0.0001) / (1 - 0.0001)
        x_tensor = x_tensor * (y_max - y_min + 1e-8) + y_min


    elif norm == 'softmax':
        raise NotImplementedError("Softmax normalization is not invertible without the original unnormalized input.")

    # Split real/imag
    B, C2, H, W = x_tensor.shape
    C = C2 // 2

    if stack_type == 'alternate':
        real = x_tensor[:, 0::2]  # (B, C, H, W)
        imag = x_tensor[:, 1::2]  # (B, C, H, W)
    else:
        real, imag = torch.tensor_split(x_tensor, 2, dim=1)

    complex_tensor = torch.complex(real, imag)  # (B, C, H, W)

    if shift:
        complex_tensor = torch.fft.ifftshift(complex_tensor, dim=1)

    ifft_result = torch.fft.ifft(complex_tensor, dim=1)

    return ifft_result.real  # (B, C, H, W)


class InverseFourierSpectralTransform:
    """
    Inverse FFT transform class for hyperspectral data transformed with real+imag FFT.
    Supports reversing min-max normalization using metadata.
    """
    def __init__(self, 
                 norm: str = 'minmax',
                 channel_first: bool = True, 
                 device: str = 'cpu', 
                 shift: bool = True, 
                 stack_type: str = 'alternate'):
        self.channel_first = channel_first
        self.norm = norm
        self.device = device
        self.shift = shift
        self.stack_type = stack_type

    def __call__(self, cube: torch.Tensor, meta: dict) -> torch.Tensor:
        """
        Apply inverse FFT transform to a normalized real+imag transformed cube.

        Args:
            cube (torch.Tensor): Transformed tensor (2C, H, W) or (H, W, 2C)
            meta (dict): Metadata containing normalization parameters (e.g., 'fft_ymin', 'fft_ymax')

        Returns:
            torch.Tensor: Reconstructed real-valued tensor.
        """
        y_transformed= inverse_fft_from_real_imag(
            y_transformed=cube,
            meta=meta,
            norm=self.norm,
            device=self.device,
            channel_first=self.channel_first,
            shift=self.shift,
            stack_type=self.stack_type
        )
        return y_transformed #torch.clip(y_transformed,0.0001,1.0001) #this is specific to hyperskin dataset

    def __str__(self):
        return f"InverseFourierSpectralTransform(norm={self.norm}, device={self.device}, shift={self.shift}, stack_type={self.stack_type})"