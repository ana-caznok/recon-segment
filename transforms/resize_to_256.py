import torch
import numpy as np
import cv2
from typing import Union, Tuple, Dict, Any


class ResizeTo256:
    """
    Resize transform that converts any image with spatial dimensions > 256 to 256x256.
    If one dimension is less than 256, it will also upscale that dimension to maintain final shape.
    Works for input x, target y, and optionally mask in metadata.
    """

    def __init__(self, interpolation=cv2.INTER_AREA):
        """
        Args:
            interpolation: OpenCV interpolation method (default is INTER_AREA for downsampling).
        """
        self.interpolation = interpolation

    def resize(self, arr: np.ndarray) -> np.ndarray:
        """
        Resize a [C, H, W] array to [C, 256, 256] using OpenCV.
        """
        C, H, W = arr.shape
        resized = np.zeros((C, 256, 256), dtype=arr.dtype)
        for c in range(C):
            resized[c] = cv2.resize(arr[c], (256, 256), interpolation=self.interpolation)
        return resized

    def __call__(self, 
                 x: np.ndarray, 
                 y: np.ndarray,
                 m: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:

        # Resize input
        if x.shape[1] != 256 or x.shape[2] != 256:
            x = self.resize(x)

        # Resize target
        if y.shape[1] != 256 or y.shape[2] != 256:
            y = self.resize(y)

        # Resize mask if available
        if isinstance(m["mask"], (np.ndarray, torch.Tensor)):
            mask = m["mask"]
            if torch.is_tensor(mask):
                mask = mask.numpy()
            if mask.shape[1] != 256 or mask.shape[2] != 256:
                m["mask"] = self.resize(mask)

        return x, y, m

    def __str__(self) -> str:
        return f"Resize input, target and mask to 256x256"
