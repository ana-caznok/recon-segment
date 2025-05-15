import os
import h5py
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, Any


class NTIRE2022LikeDataset(Dataset):
    """
    Dataset for NTIRE2022-style HSI + RGB image pairs.
    Uses flexible loading logic similar to 'read_hyp_imgs' with proper formatting.
    """

    def __init__(self, data_root: str, split: str = "train", transform=None, bgr2rgb: bool = True):
        super().__init__()

        self.bgr2rgb = bgr2rgb
        self.transform = transform
        self.split = split.lower()

        # Set up directories
        self.hyper_folder = os.path.join(data_root, f"{self.split.capitalize()}_spectral")
        self.bgr_folder = os.path.join(data_root, f"{self.split.capitalize()}_RGB")
        self.list_path = os.path.join(data_root, "split_txt", f"{self.split}_list.txt")

        # Load sample identifiers (e.g., ["0001", "0002", ...])
        with open(self.list_path, 'r') as f:
            self.sample_ids = [line.strip() for line in f]
        self.sample_ids.sort()

    def __len__(self) -> int:
        return len(self.sample_ids)

    def load_imgs(self, sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads one hyperspectral and one RGB image using flexible logic.

        Args:
            sample_id: ID string for a sample (e.g., '0001')

        Returns:
            Tuple of (hyper_image [C, H, W], rgb_image [3, H, W])
        """
        # Build full paths to the files
        hyper_path = os.path.join(self.hyper_folder, sample_id + ".mat")
        rgb_path = os.path.join(self.bgr_folder, sample_id + ".jpg")

        # === Load Hyperspectral Image ===
        ext = hyper_path.split('.')[-1]
        if ext == 'mat':
            try:
                # Try general loading of all datasets in file
                with h5py.File(hyper_path, 'r') as f:
                    arrays = [np.array(v) for k, v in f.items()]
                    img = np.array(arrays).reshape(arrays[0].shape[0], 1024, 1024)
                    img = np.swapaxes(img, 1, 2)  # [C, W, H] → [C, H, W]
            except:
                # Fallback: directly get 'cube' dataset
                with h5py.File(hyper_path, 'r') as f:
                    img = np.array(f.get('cube'))
                    img = np.swapaxes(img, 1, 2)  # [C, W, H] → [C, H, W]
        elif ext == 'npy':
            img = np.load(hyper_path)
        else:
            raise ValueError(f"Unsupported hyperspectral format: {ext}")
        hyper = img.astype(np.float32)

        # === Load RGB Image ===
        ext_rgb = rgb_path.split('.')[-1]
        if ext_rgb == 'jpg':
            try:
                img = cv2.imread(rgb_path)  # [H, W, C] in BGR
                if self.bgr2rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

                img = img.astype(np.float32)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize

                img = np.swapaxes(img, 0, 2)  # [C, W, H]
                img = np.swapaxes(img, 1, 2)  # [C, H, W]
            except Exception as e:
                print("Error loading image:", rgb_path)
                raise e
        else:
            raise ValueError(f"Unsupported RGB format: {ext_rgb}")
        rgb = img.astype(np.float32)

        return hyper, rgb

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Load and return the sample at given index.

        Returns:
            input_tensor: Hyperspectral input [C, H, W]
            target_tensor: RGB image [3, H, W]
            metadata: Dict with keys 'ID', 'bbox', 'mask'
        """
        sample_id = self.sample_ids[idx]

        # Load both images using flexible logic
        hyper, rgb = self.load_imgs(sample_id)

        metadata: Dict[str, Any] = {
            "ID": sample_id,
            "bbox": "None",  # No bounding box provided
            "mask": "None"   # No mask provided
        }

        # Apply transform if provided
        if self.transform is not None:
            rgb, hyper, metadata = self.transform(rgb, hyper, metadata)

        # Convert to torch tensors
        rgb_tensor = torch.from_numpy(rgb).float()
        hyper_tensor = torch.from_numpy(hyper).float()

        return rgb_tensor, hyper_tensor, metadata
