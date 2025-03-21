
import torch
import cv2 
import h5py
import pickle
import numpy as np 
import PIL.Image as Image
import matplotlib.pyplot as plt

from typing import Any, Callable, Dict, List, Optional, Tuple

import glob


import os
from typing import Any, Callable, List, Optional, Tuple

import torch

# from ..utils import _log_api_usage_once


class HSIDataset(torch.utils.data.Dataset):
    """
    Base Class For making datasets which are compatible with torchvision.
    It is necessary to override the ``__getitem__`` and ``__len__`` method.

    Args:
        root (string): Root directory of dataset.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    .. note::

        :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.
    """

    _repr_indent = 4

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        crop_size: int = 128,
        stride: int = 8,
    ) -> None:
        # _log_api_usage_once(self)
        # if isinstance(root, torch._six.string_classes):
        #     root = os.path.expanduser(root)
        self.root = root

        self.crop_size = crop_size
        self.stride = stride
        
        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def extra_repr(self) -> str:
        return ""


class StandardTransform:
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, "Target transform: ")

        return "\n".join(body)


class BaselineRGBVISDataset(HSIDataset):
    resolution = {
        'height': 1024,
        'width': 1024,
        'bands': 31,
    }
    
    def __init__(self, 
                input_dir: str, 
                output_dir: str, 
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,):
        super().__init__(root = input_dir, transform=transform, target_transform=target_transform, crop_size = 4, stride = 4)

        # files location
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_files = sorted(glob.glob(self.input_dir + "/*.npy"))
        self.output_files = sorted(glob.glob(self.output_dir + "/*.npy"))

        # total data
        self.input_files = np.asarray(self.input_files)
        self.output_files = np.asarray(self.output_files)

        self.total_files = len(self.input_files)


    def loadData(self, in_path, out_path):
        with open(in_path, 'rb') as f:
            in_ = np.load(f)
        with open(out_path, 'rb') as f:
            out_ = np.load(f)

        return in_, out_

    def __getitem__(self, idx):
        in_, out_ = self.loadData(self.input_files[idx], self.output_files[idx])

        if self.transform is not None:
            all = np.concatenate([in_, out_], axis = -1)
            all = self.transform(all)
            in_ = all[:3, :, :]
            out_ = all[3:, :, :]

        return in_, out_

    def __len__(self):
        return self.total_files
   


class BaselineMSINIRDataset(HSIDataset):
    resolution = {
        'height': 1024,
        'width': 1024,
        'bands': 31,
    }
    
    def __init__(self, 
                input_dir: str, 
                output_dir: str, 
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,):
        super().__init__(root = input_dir, transform=transform, target_transform=target_transform, crop_size = 4, stride = 4)

        # files location
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_files = sorted(glob.glob(self.input_dir + "/*.npy"))
        self.output_files = sorted(glob.glob(self.output_dir + "/*.npy"))

        # total data
        self.input_files = np.asarray(self.input_files)
        self.output_files = np.asarray(self.output_files)

        self.total_files = len(self.input_files)


    def loadData(self, in_path, out_path):
        with open(in_path, 'rb') as f:
            in_ = np.load(f)
        with open(out_path, 'rb') as f:
            out_ = np.load(f)

        return in_, out_

    def __getitem__(self, idx):
        in_, out_ = self.loadData(self.input_files[idx], self.output_files[idx])

        if self.transform is not None:
            all = np.concatenate([in_, out_], axis = -1)
            all = self.transform(all)
            in_ = all[:4, :, :]
            out_ = all[4:, :, :]

        return in_, out_

    def __len__(self):
        return self.total_files
   