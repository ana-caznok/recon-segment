'''
Some additional custom implementations replicating nnunet transforms according to its supplementary material
https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf

There is no parametrization to keep it as close as possible to the original implementation. Torchio is used for implementation of most transforms.

This is modified from the original version in diedre_phd
'''
import torch
import random
import threading
import numpy as np
import torchio as tio
from torch.nn import functional as F
from torchvision.transforms.transforms import InterpolationMode
from torchvision.transforms import RandomAffine
from torchvision.transforms import Compose
from .rainbow_transform import RainbowTransform
from typing import Optional, Union, Tuple, Dict, Any


class ConditionalCropOrPad():
    '''
    Dynamically crops to 512x512 axial slices or to avoid less than 128 number of slices
    '''
    def __init__(self, min_shape):
        self.min_shape = min_shape
    
    def __call__(self, x, y):
        with torch.no_grad():
            _, W, H, D = x.shape
            N = y.shape[0]

            # If axial slice dimension is different then target and number of slices W is less, crop 
            low_resolution = W < self.min_shape[0]
            if low_resolution or H != self.min_shape[1] or D != self.min_shape[2]:
                if low_resolution:
                    target_shape = self.min_shape
                else:
                    target_shape = (W,) + self.min_shape[1:]

                self.crop_or_pad_zero = tio.CropOrPad(target_shape, 0)
                self.crop_or_pad_one = tio.CropOrPad(target_shape, 1)

                x = self.crop_or_pad_zero(x)
                y_new = []
                for n in range(N):
                    if n == 0:
                        y_new.append(self.crop_or_pad_one(y[n:n+1]))
                    else:
                        y_new.append(self.crop_or_pad_zero(y[n:n+1]))
                y = torch.cat(y_new, dim=0)
            
            return x, y

    def __str__(self):
        return f"ConditionalCropOrPad {self.min_shape}"


class nnUNetTransform():
    '''
    Defined in Page 35 of nnunet published supplementary file.
    
    https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf

    Number after class name represent the item in page 35 and order of application.
    '''
    def __init__(self, p, verbose=False):
        '''
        p: probability of applying transform. If None, will always call transform.
        '''
        self.verbose = verbose
        self.p = p
        _ = str(self)
            
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.p is None:
                if self.verbose:
                    print(f"Calling transform {self.__class__} because p is None. There might be internal randomization.\n")
                return self.transform(x)
            else:
                p = random.random()
                if p <= self.p:
                    if self.verbose:
                        print(f"Calling transform {self.__class__} because {p} < {self.p}\n")
                    return self.transform(x)
                else:
                    if self.verbose:
                        print(f"NOT Calling transform {self.__class__} because {p} > {self.p}\n")
                    return x
    
    def __str__(self):
        raise NotImplementedError("Please define __str__ in nnUNetTransforms")


# Step 0: have intensities on nnunet expected range
class HUNormalize0(nnUNetTransform):
    '''
    Simplified version of CTHUClipNorm without parametrization

    Note that we have observed strangely good raw HU performance.

    Hypothesis: 
        1)This might be due to raw hu indirectly doing augmentation in the visualized range.
        2)Presence of some HU values might "hint" the network on the style of annotation of a dataset.

    Nevertheless, we doing hu normalization for nnUNet-like augmentation due to it being essential to some parametrization.
    '''
    def __init__(self, verbose=False):
        raise DeprecationWarning("HUNormalize was not correctly following nnUNet aug strategy")
        super().__init__(p=None, verbose=verbose)
        self.vmin = -1024
        self.vmax = 600

    def transform(self, x):
        return (((torch.clip(x, self.vmin, self.vmax) - self.vmin)/(self.vmax - self.vmin))*2) - 1
    
    def __str__(self):
        return "Clip into the -1024, 600 range, min max normalize and extend it to -1, 1"


# 1
from torchio.transforms import RandomAffine as tioRandomAffine
class RotationAndScaling1(nnUNetTransform):
    def __init__(self, interpolation, verbose=False, fill=0, degree_2d=180, degree_3d=15, scale=(0.7, 1.4)):
        self.interpolation = interpolation
        self.scale = scale
        self.degree_2d = degree_2d
        self.degree_3d = degree_3d
        self.fill = fill
        super().__init__(p=0.2, verbose=verbose)  # slight deviation from nnunet definition for optimization
        self.augmentation = tioRandomAffine(scales=scale,
                                            degrees=(-degree_3d, degree_3d),  # ansiotropic range, higher creates too much background
                                            image_interpolation=interpolation,
                                            default_pad_value=fill,
                                            isotropic=True)  # not sure about this but makes sense with "zoom out/in" terminology 
        
        interpolation_2d = {"linear": InterpolationMode.BILINEAR, "nearest": InterpolationMode.NEAREST}
        self.augmentation2d = RandomAffine(degrees=(-degree_2d, degree_2d), translate=None, scale=scale, interpolation=interpolation_2d[interpolation], fill=fill)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return self.augmentation(x)
        else:
            return self.augmentation2d(x)
    
    def __str__(self):
        return f"1: RotationAndScaling with 0.2 application probability and scale sampling from {self.scale} and degrees 2D {self.degree_2d}/3D {self.degree_3d}. Interpolation: {self.interpolation}. Fill {self.fill}"


# 2
from torchio.transforms import RandomNoise as tioRandomNoise
class GaussianNoise2(nnUNetTransform):
    def __init__(self, verbose=False):
        super().__init__(p=0.15, verbose=verbose)
        self.augmentation = tioRandomNoise(mean=0, std=(0, 0.02))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return self.augmentation(x)
        else:
            return self.augmentation(x.unsqueeze(1)).squeeze(1)
    
    def __str__(self):
        return f"2: nnUNet Gaussian Noise 0.15 probability with mean 0 and U(0, 0.02) standard deviation"


# 3
from torchio.transforms import RandomBlur as tioRandomBlur
class GaussianBlur3(nnUNetTransform):
    def __init__(self, verbose=False):
        super().__init__(p=1, verbose=verbose)
        self.augmentation = tioRandomBlur((0, 1))  # very light, was 0.5 1.5
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return self.augmentation(x)
        else:
            return self.augmentation(x.unsqueeze(1)).squeeze(1)
    
    def __str__(self):
        return "3: nnUNet Gaussian Blur 0.2 probability with mean 0 and U(0, 1) kernel standard deviation"


# 4
class Brightness4(nnUNetTransform):
    def __init__(self, verbose=False):
        super().__init__(p=0.15, verbose=verbose)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return random.uniform(0.8, 1.2)*x
        
    def __str__(self):
        return "4: nnUNet brightness 0.15 probability with U(0.8, 1.2) brightness"
    

# 5
class Contrast5(nnUNetTransform):
    def __init__(self, verbose=False):
        super().__init__(p=0.15, verbose=verbose)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x_min, x_max = x.min(), x.max()
        return torch.clip(random.uniform(0.8, 1.2)*x, min=x_min, max=x_max)
        
    def __str__(self):
        return "5: nnUNet contrast 0.15 probability with U(0.8, 1.2) factor clipped to previous min, max range"
    

#6
class SimulationOfInterpolationArtifacts(nnUNetTransform):
    def __init__(self, verbose=False, mask=False):
        self.mask = mask
        super().__init__(p=0.25, verbose=verbose)
        
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # channel+3D input, therefore simulate 5D batch with unsqueeze
        
        # indirect asseertion of x ndims
        if x.ndim == 4:
            _, Z, Y, X = x.shape  
            original_shape: Union[Tuple[int, int], Tuple[int, int, int]] = (Z, Y, X)
            mode = "trilinear"
            align_corners: Optional[bool] = True
        else:
            _, Y, X = x.shape  
            original_shape = (Y, X)
            mode = "bilinear"
            align_corners = None

        try:
            scale_factor = random.uniform(0.8, 1.2)
            x_cache = F.interpolate(x.unsqueeze(0), scale_factor=scale_factor, mode="nearest")
            x = F.interpolate(x_cache, size=original_shape, mode=mode, align_corners=align_corners).squeeze(0)
            if self.mask:
                x = (x > 0.5).float()
            assert (x.shape[0] == 4 or x.shape[0] == 61) and len(x.shape) == len(original_shape) + 1, f"Unexpected interpolation result {x_cache.shape}/{x.shape}"
        except Exception as e:
            print(f"SimulationOfInterpolationArtifacts error:\n{e}\noriginal_shape: {original_shape} scale_factor: {scale_factor}")
            print(f"Returning original image {x.shape}")

        return x
    
    def __str__(self):
        return f"6: nnUNet SimulationOfInterpolationArtifacts with 0.25 probability of nearest downsampling in the U(0.8, 1.2) range and trilinear upsampling back to original size mask: {self.mask}"


# 7
from torchio.transforms import RandomGamma as tioRandomGamma
class GammaAugmentation7(nnUNetTransform):
    def __init__(self, verbose=False):
        super().__init__(p=0.15, verbose=verbose)
        
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified gamma for rainbowai
        gamma = random.uniform(0.8, 1.2)
        x = x**gamma
        return x
    
    def __str__(self):
        return "7: nnUNet Gamma Augmentation with probability 0.15 and gamma parameter between 0.8 and 1.2"


# 8
from torchio.transforms import RandomFlip as tioRandomFlip
class Mirroring8(nnUNetTransform):
    def __init__(self, verbose=False):
        super().__init__(p=None, verbose=verbose)
        self.augmentation = tioRandomFlip(axes=(0, 1, 2), flip_probability=0.5)
        
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return self.augmentation(x)
        else:
            return self.augmentation(x.unsqueeze(1)).squeeze(1)  # avoid flipping the channel dimension on 2D images
    
    def __str__(self):
        return "8: nnUNet Mirroring with probability 0.5 for each axis"
    
# EXTRA fix 0s introduced to onehot
class FixTargetAfterTransform(nnUNetTransform):
    def __init__(self, verbose=False):
        super().__init__(p=None, verbose=verbose)

    def transform(self, y):
        # Set spatial locations where onehot check failed to be BG.
        # Skip binary masks
        if y.shape[0] > 1:
            projection = y.sum(dim=0).long()  # guarantees zero presence
            y[0, projection == torch.zeros_like(projection)] = 1.0

        return y
    
    def __str__(self):
        return "Fix no label area that might be added by transformations, setting it to BG"
    

class RainbowTransformsnnUNet():
    '''
    Updated version of nnunet transforms specific for the RainbowAI problem, adapted from diedre_phd
    '''
    TRHEAD_SAFE_LOCK = threading.Lock()
    def __init__(self, dim, verbose=False):
        self.dim = dim
        
        # A lot of transforms didnt work as kind of expected. Attempt simulation of half patch with 50%
        transform = Compose([Mirroring8(verbose=verbose),
                             RotationAndScaling1(interpolation="linear", verbose=verbose, fill=0, degree_2d=72, scale=(0.8, 1.2)),
                             GaussianNoise2(verbose=verbose),
                             GaussianBlur3(verbose=verbose),
                             Brightness4(verbose=verbose),  
                             Contrast5(verbose=verbose),  
                             SimulationOfInterpolationArtifacts(verbose=verbose, mask=False),  # causing nans
                             GammaAugmentation7(verbose=verbose),  # causing nans
                            ])
        
        # Reproducing spatial transforms on target but not intensity ones 
        target_transform = Compose([Mirroring8(verbose=verbose),
                                    RotationAndScaling1(interpolation="linear", verbose=verbose, fill=0, degree_2d=72, scale=(0.8, 1.2)),
                                    GaussianNoise2(verbose=verbose),
                                    GaussianBlur3(verbose=verbose),
                                    Brightness4(verbose=verbose),
                                    Contrast5(verbose=verbose),
                                    SimulationOfInterpolationArtifacts(verbose=verbose, mask=False),  # causing nans
                                    GammaAugmentation7(verbose=verbose),  # causing nans
                                   ])
        
        mask_transform = Compose([Mirroring8(verbose=verbose),
                                  RotationAndScaling1(interpolation="nearest", verbose=verbose, fill=0, degree_2d=72, scale=(0.8, 1.2)),
                                 ])
        
        self.transform = RainbowTransform(transform, target_transform, mask_transform)
        
    def __call__(self, x: np.ndarray, y: np.ndarray, m: Dict[str, Any]):
        RainbowTransformsnnUNet.TRHEAD_SAFE_LOCK.acquire()  # Avoid RNG race conditions with patcher and nnunet transforms. IO is still parallel.
        
        if self.dim == "2d":
            assert x.ndim == 3
        elif self.dim == "3d":
            assert x.ndim == 4

        x, y, m["mask"] = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m["mask"])
        x, y, m["mask"] = self.transform(x, y, m["mask"])
        x, y, m["mask"] = torch.clip(x, 0, 1), torch.clip(y, 0, 1), torch.clip(m["mask"], 0, 1)
        x, y, m["mask"] = x.numpy(), y.numpy(), m["mask"].numpy()
        RainbowTransformsnnUNet.TRHEAD_SAFE_LOCK.release()

        return x, y, m
    
    def __str__(self):
        return f'nnUNet transforms light 0.8/1.2 changes {self.dim}\n' + str(self.transform) + '\n'
    
