'''
The father of most segmentation transforming
'''
import random
import numpy as np
import torch
import threading


class RainbowTransform():
    '''
    Applies a torchvision transform into image and segmentation target
    '''
    # thread-safe lock. Multiple processes will have their own random instances, but if you use this with multiple threads,
    # you don't want RNG seed rolling race conditions
    THREAD_SAFE_LOCK = threading.Lock()  
    def __init__(self, transform, target_transform, mask_transform):
        self.transform = transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform
    
    def __call__(self, image, seg_target, mask_target):
        '''
        Precisa-se fixar a seed para mesma transformada ser aplicada
        tanto na máscara quanto na imagem. 
        '''
        RainbowTransform.THREAD_SAFE_LOCK.acquire()
        
        # Gerar uma seed aleatória
        seed = random.randint(0, 2147483647)

        # Fixar seed e aplicar transformada na imagem
        if self.transform is not None:
            random.seed(seed) 
            torch.manual_seed(seed) 
            np.random.seed(seed)
            image = self.transform(image)
            
        # Fixar seed e aplicar transformada na máscara
        if self.target_transform is not None:
            random.seed(seed) 
            torch.manual_seed(seed) 
            np.random.seed(seed)
            seg_target = self.target_transform(seg_target)

        if self.mask_transform is not None:
            random.seed(seed) 
            torch.manual_seed(seed) 
            np.random.seed(seed)
            mask_target = self.mask_transform(mask_target)
        
        RainbowTransform.THREAD_SAFE_LOCK.release()

        return image, seg_target, mask_target

    def __str__(self):
        return f"  Image Transform: {self.transform}\n  Target Transform: {self.target_transform}\n Mask Transform: {self.mask_transform}"
