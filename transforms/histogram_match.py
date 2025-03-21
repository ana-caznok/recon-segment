import torch
import numpy as np
from typing import Union, Tuple, Dict, Any
import numpy as np 
import glob
import skimage 
from skimage import exposure
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
import h5py 
import os 
import time 

#ID É UM STRING, é o nome do arquivo 
#idx é o índice/número 
def load_h5(h5file_path, ID) -> Tuple[np.ndarray, np.ndarray, str]:
        with h5py.File(h5file_path, 'r') as h5file:
            # 31 canais, index 0 30
            vis = h5file[ID][4:4 + 31] # indices 4 ate 34 incluidos
            nir = h5file[ID][4 + 31 + 1:]  # skip first channel from nir, indices 36 ate o fim incluidos
            msi_input = h5file[ID][:4]  # indices 0 a 3
            target = np.concatenate((vis, nir), axis=0)

        return msi_input, target, ID

def load_refs(base_path, use_h5=False): 

    if use_h5 ==False: 
        #list containing a few val images 
        list_refs = ['p008', 'p020', 'p021', 'p022', 'p047', 'p045', 'p037', 'p032', 'p024', 'p010', 'p032', 'p026', 'p044', 'p042', 'p012','p031', 'p051', 'p011', 'p033']
        random_ref = list_refs[np.random.randint(0,len(list_refs))]
        raw_paths = sorted(glob.glob(base_path + '/Hyper-Skin(MSI, NIR)/train/MSI_CIE/' + random_ref + '*'))
        img_num = np.random.randint(0,len(raw_paths))
        raw_path = raw_paths[img_num]
        name = raw_path.split('/')[-1].split('.')[0]

        vis_path = glob.glob(base_path + '/Hyper-Skin(RGB, VIS)/train/VIS/' + name + '*')[0]
        nir_path = glob.glob(base_path + '/Hyper-Skin(MSI, NIR)/train/NIR/' + name + '*')[0]

        #só funciona caso iremos rodar as transformadas com imagens downsampled 
        ref_raw = np.load(raw_path)
        ref_cube = np.concatenate([np.load(vis_path), np.load(nir_path)[1:]])

    else: 
        
        list_refs = ['p024_smile_left', 'p010_smile_left', 'p026_smile_front', 'p032_neutral_left', 'p010_smile_left', 'p044_smile_right', 'p026_neutral_front', 'p026_smile_left', 'p027_neutral_left', 'p029_smile_front', 'p026_smile_left', 'p042_neutral_left', 'p003_smile_left', 'p043_smile_left', 'p042_neutral_left', 'p003_smile_left', 'p042_neutral_left', 'p012_neutral_front', 'p012_smile_left', 'p012_neutral_right', 'p012_smile_front', 'p012_smile_left', 'p012_neutral_right', 'p031_neutral_left', 'p011_neutral_right', 'p015_neutral_front', 'p011_neutral_right', 'p010_neutral_front', 'p031_smile_left', 'p051_neutral_right', 'p010_neutral_front', 'p018_smile_left', 'p051_neutral_right', 'p033_smile_left', 'p033_neutral_right', 'p051_smile_left', 'p033_smile_left', 'p033_neutral_right', 'p037_smile_front', 'p033_smile_left', 'p037_smile_right', 'p037_smile_front', 'p033_smile_front', 'p037_smile_front', 'p037_smile_front', 'p033_smile_left', 'p032_neutral_right', 'p037_smile_front', 'p033_smile_left', 'p051_smile_left', 'p008_neutral_left', 'p008_neutral_front', 'p008_neutral_right', 'p008_smile_right', 'p008_smile_left']
        name = np.random.choice(list_refs)
        h5file_path = base_path + '/train.hdf'
        ref_raw, ref_cube, ID = load_h5(h5file_path, name) 

    return ref_raw, ref_cube

def hist_match(raw, cube, base_path, prob, seed, h5=False):
    np.random.seed(seed=seed)

    if np.random.random()<prob: 
        ref_raw, ref_cube = load_refs(base_path, use_h5 = h5)
        raw  = skimage.exposure.match_histograms(raw, ref_raw , channel_axis=0)
        cube = skimage.exposure.match_histograms(cube,ref_cube, channel_axis=0)

    return raw, cube 


class HistMatch():
    '''
    Real histogram matching
    '''
    def __init__(self, 
                 base_path: str, 
                 prob: float):
        
        self.base_path = base_path #ATENÇÃO, ISSO AQUI SÓ FUNCIONA EM IMAGENS DOWNSAMPLED NO PC DA ANA
        self.prob = prob 
        assert prob < 1

    def __call__(self, 
                 x: np.ndarray, 
                 y: np.ndarray,
                 m: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        
        seed = np.random.randint(0,2000)
        x, y = hist_match(x, y, self.base_path, self.prob, seed, h5=False)

        return x, y, m

    def __str__(self) -> str:
        return f"Histogram matching: {self.prob}"
    

class HistMatch_h5():
    '''
    Real histogram matching
    '''
    def __init__(self, 
                 base_path: str, 
                 prob: float):
        
        self.base_path = os.getenv("ICASP_H5")
        self.prob = prob 
        assert prob < 1

    def __call__(self, 
                 x: np.ndarray, 
                 y: np.ndarray,
                 m: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        
        seed = np.random.randint(0,2000)
        x, y = hist_match(x, y, self.base_path, self.prob, seed, h5=True)

        return x, y, m

    def __str__(self) -> str:
        return f"Histogram matching: {self.prob}"
    