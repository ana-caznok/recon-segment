import os
import glob
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from typing import Optional, Callable, List, Dict, Tuple, Any
from tqdm import tqdm
from sklearn.model_selection import KFold
import json


class FixedDataset(Dataset):  
    '''
    Simply read data and apply a transform. 
    '''  
    ORIGINAL_VAL_LIST: List[str] = ["p009", "p046", "p045", "p034", "p021"]  # internal test reference, not the same as fold validation
                           
    def __init__(self, 
                 mode: str,
                 base_path: str,
                 transform: Optional[Callable] = None,  # real time processing
                 preprocessing: Optional[str] = None, # pre-processing
                 fold: int = None,
                 **kwargs):
        super().__init__()
  
        for k, v in kwargs:
            print(f"WARNING: Ignoring dataset argument {k}: {v}")

        self.mode = mode
        self.transform = transform
        self.base_path = base_path
        self.preprocessing = preprocessing
        self.fold = fold

        if self.mode == "test" and fold is not None:
            raise ValueError("Can't test and use folds at the same time")
    
        if self.preprocessing == "None":
            print("WARNING: interpreting None string as None type")
            self.preprocessing = None
        
        if self.preprocessing is None:
            fmt = "*.mat"
        elif self.preprocessing == "downsampled":
            fmt = "*.npy"
        elif self.preprocessing == "h5py" or self.preprocessing=="downsample_input_4":
            print("Loading from hdf files")
            fmt = "*.hdf"
            if self.fold is None:
                self.h5file_path = os.path.join(self.base_path, f"{self.mode}.hdf")
            else:
                print(f"Using fold: {self.fold}")
                self.h5file_path = os.path.join(self.base_path, "train.hdf")
                
            with h5py.File(self.h5file_path, 'r') as h5file:
                self.keys = list(h5file.keys())
                if self.fold is not None:
                    full_person_list = sorted(list(set([key.split('_')[0] for key in self.keys])))
                    kfolder = KFold(5, shuffle=False)
                    splits = [x for x in kfolder.split(full_person_list)]
                    _, val_idx = splits[self.fold]
                    self.val_persons = np.array(full_person_list)[val_idx].tolist()
                    print(f"Replacing default VAL_LIST with validation for fold {self.fold}. Validation persons:")
                    print(self.val_persons)
                    self.keys = self.filter_files(self.keys, val_list=self.val_persons)[mode]
                    print(f"Fold {self.fold} {mode}: {self.keys}")
                    
                    if self.mode == "test":
                        raise ValueError("Shouldnt be k-folding in test")
                    
            self.total_files = len(self.keys)

        self.fmt = fmt

        if mode != 'test':
            if self.preprocessing != "h5py":
                self.msi_files: List[str] = sorted(glob.glob(os.path.join(self.base_path, "Hyper-Skin(MSI, NIR)", "train", "MSI_CIE", fmt)))
                self.vis_files: List[str] = sorted(glob.glob(os.path.join(self.base_path, "Hyper-Skin(RGB, VIS)", "train", "VIS", fmt)))
                self.nir_files: List[str] = sorted(glob.glob(os.path.join(self.base_path, "Hyper-Skin(MSI, NIR)", "train", "NIR", fmt)))
            
                # Filter by fixed validation list and mode
                self.msi_files = self.filter_files(self.msi_files, val_list=FixedDataset.ORIGINAL_VAL_LIST)[mode]
                self.vis_files = self.filter_files(self.vis_files, val_list=FixedDataset.ORIGINAL_VAL_LIST)[mode]
                self.nir_files = self.filter_files(self.nir_files, val_list=FixedDataset.ORIGINAL_VAL_LIST)[mode]

                self.total_files = len(self.msi_files)
                assert len(self.msi_files) == len(self.vis_files) == len(self.nir_files) == self.total_files, f"Length {self.total_files} is not matching among 3 folders"
            
            # Temporarily removed, buggy
            self.bb = []
            self.bb_names = []
            try:
                self.bb_names, self.bb = self.read_BB(mode, downsample = self.preprocessing == "downsampled")
            except FileNotFoundError as error:
                print(f"ERROR: Didn't found boundingbox files!: {error}")
                print("Continuinig without loading bounding boxes.")
            
            self.mask_files: List[str] = sorted(glob.glob(os.path.join(self.base_path, "masks_full_size_large-model", '*.npy')))
            
            if self.fold is None:
                self.mask_files = self.filter_files(self.mask_files, val_list=FixedDataset.ORIGINAL_VAL_LIST)[mode]
            else:
                # When using folds, first remove masks from internal test, old validation
                self.mask_files = self.filter_files(self.mask_files, val_list=FixedDataset.ORIGINAL_VAL_LIST)["train"]
                # Then, now take the proper fold
                self.mask_files = self.filter_files(self.mask_files, val_list=self.val_persons)[mode]
            
            nmasks = len(self.mask_files)
            if nmasks != self.total_files:
                raise ValueError(f"Number of masks is not number of files...? {nmasks} / {self.total_files}")
            
            print(f"Number of face masks {len(self.mask_files)}.", end=" ")
            if self.preprocessing == "h5py":
                for key, mask_path in tqdm(zip(self.keys, self.mask_files), desc="Checking face mask integrity..."):
                    mask_key = os.path.basename(mask_path).replace("-mask.npy", '')
                    comparison_string = f"{key} not equal to {mask_key} in mask"
                    assert key == mask_key, comparison_string

        if mode == "test":
            fmt = '*.mat'
            self.msi_files: List[str] = sorted(glob.glob(os.path.join(self.base_path, "Hyper-Skin(MSI, NIR)", "test", '*.mat')))
            self.vis_files: List[str] = ['No_Target']*len(self.msi_files)
            self.nir_files: List[str] = ['No_Target']*len(self.msi_files)
            self.bb = ['No_BB']*len(self.msi_files)
            self.total_files = len(self.msi_files)

        print(self.init_str())

    def init_str(self):
        return f"initialized {self.total_files} images {self.mode} DatasetRaw with preprocessing: {self.preprocessing} in {self.base_path} with transform {self.transform}. Fold: {self.fold}"

    def load_masks(self, idx: int): 
        m_file = self.mask_files[idx]
        
        mask_ID = os.path.basename(m_file).replace("-mask.npy", '')
        mask = np.load(m_file)

        # Workaround, downsamples face only in downsampled preprocessing or downsample of both input and target
        if self.preprocessing == 'downsampled': 
            mask = mask[::4,::4]

        return mask, mask_ID

    def filter_files(self, files: List[str], val_list: List[str]) -> Dict[str, List[str]]:
        val_files = []
        train_files = []
        for file in files:
            val = False/mnt/datassd/icasp/RainbowAI/
            for val_idx in val_list:
                if val_idx in file:
                    val_files.append(file)
                    val = True
            
            if not val:
                train_files.append(file)
        
        return {"train": train_files, "val": val_files}

    def loadCube(self, cube_path: str) -> np.ndarray:
        '''
        return cube channel first.
        Original transpose: 2, 1, 0
        Channel first, but keep face orientation: 0, 2, 1
        intensity range: (0, 1)
        '''
        with h5py.File(cube_path, 'r') as f:
            array: np.ndarray = f['cube'][:].transpose(0, 2, 1)  # correct face orientation and channel first
            return array.squeeze()
            
    
    def loadPreprocessing(self, path: str) -> np.ndarray:
        return np.load(path)

    def loadData(self, msi_path: str, vis_path: str, nir_path: str) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Convenção: remove first channel from NIR for 61 total channels (31 + (31 - 1))
        '''
        if self.preprocessing is None:
            msi_input = self.loadCube(msi_path)
            if self.mode != "test":
                target = np.concatenate([self.loadCube(vis_path), self.loadCube(nir_path)[1:]], axis=0) #new, output 61
            else:
                target = "None"
        elif self.preprocessing == "downsampled":
            msi_input = self.loadPreprocessing(msi_path)
            target = np.concatenate([self.loadPreprocessing(vis_path), self.loadPreprocessing(nir_path)[1:]], axis=0) #new, output 61

        return msi_input, target 
    
    def read_BB(self, r_mode: str, downsample: bool = True) -> Tuple[List[str], List[List[int]]]: 
        '''
        WARNING: Not set to work with folds
        '''
        face_path = os.path.join(self.base_path, "bounding-boxes/")
        file_name = 'faces_bb_NIR_w-rgb_BEST.json'
      
        mode_dict = {'train': True, 
                     'val': False}

        with open(face_path + file_name) as json_file:
            faceseg_BB = json.load(json_file)

        key_array = np.asarray([k  for  k in  faceseg_BB.keys()])
        values_array = np.asarray([np.asarray(v)  for  v in  faceseg_BB.values()])

        name_list = []
        for k in key_array: 
            name_list.append(k.split('_')[0])
        
        mask = np.isin(name_list, np.asarray(FixedDataset.ORIGINAL_VAL_LIST), invert=mode_dict[r_mode])

        key_array = key_array[mask]
        values_array = values_array[mask]
        
        if downsample == True: 
            values_array = values_array//4

        return key_array, values_array

    def load_h5(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]: #TODO ainda falta arrumar output 61 para leitura h5, ainda não entendi como foi feito
        ID = self.keys[idx]
        with h5py.File(self.h5file_path, 'r') as h5file:
            # 31 canais, index 0 30
            vis = h5file[ID][4:4 + 31] # indices 4 ate 34 incluidos
            nir = h5file[ID][4 + 31 + 1:]  # skip first channel from nir, indices 36 ate o fim incluidos
            msi_input = h5file[ID][:4]  # indices 0 a 3
            target = np.concatenate((vis, nir), axis=0)

        return msi_input, target, ID

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # Read preprocessed or raw
        if self.preprocessing == "h5py":
            msi_input, target, ID = self.load_h5(idx)
        else:
            msi_input, target = self.loadData(self.msi_files[idx], self.vis_files[idx], self.nir_files[idx])
            ID = os.path.basename(self.msi_files[idx]).replace(self.fmt[1:], '')
        
        try:
            mask_input, mask_ID = self.load_masks(idx)
            mask_input = np.expand_dims(mask_input, 0) 
        except: 
            mask_input = "None"
            mask_ID = None

        if mask_ID is not None:
            assert mask_ID == ID, f"Mask is not the same as person: {mask_ID} / {ID}"

        try:
            img_bb = self.bb[idx]
        except IndexError:
            img_bb = "None"
        
        metadata = {"ID": ID, "bbox": img_bb, "mask": mask_input}

        if self.transform is not None:
            msi_input, target, metadata = self.transform(msi_input, target, metadata)

        # Torch conversion is not in transforms from transform_factory
        msi_input = torch.from_numpy(msi_input).float()
        
        # Target might be "None" in test
        if not isinstance(target, str):
            target = torch.from_numpy(target).float()

        if mask_ID is not None:
            metadata["mask"] = torch.from_numpy(metadata["mask"]).float()

        return msi_input, target, metadata

    def __len__(self):
        return self.total_files
    

if __name__ == "__main__":
    '''
    Debug: save dataset output as nii.gz to visualize with ITK-Snap
    '''
    import SimpleITK as sitk
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed", action="store_true")
    parser.add_argument("--fmt", type=str)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--see_stuff", action="store_true")
    
    args = parser.parse_args()

    if args.preprocessed:
        if args.fmt == "h5py":
            datasets = {mode: FixedDataset(mode, os.getenv("ICASP_H5"), transform=None, preprocessing="h5py", fold=args.fold) for mode in ["train", "val"]}
        else:
            datasets = {mode: FixedDataset(mode, os.getenv("ICASP_DOWNSAMPLED"), transform=None, preprocessing="downsampled", fold=args.fold) for mode in ["train", "val"]}
    else:    
        datasets = {mode: FixedDataset(mode, os.getenv("ICASP"), transform=None, fold=args.fold) for mode in ["train", "val"]}

    print("Load testing...")
    for mode, dataset in tqdm(datasets.items()):
        for x, y, m in tqdm(dataset):
            print(x.max())
            print(x.min())
            print(y.max())
            print(y.min())
            if args.see_stuff:
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.title("Input")
                plt.imshow(x.mean(0).numpy(), cmap="gray")
                plt.subplot(1, 3, 2)
                plt.title("Target")
                plt.imshow(y.mean(0).numpy(), cmap="gray")
                plt.subplot(1, 3, 3)
                plt.title("Face")
                plt.imshow(m["mask"][0].numpy(), cmap="gray")
                plt.show()
                
    quit()
    if args.fmt == "nii.gz":
        for mode, dataset in tqdm(datasets.items()):
            out_folder = os.path.join(dataset.base_path, f"preprocessed_{args.fmt}", mode)
            os.makedirs(out_folder, exist_ok=True)
            for x, y, m in tqdm(dataset):
                out_path = os.path.join(out_folder, f"{m['ID']}_msi.nii.gz")
                sitk.WriteImage(sitk.GetImageFromArray(x.numpy()), out_path)
                tqdm.write(f"Saved {out_path}")
                
                out_path = os.path.join(out_folder, f"{m['ID']}_visnir.nii.gz")
                sitk.WriteImage(sitk.GetImageFromArray(y.numpy()), out_path)
                tqdm.write(f"Saved {out_path}")
    elif args.fmt == "h5py":
        raise ValueError("We already did that")
        for mode, dataset in tqdm(datasets.items()):
            out_folder = os.path.join(dataset.base_path, f"preprocessed_{args.fmt}")
            os.makedirs(out_folder, exist_ok=True)
            with h5py.File(os.path.join(out_folder, f"{mode}.hdf"), 'w') as h5_file:
                for i, (x, y, m) in tqdm(enumerate(dataset), total=len(dataset)):
                    h5_file.create_dataset(name=m["ID"], data=np.concatenate([x.numpy(), y.numpy()], axis=0, dtype=np.float32))
                
