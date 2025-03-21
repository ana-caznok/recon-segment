import os
import h5py
import torch
import numpy as np
from typing import Dict, Any, Union
from multiprocessing import Queue, Process, cpu_count


class SaveChallengeFMT():
    '''
    Saves either a torch tensor or numpy array in challenge format
    Assumes input follows orientation of fixed dataset
    '''
    SUPPORTED_FMTS = [".npy", ".npz", ".mat"]
    def __init__(self, save_dir=None, fmt=".npy"):
        '''
        save_dir: where to save
        fmt: .npy or .npz or .mat to save
        '''
        self.save_dir = save_dir
        self.fmt = fmt
        self.q = Queue(cpu_count())
        if fmt not in SaveChallengeFMT.SUPPORTED_FMTS:
            raise ValueError(f"SaveChallengeFMT does not support fmt: {fmt}")
        self.save_p = Process(target=self.save_worker, args=(self.q,))
        self.save_p.start()

    def join(self):
        self.q.put(None)
        print("Waiting for saving jobs to finish...")
        self.save_p.join()
        print("Done")
        
    def save_worker(self, q: Queue):
        '''
        Parallel process to save outputs
        '''
        while True:
            pkg = q.get()
            if pkg is None:
                return
            else:
                out_path, y_hat = pkg

                # Saved npy uses 64 bit float np.float64 in challenge
                y_hat = y_hat.astype(np.float64)
                if self.fmt == ".npy":
                    np.save(out_path, y_hat)
                elif self.fmt == ".npz":
                    # Warning takes a long time, not worth it 
                    np.savez_compressed(out_path, cube=y_hat)
                elif self.fmt == ".mat":
                    # TODO check if this is correct
                    with h5py.File(out_path, 'w') as mat_file:
                        mat_file.create_dataset("cube", data=y_hat)

    def __call__(self, y_hat: Union[torch.Tensor, np.ndarray], m: Dict[str, Any]):
        '''
        y_hat: post-processed output to be saved, must have 4 channels with batch size 1
        m: metadata dictionary from dataset
        '''
        assert y_hat.shape[0] == 1, "Evaluate with batch size 1!"
        assert os.path.isdir(self.save_dir), "save_dir in saver does not exist"

        # Bring to numpy three dimensional array
        if torch.is_tensor(y_hat):
            y_hat = y_hat.detach().cpu().numpy()[0]
        elif isinstance(y_hat, np.ndarray):
            y_hat = y_hat[0] 
        else:
            raise ValueError(f"y_hat in save makes no sense: {type(y_hat)}")
    
        # Bring to challenge orientation
        y_hat = np.swapaxes(y_hat, 1,2) 
        y_hat = np.swapaxes(y_hat, 0,2) 
        # CHANNEL LAST
        
        if y_hat.shape[-1] != 61: 
            raise ValueError("Number of channels is not 61")

        # Save
        out_path = os.path.join(self.save_dir, f"{m['ID'][0]}{self.fmt}")

        qsize = self.q.qsize()
        if qsize > 0:
            print(f"Adding save job to the queue, number of arrays yet to be saved: {qsize}")
        self.q.put((out_path, y_hat))
