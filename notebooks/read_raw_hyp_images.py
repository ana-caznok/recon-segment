#read_raw_hyp_images

import h5py
import numpy as np
import cv2

def read_hyp_imgs(d_file): 
    fend = d_file.split('.')[-1]

    if fend =='mat': 
        
        filepath = d_file
        arrays = []
        f = h5py.File(filepath)

        for k, v in f.items():
            arrays.append(np.array(v))
    
        img = np.array(arrays).reshape(v.shape[0], 1024, 1024)
        img = np.swapaxes(img,1,2) #colocando o rosto de pé em vez de deitado
        
    if fend=='jpg':
        try:  
            img = cv2.imread(d_file)
            img = np.swapaxes(img, 0,2) #colocando os canais na primeira dimensão 
            img = np.swapaxes(img, 1,2) #colocando o rosto de pé em vez de deitado
           
        except: 
            print('Error: ', d_file)

    return img