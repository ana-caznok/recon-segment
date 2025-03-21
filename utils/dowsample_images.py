#dowsample_images.py

import numpy as np 
import glob 
import matplotlib.pyplot as plt 
import skimage 
import io
import h5py
import cv2
from read_raw_hyp_images import read_hyp_imgs

def downsample(img): 

    if img.shape[2] < img.shape[0]:
        img_resize = img[::4, ::4, :]
    
    if img.shape[2] > img.shape[0]: 
        img_resize = img[:, ::4, ::4]

    return img_resize


# main 

base_path = '/mnt/datassd/icasp/data/raw/Link_2/'
output_path = '/mnt/datassd/icasp/data/preprocessed/downsampled/'


rgb_vis_dir = base_path + 'Hyper-Skin(RGB, VIS)'
rgb_dir = f'{rgb_vis_dir}/train/RGB_CIE'
vis_dir = f'{rgb_vis_dir}/train/VIS'

msi_nir_dir = base_path + 'Hyper-Skin(MSI, NIR)'
msi_dir = f'{msi_nir_dir}/train/MSI_CIE'
hsi_dir = f'{msi_nir_dir}/train/NIR' 

dir_list = [rgb_dir, vis_dir, msi_dir, hsi_dir]

for dir in dir_list: 

    files = glob.glob(dir+'/*.mat')

    if len(files)<=1: 
        files = glob.glob(dir+'/*.jpg')


    for file in files: 

        img = read_hyp_imgs(file)
        img = downsample(img)

        origin_folders = file.split('Link_2/')[-1].split('.')[0]
        save_dir = output_path + origin_folders + '.npy'
        #print(save_dir)
        np.save(save_dir, img)