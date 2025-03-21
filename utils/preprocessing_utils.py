import numpy as np
import yaml
import h5py
import matplotlib.pyplot as plt

def get_face_by_file(fname):
    return fname.split('/')[-1].split('_')[0]


def random_crop(img, cube, width, height):
    '''
        It assumes img and cube have the channel in last dim, first two are spatial

            img: [x,y, c1]
            cube: [x,y, c2]

    '''
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == cube.shape[0]
    assert img.shape[1] == cube.shape[1]
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    cube = cube[y:y+height, x:x+width]
    return img, cube

def train_val_split(files, val_faces = ['p009', 'p046', 'p045', 'p034', 'p021']):
    '''
        Given the people you dont want in train, it splits all files

    '''
    train_list = []
    val_list = []
    
    for fi in files:
        f = get_face_by_file(fi)
        if(f not in val_faces):
            train_list.append(fi)
        else:
            val_list.append(fi)
            
    return train_list, val_list


def read_yaml(file: str):
    with open(file, "r") as yaml_file:
        configurations = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return configurations

def check_pairs(img, cube):
    if(img.split('/')[-1].split('.')[0] == cube.split('/')[-1].split('.')[0]):
        return True
    else: False

def loadCube(cube_path):
    '''
    return cube in (h, w, c=31 or 4)
    range: (0, 1)
    '''
    with h5py.File(cube_path, 'r') as f:
        cube = np.squeeze(np.float32(np.array(f['cube'])))
        cube = np.transpose(cube, [2,1,0]) 
        f.close()
    return cube

def loadRGB(rgb_path):
    '''
    return rgb img in (h, w, c=3)
    range: (0, 1)
    '''

    rgb = plt.imread(rgb_path)
    rgb = np.float32(rgb)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    return rgb