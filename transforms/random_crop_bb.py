import numpy as np
from typing import Tuple, Dict, Any


def random_crop_prob_BB(img, cube, patch_size, img_bb, prob_in_bb=0.85, mask = 'None'):

    if isinstance(mask, str) and mask == 'None': 
       mask = np.ones_like(cube) 

    img_size = img.shape[1]
    width = patch_size
    height = patch_size

    all_x = np.arange(0, img_size - width) #não podemos pegar valores de x cujo patch caia fora das dimensões da imagem 
    all_y = np.arange(0, img_size- height) #não podemos pegar valores de y cujo patch caia fora das dimensões da imagem 

    x_start = img_bb[0]-patch_size//2
    y_start = img_bb[1]-patch_size//2

    delta_x =  img_bb[2]-patch_size//2 
    delta_y =  img_bb[3]-patch_size//2


    if img_bb[0] < patch_size//2: 
        x_start = img_bb[0]

    if img_bb[1] < patch_size//2: 
        y_start = img_bb[1]
    
    if delta_x < 0: 
        delta_x = img_bb[2] 
        delta_y = img_bb[3] 
    
        if img_bb[0] + delta_x > img_size-width: #fazendo um shift de meio patch pra direita caso isso aconteça 
            x_start = x_start - patch_size//2
            img_bb[0] = x_start
        
        if img_bb[1] + delta_y > img_size-height: 
            y_start = y_start - patch_size//2
            img_bb[1] = y_start
        

    #array de coordenadas cujo centro do patch caem no bounding box 
    x_in_bb = np.arange(x_start, img_bb[0] + delta_x) 
    y_in_bb = np.arange(y_start, img_bb[1] + delta_y) 

    prob = np.random.random()

    if prob <= prob_in_bb: 
        inside_bb = True
        outside_bb = False #caso o valor tirado seja < prob, não iremos inverter a máscara

    else: 
        inside_bb = False
        outside_bb = True

    #mascara de valores que caem dentro do bounding box se invert = False
    #mascara de valores que caem fora do bounding box se invert = True
    condition_x = np.isin(all_x, x_in_bb, invert = outside_bb) 
    condition_y = np.isin(all_y, y_in_bb, invert = outside_bb) 

    filtered_x = all_x[condition_x]
    filtered_ỳ = all_y[condition_y]

    array_in = np.meshgrid(filtered_x, filtered_ỳ) #forma uma malha de pontos

    x_ind_max = min(array_in[0].shape[0], array_in[1].shape[0])
    y_ind_max = min(array_in[0].shape[1], array_in[1].shape[1])

    try: 
        ind_x = np.random.randint(low=0, high = x_ind_max) #dois indices aleatórios dentre os pontos que pertencem (ou não) a lista dada 
        ind_y = np.random.randint(low=0, high = y_ind_max) #dois indices aleatórios dentre os pontos que pertencem (ou não) a lista dada 
        x = array_in[0][ind_x][ind_y]
        y = array_in[1][ind_x][ind_y]  # aqui estão os meus dois pontos aleatorios 
    
    except: 
        print('inside bb=', inside_bb)
        print('outside bb=', outside_bb)
        print('img bb: ', img_bb[0], img_bb[1], img_bb[2], img_bb[3])
        print('start x,y: ', x_start, y_start)
        print('end x,y: ', delta_x, delta_y)
        x = img_size//2 - patch_size//2
        y = img_size//2 - patch_size//2

    img_crop = img[:, y:y+height, x:x+width]

    if cube.shape[1] != img.shape[1]:  #extras para caso o downsample seja feito apenas no dado de input, cube e mask recebem o mesmo input
        F = int(np.sqrt(cube.shape[1]//img_crop.shape[1]))
        y = y*F
        x = x*F
        height = height*F
        width = width*F
    
    img = img_crop
    cube = cube[:, y:y+height, x:x+width]
    mask = mask[:, y:y+height, x:x+width]
        
    return img, cube, mask


class RandomCropBB():
    '''
    Real time random cropping
    '''
    def __init__(self, 
                 patch_size: int,
                 prob_in_bb:int
                 ):
        
        self.patch_size = patch_size
        self.width = patch_size
        self.height = patch_size
        self.prob_in_bb = prob_in_bb

    def __call__(self, 
                 img: np.ndarray, 
                 cube: np.ndarray,
                 m: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        
        if 'mask' in m.keys(): 
            mask_input = m['mask']

        else: 
            mask_input = 'None'
        
        img, cube, m['mask']  = random_crop_prob_BB(img, cube, self.patch_size, m['bbox'], prob_in_bb = self.prob_in_bb, mask = mask_input)
        
        return img, cube, m

    def __str__(self) -> str:
        return f"RandomCrop with width: {self.width} and height {self.height}"
    

class RandomCropCenter():
    '''
    Reason: avoid transforming face bbox

    Real time random cropping on x, y, m
    
    Probability of being in center crop? 
    
    Shapes de saida: x.shape = y.shape/4, m["mask"].shape/4
    '''
    def __init__(self, 
                 patch_size: int,
                 prob_in_bb:int
                 ):
        
        self.patch_size = patch_size
        self.width = patch_size
        self.height = patch_size
        self.prob_in_bb = prob_in_bb

    def __call__(self, 
                 img: np.ndarray, 
                 cube: np.ndarray,
                 m: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        
        x_center, y_center = img.shape[1]//2, img.shape[2]//2 #pegando o centro da imagem MSI
        bb_size = int(img.shape[-1]*0.5) #o tamanho do rosto
        bbox_center = [x_center - bb_size//2, y_center-bb_size//2, bb_size, bb_size]

        if 'mask' in m.keys(): 
            mask_input = m['mask']
        else: 
            mask_input = 'None'
        
        img, cube, m['mask']  = random_crop_prob_BB(img, cube, self.patch_size, bbox_center, prob_in_bb = self.prob_in_bb, mask = mask_input)
        return img, cube, m


    def __str__(self) -> str:
        return f"RandomCrop with width: {self.width} and height {self.height}"
    