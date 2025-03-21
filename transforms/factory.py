import os
from transforms import RandomCrop, Compose, Downsample, RandomCropBB, DownsampleInput, RainbowTransformsnnUNet, RandomCropCenter, HistMatch, HistMatch_h5, RGB2Pseudo_Hyp, Spectrogram4D, FourierSpectralTransform


def transform_factory(index: str):
    '''
    Documents used transform using an index string
    '''
    if index == "downsample_4_crop_64":
        return Compose([Downsample(factor=4), RandomCrop(width=64, height=64)])
    elif index == "downsample_4":
        return Downsample(factor=4)
    elif index == "downsample_2":
        return Downsample(factor=2)
    
    elif index == "random_patch_64":
        return RandomCrop(width=64, 
                          height=64)
    elif index == "random_patch_128":
        return RandomCrop(width=128, 
                          height=128)
    elif index == "random_patch_256":
        return RandomCrop(width=256, 
                          height=256)
    elif index == "random_patch_512":
        return RandomCrop(width=512, 
                          height=512)
    
    elif index == 'random_crop_bb_downs_p085': 
        return RandomCropBB(patch_size = 64, prob_in_bb = 0.85)
    
    elif index == 'rgb2hyp_D40': 
        base_path = '/media/ana-caznok/SSD-08/recon-segment/'
        return RGB2Pseudo_Hyp(base_path, 'D40') 
    elif index == 'rgb2hyp_normal': 
        base_path = '/media/ana-caznok/SSD-08/recon-segment/'
        return RGB2Pseudo_Hyp(base_path, 'norm')
    
    elif index =='fft_D40_x_gpu':
        base_path = '/media/ana-caznok/SSD-08/recon-segment/'
        return Compose([RGB2Pseudo_Hyp(base_path, 'D40'),FourierSpectralTransform('minmax',False, True , 'cuda')])
    elif index =='fft_D40_x_cpu':
        base_path = '/media/ana-caznok/SSD-08/recon-segment/'
        return Compose([RGB2Pseudo_Hyp(base_path, 'D40'),FourierSpectralTransform('minmax',False, True, 'cpu')])
    
    elif index =='fft_D40_cube_gpu':
        base_path = '/media/ana-caznok/SSD-08/recon-segment/'
        return Compose([RGB2Pseudo_Hyp(base_path, 'D40'),FourierSpectralTransform('minmax',True, True, 'cuda')])
    
    elif index=='stft_D40_x_gpu': 
        base_path = '/media/ana-caznok/SSD-08/recon-segment/'
        return Compose([Downsample(factor=4), RGB2Pseudo_Hyp(base_path, 'D40'),Spectrogram4D(2, 5,32,'abs',device='cuda')])
    elif index=='stft_D40_x_cpu': 
        base_path = '/media/ana-caznok/SSD-08/recon-segment/'
        return Compose([Downsample(factor=4), RGB2Pseudo_Hyp(base_path, 'D40'),Spectrogram4D(2, 5,32,'abs',device='cpu')])

    
    elif index == 'hist_match_h5_04_and_patch_512': 
        base_path = os.getenv('ICASP_H5')
        return Compose([HistMatch_h5(base_path, 0.4), RandomCrop(width=512, height=512)])
    else:
        print("WARNING: Using no transforms")
        return None
    