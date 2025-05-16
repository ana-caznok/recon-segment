import os
from transforms import RandomCrop, Compose, Downsample, RandomCropBB, HistMatch_h5, RGB2Pseudo_Hyp, Spectrogram4D, FourierSpectralTransform, DownsampleInput, RainbowTransformsnnUNet, RandomCropCenter, HistMatch
from transforms import ResizeTo256
from crop_channels import ChannelCrop
def get_base_path():
    """Get the base path for loading data."""
    try: 
        base_path = os.environ.get("CODE_PATH") + '/'
    except: 
        base_path = '/media/ana-caznok/SSD-08/recon-segment/'
    return base_path 


def rgb2hyp_transf(rgb2hyp_file='D40', normalize=False):
    """Helper to create RGB to Pseudo-Hyperspectral transform."""
    return RGB2Pseudo_Hyp(get_base_path(), rgb2hyp_file, normalize) # PROVAVELMENTE NORMALIZE PRECISE SER SEMPRE TRUE


def fourier_transform(rgb2hyp_file='D40', pseudohyp_norm = False,downsample_factor=None, resize=False, fft_norm='abs',
                      double=False, cube=True, shift=False, 
                      device='cuda', invert=False, stack_type='alternate'):
    """Compose a transform including optional downsampling, RGB2Hyp, and spectral FFT."""
    transforms = []
    if downsample_factor:
        transforms.append(Downsample(factor=downsample_factor))  # Downsample if specified

    if resize: 
        transforms.append(ResizeTo256())
    
    transforms.append(rgb2hyp_transf(rgb2hyp_file, pseudohyp_norm))  # Convert RGB to hyperspectral
    transforms.append(FourierSpectralTransform(
        norm=fft_norm,
        double=double,
        transf_cube=cube,
        channel_first=True,
        device=device,
        shift=shift,
        stack_type=stack_type,
        invert=invert, 
        return_numpy=False
    ))
    return Compose(transforms)  # Compose all into one transform


def transform_factory(index: str):
    """
    Factory to create and return a specific image transformation pipeline
    based on the input index string.
    """
    base_path = get_base_path()  # Resolve base path
    string_split = index.split('_')  # Split the string by underscore for parsing

    # Basic predefined transforms
    if index == "downsample_4_crop_64":
        return Compose([Downsample(factor=4), RandomCrop(width=64, height=64)])
    elif index == "downsample_4":
        return Downsample(factor=4)
    elif index == "downsample_2":
        return Downsample(factor=2)

    # Random spatial cropping
    elif index == "random_patch_64":
        return RandomCrop(width=64, height=64)
    elif index == "random_patch_128":
        return RandomCrop(width=128, height=128)
    elif index == "random_patch_256":
        return RandomCrop(width=256, height=256)
    elif index == "random_patch_512":
        return RandomCrop(width=512, height=512)

    # Bounding-box-based random crop
    elif index == 'random_crop_bb_downs_p085': 
        return RandomCropBB(patch_size=64, prob_in_bb=0.85)
    
    elif index=='resize256': 
        return ResizeTo256()
    

    elif 'rgb2hyp' in string_split:
        # Determine downsample factor from prefix (e.g. 'downs4')
        if 'downs' in string_split[0]:
            factor = int(string_split[0].split('s')[1])
        else:
            factor = None

        # Choose RGB2Hyp source
        if 'D40' in index:
            rgb2hyp_file = 'D40'
        elif 'normal' in index:
            rgb2hyp_file = 'cie'
        elif 'interp' in index: 
            rgb2hyp_file = 'interp'
        else:
            rgb2hyp_file = 'cie'

        if 'norm01' in index: 
            normalization = True
        else: 
            normalization = False

        return Compose([Downsample(factor=factor), rgb2hyp_transf(rgb2hyp_file, normalization)])

    # FFT-based transforms
    elif 'fft' in string_split:
        pseudohyp_norm = False
        resize=False

        # Determine downsample factor from prefix (e.g. 'downs4')
        if 'downs' in string_split[0]:
            factor = int(string_split[0].split('s')[1])
        else:
            factor = None

        if 'resize256' in index: 
            resize=True
        # Choose RGB2Hyp source
        if 'D40' in index:
            rgb2hyp_file = 'D40'
        elif 'normal' in index:
            rgb2hyp_file = 'cie'
        elif 'interp' in index: 
            #_interp15_
            interp_factor = index.split('interp')[1].split('_')[0]
            rgb2hyp_file = 'interp' + interp_factor
            
        else:
            rgb2hyp_file = 'cie'

        # Use cube format (3D) or not
        if 'cube' in index:
            transform_cube = True
        elif 'x' in index:
            transform_cube = False
        else:
            transform_cube = False

        # Select device
        device = 'cuda' if 'gpu' in index or 'cuda' in index else 'cpu'

        # Use double (real+imag channels)
        double = 'double' in index

        # Frequency shift toggle
        shift_frequencies = 'shift' in index

        # Stack output as alternate or continuous
        stack_type = 'continuous' if 'continuous' in index else 'alternate'

        # Invert toggle for FFT output
        invert = 'invert' in index

        # Choose normalization method
        if 'minmax' in index:
            fft_norm = 'minmax'
            if 'byc' in index:
                fft_norm = fft_norm + '-byc' #by channel is actually by pixel, by spectral 
            if 'byimg' in index:
                fft_norm = fft_norm + '-byimg' #by channel is actually by channel, by img channel
        elif 'softmax' in index:
            fft_norm = 'softmax'
        elif 'none' in index:
            fft_norm = 'None'
            pseudohyp_norm = True
        else:
            fft_norm = 'abs'

        # Compose the full FFT transform
        fft_transform = fourier_transform(
                        rgb2hyp_file,
                        pseudohyp_norm,
                        downsample_factor=factor,
                        resize=resize,
                        fft_norm=fft_norm,
                        double=double,
                        cube=transform_cube,
                        shift=shift_frequencies,
                        device=device,
                        invert=invert,
                        stack_type=stack_type
                        )

        if 'crop-channels' in index: 
            fft_transform = Compose([ChannelCrop(31), fft_transform])
            
        return fft_transform

       

    # STFT (spectrogram-based) transforms
    elif index == 'stft_D40_x_gpu':
        return Compose([
            Downsample(factor=4),  # Downsample
            rgb2hyp_transf('D40'),  # Convert to hyperspectral
            Spectrogram4D(2, 5, 32, 'abs', device='cuda')  # Apply STFT
        ])
    elif index == 'stft_D40_x_cpu':
        return Compose([
            Downsample(factor=4),
            rgb2hyp_transf('D40'),
            Spectrogram4D(2, 5, 32, 'abs', device='cpu')
        ])

    # Histogram matching and patch extraction
    elif index == 'hist_match_h5_04_and_patch_512':
        base_path = os.getenv('ICASP_H5')
        return Compose([
            HistMatch_h5(base_path, 0.4),
            RandomCrop(width=512, height=512)
        ])

    # Default/fallback case
    else:
        print("WARNING: Using no transforms")
        return None
