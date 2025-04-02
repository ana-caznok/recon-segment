import os
from transforms import Compose, Downsample
from transforms.ifft_transform import InverseFourierSpectralTransform

def get_base_path():
    """Get the base path for loading data."""
    try: 
        base_path = os.environ.get("CODE_PATH") + '/'
    except: 
        base_path = '/media/ana-caznok/SSD-08/recon-segment/'
    return base_path 


def inverse_transform_factory(index: str, 
                              output: bool):
    """
    Factory to create and return a specific image transformation pipeline
    based on the input index string.
    """
    base_path = get_base_path()  # Resolve base path
    string_split = index.split('_')  # Split the string by underscore for parsing

    if 'shift' in index: 
        shift_fft = True
    else: 
        shift_fft = False
    if 'continuous' in index:
        stack_type = 'continuous'
    else: 
        stack_type = 'alternate'

    if output == True:
        my_norm = 'None'
    elif 'minmax' in index: 
        my_norm = 'minmax'
    else: 
        my_norm = 'None'

    ifft_function  = InverseFourierSpectralTransform(
                    norm = my_norm,
                    channel_first = True, 
                    device = 'cuda', 
                    shift = shift_fft, 
                    stack_type = stack_type)
    

    return ifft_function
