from .save_challenge_fmt import SaveChallengeFMT
from .upsample import TorchInterpolation


def select_post_processing(index: str):
    '''
    Index post processing depending on string
    '''

    if "save" in index:
        # last 3 characters will select format, example, save_npy will save as .npy
        fmt = f'.{index[-3:]}'
        return SaveChallengeFMT(fmt=fmt)
    elif "upsample" in index:
        # will use last character as upsample factor in spatial resolution
        try:
            factor = int(index[-1])
        except Exception as e:
            print("ERROR: When using upsample post_processing, give factor as last digit, example: upsample_4")
            raise e
        
        return TorchInterpolation(factor=factor, order=3)
    else:
        raise ValueError(f"Unsupported index {index} for select_post_processing")
