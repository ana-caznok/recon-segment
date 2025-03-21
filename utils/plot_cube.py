'''
Utility to plot the multispectral final image with 60+ channels with matplotlib
'''
import math
import numpy as np
import matplotlib.pyplot as plt


def plot_cube(cube: np.ndarray, name: str, show: bool):
    '''
    Saves a summarized view of a cube as a 8x8 mosaic
    ''' 
    assert cube.shape[0] == 1, "cube has to be of batch size 1"

    channel_first = np.argmin(cube.shape) == 0
    if channel_first:
        C = cube.shape[1]
    else:
        C = cube.shape[-1]

    side = math.ceil(math.sqrt(C))
    plt.figure(figsize=(16, 16))
    for y in range(side):
        for x in range(side):
            idx = y*side + x
            
            plt.subplot(side, side, idx + 1)
            try:
                if channel_first:
                    img = cube[0, idx, :, :]
                else:
                    img = cube[0, :, :, idx]
            except:
                continue
            plt.imshow(img, cmap="gray", vmax=1, vmin=0)
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{name}.jpg", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

    return
