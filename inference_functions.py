import torch 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
from metrics.sam import SAMScore, Intensity_SAMScore

def plot_channel_differences(y: torch.Tensor, out: torch.Tensor, channels: list,model_name:str, base_path = "/media/ana-caznok/SSD-08/recon-segment/"):
    """restormer_fft2ifft2hsi_noclip.yaml
    Plots the difference between corresponding channels of `y` and `out` for the specified channels.

    Parameters:
    - y (torch.Tensor): Tensor of shape (61, 256, 256)
    - out (torch.Tensor): Tensor of shape (1, 61, 256, 256)
    - channels (list of int): List of channel indices to plot
    """
    # Remove the batch dimension from `out`, now shape (61, 256, 256)
    if y.type() != out.type():
        out = out.detach().cpu()
    out = out.squeeze(0)

    # Verify tensor shapes
    assert y.shape == out.shape == (61, 256, 256), "Shape mismatch: Expected (61, 256, 256)"

    # Compute the difference for each selected channel and store in a list
    differences = [np.abs((y[c] - out[c]).detach().cpu().numpy()) for c in channels]

    # Determine global min and max for consistent color scale
    global_min = 0 #min(diff.min() for diff in differences)
    global_max = 1 #max(diff.max() for diff in differences)

    # Determine subplot layout
    n_channels = len(channels)
    ncols = min(4, n_channels)
    nrows = (n_channels + ncols - 1) // ncols

    # Create figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten() if n_channels > 1 else [axes]

    # Plot each channel difference
    for idx, (channel, diff) in enumerate(zip(channels, differences)):
        ax = axes[idx]
        im = ax.imshow(diff, cmap='magma', vmin=global_min, vmax=global_max)
        ax.set_title(f'Channel {channel} Difference')
        ax.axis('off')

    # Turn off unused subplots if any
    for idx in range(len(differences), len(axes)):
        axes[idx].axis('off')

    # Add a common colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    fig.suptitle(model_name)

    # Adjust layout to prevent overlap
    #plt.tight_layout()
    plt.savefig(base_path + 'plots/' + 'err_' + model_name + '.png')

    plt.show()

def plot_output_channels(
    out: torch.Tensor,
    channels: list,
    model_name: str,
    base_path: str = "/media/ana-caznok/SSD-08/recon-segment/"):
    
    # Remove batch dimension, resulting shape (C, H, W)
    out = out.squeeze(0)

    # Check if output has correct dimensions
    assert out.ndim == 3, "Expected tensor shape (C, H, W) after squeezing batch dimension"

    # Extract the selected channels and move to CPU
    selected_images = [out[c].detach().cpu().numpy() for c in channels]

    # Determine layout for subplots
    n_channels = len(channels)
    ncols = min(4, n_channels)
    nrows = (n_channels + ncols - 1) // ncols

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten() if n_channels > 1 else [axes]

    # Plot each selected channel
    for idx, (channel, img) in enumerate(zip(channels, selected_images)):
        ax = axes[idx]
        im = ax.imshow(img, cmap='gray')
        ax.set_title(f'Channel {channel}')
        ax.axis('off')

    # Turn off unused axes if there are any
    for idx in range(len(selected_images), len(axes)):
        axes[idx].axis('off')

    # Add colorbar for last image
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    fig.suptitle(f'Output Channels - {model_name}')

    # Save the figure
    plt.savefig(base_path + 'plots/' + 'out_channels_' + model_name + '.png')
    plt.show()

def plot_spectral_input(x, output, y, tp, 
                        model_name, key,
                        points = [(148, 140), (40, 40)], 
                        base_path: str = "/media/ana-caznok/SSD-08/recon-segment/"):
    
    # Squeeze output if it has a batch dimension
    if output.ndim == 4:
        output = output.squeeze(0)
        
    if y.type() != output.type():
       output = output.detach().cpu().numpy()
    # Move tensors to numpy
    y = y.detach().cpu().numpy()
    output = output.detach().cpu().numpy()
    x = x.detach().cpu().numpy()

    if tp =='msi':
        wavl = np.linspace(400,1000,61)
        rgbnir = [420,500,630,960] 
        _,wavl_idx = np.unique(wavl,return_index=True)
        rgbnir_idx = wavl_idx[np.isin(wavl, rgbnir)]
        x_plot = rgbnir_idx
        x_min,x_max = wavl_idx[0], wavl_idx[-1]
        x_domain = 'wavelenght'

    else:
        x_plot = np.arange(0,x.shape[0],1)
        x_min,x_max = 0, x.shape[0]

    # Create 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    input_colors = ['hotpink', 'red']
    my_colors = ['turquoise', 'teal', 'mediumorchid', 'darkorchid']

    # --- TOP LEFT: Middle band of input x ---
    mid_band = x.shape[0] // 2
    axs[0, 0].imshow(x[mid_band, :, :], cmap='gray')
    axs[0, 0].set_title(f"Input Image - Band {mid_band}")

    # Mark selected points
    for i, (row, col) in enumerate(points):
        axs[0, 0].plot(col, row, '*', color=input_colors[i], markersize=10)

    # --- TOP RIGHT: Spectral values from input x ---
    for i, (row, col) in enumerate(points):
        axs[0, 1].plot(x_plot, x[:, row, col],'o', color=input_colors[i], label=f'Input ({row},{col})')

    axs[0, 1].set_title("Spectral Signature of Input (x)")
    axs[0, 1].set_xlabel("Band Index")
    axs[0, 1].set_ylabel("Intensity")
    axs[0,1].set_xlim([x_min,x_max])
    axs[0, 1].legend()

    # --- BOTTOM LEFT: GT image with points ---
    axs[1, 0].imshow(y[30, :, :], cmap='gray')
    axs[1, 0].set_title("GT Band 30 with Points")
    for i, (row, col) in enumerate(points):
        axs[1, 0].plot(col, row, '*', color=my_colors[i*2 + 1], markersize=10)

    # --- BOTTOM RIGHT: GT vs Output Spectra ---
    for i, (row, col) in enumerate(points):
        axs[1, 1].plot(y[:, row, col], color=my_colors[i*2], label=f'GT ({row},{col})')
        axs[1, 1].plot(output[:, row, col], color=my_colors[i*2 + 1], linestyle='-.', label=f'Output ({row},{col})')
    axs[1, 1].set_title("Spectral Comparison (GT vs Output)")
    axs[1, 1].set_xlabel("Band Index")
    axs[1, 1].set_ylabel("Intensity")
    axs[1, 1].legend()

    # Overall figure title and layout adjustment
    fig.suptitle(model_name, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(base_path + 'plots/in-out_' + model_name + 'png')
    plt.show()

def plot_spectral_diff(y, output, model_name, points=[(148, 140), (40, 40)]):
    """
    Plots spectral differences between ground truth and model output at given spatial points.

    Parameters:
        y (Tensor or ndarray): Ground truth hyperspectral data of shape (bands, H, W).
        output (Tensor or ndarray): Model output data, expected to be (1, bands, H, W) or (bands, H, W).
        model_name (str): Title for the plot.
        points (list of tuple): List of (row, col) coordinates for which to plot spectra.
    """
    
    # Remove batch dimension if present
    if output.ndim == 4:
        output = output.squeeze(0)
    y = y.detach().cpu().numpy()
    output = output.detach().cpu().numpy()
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Show a grayscale image at an arbitrary band (e.g., 30)
    axs[0].imshow(y[30, :, :], cmap='gray')
    axs[0].set_title("Spatial Locations")
    
    my_colors = ['teal', 'darkorange']
    
    for i, (row, col) in enumerate(points):
        # Mark point on the image
        axs[0].plot(col, row, '*', color=my_colors[i], markersize=10)
        
        # Plot ground truth and output spectra at this point
        axs[1].plot(y[:, row, col], color=my_colors[i], label=f'GT ({row},{col})')
        axs[1].plot(output[:, row, col], color=my_colors[i], linestyle='-.', label=f'Output ({row},{col})')
    
    axs[1].set_title("Spectral Comparison")
    axs[1].set_xlabel("Spectral Band")
    axs[1].set_ylabel("Intensity")
    axs[1].legend()
    
    fig.suptitle(model_name)
    plt.tight_layout()
    plt.show()

def replace_zeros_with_small_value(tensor: torch.Tensor, small_value: float = 0.0001) -> torch.Tensor:

    # Clone the tensor to avoid modifying the original tensor in-place
    result = tensor.clone()

    # Create a boolean mask where the tensor is zero
    zero_mask = result == 0

    # Replace all values in result where mask is True with the small_value
    result[zero_mask] = small_value

    return result
def calc_plot_ssim(y, out, model_name, base_path: str = "/media/ana-caznok/SSD-08/recon-segment/"):

    # Ensure both tensors are on CPU and detached for visualization
    if y.type() != out.type():
        out = out.detach().cpu()

    #y = y.detach().cpu()

    # Compute SAM and Intensity-SAM with corresponding maps
    ssim_func = StructuralSimilarityIndexMeasure()
    channel_nb = y.shape[1]
    ssim = []
    for c in range(channel_nb):
        ssim_c = ssim_func(y[0,c,:,:].unsqueeze(0).unsqueeze(0),out[0,c,:,:].unsqueeze(0).unsqueeze(0)).detach().cpu().numpy()
        ssim.append(ssim_c)

    ssim = np.array(ssim)

    plt.plot(ssim)
    plt.title(model_name)
    plt.ylabel('SSIM')
    plt.xlabel('channel')
    plt.savefig(base_path + 'plots/spectral-ssim_' + model_name + '.png')
    plt.show()

    return ssim

def plot_all_models(data_dict, base_path: str = "/media/ana-caznok/SSD-08/recon-segment/"):
    """
    Plots all arrays in a single figure with each line labeled by its dictionary key.

    Parameters:
    - data_dict (dict): A dictionary where keys are model names and values are 1D arrays.
    """
    plt.figure(figsize=(8, 8))
    
    # Plot each array with the key as legend
    for key, array in data_dict.items():
        plt.plot(array, label=key)

    # Add legend and labels
    plt.legend()
    plt.title('All Models Comparison')
    plt.xlabel('Channels')
    plt.ylabel('SSIM')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(base_path + 'spectral-compare_ssim.png')
    plt.show()


def plot_grouped_models(data_dict, base_path: str = "/media/ana-caznok/SSD-08/recon-segment/", only_vis=False):
    """
    Plots subplots grouped by model types ('restormer' and 'unet').
    Each subplot contains two lines: one for 'msi2hsi' and another for 'fft2ifft2hsi'.

    Parameters:
    - data_dict (dict): A dictionary where keys are model-transformation identifiers and values are 1D arrays.
    """
    # Initialize subplots for each model type
    lbl_dict = {}
    for d in data_dict.keys(): 
        if 'interp31' in d: 
            lbl_dict[d] = 'Contains NIR info ' + d
        elif 'msi' in d: 
            lbl_dict[d] = 'Contains NIR info ' + d
        else:
            lbl_dict[d] = 'Pseudo_hyper from RGB ' + d
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    model_types = ['restormer', 'unet']

    for idx, model in enumerate(model_types):
        ax = axs[idx]
        for key, array in data_dict.items():
            # Check if the key contains the current model and a valid transformation
            if model in key and ('msi2hsi' in key or 'fft2ifft2hsi' in key):
                ax.plot(array, label=lbl_dict[key])
        
        # Set title and labels for the subplot
        ax.set_title(f'{model.upper()} Models')
        ax.set_xlabel('Channels')
        if idx == 0:
            ax.set_ylabel('SSIM')
        ax.grid(True)
        ax.legend()

    plt.suptitle('Model Comparison by Transformation Type')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(base_path + 'spectral_compare_nir_info.png')
    plt.show()

def calc_plot_sam(y, out, m, model_name, base_path: str = "/media/ana-caznok/SSD-08/recon-segment/"):
    """
    Calculate and plot SAM and Intensity-SAM score maps side-by-side with a shared intensity scale bar.
    """

    # Ensure both tensors are on CPU and detached for visualization
    if y.type() != out.type():
        out = out.detach().cpu()
        #mask = mask.cpu().detach()

    #y = y.detach().cpu()
    mask = m['mask'].unsqueeze(0)
    mask = mask.cpu().detach()
    masked_y = replace_zeros_with_small_value(mask*y,0.001)
    masked_out = replace_zeros_with_small_value(mask*out,0.002)
    print(masked_out.shape)
    # Compute SAM and Intensity-SAM with corresponding maps
    sam_func = SAMScore()
    isam_func = Intensity_SAMScore()

    sam, sam_map = sam_func(y, out, return_maps=True)
    #face_sam, face_sam_map = sam_func(masked_y, masked_out)
    isam, isam_map = isam_func(y, out, return_maps=True)

    # Convert to numpy for plotting
    sam_map_np = sam_map[0].detach().cpu().numpy()
    isam_map_np = isam_map[0].detach().cpu().numpy()
    #face_sam_map = face_sam_map[0].detach().cpu().numpy()

    # Determine shared color scale limits
    #vmin = min(sam_map_np.min(), isam_map_np.min())
    #vmax = max(sam_map_np.max(), isam_map_np.max())
    vmin=0
    vmax=1.01

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Display SAM map
    im0 = axs[0].imshow(sam_map_np, cmap='magma', vmin=vmin, vmax=vmax)
    axs[0].set_title(f'SAM = {sam:.4f}')

    # Display ISAM map
    #im1 = axs[1].imshow(face_sam_map, cmap='magma', vmin=vmin, vmax=vmax)
    #axs[1].set_title(f'Face SAM = {face_sam:.4f}')

    # Display ISAM map
    im2 = axs[1].imshow(isam_map_np, cmap='magma', vmin=vmin, vmax=vmax)
    axs[1].set_title(f'Intensity SAM = {isam:.4f}')

    # Add a single colorbar for both subplots
    fig.colorbar(im0, ax=axs, orientation='vertical', fraction=0.02, pad=0.04, label='Score Intensity')

    fig.suptitle(model_name)

    plt.savefig(base_path + 'plots/spatial-sam_' + model_name + '.png')

    #plt.tight_layout()
    plt.show()
    