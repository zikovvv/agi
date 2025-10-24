import matplotlib.pyplot as plt
import numpy as np
# from get_ds import dataset
from typing import Dict, Tuple, List, Union
import torch
from PIL import Image
import matplotlib.cm as cm


def show_examples(example : List[Tuple[np.ndarray, np.ndarray]]) :
    n = len(example)
    fig, axs = plt.subplots(2, n, figsize=(3*n, 6))
    for i, (inp, out) in enumerate(example) :
        # Get the unique values from both input and output to create consistent color mapping
        all_values = np.unique(np.concatenate([inp.flatten(), out.flatten()]))
        vmin, vmax = all_values.min(), all_values.max()
        axs[0, i].imshow(inp, cmap='tab20', vmin=vmin, vmax=vmax)
        axs[0, i].set_title('Input')

        axs[1, i].imshow(out, cmap='tab20', vmin=vmin, vmax=vmax)
        axs[1, i].set_title('Output')
    plt.show()


def grid_to_pillow_images(
    grid: np.ndarray,
    sizes: List[int] = None
) -> Union[Image.Image, List[Image.Image]]:
    """
    Convert a grid of integers to Pillow Image(s) in different sizes.
    
    Args:
        grid: 2D numpy array of integers representing the grid
        sizes: List of widths (in pixels) for the left side. 
               Default is [128]. Height is calculated proportionally.
    
    Returns:
        Single PIL Image if sizes has one element, otherwise list of PIL Images
    """
    if sizes is None:
        sizes = [128]
    
    # Get the grid dimensions
    grid_height, grid_width = grid.shape
    
    # Get unique values for color mapping
    unique_values = np.unique(grid)
    vmin, vmax = unique_values.min(), unique_values.max()
    
    # Get the tab20 colormap
    cmap = cm.get_cmap('tab20')
    
    # Normalize grid values to [0, 1] range for colormap
    if vmax > vmin:
        normalized_grid = (grid - vmin) / (vmax - vmin)
    else:
        normalized_grid = np.zeros_like(grid, dtype=float)
    
    # Apply colormap to get RGBA values
    colored_grid = cmap(normalized_grid)
    
    # Convert to RGB (0-255 range)
    rgb_grid = (colored_grid[:, :, :3] * 255).astype(np.uint8)
    
    # Create PIL Image from the RGB array
    base_image = Image.fromarray(rgb_grid, mode='RGB')
    
    # Generate images for each size
    result_images = []
    for width in sizes:
        # Calculate height proportionally
        height = int(width * grid_height / grid_width)
        
        # Resize using NEAREST to preserve grid appearance
        resized_image = base_image.resize((width, height), Image.NEAREST)
        result_images.append(resized_image)
    
    # Return single image if only one size, otherwise return list
    if len(result_images) == 1:
        return result_images[0]
    else:
        return result_images




import random
import wandb

def plot_batch(
    data: List[torch.Tensor],
    height: int,
    width: int,
    show_to_window: bool,
    cmap: str = "tab20",
    cell_inches: float = 2.0,      # size of each small image (inches)
    dpi: int = 400,                # higher -> sharper
) -> None:
    fields = [f.view(f.shape[0], height, width) for f in data]

    B, H, W = fields[0].shape
    nrows, ncols = len(data), B
    fig = plt.figure(figsize=(cell_inches * ncols, cell_inches * nrows), dpi=dpi)
    gs = fig.add_gridspec(nrows, ncols, wspace=0.0, hspace=0.0)

    # Helper to render one small image without axes chrome
    def _show(ax, arr):
        # Ensure numpy
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        ax.imshow(arr, vmin=0, vmax=11, cmap=cmap)
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="box")

    # Fill grid
    for j in range(ncols):
        for i in range(nrows):
            _show(fig.add_subplot(gs[i, j]), fields[i][j])

    # Absolutely no outer padding
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Send to Weights & Biases if a run is active
    try:
        run = getattr(wandb, "run", None)
        if run is not None and not getattr(run, "_is_finished", False):
            wandb.log({"plot_batch": wandb.Image(fig)})
    except Exception:
        pass
    if show_to_window:
        plt.show()
    plt.close()
    

