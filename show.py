import matplotlib.pyplot as plt
import numpy as np
# from get_ds import dataset
from typing import Dict, Tuple, List
import torch


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
    

