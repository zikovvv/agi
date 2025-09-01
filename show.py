import matplotlib.pyplot as plt
import numpy as np
# from get_ds import dataset
from typing import Dict, Tuple, List

import torch

from get_ds import get_ds

def show_example(example : List[Tuple[np.ndarray, np.ndarray]]) :
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


def plot_eval_batch(
    to_show : List[torch.Tensor],
    max_field_width : int = 40,
    *,
    max_color: int = 30,
    cmap: str = "tab20",
    cell_inches: float = 2.0,      # size of each small image (inches)
    dpi: int = 400,                # higher -> sharper
    show_row_labels: bool = True,  # overlay row labels instead of titles (no extra space)
) -> None:
    """
    4 x B grid with NO whitespace.
    Rows: [Input ids, Actual labels, Predicted labels, Predicted on inputs]
    Columns: all batch elements.
    """
    max_field_area = max_field_width * max_field_width
    # for a in to_show :
    #     print(a.shape)
    fields = [
        f.view(f.shape[0], max_field_width, max_field_width) for f in to_show
    ]

    B, H, W = fields[0].shape
    nrows, ncols = len(to_show), B
    fig = plt.figure(figsize=(cell_inches * ncols, cell_inches * nrows), dpi=dpi)
    gs = fig.add_gridspec(nrows, ncols, wspace=0.0, hspace=0.0)

    # Helper to render one small image without axes chrome
    def _show(ax, arr):
        # Ensure numpy
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        ax.imshow(arr, vmin=0, vmax=11,  cmap=cmap)#, interpolation="nearest")
        ax.set_axis_off()
        # Keep pixels square & tight
        ax.set_aspect("equal", adjustable="box")

    # Fill grid
    for j in range(ncols):
        for i in range(nrows) :
            _show(fig.add_subplot(gs[i, j]), fields[i][j])

    # Absolutely no outer padding
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    plt.show()

# def main():
#     dataset = get_ds()
#     for example in random.sample(list(dataset.values()), len(dataset)):
#         show_example(example)
#         if plt.waitforbuttonpress():
#             plt.close('all')
#             break
# if __name__ == "__main__":
#     main()