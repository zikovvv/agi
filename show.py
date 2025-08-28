import matplotlib.pyplot as plt
import numpy as np
# from get_ds import dataset
from typing import Dict, Tuple, List

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
        
# def main():
#     dataset = get_ds()
#     for example in random.sample(list(dataset.values()), len(dataset)):
#         show_example(example)
#         if plt.waitforbuttonpress():
#             plt.close('all')
#             break
# if __name__ == "__main__":
#     main()