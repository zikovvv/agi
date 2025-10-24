# grid_to_pillow_images Function

## Overview
The `grid_to_pillow_images` function converts a grid of integers (like ARC puzzle grids) into Pillow images with customizable sizes.

## Function Signature
```python
def grid_to_pillow_images(
    grid: np.ndarray,
    sizes: List[int] = None
) -> Union[Image.Image, List[Image.Image]]
```

## Parameters
- **grid** (`np.ndarray`): A 2D numpy array of integers representing the grid to be visualized
- **sizes** (`List[int]`, optional): List of widths in pixels. Default is `[128]`. Height is calculated proportionally to maintain the grid's aspect ratio.

## Returns
- **Single `PIL.Image.Image`**: When `sizes` contains only one element
- **`List[PIL.Image.Image]`**: When `sizes` contains multiple elements

## Features
1. **Default size**: 128px width (as specified in requirements)
2. **Proportional scaling**: Height is automatically calculated to preserve the grid's aspect ratio
3. **Consistent coloring**: Uses the `tab20` colormap, matching the existing `show_examples` function
4. **Grid preservation**: Uses NEAREST neighbor interpolation to maintain sharp grid appearance
5. **Flexible output**: Returns single image or list based on number of requested sizes

## Usage Examples

### Example 1: Default Size (128px)
```python
import numpy as np
from show import grid_to_pillow_images

grid = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])

img = grid_to_pillow_images(grid)
print(img.size)  # (128, 128) for a square grid
img.save('grid.png')
```

### Example 2: Custom Single Size
```python
grid = np.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [2, 2, 3, 3],
    [2, 2, 3, 3]
])

img = grid_to_pillow_images(grid, sizes=[256])
print(img.size)  # (256, 256)
img.save('grid_256.png')
```

### Example 3: Multiple Sizes
```python
grid = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 2, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

imgs = grid_to_pillow_images(grid, sizes=[64, 128, 256])
for i, img in enumerate(imgs):
    img.save(f'grid_size_{i}.png')
```

### Example 4: Rectangular Grid (Aspect Ratio Preserved)
```python
grid = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8],
    [1, 2, 3, 4, 5, 6, 7, 8]
])  # 2x8 grid, ratio 1:4

img = grid_to_pillow_images(grid, sizes=[160])
print(img.size)  # (160, 40) - aspect ratio preserved
```

## Implementation Details

### Color Mapping
The function uses matplotlib's `tab20` colormap to convert integer values to RGB colors:
1. Grid values are normalized to [0, 1] range
2. The `tab20` colormap is applied to get RGBA values
3. Alpha channel is discarded, RGB values are scaled to [0, 255]

### Scaling Algorithm
1. Base image is created from the colormap-processed grid
2. For each requested width:
   - Height is calculated as: `height = int(width * grid_height / grid_width)`
   - Image is resized using `Image.NEAREST` to preserve sharp grid appearance

### Edge Cases
- **Single value grids**: Handled by checking if `vmax > vmin` before normalization
- **Non-square grids**: Aspect ratio is always preserved
- **Large grids**: No size limitations, but larger grids will result in larger base images

## Dependencies
- `numpy`: For array operations
- `PIL (Pillow)`: For image creation and manipulation
- `matplotlib`: For colormap support

These dependencies are included in `requirements.txt`.

## Related Functions
- `show_examples()`: Displays multiple input/output pairs using matplotlib
- `plot_batch()`: Plots batches of tensors with wandb integration

## Testing
See `example_grid_to_pillow.py` for comprehensive usage examples and test cases.
