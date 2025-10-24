#!/usr/bin/env python3
"""
Example script demonstrating the use of grid_to_pillow_images function.

This script shows how to convert a grid of integers (like ARC puzzle grids)
to Pillow images in different sizes.
"""

import numpy as np
from show import grid_to_pillow_images

# Example 1: Simple grid with default size
print("Example 1: Simple grid with default size (128px width)")
grid = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])
img = grid_to_pillow_images(grid)
print(f"Grid shape: {grid.shape}")
print(f"Image size: {img.size}")
# img.save('grid_default.png')  # Uncomment to save
print()

# Example 2: Grid with single custom size
print("Example 2: Grid with custom size (256px width)")
grid = np.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [2, 2, 3, 3],
    [2, 2, 3, 3]
])
img = grid_to_pillow_images(grid, sizes=[256])
print(f"Grid shape: {grid.shape}")
print(f"Image size: {img.size}")
# img.save('grid_256.png')  # Uncomment to save
print()

# Example 3: Grid with multiple sizes
print("Example 3: Grid with multiple sizes [64, 128, 256]")
grid = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 2, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])
imgs = grid_to_pillow_images(grid, sizes=[64, 128, 256])
print(f"Grid shape: {grid.shape}")
print(f"Number of images: {len(imgs)}")
for i, img in enumerate(imgs):
    print(f"  Image {i}: size={img.size}")
    # img.save(f'grid_size_{i}.png')  # Uncomment to save
print()

# Example 4: Rectangular grid (aspect ratio preserved)
print("Example 4: Rectangular grid (aspect ratio preserved)")
grid = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8],
    [1, 2, 3, 4, 5, 6, 7, 8]
])
img = grid_to_pillow_images(grid, sizes=[160])
print(f"Grid shape: {grid.shape} (height=2, width=8, ratio=1:4)")
print(f"Image size: {img.size} (width=160, height should be 40)")
# img.save('grid_rectangular.png')  # Uncomment to save
print()

print("All examples completed successfully!")
print("Uncomment the .save() lines to save the images to disk.")
