"""
IO Utilities for Handling Images and Tensors

This module provides a set of functions for displaying images, 
converting between PyTorch tensors and NumPy arrays, 
and saving arrays as MATLAB .mat files.
"""

from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from scipy.io import savemat
from pathlib import Path

__all__ = [
    'glance',
    'save_array_to_img',
    'save_array_to_mat',
]

def _tensor_to_numpy(
        image: Union[np.ndarray, torch.Tensor, Image.Image], 
        clip: bool = False
    ) -> np.ndarray:
    """
    Convert a PyTorch tensor or NumPy array to a NumPy array.

    If the input is a PyTorch tensor, it is assumed to be in 
    the range [0, 1] and is scaled to [0, 255]. 

    If the input is a NumPy array, it is assumed to be in the 
    range [0, 255] by default.
    """
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 4:
            if image.shape[0] != 1:
                raise ValueError("Batch number should be 1.")
            image = image[0,:,:,:]
        if len(image.shape) == 3:
            image = image.permute(1, 2, 0)
            if image.shape[-1] == 1:
                image = image[:,:,0]
        elif len(image.shape) != 2:
            raise ValueError("Image should be an image.")
        image_array = image.detach().cpu().numpy() * 255.0

    else:
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.ndim != 3 and image.ndim != 2:
            raise ValueError("Image should be an image.")
        image_array = image

    if clip:
        image_array = image_array.clip(0.0, 255.0)

    return image_array

def glance(
        image: Union[np.ndarray, torch.Tensor, Image.Image], 
        clip: bool = False,
        title: str = "",
        hide_axis: bool = True):
    """
    Display a PyTorch tensor or NumPy array as an image.

    Can input:
    * tensor: 4 dims, batch number should only be 1. channel can be 3 or 1.
    * tensor: 3 dims, channel can be 3 or 1.
    * tensor: 2 dims.
    * ndarray: 2 dims, channel can be 3 or 1.
    * ndarray: 2 dims.
    """
    # transform torch.tensor to np.array
    image = _tensor_to_numpy(image,clip)

    # show image with PIL.Image
    plt.imshow(image.astype(np.uint8), 
               cmap='gray' if image.ndim == 2 else 'viridis')
    if title != "": plt.title(title)
    if hide_axis: plt.axis('off')
    plt.show()

def save_array_to_img(
        image: Union[np.ndarray, torch.Tensor, Image.Image], 
        filename: Union[str, Path], 
        clip: bool = False):
    # transform torch.tensor to np.array
    image = _tensor_to_numpy(image,clip)
    
    # save image
    Image.fromarray(
        obj = image.astype(np.uint8),
        mode = 'L' if image.ndim == 2 else 'RGB'
    ).save(filename)

def save_array_to_mat(
        image: Union[np.ndarray, torch.Tensor, Image.Image], 
        base_filename: str = 'glance', 
        clip: bool = False,
        log: bool = False):
    """
    Save a NumPy array or PyTorch tensor as MATLAB .mat files.
    """
    # transform torch.tensor to np.array
    image_array = _tensor_to_numpy(image,clip)

    # Save Image
    if image_array.ndim == 2:
        savemat(f"{base_filename}_gray.mat", {'gray': image_array})
        if log:
            print(f"Gray image have saved as {base_filename}_gray.mat")
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        savemat(f"{base_filename}_red.mat", {'red': image_array[:, :, 0]})
        savemat(f"{base_filename}_green.mat", {'green': image_array[:, :, 1]})
        savemat(f"{base_filename}_blue.mat", {'blue': image_array[:, :, 2]})
        if log:
            print(f"RGB image have saved as {base_filename}_red.mat, {base_filename}_green.mat and {base_filename}_blue.mat")
    else:
        raise ValueError("Image array should be 2D(Gray) or 3D (RGB).")


if __name__ == "__main__":
    import os
    
    # Load Image
    rgb_path = Path(os.path.dirname(__file__), "image/rgb.png")
    gray_path = Path(os.path.dirname(__file__), "image/gray.png")
    rgb_im = Image.open(rgb_path)
    gray_im = np.array(Image.open(gray_path))

    # We can Auto-recgonice RGB and Gray images!
    # We can Auto-recgonice Image.Image, torch.Tensor, np.ndaray!

    # use matplotlib to show image
    glance(rgb_im, title=f'RGB - {type(rgb_im)}')
    glance(gray_im, title=f'GRAY - {type(gray_im)}')

    # the same of save_array_to_mat and save_array_to_img