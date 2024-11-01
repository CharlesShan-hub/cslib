"""
IO Utilities for Handling Images and Tensors

This module provides a set of functions for displaying images, 
converting between PyTorch tensors and NumPy arrays, 
and saving arrays as MATLAB .mat files.
"""

from typing import Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torch
from scipy.io import savemat
from pathlib import Path
from torchvision.transforms import ToTensor
from skimage import color


__all__ = [
    'to_tensor',
    'to_image',
    'to_numpy',
    'glance',
    'path_to_gray',
    'path_to_rgb',
    'path_to_ycbcr',
    'rgb_to_ycbcr',
    'ycbcr_to_rgb',
    'save_array_to_img',
    'save_array_to_mat',
]


CLIP_MIN = 0.0
CLIP_MAX = 1.0


def _clip(
        image: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(image, np.ndarray):
        return image.clip(min=CLIP_MIN, max=CLIP_MAX)
    elif isinstance(image, torch.Tensor):
        return image.clamp(min=CLIP_MIN, max=CLIP_MAX)


def _tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
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
    return image.detach().cpu().numpy()
    

def _tensor_to_image(image: torch.Tensor) -> Image.Image:
    image_array = _tensor_to_numpy(image)
    return _numpy_to_image(image_array)


def _image_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image)/255.0


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    return ToTensor()(image)/255.0


def _numpy_to_image(image: np.ndarray) -> Image.Image:
    image = image * 255.0
    if len(image.shape) == 2:
        return Image.fromarray(image.astype(np.uint8), mode="L")
    else:
        return Image.fromarray(image.astype(np.uint8), mode="RGB") 


def _numpy_to_tensor(image: np.ndarray) -> torch.Tensor:
    return ToTensor()(image)


def to_tensor(
        image: Union[np.ndarray, torch.Tensor, Image.Image], 
        clip: bool = False
    ) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        image = _numpy_to_tensor(image)
    elif isinstance(image, Image.Image):
        image = _image_to_tensor(image)
    return _clip(image) if clip else image # type: ignore


def to_numpy(
        image: Union[np.ndarray, torch.Tensor, Image.Image], 
        clip: bool = False
    ) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = _tensor_to_numpy(image)
    elif isinstance(image, Image.Image):
        image = _image_to_numpy(image)
    return _clip(image) if clip else image # type: ignore


def to_image(
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        clip: bool = False
    ) -> Image.Image:
    if isinstance(image, np.ndarray):
        image = _clip(image) if clip else image
        image = _numpy_to_image(image) # type: ignore
    elif isinstance(image, torch.Tensor):
        image = _clip(image) if clip else image
        image = _tensor_to_image(image) # type: ignore
    return image


def path_to_gray(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from the given path and convert it to Gray format.
    
    Output: Gary image, range from 0 to 1, channel number is 1
    """
    image = np.array(Image.open(path))
    if len(image.shape) == 3:
        return color.rgb2gray(image)
    elif len(image.shape) != 2:
        raise ValueError(f"Wrong shape of image: {image.shape}")
    return image/255.0


def path_to_rgb(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from the given path and convert it to RGB format.

    Output: RGB image, range from 0 to 1, channel number is 3
    """
    image = np.array(Image.open(path))
    if len(image.shape) == 2:
        return color.gray2rgb(image)/255.0
    elif len(image.shape) != 3:
        raise ValueError(f"Wrong shape of image: {image.shape}")
    return image/255.0


def path_to_ycbcr(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from the given path and convert it to YCbCr format.

    Output: YCbCr image, range from 0 to 1, channel number is 3
    """
    image = np.array(Image.open(path))
    if len(image.shape) == 2:
        image = color.gray2rgb(image)
    return color.rgb2ycbcr(image)


def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """
    Load RGB format Image and convert to YCbCr format.

    Input & Output: YCbCr image, range from 0 to 1, channel number is 3
    """
    return color.rgb2ycbcr(image)


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Load YCbCr format Image and convert to RGB format.

    Input & Output: YCbCr image, range from 0 to 1, channel number is 3
    """
    return color.ycbcr2rgb(image)


def glance(
        image: Union[np.ndarray, torch.Tensor, Image.Image, list, tuple], 
        annotations: Union[list, tuple] = (),
        clip: bool = False,
        title: Union[str, list] = "",
        hide_axis: bool = True,
        shape: tuple = (1,1)):
    """
    Display a PyTorch tensor or NumPy array as an image.

    Can input:
    * tensor: 4 dims, batch number should only be 1. channel can be 3 or 1.
    * tensor: 3 dims, channel can be 3 or 1.
    * tensor: 2 dims.
    * ndarray: 2 dims, channel can be 3 or 1.
    * ndarray: 2 dims.
    * Image: auto convert to numpy.
    """
    # transfrom batch tensor to list
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 4 and image.shape[0] > 1:
            image = [i.unsqueeze(0) for i in image]
    # transform torch.tensor to np.array
    if isinstance(image,list) or isinstance(image, tuple):
        if shape[0]*shape[1] != len(image):
            shape = (1,len(image)) 
        image = [to_numpy(i,clip) for i in image]
    else:
        image = [to_numpy(image,clip)]

    # show image with PIL.Image
    (H,W) = shape
    for k in range(H*W):
        plt.subplot(H,W,k+1)
        plt.imshow((image[k]*255).astype(np.uint8), 
                cmap='gray' if image[k].ndim == 2 else 'viridis')
        if len(annotations)>0:
            if hasattr(annotations[k-1],'boxes'):
                for anno in annotations[k-1]['boxes']:
                    x_min, y_min, x_max, y_max = [anno[i] for i in range(4)]
                    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                linewidth=1, edgecolor='r', facecolor='none')
                    plt.gca().add_patch(rect)
        if title != "": plt.title(title[k] if isinstance(title,list) else title)
        if hide_axis: plt.axis('off')
    plt.show()


def save_array_to_img(
        image: Union[np.ndarray, torch.Tensor, Image.Image], 
        filename: Union[str, Path], 
        clip: bool = False
    ) -> None:
    to_image(image,clip).save(filename)


def save_array_to_mat(
        image: Union[np.ndarray, torch.Tensor, Image.Image], 
        base_filename: str = 'glance', 
        clip: bool = False,
        log: bool = False
    ) -> None:
    """
    Save a NumPy array or PyTorch tensor as MATLAB .mat files.
    """
    # transform torch.tensor to np.array
    image_array = to_numpy(image,clip)

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

