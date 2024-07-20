from PIL import Image
from skimage import color
import numpy as np

__all__ = [
    'path_to_gray',
    'path_to_rgb',
]

def path_to_gray(path: str) -> Image.Image:
    """
    Load an image from the given path and convert it to Gray format.
    
    Output: Gary image, range from 0 to 255, channel number is 2
    """
    image = np.array(Image.open(path))
    if len(image.shape) == 3:
        image = color.rgb2gray(image)*255.0
    return Image.fromarray(image.astype(np.uint8))


def path_to_rgb(path: str) -> Image.Image:
    """
    Load an image from the given path and convert it to RGB format.

    Output: RGB image, range from 0 to 255, channel number is 3
    """
    image = np.array(Image.open(path))
    if len(image.shape) == 2:
        image = color.gray2rgb(image)
    return Image.fromarray(image.astype(np.uint8))