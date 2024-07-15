from PIL import Image
from skimage import color
import numpy as np

def load_gray_from_path(path: str) -> Image.Image:
    """
    Load an image from the given path and convert it to Gray format.

    Parameters:
    path (str): The path to the image file.

    Returns:
    Image: The loaded image in Gray format.

    Raises:
    ValueError: If the image mode is not supported (only RGB and grayscale images are supported).
    """
    image = Image.open(path)
    if len(image.size) == 3:  # 灰度图像转换灰度图像到RGB
        image = color.rgb2gray(image)
    # elif image.mode != 'gray':  # 如果不是RGB也不是灰度，则抛出错误
    #     print(image.mode)
    #     raise ValueError("Unsupported image mode. Only RGB and grayscale images are supported.")
        return Image.fromarray(image.astype(np.uint8))
    return image