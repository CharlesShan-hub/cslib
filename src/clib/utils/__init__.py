from typing import Dict, Any
from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color

def path_load_test():
    print("Hello World!")

def glance(tensor):
    # 处理色彩
    if len(tensor.shape)==3:
        image = tensor.permute(1, 2, 0)
        if image.shape[-1] == 1:
            image = image[:,:,0].detach().cpu().numpy()
        else:
            image = image.detach().cpu().numpy()
    else:
        image = tensor.detach().cpu().numpy()

    # 使用 PIL 显示图片
    if len(image.shape) == 2:
        plt.imshow((image * 255).astype(np.uint8),cmap='gray')
    else:
        plt.imshow((image * 255).astype(np.uint8))
    plt.show()

def save_tensor_to_img(tensor,filename):
    # 处理色彩
    if len(tensor.shape)==3:
        image = tensor.permute(1, 2, 0)
        if image.shape[-1] == 1:
            image = image[:,:,0].detach().cpu().numpy()
        else:
            image = image.detach().cpu().numpy()
    else:
        image = tensor.detach().cpu().numpy()
    
    # 保存图片
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.save(filename)

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

def load_rgb_from_path(path: str) -> Image.Image:
    """
    Load an image from the given path and convert it to RGB format.

    Parameters:
    path (str): The path to the image file.

    Returns:
    Image: The loaded image in RGB format.

    Raises:
    ValueError: If the image mode is not supported (only RGB and grayscale images are supported).
    """
    image = Image.open(path)
    if len(image.size) == 2:  # 灰度图像转换灰度图像到RGB
        image = color.gray2rgb(image)
    elif image.mode != 'RGB':  # 如果不是RGB也不是灰度，则抛出错误
        raise ValueError("Unsupported image mode. Only RGB and grayscale images are supported.")
    return Image.fromarray(image.astype(np.uint8))

class Options(object):
    """
    Options class for DeepFuse.

    This class provides a way to define and update command line arguments.

    Attributes:
        opts (argparse.Namespace): A namespace containing the parsed command line arguments.

    Methods:
        INFO(string): Print an information message.
        presentParameters(args_dict): Print the parameters setting line by line.
        update(parmas): Update the command line arguments.
    """

    def __init__(self, name: str = 'Undefined', params: Dict[str, Any] = {}):
        self.opts = Namespace()
        self.name = name
        if len(params) > 0:
            self.update(params)

    def INFO(self, string: str):
        """
        Print an information message.

        Args:
            string (str): The message to be printed.
        """
        print("[ %s ] %s" % (self.name,string))

    def presentParameters(self, args_dict: Dict[str, Any]):
        """
        Print the parameters setting line by line.

        Args:
            args_dict (Dict[str, Any]): A dictionary containing the command line arguments.
        """
        self.INFO("========== Parameters ==========")
        for key in sorted(args_dict.keys()):
            self.INFO("{:>15} : {}".format(key, args_dict[key]))
        self.INFO("===============================")

    def update(self, parmas: Dict[str, Any] = {}):
        """
        Update the command line arguments.

        Args:
            parmas (Dict[str, Any]): A dictionary containing the updated command line arguments.
        """
        for (key, value) in parmas.items():
            setattr(self.opts, key, value)
    
    def parse(self, parmas: Dict[str, Any] = {}, present: bool = True):
        """
        Update the command line arguments. Can also present into command line.
        
        Args:
            parmas (Dict[str, Any]): A dictionary containing the updated command line arguments.
            present (bool) = True: Present into command line.
        """
        self.update(parmas)
        if present:
            self.presentParameters(vars(self.opts))
        return self.opts
