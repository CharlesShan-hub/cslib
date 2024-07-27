from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from scipy.io import savemat

__all__ = [
    'glance',
    'save_tensor_to_img',
    'save_tensor_to_mat',
    'save_array_to_mat',
]

def _tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    if len(image.shape) == 4:
        if image.shape[0] != 1:
            raise ValueError("Batch number should be 1.")
        image = image[0,:,:,:]
    if len(image.shape) == 3:
        image = image.permute(1, 2, 0)
        if image.shape[-1] == 1:
            image = image[:,:,0]
    if len(image.shape) != 2:
        raise ValueError("Image should be an image.")
    return image.detach().cpu().numpy() * 255.0

def glance(image: Union[np.ndarray, torch.Tensor], clip: bool = False):
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
    if isinstance(image, torch.Tensor):
        image = _tensor_to_numpy(image)
    elif isinstance(image, np.ndarray):
        if image.ndim != 3 and image.ndim != 2:
            raise ValueError("Image should be an image.")
    if clip:
        image = image.clip(0.0, 255.0)

    # show image with PIL.Image
    assert isinstance(image, np.ndarray)
    plt.imshow(image.astype(np.uint8), 
               cmap='gray' if image.ndim == 2 else 'rgb')
    plt.show()

def save_tensor_to_img(tensor,filename):
    # 确保张量在CPU上，并且没有梯度
    tensor = tensor.detach().cpu()
    
    # 处理色彩
    if len(tensor.shape) == 3:  # C x H x W
        # 如果是灰度图像（C=1），则去掉通道维度
        if tensor.shape[0] == 1:
            image = tensor.squeeze(0)
        else:  # 如果是彩色图像（C=3），则转置张量
            image = tensor.permute(1, 2, 0)
    elif len(tensor.shape) == 2:  # H x W
        image = tensor
    else:
        raise ValueError("Unsupported tensor shape: {}".format(tensor.shape))
    
    # 将张量值从 [0, 1] 转换到 [0, 255] 并转换为 np.uint8
    image = (image * 255).numpy().astype(np.uint8)
    
    # 保存图片
    image = Image.fromarray(image)
    image.save(filename)

def save_array_to_mat(image_array, base_filename='glance'):
    """
    将图像数组保存为 .mat 文件，每个颜色通道一个文件。
    
    参数:
    image_array (numpy.ndarray): 输入的图像数组，形状为 (高, 宽) 对于灰度图像，形状为 (高, 宽, 3) 对于 RGB 图像。
    base_filename (str): 保存文件的基础名称，不包括扩展名。
    """
    # 确保图像数组是 NumPy 数组
    if not isinstance(image_array, np.ndarray):
        raise TypeError("图像数组必须是 NumPy 数组。")
    
    # 获取图像的维度
    if image_array.ndim == 2:
        # 灰度图像
        savemat(f"{base_filename}_gray.mat", {'gray': image_array})
        print(f"灰度图像已保存为 {base_filename}_gray.mat")
    
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        # RGB 图像
        red_channel = image_array[:, :, 0]
        green_channel = image_array[:, :, 1]
        blue_channel = image_array[:, :, 2]
        
        savemat(f"{base_filename}_red.mat", {'red': red_channel})
        savemat(f"{base_filename}_green.mat", {'green': green_channel})
        savemat(f"{base_filename}_blue.mat", {'blue': blue_channel})
        
        print(f"RGB 图像的通道已保存为 {base_filename}_red.mat, {base_filename}_green.mat 和 {base_filename}_blue.mat")
    
    else:
        raise ValueError("图像数组必须是 2D (灰度图像) 或 3D (RGB 图像) 数组。")

def save_tensor_to_mat(tensor, base_filename='glance'):
    save_array_to_mat(_tensor_to_numpy(tensor), base_filename)

