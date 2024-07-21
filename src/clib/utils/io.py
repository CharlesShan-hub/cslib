import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

__all__ = [
    'glance',
    'save_tensor_to_img'
]

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

