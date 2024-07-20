import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .color import *
from .config import Options

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

