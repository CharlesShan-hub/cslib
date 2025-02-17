from typing import Union
import numpy as np
import torch
from PIL import Image
from skimage import color

def rgb_to_ycbcr(
        image: Union[np.ndarray, torch.Tensor, Image.Image]
    ) -> Union[np.ndarray, torch.Tensor, Image.Image]:
    if isinstance(image, np.ndarray):
        return color.rgb2ycbcr(image)
    elif isinstance(image, Image.Image):
        return image.convert('YCbCr')
    else:
        assert image.ndim in [3, 4], "Input must be a 3-channel RGB image or a batch of 3-channel RGB images."
        assert image.shape[-3] == 3, "Input must have 3 channels for RGB."
        rgb_to_ycbcr_matrix = torch.tensor([
            [65.481, 128.553, 24.966], [-37.797, -74.203, 112.0], [112.0, -93.786, -18.214]
        ], dtype=image.dtype, device=image.device)
        offset = torch.tensor([16, 128, 128], dtype=image.dtype, device=image.device)
        if image.ndim == 4:
            return torch.einsum('nchw,cd->ndhw', image, rgb_to_ycbcr_matrix.T) + offset.view(1, 3, 1, 1)
        else:
            return torch.einsum('chw,cd->dhw', image, rgb_to_ycbcr_matrix.T) + offset.view(3, 1, 1)


def ycbcr_to_rgb(
        image: Union[np.ndarray, torch.Tensor, Image.Image]
    ) -> Union[np.ndarray, torch.Tensor, Image.Image]:
    if isinstance(image, np.ndarray):
        return color.ycbcr2rgb(image)
    elif isinstance(image, Image.Image):
        return image.convert('RGB')
    else:
        assert image.ndim in [3, 4], "Input must be a 3-channel RGB image or a batch of 3-channel RGB images."
        assert image.shape[-3] == 3, "Input must have 3 channels for RGB."
        ycbcr_to_rgb_matrix = torch.tensor([
            [65.481, 128.553, 24.966], [-37.797, -74.203, 112.0], [112.0, -93.786, -18.214]
        ], dtype=image.dtype, device=image.device).inverse()
        offset = torch.tensor([-16, -128, -128], dtype=image.dtype, device=image.device)
        if image.ndim == 4:
            return (torch.einsum('nchw,cd->ndhw', image + offset.view(1, 3, 1, 1), ycbcr_to_rgb_matrix.T))
        else:
            return (torch.einsum('chw,cd->dhw', image + offset.view(3, 1, 1), ycbcr_to_rgb_matrix.T))


# result 的形状为 (N, C, H, W)

# from clib.metrics.fusion import ir,vis
from clib.utils import path_to_rgb, to_tensor, glance, to_numpy
ir = path_to_rgb('/Users/kimshan/Public/project/paper/ir_250423.jpg')
vis = path_to_rgb('/Users/kimshan/Public/project/paper/vis_250423.jpg')
    

glance([
    rgb_to_ycbcr(vis), 
    rgb_to_ycbcr(to_tensor(vis)),
    ycbcr_to_rgb(rgb_to_ycbcr(vis)), 
    ycbcr_to_rgb(rgb_to_ycbcr(to_tensor(vis))),
    # to_numpy(rgb_to_ycbcr(to_tensor(vis))),
    # rgb_to_ycbcr(to_tensor(vis).unsqueeze(0)),
])

# breakpoint()

# # 单张图像
# image_single = torch.rand(3, 256, 256)
# ycbcr_single = rgb_to_ycbcr_f(image_single)
# print("Single image shape:", ycbcr_single.shape)

# # 批量图像
# image_batch = torch.rand(4, 3, 256, 256)
# ycbcr_batch = rgb_to_ycbcr_f(image_batch)
# print("Batch image shape:", ycbcr_batch.shape)