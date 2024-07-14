from skimage import color
from PIL import Image
import torch
import numpy as np

L1_NORM = lambda b: torch.sum(torch.abs(b))

def weightedFusion(cr1, cr2, cb1, cb2):
    """
        Perform the weighted fusing for Cb and Cr channel (paper equation 6)

        Arg:    cr1     (torch.Tensor)  - The Cr slice of 1st image
                cr2     (torch.Tensor)  - The Cr slice of 2nd image
                cb1     (torch.Tensor)  - The Cb slice of 1st image
                cb2     (torch.Tensor)  - The Cb slice of 2nd image
        Ret:    The fused Cr slice and Cb slice
    """
    # Fuse Cr channel
    cr_up = (cr1 * L1_NORM(cr1 - 127.5) + cr2 * L1_NORM(cr2 - 127.5))
    cr_down = L1_NORM(cr1 - 127.5) + L1_NORM(cr2 - 127.5)
    cr_fuse = cr_up / cr_down

    # Fuse Cb channel
    cb_up = (cb1 * L1_NORM(cb1 - 127.5) + cb2 * L1_NORM(cb2 - 127.5))
    cb_down = L1_NORM(cb1 - 127.5) + L1_NORM(cb2 - 127.5)
    cb_fuse = cb_up / cb_down

    return cr_fuse, cb_fuse

def load_ycbcr_from_path(path: str) -> Image.Image:
    """
    Load an image from the given path and convert it to YCbCr format.

    Parameters:
    path (str): The path to the image file.

    Returns:
    Image: The loaded image in YCbCr format.

    Raises:
    ValueError: If the image mode is not supported (only RGB and grayscale images are supported).
    """
    image = Image.open(path)
    if len(image.size) == 2:  # 灰度图像转换灰度图像到RGB
        image = color.gray2rgb(image)
    elif image.mode != 'RGB':  # 如果不是RGB也不是灰度，则抛出错误
        raise ValueError("Unsupported image mode. Only RGB and grayscale images are supported.")
    image = color.rgb2ycbcr(image)
    return Image.fromarray(image.astype(np.uint8))

def change_ycbcr_to_rgb(tensor):
    im = tensor[0,:,:,:].permute(1, 2, 0).numpy().astype(np.uint8)
    image_rgb = color.ycbcr2rgb(im)
    image_rgb[:,:,0:1] -= np.min(image_rgb[:,:,0:1])
    image_rgb[:,:,0:1] /= np.max(image_rgb[:,:,0:1])
    image_rgb[:,:,0:1] *= 255
    image_rgb[:,:,1:2] -= np.min(image_rgb[:,:,1:2])
    image_rgb[:,:,1:2] /= np.max(image_rgb[:,:,1:2])
    image_rgb[:,:,1:2] *= 255
    image_rgb[:,:,2:3] -= np.min(image_rgb[:,:,2:3])
    image_rgb[:,:,2:3] /= np.max(image_rgb[:,:,2:3])
    image_rgb[:,:,2:3] *= 255
    
    # print(image_rgb,'.....')
    # import matplotlib.pyplot as plt
    # plt.subplot(1,4,1)
    # plt.imshow(im[:,:,0],cmap='gray')
    # plt.title("Y")
    # plt.subplot(1,4,2)
    # plt.imshow(im[:,:,1],cmap='gray')
    # plt.title("Cb")
    # plt.subplot(1,4,3)
    # plt.imshow(im[:,:,2],cmap='gray')
    # plt.title("Cr")
    # plt.subplot(1,4,4)
    # plt.imshow((np.abs(image_rgb)*255*255).astype(np.uint8))
    # plt.title("RGB")
    # plt.show()

    return image_rgb

def test():
    path = '/Volumes/Charles/DateSets/Fusion/Toy/vis/00449.png'
    im = np.array(load_ycbcr_from_path(path))
    image_rgb = color.ycbcr2rgb(im)

    import matplotlib.pyplot as plt
    plt.subplot(1,4,1)
    plt.imshow(im[:,:,0],cmap='gray')
    plt.subplot(1,4,2)
    plt.imshow(im[:,:,1],cmap='gray')
    plt.subplot(1,4,3)
    plt.imshow(im[:,:,2],cmap='gray')
    plt.subplot(1,4,4)
    plt.imshow((image_rgb*255).astype(np.uint8))
    plt.show()
    print(image_rgb*255)
# test()
