# from typing import Union
# from pathlib import Path
# from PIL import Image
# from skimage import color
# import numpy as np

# __all__ = [
#     'path_to_gray',
#     'path_to_rgb',
#     'path_to_ycbcr',
#     'ycbcr_to_rgb',
# ]

# def path_to_gray(path: Union[str, Path]) -> np.ndarray:
#     """
#     Load an image from the given path and convert it to Gray format.
    
#     Output: Gary image, range from 0 to 1, channel number is 1
#     """
#     image = np.array(Image.open(path))
#     if len(image.shape) == 3:
#         return color.rgb2gray(image)
#     elif len(image.shape) != 2:
#         raise ValueError(f"Wrong shape of image: {image.shape}")
#     return image/255.0


# def path_to_rgb(path: Union[str, Path]) -> np.ndarray:
#     """
#     Load an image from the given path and convert it to RGB format.

#     Output: RGB image, range from 0 to 1, channel number is 3
#     """
#     image = np.array(Image.open(path))
#     if len(image.shape) == 2:
#         return color.gray2rgb(image)/255.0
#     elif len(image.shape) != 3:
#         raise ValueError(f"Wrong shape of image: {image.shape}")
#     return image/255.0


# def path_to_ycbcr(path: Union[str, Path]) -> Image.Image:
#     """
#     Load an image from the given path and convert it to YCbCr format.
#     """
#     image = np.array(Image.open(path))
#     if len(image.shape) == 2:
#         image = color.gray2rgb(image)
#     image = color.rgb2ycbcr(image)
#     return Image.fromarray(image.astype(np.uint8), mode='YCbCr')


# # def path_to_gray(path: Union[str, Path]) -> Image.Image:
# #     """
# #     Load an image from the given path and convert it to Gray format.
    
# #     Output: Gary image, range from 0 to 255, channel number is 1
# #     """
# #     image = np.array(Image.open(path))
# #     if len(image.shape) == 3:
# #         image = color.rgb2gray(image)*255.0
# #     return Image.fromarray(image.astype(np.uint8), mode="L")


# # def path_to_rgb(path: Union[str, Path]) -> Image.Image:
# #     """
# #     Load an image from the given path and convert it to RGB format.

# #     Output: RGB image, range from 0 to 255, channel number is 3
# #     """
# #     image = np.array(Image.open(path))
# #     if len(image.shape) == 2:
# #         image = color.gray2rgb(image)
# #     return Image.fromarray(image.astype(np.uint8), mode="RGB")


# # def path_to_ycbcr(path: Union[str, Path]) -> Image.Image:
# #     """
# #     Load an image from the given path and convert it to YCbCr format.
# #     """
# #     image = np.array(Image.open(path))
# #     if len(image.shape) == 2:
# #         image = color.gray2rgb(image)
# #     image = color.rgb2ycbcr(image)
# #     return Image.fromarray(image.astype(np.uint8), mode='YCbCr')


# # def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
# #     """
# #     Load YCbCr format Image and convert to RGB format.
# #     """
# #     image_np = np.array(image)*1.0
# #     image_rgb = color.ycbcr2rgb(image_np)*255.0
# #     return Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")


# # def rgb_to_ycbcr(image: Image.Image) -> Image.Image:
# #     """
# #     Load RGB format Image and convert to YCbCr format.
# #     """
# #     image_np = np.array(image)*1.0
# #     image_rgb = color.rgb2ycbcr(image_np)*255.0
# #     return Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")