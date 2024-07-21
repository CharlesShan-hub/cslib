# import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from skimage import color as c

from clib.utils import *
from clib.metrics.fusion import ir,vis
import clib.transforms as ctransforms

# path = "/Volumes/Charles/DateSets/Fusion/Toy/ir/105.png"
# image = np.array(Image.open(path))
# image = c.gray2rgb(image)
# plt.imshow(np.array(image))
# plt.show()

# image = c.rgb2ycbcr(image)
# plt.imshow(np.array(image))
# plt.show()

image = path_to_rgb("/Volumes/Charles/DateSets/Fusion/Toy/ir/105.png")
path = "/Volumes/Charles/DateSets/SR/Set5/X2/GT/woman.png"
# image1 = path_to_ycbcr(path)
# image2 = ycbcr_to_rgb(image1)
image1 = transforms.ToTensor()(path_to_rgb(path))
image2 = ctransforms.rgb_to_ycbcr(image1)
image3 = ctransforms.ycbcr_to_rgb(image2)

plt.subplot(1,2,1)
plt.imshow(transforms.ToPILImage()(image1))
plt.subplot(1,2,2)
plt.imshow(transforms.ToPILImage()(image3))
plt.show()

# a = np.arange(0,255)
# print(a)