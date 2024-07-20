# import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from skimage import color as c

from clib.utils import *
from clib.metrics.fusion import ir,vis

# path = "/Volumes/Charles/DateSets/Fusion/Toy/ir/105.png"
# image = np.array(Image.open(path))
# image = c.gray2rgb(image)
# plt.imshow(np.array(image))
# plt.show()

# image = c.rgb2ycbcr(image)
# plt.imshow(np.array(image))
# plt.show()

image = path_to_ycbcr("/Volumes/Charles/DateSets/Fusion/Toy/ir/105.png")
image = ycbcr_to_rgb(image)
plt.imshow(np.array(image))
plt.show()
