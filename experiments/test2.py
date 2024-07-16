import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from clib.utils import glance
from clib.metrics.fusion import ir,vis

from clib.model.fusion.DenseFuse.utils import load_rgb_from_path

path = '/Users/kimshan/Downloads/TNO_Image_Fusion_Dataset/DHV_images/Fire_sequence/part_2/dhv/DHVheli0.bmp'
img = load_rgb_from_path(path)
print(img)