# import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from clib.utils import path_to_rgb,path_to_gray,glance
from clib.metrics.fusion import ir,vis

a = path_to_rgb("/Volumes/Charles/DateSets/Fusion/Toy/fused/DenseFuse/00449.png")
# b = path_to_rgb("/Volumes/Charles/DateSets/Fusion/RoadScene/vis_rgb/FLIR_00233.png")
# c = path_to_gray("/Volumes/Charles/DateSets/Fusion/RoadScene/ir/FLIR_00006.png",3)
# d = path_to_gray("/Volumes/Charles/DateSets/Fusion/RoadScene/vis_rgb/FLIR_00233.png",3)
# [a,b,c,d] = [np.array(i) for i in [a,b,c,d]]


plt.imshow(np.array(a))
plt.show()


# from clib.model.fusion.DenseFuse.utils import load_rgb_from_path
# path = '/Users/kimshan/Downloads/TNO_Image_Fusion_Dataset/DHV_images/Fire_sequence/part_2/dhv/DHVheli0.bmp'
# img = load_rgb_from_path(path)
# print(img)