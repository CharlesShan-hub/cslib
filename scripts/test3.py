import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

from clib.utils import *
from clib.metrics.fusion import ir,vis

# tensor_to_image(vis).show()

# plt.imshow(tensor_to_numpy(vis))
# plt.show()

# resize to imagenet size 
# transform = Compose([Resize((224, 224)), ToTensor()])
transform = Compose([Resize((224, 224))])
x = transform(vis)
# x = x.unsqueeze(0) # add batch dim
# print(x.shape) (1,1,224,224)

patch_size = 16 # 16 pixels
pathes = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
breakpoint()