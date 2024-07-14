import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from clib.utils import glance
from clib.metrics.fusion import ir,vis


def to_rgb(image):
    c,_,_ = image.shape
    if c == 1:
        image = image.repeat(3, 1, 1)
    return image

transform_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ]).float()

def rgb_to_ycbcr(tensor):
    
    # 应用转换矩阵
    ycbcr_tensor = torch.einsum('kc,cij->kij', transform_matrix,tensor)

    # glance(ycbcr_tensor[0,:,:])
    # glance(ycbcr_tensor[1,:,:])
    # glance(ycbcr_tensor[2,:,:])
    glance(ycbcr_tensor)

    return ycbcr_tensor

def ycbcr_to_rgb(tensor):

    print(transform_matrix.inverse())

    # 应用转换矩阵
    rgb_tensor = torch.einsum('kc,cij->kij', torch.pinverse(transform_matrix),tensor)
    # glance(rgb_tensor[0,:,:])
    # glance(rgb_tensor[1,:,:])
    # glance(rgb_tensor[2,:,:])
    glance(rgb_tensor)

    return rgb_tensor

    
trans = transforms.Compose([
                # transforms.ToTensor(),
                # transforms.Resize((opts.W, opts.H)),
                transforms.Lambda(to_rgb),
                transforms.Lambda(rgb_to_ycbcr),
                transforms.Lambda(ycbcr_to_rgb),
                # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])

trans(vis[0,:,:,:])

# def tttt():
#     transform_matrix = torch.tensor([
#         [0.299, 0.587, 0.114],
#         [-0.168736, -0.331264, 0.5],
#         [0.5, -0.418688, -0.081312]
#     ]).double()

#     # print(torch.matmul(transform_matrix,transform_matrix.inverse()))
    
#     for i in range(256):
#         # c = torch.tensor([[i],[i],[i]]).double()
#         c = torch.tensor([[i],[i],[i]]).double()
#         color = torch.einsum('ij,jk->ik', transform_matrix,c)
#         color = torch.einsum('ij,jk->ik', transform_matrix.inverse(),color)
#         print(i,color)

# tttt()