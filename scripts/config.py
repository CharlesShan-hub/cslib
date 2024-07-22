import os
import torch
from pathlib import Path
# devise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset base path
data_base_path_list = [
    '/root/autodl-fs/DateSets',
    '/Volumes/Charles/DateSets',
    '/Users/kimshan/resources/DataSets'
]

# 遍历路径列表，找到第一个可用的路径
data_base_path = None
for path in data_base_path_list:
    if os.path.exists(path) and os.path.isdir(path):
        data_base_path = path
        break
assert(data_base_path is not None)

TorchVisionPath = Path(data_base_path, "torchvision").__str__()
FusionPath = Path(data_base_path, "Fusion").__str__()
SRPath = Path(data_base_path, "SR").__str__()
ModelBasePath = Path(data_base_path, "Model").__str__()

# network config
opts = {
    'DeepFuse':{
        'device': device,
        'pre_trained': Path(ModelBasePath,'DeepFuse','DeepFuse_model.pth'), # pytorch复现版的, https://github.com/SunnerLi/DeepFuse.pytorch
    },
    'DenseFuse_gray':{
        'device': device,
        'pre_trained': Path(ModelBasePath,'DenseFuse','densefuse_gray.model'), # https://github.com/hli1221/densefuse-pytorch
        'color': 'gray',
    },
    'DenseFuse_rgb':{
        'device': device,
        # 'pre_trained': Path(ModelBasePath,'DenseFuse','densefuse_rgb.model'), # https://github.com/hli1221/densefuse-pytorch
        'pre_trained': Path(ModelBasePath,'DenseFuse','densefuse_gray.model'),
        'color': 'color', # 看了论文,rgb 的 densefuse 是三个通道分别进行单通道融合然后拼起来!
    },
    'CDDFuse':{
        'device': device,
        'pre_trained': Path(ModelBasePath,'CDDFuse','CDDFuse_IVF.pth'), # https://github.com/Zhaozixiang1228/MMIF-CDDFuse
    },
    'AUIF':{
        'device': device,
        'pre_trained': [ # https://github.com/Zhaozixiang1228/IVIF-AUIF-Net
            Path(ModelBasePath,'AUIF','TCSVT_Encoder_Base.model'),
            Path(ModelBasePath,'AUIF','TCSVT_Encoder_Detail.model'),
            Path(ModelBasePath,'AUIF','TCSVT_Decoder.model'),
        ],
    },
    'SRCNN2':{
        'device': device,
        'pre_trained': Path(ModelBasePath,'SRCNN','srcnn_x2-T91-7d6e0623.pth.tar') # https://github.com/Lornatang/SRCNN-PyTorch
    },
    'SRCNN3':{
        'device': device,
        'pre_trained': Path(ModelBasePath,'SRCNN','srcnn_x3-T91-919a959c.pth.tar') # https://github.com/Lornatang/SRCNN-PyTorch
    },
    'SRCNN4':{
        'device': device,
        'pre_trained': Path(ModelBasePath,'SRCNN','srcnn_x4-T91-7c460643.pth.tar') # https://github.com/Lornatang/SRCNN-PyTorch
    }
}