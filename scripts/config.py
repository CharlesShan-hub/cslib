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
    'DIDFuse':{
        'device': device,
        'pre_trained': [ # https://github.com/Zhaozixiang1228/IVIF-DIDFuse
            Path(ModelBasePath,'DIDFuse','Encoder_weight_IJCAI.pkl'),
            Path(ModelBasePath,'DIDFuse','Decoder_weight_IJCAI.pkl'),
        ],
    },
    'SwinFuse':{
        'device': device,
        'pre_trained': Path(ModelBasePath,'SwinFuse','Final_epoch_50_Mon_Feb_14_17_37_05_2022_1e3.model'), # https://github.com/Zhishe-Wang/SwinFuse
    },
    'MFEIF':{
        'device': device,
        'pre_trained': Path(ModelBasePath,'MFEIF','default.pth'), # https://github.com/JinyuanLiu-CV/MFEIF
    },
    'Res2Fusion':{
        'device': device,
        # 'pre_trained': Path(ModelBasePath,'Res2Fusion','Final_epoch_4_1e0.model'), # https://github.com/Zhishe-Wang/Res2Fusion
        # 'pre_trained': Path(ModelBasePath,'Res2Fusion','Final_epoch_4_1e1.model'),
        # 'pre_trained': Path(ModelBasePath,'Res2Fusion','Final_epoch_4_1e2.model'),
        'pre_trained': Path(ModelBasePath,'Res2Fusion','Final_epoch_4_1e3.model'),
    },
    'UNFusion_l1_mean':{
        'device': device,
        'pre_trained': Path(ModelBasePath,'UNFusion','UNFusion.model'),
        'fusion_type': 'l1_mean',#['l1_mean', 'l2_mean', 'linf']
    },
    'UNFusion_l2_mean':{
        'device': device,
        'pre_trained': Path(ModelBasePath,'UNFusion','UNFusion.model'),
        'fusion_type': 'l2_mean',#['l1_mean', 'l2_mean', 'linf']
    },
    'UNFusion_linf':{
        'device': device,
        'pre_trained': Path(ModelBasePath,'UNFusion','UNFusion.model'),
        'fusion_type': 'linf',#['l1_mean', 'l2_mean', 'linf']
    },
    'CoCoNet':{
        'device': device,
        'pre_trained': Path(ModelBasePath,'CoCoNet','latest.pth'), # https://github.com/runjia0124/CoCoNet
    },
    'DDFM':{
        'device': device,
        'pre_trained': Path(ModelBasePath,'DDFM','256x256_diffusion_uncond.pt'), # https://github.com/openai/guided-diffusion
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
    },
    'GAN':{
        'dataset_path': Path(TorchVisionPath),
        'images_path': Path(ModelBasePath,'GAN','images'),
        'models_path': Path(ModelBasePath,'GAN','epoch200'),
        'pre_trained': Path(ModelBasePath,'GAN','epoch200','generator.pth')
    }
}