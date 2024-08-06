from collections import UserDict
import torch
from pathlib import Path

class ConfigDict(UserDict):
    def __init__(self,_dict):
        super().__init__(_dict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root_path_list = [
            '/root/autodl-fs/DateSets',
            '/Volumes/Charles/DateSets',
            '/Users/kimshan/resources/DataSets'
        ]
        self.DataRootPath = None
        for path in self.data_root_path_list:
            if Path(path).exists() and Path(path).is_dir():
                self.DataRootPath = path
                break
        assert(self.DataRootPath is not None)
        self.TorchVisionPath = Path(self.DataRootPath, "torchvision").__str__()
        self.FusionPath = Path(self.DataRootPath, "Fusion").__str__()
        self.SRPath = Path(self.DataRootPath, "SR").__str__()
        self.ModelBasePath = Path(self.DataRootPath, "Model").__str__()

    def __setitem__(self, key, value):
        check_list = [
            'device','DataRootPath','TorchVisionPath',
            'FusionPath','SRPath','ModelBasePath',
        ]
        for item in check_list:
            if item not in value:
                value[item] = getattr(self,item)
        super().__setitem__(key, value)
        

opts = ConfigDict({})
opts['LeNet'] = {
    'models_path': Path(opts.ModelBasePath,'LeNet','MNIST','new'),
    'pre_trained': Path(opts.ModelBasePath,'LeNet','MNIST','9839_m1_d003','model.pth'),
}
opts['AlexNet'] = {
    'models_path': Path(opts.ModelBasePath,'AlexNet','MNIST','new'),
    'pre_trained': Path(opts.ModelBasePath,'AlexNet','MNIST','9839_m1_d003','model.pth'),
}

print(opts['AlexNet']['device']) # cpu
print(opts['LeNet']['pre_trained']) # /Users/kimshan/resources/DataSets/Model/LeNet/MNIST/9839_m1_d003/model.pth