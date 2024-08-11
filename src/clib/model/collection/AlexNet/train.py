import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .utils import BaseTrainer
from .config import TrainOptions
from .model import AlexNet as Model

class Trainer(BaseTrainer):
    def __init__(self, opts, **kwargs):
        super().__init__(opts,TrainOptions,**kwargs)
    
    def default_model(self):
        return Model(num_classes=self.opts.num_classes)
    
    def default_criterion(self):
        return nn.CrossEntropyLoss()
    
    def default_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=self.opts.lr, momentum=self.opts.momentum)
    
    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            lambda x: x.repeat(3, 1, 1),  # 自定义转换，将单通道复制三次
            transforms.Normalize((0.1307,), (0.3081,))  # 标准化
        ])
    
    def default_train_loader(self):
        train_dataset = datasets.MNIST(root=self.opts.TorchVisionPath, train=True, download=True, transform=self.transform)
        return DataLoader(dataset=train_dataset, batch_size=self.opts.batch_size, shuffle=True)
    
    def default_test_loader(self):
        test_dataset = datasets.MNIST(root=self.opts.TorchVisionPath, train=False, download=True, transform=self.transform)
        return DataLoader(dataset=test_dataset, batch_size=self.opts.batch_size, shuffle=False)

    
def train(opts = {}, **kwargs):
    trainer = Trainer(opts, **kwargs)
    trainer.train()
