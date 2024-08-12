import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from .utils import BaseTrainer
from .config import TrainOptions
from .model import LeNet as Model

class Trainer(BaseTrainer):
    def __init__(self, opts, **kwargs):
        super().__init__(opts,TrainOptions,**kwargs)
    
    def default_model(self):
        return Model(use_max_pool=self.opts.use_max_pool,use_relu=self.opts.use_relu)
    
    def default_criterion(self):
        return nn.CrossEntropyLoss()
    
    def default_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.opts.lr)
    
    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def default_train_loader(self):
        dataset = datasets.MNIST(root=self.opts.TorchVisionPath, train=True, download=True, transform=self.transform)
        val_size = int(self.opts.val * len(dataset)) 
        train_size = len(dataset) - val_size
        train_dataset, _ = random_split(dataset, [train_size, val_size])
        return DataLoader(dataset=train_dataset, batch_size=self.opts.batch_size, shuffle=True)
    
    def default_val_loader(self):
        dataset = datasets.MNIST(root=self.opts.TorchVisionPath, train=True, download=True, transform=self.transform)
        val_size = int(self.opts.val * len(dataset))
        train_size = len(dataset) - val_size
        _, val_size = random_split(dataset, [train_size, val_size])
        return DataLoader(dataset=val_size, batch_size=self.opts.batch_size, shuffle=False)
    
    def default_test_loader(self):
        test_dataset = datasets.MNIST(root=self.opts.TorchVisionPath, train=False, download=True, transform=self.transform)
        return DataLoader(dataset=test_dataset, batch_size=self.opts.batch_size, shuffle=False)

    
def train(opts = {}, **kwargs):
    trainer = Trainer(opts, **kwargs)
    trainer.train()