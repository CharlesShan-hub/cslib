import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .utils import BaseTrainer
from .config import TrainOptions
from .model import LeNet as Model

class Trainer(BaseTrainer):
    def __init__(self, opts, **kwargs):
        super().__init__(opts,TrainOptions,**kwargs)
    
    def default_model(self):
        return Model(use_max_pool=False,use_relu=False)
    
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
        train_dataset = datasets.MNIST(root=self.opts.TorchVisionPath, train=True, download=True, transform=self.transform)
        return DataLoader(dataset=train_dataset, batch_size=self.opts.batch_size, shuffle=True)
    
    def default_test_loader(self):
        test_dataset = datasets.MNIST(root=self.opts.TorchVisionPath, train=False, download=True, transform=self.transform)
        return DataLoader(dataset=test_dataset, batch_size=self.opts.batch_size, shuffle=False)

    
def train(opts = {}, model = None, criterion = None, optimizer = None,
          train_loader = None, test_loader = None):
    trainer = Trainer(opts)
    print(trainer.opts.__str__())

    # trainer.train()
