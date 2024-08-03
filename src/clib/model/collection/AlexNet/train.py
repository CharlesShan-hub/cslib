import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .config import TrainOptions
from .model import AlexNet as Model
def train(opts={}):
    opts = TrainOptions().parse(opts)
    model = Model().to(opts.device)
