"""
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    Paper: https://doi.org/10.48550/arXiv.1409.1556
    ArXiv: https://arxiv.org/abs/1409.1556
    Author: Charles Shan
"""
from .model import VGG as Model, load_model
from .inference import inference
from .train import train
