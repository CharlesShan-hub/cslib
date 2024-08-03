"""
    ImageNet Classification with Deep Convolutional Neural Networks
    Paper: https://dl.acm.org/doi/10.1145/3065386
    Author: Charles Shan
"""
from .model import AlexNet as Model, load_model
from .inference import inference
from .train import train
