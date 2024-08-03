"""
    Gradient-based learning applied to document recognition
    Paper: https://hal.science/hal-03926082/document
    Author: Charles Shan
"""
from .model import LeNet as Model, load_model
from .inference import inference
from .train import train
