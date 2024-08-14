import torch
import torch.nn as nn

def load_model(opts):
    model = VGG().to(opts.device)
    params = torch.load(opts.pre_trained, map_location=opts.device)
    model.load_state_dict(params)
    return model

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
    
    def forward(self, x):
        return x
