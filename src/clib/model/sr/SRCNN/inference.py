import torch
import torchvision.transforms as transforms

from ....utils import glance,load_rgb_from_path
from ....transforms import to_rgb,rgb_to_ycbcr
from .utils import *

def inference(model, iml, imh, opts):
    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((opts.W, opts.H), antialias=True), # type: ignore
        transforms.Lambda(to_rgb),
        transforms.Lambda(rgb_to_ycbcr),
        # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    [iml, imh] = [load_rgb_from_path(im) for im in [iml,imh]]
    [iml, imh] = [torch.unsqueeze(trans(im), 0)for im in [iml,imh]] # type: ignore
    [iml, imh] = [im.to(opts.device) for im in [iml,imh]]
