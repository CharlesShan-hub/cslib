import torch
import torchvision.transforms as transforms

from .utils import *

def inference(model,im1,im2,opts):
    if hasattr(opts, "height") and hasattr(opts, "width"):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([opts.height, opts.width], interpolation=transforms.InterpolationMode.NEAREST),
        ])
    else:
        trans = transforms.Compose([transforms.ToTensor(),])

    [im1, im2] = [path_to_gray(im) for im in [im1,im2]]
    [im1, im2] = [torch.unsqueeze(trans(im), 0) for im in [im1,im2]] # type: ignore
    assert(im1.shape == im2.shape)
    [im1, im2] = [im.to(opts.device) for im in [im1,im2]]
    
    model.eval()
    with torch.no_grad():
        imf = model.forward(im1,im2)
    
    return imf[0,:,:,:]