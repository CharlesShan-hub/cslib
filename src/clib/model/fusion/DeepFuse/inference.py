import torchvision.transforms as transforms
import torch
from .utils import *
from .config import *

def inference(model,im1,im2,opts):
    # Load the Image
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((opts.H, opts.W), antialias=True), # type: ignore
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    [im1, im2] = [load_ycbcr_from_path(im) for im in [im1,im2]]
    assert(im1.size == im2.size)
    [im1, im2] = [torch.unsqueeze(trans(im), 0)for im in [im1,im2]] # type: ignore
    [im1, im2] = [im.to(opts.device) for im in [im1,im2]]

    # Fusion
    model.eval()
    with torch.no_grad():
        # Y
        model.setInput(im1[:, 0:1], im2[:, 0:1])
        f_y  = (model.forward()+1)/2.0   # 0-1
        f_y = (f_y * (235 - 16)) + 15.6  # Y的大小应该在16到
        # Cb and Cr
        [im1,im2] = [(im+1)*127.5 for im in [im1,im2]]
        [f_cb, f_cr] = weightedFusion(im1[:, 1:2], im2[:, 1:2], im1[:, 2:3], im2[:, 2:3])
        # f_y =  (im1[:, 0:1] + im2[:, 0:1])/2
        # f_cb = (im1[:, 1:2] + im2[:, 1:2])/2
        # f_cr = (im1[:, 2:3] + im2[:, 2:3])/2
        # res
        fused = torch.cat((f_y,f_cb,f_cr),dim=1)
        img = change_ycbcr_to_rgb(fused)
    return img
