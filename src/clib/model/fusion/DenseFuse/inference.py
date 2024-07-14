import torch
import torchvision.transforms as transforms

# from ....utils import glance
from .utils import *

def inference(model,im1,im2,opts):
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    #[im1, im2] = [load_rgb_from_path(im) for im in [im1,im2]]
    [im1, im2] = [load_gray_from_path(im) for im in [im1,im2]]
    assert(im1.size == im2.size)
    [im1, im2] = [torch.unsqueeze(trans(im), 0)for im in [im1,im2]] # type: ignore
    [im1, im2] = [im.to(opts.device) for im in [im1,im2]]
    en_r = model.encoder(im1)
    # vision_features(en_r, 'ir')
    en_v = model.encoder(im2)
    # # vision_features(en_v, 'vi')
    # fusion
    f = model.fusion(en_r, en_v)#, strategy_type=opts.strategy_type)
    # f = en_v
    # decoder
    img_fusion = model.decoder(f)
    # glance(img_fusion[0][0,:,:,:])
    # print(img_fusion.shape)
    return img_fusion[0][0,:,:,:]