'''
训练融合网络
'''

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from clib.model.fusion import DeepFuse as Method
from clib.data.fusion import FusionToy
import config

opts = Method.TrainOptions().parse(config.opts['DeepFuse'])

dataset = FusionToy(
        root_dir=Path(config.FusionPath,'Toy'),
        transform=Method.train_trans(opts),
        only_path=False
    )

dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False)

model = Method.model(device=opts.device)

Method.train(model,dataloader,opts)