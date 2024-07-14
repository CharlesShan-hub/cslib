'''
生成融合图片
'''

import torch
from clib.model.fusion import DeepFuse as Method
from clib.data.fusion import FusionToy
from torch.utils.data import DataLoader
import config
from pathlib import Path

opts = Method.TestOptions().parse(config.opts['DeepFuse'])

dataset = FusionToy(root_dir=Path(config.FusionPath,'Toy'),img_id=None)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = Method.model(device=config.device)
model.load_state_dict(torch.load(opts.pre_trained, map_location=opts.device)['model'])

for batch in dataloader:
    img = Method.inference(model,batch['ir'][0],batch['vis'][0],config.device)
    break