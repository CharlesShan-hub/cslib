import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import click

import clib.model.sr as sr
import clib.data.sr as sr_data
from clib.utils import save_tensor_to_img
import config

# a = sr_data.Set5(root_dir=Path(config.SRPath, 'Set5'),upscale_factor=2,img_id=['001'])

opts = sr.TestOptions().parse(config.opts['SRCNN'])
sr.load_model(opts)