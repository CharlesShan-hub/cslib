from typing import List, Optional, Tuple, Union, Callable
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from pathlib import Path

class Set5(Dataset):
    def __init__(self, root_dir: Union[str, Path], upscale_factor: int,\
                 transform: Optional[Callable[[Image.Image], torch.Tensor]] = None, \
                 suffix: str = 'png', img_id: Optional[List[str]] = None, \
                 check: bool = True, only_path: bool = True):
        # Base Paths and Config
        self.root_dir = root_dir
        self.upscale_factor = upscale_factor
        self.upsacle_dir = Path(root_dir,f'image_SRF_{upscale_factor}')
        self.only_path = only_path

        # Set default transform if none provided
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.transform = transform

        # Get HR and LR Path
        temp = sorted([Path(i) for i in os.listdir(self.upsacle_dir) if i.endswith(f'.{suffix}')])
        self.hr_paths = [i for i in temp if os.path.splitext(i.name)[0].split('_')[-1] == 'HR']
        self.lr_paths = [i for i in temp if os.path.splitext(i.name)[0].split('_')[-1] == 'LR']
        if img_id is not None:
            self.hr_paths = [i for i in self.hr_paths if  os.path.splitext(i.name)[0].split('_')[1] in img_id]
            self.lr_paths = [i for i in self.lr_paths if  os.path.splitext(i.name)[0].split('_')[1] in img_id]
            
        # Check
        assert len(self.hr_paths) == len(self.lr_paths)
        if check:
            for i,j in zip(self.hr_paths, self.lr_paths):
                assert os.path.splitext(i.name)[0].split('_')[1] == os.path.splitext(j.name)[0].split('_')[1]
        
    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        # Load images
        hr_image = self.hr_paths[idx].__str__()
        lr_image = self.lr_paths[idx].__str__()
        img_id = os.path.splitext(self.hr_paths[idx])[0].split('_')[1]

        # Apply transform if specified
        if self.only_path == False:
            hr_image = self.transform(Image.open(hr_image))
            lr_image = self.transform(Image.open(lr_image))

        # Return a dictionary with all images
        sample = {
            'hr': hr_image,
            'lr': lr_image,
            'id': img_id
        }
        return sample