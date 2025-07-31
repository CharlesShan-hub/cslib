from typing import Callable, Optional, Union
from torchvision.datasets.vision import VisionDataset
import os
from torchvision.datasets.utils import download_and_extract_archive
import shutil
from pathlib import Path
from PIL import Image

__all__ = ['RoadScene']

class RoadScene(VisionDataset):
    """
    RoadScene dataset.
    
    This datset has 221 aligned Vis and IR image pairs containing rich scenes 
    such as roads, vehicles, pedestrians and so on. These images are highly 
    representative scenes from the FLIR video. We preprocess the background 
    thermal noise in the original IR images, accurately align the Vis and IR 
    image pairs, and cut out the exact registration regions to form this dataset.

    https://github.com/hanna-xu/RoadScene
    """
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        raise ValueError("RoadScene dataset is not complete.")
        self.split = split
        self._base_folder = Path(root)
        self._src_folder = self._base_folder / 'RoadScene'
        self._src_folder.mkdir(parents=True, exist_ok=True)

