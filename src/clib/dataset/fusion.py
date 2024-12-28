from typing import List, Optional, Tuple, Union, Callable
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from pathlib import Path
from .collection.ino import INO
from .collection.m3fd import M3FD

__all__ = [
    'INO',
    'M3FD',
]

class FusionToy(Dataset):
    """
    Used for running fusion models.

    Parameters:
        root_dir (str): The root directory containing the 'ir' and 'vis' subdirectories.
        transform (Optional[Callable[[PIL.Image.Image], torch.Tensor]]): An optional transform to be applied on the input images.
        suffix (str): The file suffix for the IR and Vis images. Default is 'png'.
        img_id (Optional[List[str]]): A list of image IDs to filter the dataset. Default is None.
        check (bool): Whether to check the integrity of the dataset. Default is True.
        only_path (bool): Whether to only return the paths of the images. Default is True.

    Attributes:
        root_dir (Path): The root directory of the dataset.
        ir_dir (Path): The directory containing the IR images.
        vis_dir (Path): The directory containing the Vis images.
        fused_dir (Path): The directory where the fused images will be saved.
        ir_paths (List[Path]): The sorted list of IR image paths.
        vis_paths (List[Path]): The sorted list of Vis image paths.
        transform (Callable[[PIL.Image.Image], torch.Tensor]): The transform to be applied on the input images.
        only_path (bool): Whether to only return the paths of the images.

    Examples:
        >>> dataset = FusionToy('path_to_dataset', img_id=['0001', '0002'])
        >>> print(len(dataset))
        >>> print(dataset[0]['ir'].shape)
    """
    def __init__(self, root_dir: Union[str, Path], \
                 transform: Optional[Callable[[Image.Image], torch.Tensor]] = None, \
                 suffix: str = 'png', img_id: Optional[List[str]] = None, \
                 check: bool = True, only_path: bool = True):
        # Base Paths and Config
        self.root_dir = root_dir
        self.ir_dir = Path(root_dir,'ir')
        self.vis_dir = Path(root_dir,'vis')
        self.fused_dir = Path(root_dir,'fused')
        self.only_path = only_path

        # Set default transform if none provided
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.transform = transform

        # Get Ir and Vis Path
        self.ir_paths = sorted([Path(self.ir_dir, img) for img in \
                             os.listdir(self.ir_dir) if img.endswith(f'.{suffix}')])
        self.vis_paths = sorted([Path(self.vis_dir, img) for img in \
                             os.listdir(self.vis_dir) if img.endswith(f'.{suffix}')])
        assert len(self.ir_paths) == len(self.vis_paths)

        # Load part of Images
        if img_id is not None:
            self.ir_paths = [path for path in self.ir_paths if\
                              os.path.splitext(path.name)[0] in img_id]
            self.vis_paths = [path for path in self.vis_paths if\
                              os.path.splitext(path.name)[0] in img_id]

        # Check
        if check:
            for i,j in zip(self.ir_paths, self.vis_paths):
                assert i.name == j.name
        
    def __len__(self):
        return len(self.ir_paths)

    def __getitem__(self, idx):
        # Load images
        ir_image = self.ir_paths[idx].__str__()
        vis_image = self.vis_paths[idx].__str__()
        img_id, _ = os.path.splitext(self.ir_paths[idx].name)

        # Apply transform if specified
        if self.only_path == False:
            ir_image = self.transform(Image.open(ir_image))
            vis_image = self.transform(Image.open(vis_image))

        # Return a dictionary with all images
        sample = {
            'ir': ir_image,
            'vis': vis_image,
            'id': img_id
        }
        return sample


class MetricsToy(Dataset):
    '''
    Used for Test Fusion Metrics
    '''
    def __init__(self, root_dir: Path, transform: Optional[transforms.Compose] = None,
                 suffix: str = 'png', method: Optional[Union[str, List[str]]] = None,
                 img_id: Optional[Union[str, List[str]]] = None, check: bool = True):
        """
        Args:
            root_dir (Path): Directory with all the images.
            transform (transforms.Compose, optional): Optional transform to be applied on a sample.
            suffix (str): Suffix of the images.
            method (Union[str, List[str]], optional): Fused methods. Defaults to None.
            img_id (Union[str, List[str]], optional): Image IDs. Defaults to None.
            check (bool): Whether to check the existence of IR and VIS images. Defaults to True.
        """
        # Base Paths
        self.root_dir = root_dir
        self.ir_dir = Path(root_dir,'ir')
        self.vis_dir = Path(root_dir,'vis')
        self.fused_dir = Path(root_dir,'fused')

        # Set default transform if none provided
        if transform is None:
            transform = transforms.Compose([
                # transforms.Resize((256, 256)),  # Uncomment this line if you want to resize images
                transforms.ToTensor(),
            ])
        self.transform = transform
        
        # Get Fused Methods
        if method == None:
            self.fused_folders = [path for path in os.listdir(self.fused_dir) \
                             if os.path.isdir(Path(self.fused_dir,path))]
        elif isinstance(method, str):
            method = [method]
        else: # Is list
            for m in method:
                assert(Path(self.fused_dir, m).exists())
            self.fused_folders = method

        # Get Fused Paths
        self.fused_paths_dict = {}
        for method in self.fused_folders:
            all_imgs = os.listdir(Path(self.fused_dir, method))
            temp_paths = sorted([Path(self.fused_dir, method, img) for img in all_imgs \
                                 if img.endswith(f'.{suffix}')])
            if isinstance(img_id, str):
                img_id = [img_id]
            if isinstance(img_id, list):
                self.fused_paths_dict[method] = sorted([Path(self.fused_dir, method, img) for img in temp_paths \
                                                        if os.path.splitext(img.name)[0] in img_id])
            else: # Is None
                self.fused_paths_dict[method] = temp_paths
        self.fused_paths = []
        for _,value in self.fused_paths_dict.items():
            self.fused_paths += value
            
        # Check ir and vis
        if check:
            for _,value in self.fused_paths_dict.items():
                for path in value:
                    assert Path(self.ir_dir, path.name).exists()
                    assert Path(self.vis_dir, path.name).exists()
        
    def __len__(self):
        count = 0
        for _, value in self.fused_paths_dict.items():
            count += len(value)
        return count

    def __getitem__(self, idx):
        # Load images
        ir_image = Image.open(Path(self.ir_dir,self.fused_paths[idx].name).__str__())
        vis_image = Image.open(Path(self.vis_dir,self.fused_paths[idx].name).__str__())
        fused_image = Image.open(self.fused_paths[idx].__str__())
        method_name = self.fused_paths[idx].parent.name
        img_id, _ = os.path.splitext(self.fused_paths[idx].name)

        # Apply transform if specified
        ir_image = self.transform(ir_image)
        vis_image = self.transform(vis_image)
        fused_image = self.transform(fused_image)

        # Return a dictionary with all images
        sample = {
            'ir': ir_image,
            'vis': vis_image,
            'fused': fused_image,
            'method': method_name,
            'id': img_id
        }
        return sample