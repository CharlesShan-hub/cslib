from typing import List, Dict, Optional, Union
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image

class GeneralFusion(Dataset):
    '''
    Used for Test Fusion or Calculate Metrics
    '''
    def __init__(
            self, 
            ir_dir: Union[str, Path], 
            vis_dir: Union[str, Path], 
            fused_dir: Optional[Union[str, Path]], 
            transform: Optional[transforms.Compose] = None,
            suffix: str = 'png', 
            algorithms: Optional[Union[str, List[str]]] = None,
            img_id: Optional[Union[str, List[str]]] = None, 
        ):
        """
        Args:
            root_dir (Path): Directory with all the images.
            transform (transforms.Compose, optional): Optional transform to be applied on a sample.
            suffix (str): Suffix of the images.
            algorithms (Union[str, List[str]], optional): Fused algorithms. Defaults to None.
            img_id (Union[str, List[str]], optional): Image IDs. Defaults to None.
        """
        # Base Paths
        self.ir_dir = Path(ir_dir)
        self.vis_dir = Path(vis_dir)
        self.fused_dir = Path(fused_dir) if fused_dir != None else None
        
        # Enable Multiple Fused Algorithms
        if self.fused_dir is not None:
            if algorithms is None:
                self.fused_dirs = {
                    self.fused_dir.name: self.fused_dir
                }
            else:
                if isinstance(algorithms, str):
                    algorithms = [algorithms]
                self.fused_dirs = {
                    a: self.fused_dir / a for a in algorithms
                }
        
        # Check Path
        assert self.ir_dir.exists()
        assert self.vis_dir.exists()
        if self.fused_dir is not None:
            for _,p in self.fused_dirs.items():
                assert p.exists()

        # Default Transform
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transform

        # Checking and Enable Specified Image IDs
        if img_id is None or len(img_id) == 0:
            # 1. All Images
            all_ir_imgs = sorted(list(self.ir_dir.glob(f"*.{suffix}")))
            all_vis_imgs = sorted(list(self.vis_dir.glob(f"*.{suffix}")))
            all_img_id = [i.name for i in all_vis_imgs]
            if self.fused_dir is None: # doesn't need fused, assert num equals
                # 1.1 All Images without fused
                assert len(all_ir_imgs) == len(all_vis_imgs), "Number of IR and VIS must be equal"
                self.all_img_id: Union[List, Dict] = all_img_id
            else: 
                # 1.2 All Images with fused - maybe shrink range
                self.all_img_id: Union[List, Dict] = {}
                for algorithms,p in self.fused_dirs.items():
                    all_fused_imgs = sorted(list(p.glob(f"*.{suffix}")))
                    if len(all_fused_imgs) != len(all_vis_imgs):
                        # For We believe that if the lengths are equal, 
                        # the file names are the same, for saving time.
                        for i in all_fused_imgs:
                            vis_path = self.vis_dir / i.name
                            ir_path = self.ir_dir / i.name
                            if not vis_path.exists():
                                raise FileNotFoundError(f"Visible image not found: {vis_path}")
                            if not ir_path.exists():
                                raise FileNotFoundError(f"Infrared image not found: {ir_path}")
                    self.all_img_id[algorithms] = [i.name for i in all_fused_imgs]
        else:
            # 2. Enable Specified Image IDs
            all_img_id = [f'{i}.{suffix}' for i in img_id]
            for img_name in all_img_id:
                vis_path = self.vis_dir / img_name
                ir_path = self.ir_dir / img_name
                if not vis_path.exists():
                    raise FileNotFoundError(f"Visible image not found: {vis_path}")
                if not ir_path.exists():
                    raise FileNotFoundError(f"Infrared image not found: {ir_path}")
            if self.fused_dir is None: # doesn't need fused, assert num equals
                # 2.1 Specified Images without fused
                self.all_img_id: Union[List, Dict] = all_img_id
            else: 
                # 2.2 Specified Images with fused
                self.all_img_id: Union[List, Dict] = {}
                for algorithms,p in self.fused_dirs.items():
                    all_fused_imgs = sorted(list(p.glob(f"*.{suffix}")))
                    for i in all_img_id:
                        if not (p / i).exists():
                            raise FileNotFoundError(f"Fused image not found: {p / i}")
                    self.all_img_id[algorithms] = all_img_id
        
        # Build Bias Helper Dict
        self.bias_helper_dict = {}
        if isinstance(self.all_img_id, dict):
            bias = 0
            for algorithms,ids in self.all_img_id.items():
                self.bias_helper_dict[bias] = [algorithms,len(ids)]
                bias = bias + len(ids)
        
    def _recover_from_idx(self, idx):
        assert isinstance(self.all_img_id, dict)
        for bias,(algorithms,length) in self.bias_helper_dict.items():
            if bias <= idx+1 and bias+length >= idx+1:
                return idx-bias, algorithms
        raise ValueError("idx is not of range")
        
    def __len__(self):
        if isinstance(self.all_img_id, dict):
            count = 0
            for _, value in self.all_img_id.items():
                count += len(value)
            return count
        else:
            return len(self.all_img_id)

    def __getitem__(self, idx):
        if isinstance(self.all_img_id, dict):
            idx, algorithms = self._recover_from_idx(idx)
            ir_image = Image.open(((self.ir_dir / self.all_img_id[algorithms][idx])).__str__())
            vis_image = Image.open(((self.vis_dir / self.all_img_id[algorithms][idx])).__str__())
            fused_image = Image.open(((self.fused_dirs[algorithms] / self.all_img_id[algorithms][idx])).__str__())

            ir_image = self.transform(ir_image)
            vis_image = self.transform(vis_image)
            fused_image = self.transform(fused_image)

            return {
                'ir': ir_image,
                'vis': vis_image,
                'fused': fused_image,
                'algorithm': algorithms,
                'id': self.all_img_id[algorithms][idx].split('.')[0]
            }

        if isinstance(self.all_img_id, list):
            ir_image = Image.open(((self.ir_dir / self.all_img_id[idx])).__str__())
            vis_image = Image.open(((self.vis_dir / self.all_img_id[idx])).__str__())

            ir_image = self.transform(ir_image)
            vis_image = self.transform(vis_image)

            return {
                'ir': ir_image,
                'vis': vis_image,
                'id': self.all_img_id[idx].split('.')[0]
            }
        
        raise ValueError("all_img_id should be list or dict")