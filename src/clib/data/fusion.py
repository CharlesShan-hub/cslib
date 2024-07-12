import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path

class MetricsToy(Dataset):
    def __init__(self, root_dir, transform=None, suffix='png', method=None, img_id=None, check=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
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
        elif isinstance(method, list):
            for m in method:
                assert(Path(self.fused_dir, m).exists())
            self.fused_folders = method
        else:
            raise(ValueError(f'`method` should only be `str` or `list`, but {type(method)} get.'))
        
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
            elif img_id is None:
                self.fused_paths_dict[method] = temp_paths
            else:
                raise(ValueError("`img_id` should only be `str`, `list` or None"))
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

