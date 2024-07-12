from pathlib import Path
from torchvision.transforms.functional import to_tensor
from PIL import Image

__all__ = [
    'ir', 'vis', 'fused', 'cddfuse', 'densefuse', 'adf'
]

def load_demo_image():
    # 获取当前文件的绝对路径
    current_file_path = Path(__file__).resolve()
    # 定义资源文件夹的相对路径
    RESOURCE_DIR =Path(current_file_path.parent, 'resources')
    # 打开图片
    return [to_tensor(Image.open(Path(RESOURCE_DIR,f'{f}.png'))).unsqueeze(0)\
                        for f in ['vis','ir','CDDFuse','CDDFuse','DenseFuse','ADF']]

[ir, vis, fused, cddfuse, densefuse, adf] = load_demo_image()