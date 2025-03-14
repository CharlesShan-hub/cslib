import click
from tqdm import tqdm

from clib.utils import get_device
from clib.dataset.fusion import GeneralFusion

# Paths - llvip
# default_ir_dir = "/Volumes/Charles/data/vision/torchvision/llvip/infrared/test"
# default_vis_dir = "/Volumes/Charles/data/vision/torchvision/llvip/visible/test"
# default_fused_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused"
# default_db_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused"
# default_db_name = "metrics.db"

# Paths - tno
default_ir_dir = "/Volumes/Charles/data/vision/torchvision/tno/tno/ir"
default_vis_dir = "/Volumes/Charles/data/vision/torchvision/tno/tno/vis"
default_fused_dir = "/Volumes/Charles/data/vision/torchvision/tno/tno/fused"
default_db_dir = "/Volumes/Charles/data/vision/torchvision/tno/tno/fused"
default_db_name = "metrics.db"

# Fusion Images
# 1. Calculare all images in each fused_dir
defaulf_img_id = ()
# 2. Calculare for specified images
# defaulf_img_id = ('190001','190002','190003')

# Fusion Algorithms
# 1. `fused_dir` is into one algorithm
# default_algorithms = () 
# 2. `fused_dir` is the parent dir of all algorithms
# default_algorithms = ('cpfusion','datfuse','fpde','fusiongan','gtf','ifevip','piafusion','stdfusion','tardal')
default_algorithms = ('cpfusion',)

# Metrics
# 1. All Metrics
default_metrics = [
    'ce','en','te','mi','nmi','q_ncie','psnr','cc','scc','scd',
    'ssim','ms_ssim','q_s','q','q_w','q_e','q_c','q_y','mb','mae',
    'mse','rmse','nmse','ergas','d','ag','mg','ei','pfe','sd','sf',
    'q_abf','q_sf','eva','sam','asm','con','fmi','n_abf','pww',
    'q_cv','q_cb','vif'
]
# 2. VIFB
# default_metrics = [
#     'ce','en','mi','psnr','ssim','rmse','ag','ei','sf',
#     'q_abf','sd','q_cb','q_cv'
# ]
# 3. MEFB
# default_metrics = [
#     'ce','en','fmi','nmi','psnr','q_ncie','te','ag','ei',
#     'q_abf','sd','sf','q_c','q_w','q_y','q_cb','q_cv','vif'
# ]

@click.command()
@click.option('--ir_dir', default=default_ir_dir)
@click.option('--vis_dir', default=default_vis_dir)
@click.option('--fused_dir', default=default_fused_dir)
@click.option('--algorithms', default=default_algorithms, multiple=True, help='compute metrics for multiple fusion algorithms')
@click.option('--img_id', default=defaulf_img_id, multiple=True, help='compute metrics for specified images')
@click.option('--metrcis', default=default_metrics, multiple=True)
@click.option('--db_dir','-n',default=default_db_dir, help='Path to save database file.')
@click.option('--db_name','-n',default=default_db_name, help='Name of database file.')
@click.option('--suffix', default="png")
@click.option('--device', default='auto', help='auto | cuda | mps | cpu')
@click.option('--jump', default=False, help='Jump Metrics that calculated before.')
def main(**kwargs):
    device = get_device(kwargs['device'])
    dataset = GeneralFusion(
        ir_dir = kwargs['ir_dir'],
        vis_dir = kwargs['vis_dir'],
        fused_dir = kwargs['fused_dir'],
        suffix = kwargs['suffix'],
        algorithms = kwargs['algorithms'],
        img_id = kwargs['img_id'],
    )
    for idx in tqdm(range(len(dataset)), desc="Processing batches", unit="image"):
        item: dict = dataset[idx]
        ir = item["ir"].to(device)
        vis = item["vis"].to(device)
        fused = item["fused"].to(device)
        print(item['id'])


if __name__ == '__main__':
    main()