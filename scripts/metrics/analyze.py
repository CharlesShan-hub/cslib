import click
import json
from clib.utils.config import Options
from clib.metrics.fusion.utils import Database

# Paths - llvip
default_db_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused"
default_db_name = "metrics.db"

# Paths - tno
# default_db_dir = "/Volumes/Charles/data/vision/torchvision/tno/tno/fused"
# default_db_name = "metrics.db"

# Fusion Images
# 1. Calculare all images in each fused_dir
defaulf_img_id = ()
# 2. Calculare for specified images
# defaulf_img_id = ('190001','190002','190003')
# defaulf_img_id = ('39',)

# Fusion Algorithms
# 1. `fused_dir` is into one algorithm
# default_algorithms = () 
# 2. `fused_dir` is the parent dir of all algorithms
default_algorithms = ('cpfusion','datfuse','fpde','fusiongan','gtf','ifevip','piafusion','stdfusion','tardal')
# default_algorithms = ('cpfusion',)

# Metrics
default_metrics = [
    'ce', 'sf', 'ag', 'sd', 'scd', 'vif', 'psnr', 'mb', 
    'mae', 'mse', 'rmse', 'nrmse', 'mg', 'ei', 'mi'
]
# default_metrics = [
#     'ag',
# ]
# 1. All Metrics
# default_metrics = [
#     'ce','en','te','mi','nmi','q_ncie','psnr','cc','scc','scd',
#     'ssim','ms_ssim','q_s','q','q_w','q_e','q_c','q_y','mb','mae',
#     'mse','rmse','nrmse','ergas','d','ag','mg','ei','pfe','sd','sf',
#     'q_abf','q_sf','eva','sam','asm','con','fmi','n_abf','pww',
#     'q_cv','q_cb','vif'
# ]
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
@click.option('--metrics', default=default_metrics, multiple=True)
@click.option('--algorithms', default=default_algorithms, multiple=True, help='analyze metrics for multiple fusion algorithms')
@click.option('--db_dir', default=default_db_dir, help='Path to save database file.')
@click.option('--db_name', default=default_db_name, help='Name of database file.')
def main(**kwargs):
    opts = Options('Analyze Metrics',kwargs).parse({},present=True)
    database = Database(
        db_dir = opts.db_dir, 
        db_name = opts.db_name,
        metrics = opts.metrics,
        algorithms = opts.algorithms,
        mode = 'analyze' # analyze 就是检查 metrics 和 algorithms 已经存在
    )
    # print(json.dumps(database.analyze_average(), indent=4, sort_keys=True))
    print(json.dumps(database.analyze_general(), indent=4, sort_keys=False))

if __name__ == '__main__':
    main()