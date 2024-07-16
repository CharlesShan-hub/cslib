import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import click

import clib.model.fusion as fusion
import clib.data.fusion as fusion_data
from clib.utils import save_tensor_to_img
import config

'''
生成融合图片
1. 选择数据集(dataset: torch.utils.data.DataSet)
2. 选择数据集根目录(root_dir)
3. 选择融合图片保存目录(des_dir), 一般和root_dir一样就行
4. 选择融合方案(algorithm_name)
5. 选择融合方案的配置名称(algorithm_config),在 config.py 中
6. 选择预训练模型路径(pre_trained), 最好在config里边配置好
7. 选择图片名称(img_id)
'''
@click.command()
@click.option('--dataset','-n',default='FusionToy', help='Name of images dataset.')
@click.option('--root_dir','-r',default=Path(config.FusionPath, 'TNO'), help='Root directory containing the dataset.')
@click.option('--des_dir','-dr',default=Path(config.FusionPath, 'TNO'), help='Destination directory to save the results.')
@click.option('--algorithm_name','-a',default='DenseFuse', help='Fusion algorithm.')
@click.option('--algorithm_config','-ac',default='DenseFuse', help='Config name of Fusion algorithm.')
@click.option('--pre_trained','-p',default='',help='path to pretrained model.')
@click.option('--img_id','-i',default=(),multiple=True, help='Image IDs to compute metrics for.')
def main(dataset, root_dir, des_dir, algorithm_name, algorithm_config, pre_trained, img_id):
    # load Algorithm Module and Options
    if img_id == (): img_id = None
    else: img_id = [str(_id) for _id in img_id]
    assert hasattr(fusion, algorithm_name)
    assert algorithm_config in config.opts
    algorithm = getattr(fusion, algorithm_name)
    assert hasattr(fusion_data, dataset)
    FusionDataSet = getattr(fusion_data, dataset)
    opts = algorithm.TestOptions().parse(config.opts[algorithm_config])
    if pre_trained != '': setattr(opts, 'pre_trained', pre_trained)

    # Load Dataset, Dataloader and model with pre-trained params
    dataset = FusionDataSet(root_dir=Path(root_dir),img_id=img_id)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # batch size should be 1
    model = algorithm.load_model(opts)

    # run inference and save
    des_dir = Path(des_dir,'fused',algorithm_name)
    if des_dir.exists() == False: des_dir.mkdir()
    bar = tqdm(range(len(dataset)))
    for _,batch in zip(bar,dataloader):
        img_res_path = Path(des_dir,Path(batch['ir'][0]).name)
        if img_res_path.exists(): continue
        img = algorithm.inference(model,batch['ir'][0],batch['vis'][0],opts)
        save_tensor_to_img(img,img_res_path)

if __name__ == '__main__':
    main()