import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import click

import clib.model.fusion as fusion
import clib.data.fusion as fusion_data
import config

'''
生成融合图片
1. 选择数据集(dataset: torch.utils.data.DataSet)
2. 选择数据集根目录(root_dir)
3. 选择融合图片保存目录(des_dir), 一般和root_dir一样就行
4. 选择融合方案(algorithm_name)
5. 选择融合方案的配置名称(algorithm_config),在 config.py中
6. 选择预训练模型路径(pre_trained), 最好在config里边配置好
7. 选择图片名称(img_id)
8. 选择设备(device)(尽量别选,程序自动判断)
'''
@click.command()
@click.option('--dataset','-n',default='FusionToy', help='Name of images dataset.')
@click.option('--root_dir','-r',default=Path(config.FusionPath, 'Toy'), help='Root directory containing the dataset.')
@click.option('--des_dir','-dr',default=Path(config.FusionPath, 'Toy'), help='Destination directory to save the results.')
@click.option('--algorithm_name','-a',default='DenseFuse', help='Fusion algorithm.')
@click.option('--algorithm_config','-ac',default='DenseFuse', help='Config name of Fusion algorithm.')
@click.option('--pre_trained','-p',default='',help='path to pretrained model.')
@click.option('--img_id','-i',default=(),multiple=True, help='Image IDs to compute metrics for.')
@click.option('--device','-d',default=config.device, help='Device to compute metrics on.')
def main(dataset, root_dir, des_dir, algorithm_name, algorithm_config, pre_trained, img_id, device):
    # load Algorithm Module and Options
    if img_id == (): img_id = None
    assert hasattr(fusion, algorithm_name)
    assert algorithm_config in config.opts
    algorithm = getattr(fusion, algorithm_name)
    assert hasattr(fusion_data, dataset)
    FusionDataSet = getattr(fusion_data, dataset)
    opts = algorithm.TestOptions().parse(config.opts[algorithm_config])
    if pre_trained != '': setattr(opts, 'pre_trained', pre_trained)

    # Load Dataset and Dataloader
    dataset = FusionDataSet(root_dir=Path(root_dir),img_id=img_id)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # batch size should be 1
    # model = algorithm.model(device=device)
    # model.load_state_dict(torch.load(opts.pre_trained, map_location=device)['model'])
    model = algorithm.model(1,1)
    model.load_state_dict(torch.load(opts.pre_trained, map_location=device))

    # run inference and save
    des_dir = Path(des_dir,algorithm_name)
    bar = tqdm(range(len(dataset)))
    for _,batch in zip(bar,dataloader):
        img = algorithm.inference(model,batch['ir'][0],batch['vis'][0],opts)
        # save

if __name__ == '__main__':
    main()