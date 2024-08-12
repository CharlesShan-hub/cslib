from torch.utils.data import DataLoader
from pathlib import Path
import sqlite3
import click

import clib.metrics.fusion as metrics
from clib.data import fusion as fusion_data
from config import opts

'''
测试融合算法的指标
1. 选择指定的融合方案
2. 选择指定的融合指标
3. 选择指定的融合图片
4. 结果输出到数据库中
5. 可以避免重复计算，可以进行结果更新(二选一)
6. 注意需要提前组织好融合图片的存储结构
'''
@click.command()
@click.option('--database','-n',default='MetricsToy', help='Name of images database.')
@click.option('--root_dir','-r',default=Path(opts['_'].FusionPath, 'Toy'), help='Root directory containing the dataset.')
@click.option('--db_name','-n',default='metrics.db', help='Name of database file.')
@click.option('--algorithm','-a',default=(),multiple=True, help='Fusion algorithm.')
@click.option('--img_id','-i',default=(),multiple=True, help='Image IDs to compute metrics for.')
@click.option('--metric_group','-m',default='VIFB', help='Methods Group to compute metrics for.')
@click.option('--device','-d',default=opts['_'].device, help='Device to compute metrics on.')
@click.option('--update','-u',default=False, help='Update Metrics that calculated before.')
def main(database, root_dir, db_name, metric_group, algorithm, img_id, device, update):
    # Modify Params
    assert hasattr(fusion_data, database)
    [img_id, algorithm] = [None if len(item)==0 else item for item in [img_id, algorithm]]
    
    # Connect to Database
    conn = sqlite3.connect(Path(root_dir,db_name))
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fusion_metrics (
        method TEXT,
        id TEXT,
        name TEXT,
        value REAL,
        PRIMARY KEY (method, id, name)
    );
    ''')

    # Load Dataset and Dataloader
    dataset = getattr(fusion_data,database)(root_dir=root_dir,method=algorithm,img_id=img_id)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load metrics
    group = {
        'VIFB': ['en', 'ce', 'mi', 'psnr', 'ssim', 'rmse', 'ag', 'ei', 'sf', 'q_abf', 'sd', 'q_cb', 'q_cv'],
    }
    assert metric_group in group
    
    # Calculate
    for batch in dataloader:
        for (k,v) in metrics.info_summary_dict.items():
            # Skip Unneeded Metrics
            if k not in group[metric_group]: continue

            # Skip Calculated Metrics
            if update == False:
                cursor.execute('''
                SELECT value FROM fusion_metrics WHERE method=? AND id=? AND name=?;
                ''', (batch['method'][0], batch['id'][0], k))
                if cursor.fetchone(): continue # If Exist -> Skip

            # Calculate
            value = v['metric'](batch['ir'].to(device),batch['vis'].to(device),batch['fused'].to(device))
            print(f"{k} - {batch['method'][0]} - {batch['id'][0]}: {value}")
            
            # Insert or Update the database
            cursor.execute('''
            INSERT OR REPLACE INTO fusion_metrics (method, id, name, value)
            VALUES (?, ?, ?, ?);
            ''', (batch['method'][0], batch['id'][0], k, value.item()))

    conn.commit()
    conn.close()

if __name__ == '__main__':
    main()