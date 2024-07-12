import clib.metrics.fusion as metrics
from clib.data.fusion import MetricsToy
from torch.utils.data import DataLoader
from pathlib import Path
import config
import sqlite3

conn = sqlite3.connect(Path(config.FusionPath,'Toy','metrics.db'))
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

need = ['ag', 'q_abf']

dataset = MetricsToy(root_dir=Path(config.FusionPath,'Toy'),
                     method=['ADF','CDDFuse'], 
                     img_id=['48','99'])
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for batch in dataloader:
    for (k,v) in metrics.info_summary_dict.items():
        # 跳过不需要的指标
        if k not in need: continue

        # 跳过计算过的数据
        cursor.execute('''
        SELECT value FROM fusion_metrics WHERE method=? AND id=? AND name=?;
        ''', (batch['method'][0], batch['id'][0], k))
        if cursor.fetchone(): continue # 如果存在，则更新 value

        # 计算指标
        value = v['metric'](batch['ir'],batch['vis'],batch['fused'])
        print(f"{k} - {batch['method'][0]} - {batch['id'][0]}: {value}")
        
        # 插入或更新数据
        cursor.execute('''
        INSERT OR REPLACE INTO fusion_metrics (method, id, name, value)
        VALUES (?, ?, ?, ?);
        ''', (batch['method'][0], batch['id'][0], k, value.item()))

conn.commit()
conn.close()