import sqlite3
from pathlib import Path
import clib.metrics.fusion as fusion

__all__ = [
    'Database'
]

class Database:
    def __init__(self, db_dir, db_name, metrcis=[], jump=True):
        self.load_database(db_dir, db_name)
        self.load_metrics(metrcis)
        self.jump = jump
    
    def __del__(self):
        self.conn.close()
        
    def load_metrics(self, metrcis):
        for m in metrcis:
            if not hasattr(fusion, m):
                raise ValueError(f'{m} not find')
        self.metrcis = metrcis

    def load_database(self, db_dir, db_name):
        assert Path(db_dir).exists()
        self.db_dir = db_dir
        self.db_name = db_name
        self.conn = sqlite3.connect(Path(db_dir) / db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS fusion_metrics (
            algorithm TEXT,
            id TEXT,
            metric TEXT,
            value REAL,
            PRIMARY KEY (algorithm, id, metric)
        );
        ''')
    
    def compute(self, ir, vis, fused, algorithm, img_id, logging=True):
        for m in self.metrcis:
            # Check if the metric has already been calculated
            self.cursor.execute('''
            SELECT value FROM fusion_metrics WHERE algorithm=? AND id=? AND metric=?;
            ''', (algorithm, img_id, m))
            result = self.cursor.fetchone()

            if result and self.jump:
                if logging:
                    print(f"{m} \t {algorithm} \t {img_id}: {result[0]} (skipped)")
                continue  # Skip calculation if the metric already exists and jump is True

            # Calculate
            value = getattr(fusion,f'{m}_metric')(ir,vis,fused)
            if logging:
                print(f"{m} \t {algorithm} \t {img_id}: {value}")
            
            # Insert or Update the database
            self.cursor.execute('''
            INSERT OR REPLACE INTO fusion_metrics (algorithm, id, metric, value)
            VALUES (?, ?, ?, ?);
            ''', (algorithm, img_id, m, value.item()))
        self.conn.commit()
