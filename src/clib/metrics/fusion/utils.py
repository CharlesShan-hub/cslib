import sqlite3
from pathlib import Path
import clib.metrics.fusion as fusion

__all__ = [
    'Database'
]

class Database:
    def __init__(self, db_dir, db_name, metrcis=[], algorithms=[], jump=True):
        self.load_database(db_dir, db_name)
        self.load_metrics(metrcis)
        self.load_algorithms(algorithms)
        self.jump = jump

    def load_all_algorithms(self):
        self.cursor.execute("SELECT DISTINCT algorithm FROM fusion_metrics")
        self.all_algorithms = [row[0] for row in self.cursor.fetchall()]

    def load_algorithms(self, algorithms):
        self.load_all_algorithms()
        for a in algorithms:
            if a not in self.all_algorithms:
                raise ValueError(f'{a} not find')
        self.algorithms = algorithms
    
    def load_all_metrics(self):
        self.cursor.execute("SELECT DISTINCT metric FROM fusion_metrics")
        self.all_metrics = [row[0] for row in self.cursor.fetchall()]
        
    def load_metrics(self, metrics):
        for m in metrics:
            if not hasattr(fusion, m):
                raise ValueError(f'{m} not find')
        self.metrics = metrics

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
    
    def __del__(self):
        self.conn.close()
    
    def analyze_average(self):
        result = {metric: {} for metric in self.metrics}
        for metric in self.metrics:
            for alg in self.algorithms:
                self.cursor.execute(
                    "SELECT AVG(value) FROM fusion_metrics WHERE algorithm=? AND metric=?",
                    (alg, metric)
                )
                avg_value = self.cursor.fetchone()[0]
                result[metric][alg] = avg_value

        return result
    
    def analyze_general(self):
        self.load_all_algorithms()
        self.load_all_metrics()
        info = {
            "database_path": str(Path(self.db_dir) / self.db_name),
            "metrics": self.all_metrics,
            "algorithms": self.all_algorithms,
            "statistics": {}
        }
        for metric in self.all_metrics:
            info["statistics"][metric] = {}
            for alg in self.all_algorithms:
                info["statistics"][metric][alg] = {}
                # Count Number
                self.cursor.execute(
                    "SELECT COUNT(*) FROM fusion_metrics WHERE algorithm=? AND metric=?",
                    (alg, metric)
                )
                num_images = self.cursor.fetchone()[0]
                info["statistics"][metric][alg]['num'] = num_images
                
                # Statistics
                self.cursor.execute(
                    "SELECT value FROM fusion_metrics WHERE algorithm=? AND metric=?",
                    (alg, metric)
                )
                values = [row[0] for row in self.cursor.fetchall() if row[0] is not None]
                info["statistics"][metric][alg]['mean'] = sum(values) / len(values) if values else None
                # info["statistics"][metric][alg]['min'] = min(values) if values else None
                # info["statistics"][metric][alg]['max'] = max(values) if values else None
        return info
    
    def compute(self, ir, vis, fused, algorithm, img_id, logging=True, commit=True):
        for m in self.metrics:
            # Check if the metric has already been calculated
            if self.jump:
                self.cursor.execute('''
                SELECT value FROM fusion_metrics WHERE algorithm=? AND id=? AND metric=?;
                ''', (algorithm, img_id, m))
                result = self.cursor.fetchone()

                if result:
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
        if commit:
            self.conn.commit()
    
    def commit(self):
        self.conn.commit()
