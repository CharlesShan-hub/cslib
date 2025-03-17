import sqlite3
from pathlib import Path
import clib.metrics.fusion as fusion

__all__ = [
    'Database'
]

class Database:
    def __init__(self, db_dir, db_name, metrics=[], algorithms=[], jump=True, mode='compute'): # 'compute', 'analyze'
        self.load_database(db_dir, db_name)
        if mode == 'compute':
            self.update_metrics(metrics)
            self.update_algorithms(algorithms)
        elif mode == 'analyze':
            self.valid_metrics(metrics)
            self.valid_algorithms(algorithms)
        self.jump = jump

    def load_all_algorithms(self):
        ''' Load All Existing Algorithms From The Algorithms Table.
        '''
        self.cursor.execute("SELECT id, name FROM algorithms")
        rows = self.cursor.fetchall()
        self.all_algorithms = {row[1]:row[0] for row in rows}
    
    def load_all_metrics(self):
        ''' Load All Existing Metrics From The Metrics Table.'''
        self.cursor.execute("SELECT id, name FROM metrics")
        rows = self.cursor.fetchall()
        self.all_metrics = {row[1]:row[0] for row in rows}

    def valid_algorithms(self, algorithms):
        ''' Assert All Algorithms Existing In The Algorithms Table.
        '''
        self.load_all_algorithms()
        for a in algorithms:
            if a not in self.all_algorithms:
                raise ValueError(f'{a} not found in the database')
        self.algorithms = {a:self.all_algorithms[a] for a in algorithms}
    
    def valid_metrics(self, metrics):
        ''' Assert All Metrics Existing In The Metrics Table.
        '''
        self.load_all_metrics()
        for m in metrics:
            if m not in self.all_metrics:
                raise ValueError(f'{m} not found in the database')
        self.metrics = {m:self.all_metrics[m] for m in metrics}
    
    def update_algorithms(self, algorithms):
        ''' Update Algorithms To Algorithms Table
        '''
        self.load_all_algorithms()
        for a in algorithms:
            if a not in self.all_algorithms:
                self.cursor.execute("INSERT INTO algorithms (name) VALUES (?)", (a,))
                self.conn.commit()
        self.load_all_algorithms()
        self.algorithms = {a:self.all_algorithms[a] for a in algorithms}
    
    def update_metrics(self, metrics):
        ''' Update Metrics To Metrics Table
        '''
        self.load_all_metrics()
        for m in metrics:
            if m not in self.all_metrics:
                self.cursor.execute("INSERT INTO metrics (name) VALUES (?)", (m,))
                self.conn.commit()
        self.load_all_metrics()
        self.metrics = {m:self.all_metrics[m] for m in metrics}

    def load_database(self, db_dir, db_name):
        assert Path(db_dir).exists()
        self.db_dir = db_dir
        self.db_name = db_name
        self.conn = sqlite3.connect(Path(db_dir) / db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS algorithms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );
        ''')
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );
        ''')
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS fusion_metrics (
            algorithm_id INTEGER,
            metric_id INTEGER,
            image_id TEXT,
            value REAL,
            FOREIGN KEY (algorithm_id) REFERENCES algorithms (id),
            FOREIGN KEY (metric_id) REFERENCES metrics (id),
            PRIMARY KEY (algorithm_id, metric_id, image_id)
        );
        ''')
    
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def compute(self, ir, vis, fused, algorithm, img_id, logging=True, commit=True):
        algorithm_id = self.algorithms[algorithm]
        for m_name,m_id in self.metrics.items():
            # Check if the metric has already been calculated
            if self.jump:
                self.cursor.execute('''
                SELECT value FROM fusion_metrics WHERE algorithm_id=? AND image_id=? AND metric_id=?;
                ''', (algorithm_id, img_id, m_id))
                result = self.cursor.fetchone()

                if result:
                    if logging:
                        print(f"{m_name} \t {algorithm} \t {img_id}: {result[0]} (skipped)")
                    continue  # Skip calculation if the metric already exists and jump is True

            # Calculate
            value = getattr(fusion,f'{m_name}_metric')(ir,vis,fused)
            if logging:
                print(f"{m_name} \t {algorithm} \t {img_id}: {value}")
            
            # Insert or Update the database
            self.cursor.execute('''
            INSERT OR REPLACE INTO fusion_metrics (algorithm_id, image_id, metric_id, value)
            VALUES (?, ?, ?, ?);
            ''', (algorithm_id, img_id, m_id, value.item()))
        if commit:
            self.conn.commit()
    
    def commit(self):
        self.conn.commit()
    
    def analyze_average(self):
        result = {metric: {} for metric in self.metrics}
        for metric,m_id in self.metrics.items():
            for alg,a_id in self.algorithms.items():
                self.cursor.execute(
                    "SELECT AVG(value) FROM fusion_metrics WHERE algorithm_id=? AND metric_id=?",
                    (a_id, m_id)
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
        for metric,m_id in self.all_metrics.items():
            info["statistics"][metric] = {}
            for alg,a_id in self.all_algorithms.items():
                info["statistics"][metric][alg] = {}
                # Count Number
                self.cursor.execute(
                    "SELECT COUNT(*) FROM fusion_metrics WHERE algorithm_id=? AND metric_id=?",
                    (a_id, m_id)
                )
                num_images = self.cursor.fetchone()[0]
                info["statistics"][metric][alg]['num'] = num_images
                
                # Statistics
                self.cursor.execute(
                    "SELECT value FROM fusion_metrics WHERE algorithm_id=? AND metric_id=?",
                    (a_id, m_id)
                )
                values = [row[0] for row in self.cursor.fetchall() if row[0] is not None]
                info["statistics"][metric][alg]['mean'] = sum(values) / len(values) if values else None
                # info["statistics"][metric][alg]['min'] = min(values) if values else None
                # info["statistics"][metric][alg]['max'] = max(values) if values else None
        return info


if __name__ == '__main__':
    database = Database(
        db_dir='/Users/kimshan/temp',
        db_name='test.db',
        algorithms = ('cpfusion','datfuse','fpde','fusiongan','gtf','ifevip','piafusion','stdfusion','tardal'),
        metrics = ['mi','ag','mae'],
        mode='compute'
    )
    print(database.all_algorithms)