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
        self.all_algorithms_names = {row[1]:row[0] for row in rows}
        self.all_algorithms_ids = {row[0]:row[1] for row in rows}
    
    def load_all_metrics(self):
        ''' Load All Existing Metrics From The Metrics Table.'''
        self.cursor.execute("SELECT id, name FROM metrics")
        rows = self.cursor.fetchall()
        self.all_metrics_names = {row[1]:row[0] for row in rows}
        self.all_metrics_ids = {row[0]:row[1] for row in rows}

    def valid_algorithms(self, algorithms):
        ''' Assert All Algorithms Existing In The Algorithms Table.
        '''
        self.load_all_algorithms()
        for a in algorithms:
            if a not in self.all_algorithms_names:
                raise ValueError(f'{a} not found in the database')
        self.algorithms_names = {a:self.all_algorithms_names[a] for a in algorithms}
        self.algorithms_ids = {self.all_algorithms_names[a]:a for a in algorithms}
    
    def valid_metrics(self, metrics):
        ''' Assert All Metrics Existing In The Metrics Table.
        '''
        self.load_all_metrics()
        for m in metrics:
            if m not in self.all_metrics_names:
                raise ValueError(f'{m} not found in the database')
        self.metrics_names = {m:self.all_metrics_names[m] for m in metrics}
        self.metrics_ids = {self.all_metrics_names[m]:m for m in metrics}
    
    def update_algorithms(self, algorithms):
        ''' Update Algorithms To Algorithms Table
        '''
        self.load_all_algorithms()
        for a in algorithms:
            if a not in self.all_algorithms_names:
                self.cursor.execute("INSERT INTO algorithms (name) VALUES (?)", (a,))
                self.conn.commit()
        self.load_all_algorithms()
        self.algorithms_names = {a:self.all_algorithms_names[a] for a in algorithms}
        self.algorithms_ids = {self.all_algorithms_names[a]:a for a in algorithms}
    
    def update_metrics(self, metrics):
        ''' Update Metrics To Metrics Table
        '''
        self.load_all_metrics()
        for m in metrics:
            if m not in self.all_metrics_names:
                self.cursor.execute("INSERT INTO metrics (name) VALUES (?)", (m,))
                self.conn.commit()
        self.load_all_metrics()
        self.metrics_names = {m:self.all_metrics_names[m] for m in metrics}
        self.metrics_ids = {self.all_metrics_names[m]:m for m in metrics}

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

    def select_values(self, algorithm=None, metrics=None, img_id=None,
                      return_value=True, return_algorithm_id=False, 
                      return_metric_id=False, return_img_id=False):
        if not (return_value or return_algorithm_id or return_metric_id):
            raise ValueError("At least one of return_value, return_algorithm_id, or return_metric_id must be True")
        
        select_clause = []
        conditions = []
        params = []

        if return_algorithm_id:
            select_clause.append("algorithm_id")
        if return_metric_id:
            select_clause.append("metric_id")
        if return_img_id:
            select_clause.append("image_id")
        if return_value:
            select_clause.append("value")
        # alg_id, metric_id, image_id, value

        if algorithm is not None:
            conditions.append("algorithm_id = ?")
            params.append(self.all_algorithms_names[algorithm])

        if metrics is not None:
            conditions.append("metric_id = ?")
            params.append(self.all_metrics_names[metrics])

        if img_id is not None:
            conditions.append("image_id = ?")
            params.append(img_id)

        query = f"SELECT {', '.join(select_clause)} FROM fusion_metrics"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        self.cursor.execute(query, params)
        return self.cursor.fetchall()
    
    def update_values(self, algorithm, metrics, img_id, value, commit=True):
        self.cursor.execute('''
        INSERT OR REPLACE INTO fusion_metrics (algorithm_id, image_id, metric_id, value)
        VALUES (?, ?, ?, ?);
        ''', (self.all_algorithms_names[algorithm], img_id, self.all_metrics_names[metrics], value))
        if commit:
            self.conn.commit()
        
    def compute(self, ir, vis, fused, algorithm, img_id, logging=True, commit=True):
        for m_name,m_id in self.metrics_names.items():
            # Check if the metric has already been calculated
            if self.jump:
                result = self.select_values(algorithm, m_name, img_id)
                if result:
                    if logging:
                        print(f"{m_name} \t {algorithm} \t {img_id}: {result[0]} (skipped)")
                    continue  # Skip calculation if the metric already exists and jump is True

            # Calculate
            value = getattr(fusion,f'{m_name}_metric')(ir,vis,fused)
            if logging:
                print(f"{m_name} \t {algorithm} \t {img_id}: {value}")
            
            # Insert or Update the database
            self.update_values(algorithm, m_name, img_id, value.item(), commit)
    
    def commit(self):
        if self.conn is not None:
            self.conn.commit()
    
    def analyze_average(self):
        result = {metric: {} for metric in self.metrics_names}
        for metric,m_id in self.metrics_names.items():
            for alg,a_id in self.algorithms_names.items():
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
            "metrics": self.all_metrics_names,
            "algorithms": self.all_algorithms_names,
            "statistics": {}
        }
        for metric,m_id in self.all_metrics_names.items():
            info["statistics"][metric] = {}
            for alg,a_id in self.all_algorithms_names.items():
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
    
    def merge(self, other):
        # Step 1: Load all fusion metrics from the other database
        other_fusion_metrics = other.select_values(
            return_value=True,
            return_algorithm_id=True,
            return_metric_id=True,
            return_img_id=True,
        )

        # Step 2: Update algorithms and metrics in the current database
        self.update_algorithms(other.all_algorithms_names.keys())
        self.update_metrics(other.all_metrics_names.keys())
        
        # Step 3: Insert or update fusion metrics in the current database
        for row in other_fusion_metrics:
            other_alg_id, other_metric_id, image_id, value = row
            self.update_values(
                algorithm = other.all_algorithms_ids[other_alg_id],
                metrics = other.all_metrics_ids[other_metric_id],
                img_id = image_id,
                value = value,
                commit = False,
            )

        # Step 4: Commit changes
        self.conn.commit()
        print("Merge Completed.")

    def split_by_metric(self, output_dir: str) -> None:
        """
        Split the current database into separate databases for each metric.
        Each new database will contain only the fusion metrics for a single metric.
        """
        for metric_name, metric_id in self.all_metrics_names.items():
            # 1. Create a new Database instance for the metric
            output_db_path = Path(output_dir) / f"{metric_name}.db"
            new_db = Database(
                output_dir, output_db_path.name, 
                metrics=[metric_name],
                algorithms=self.all_algorithms_names.keys(),
                mode='compute'
            )

            # 2. Fetch all fusion metrics for the current metric from the current database
            fusion_metrics = self.select_values(
                metrics=metric_name,
                return_algorithm_id=True,
                return_img_id=True,
            )

            # 3. Insert the fetched metrics into the new database
            # for row in fusion_metrics:
            #     alg_id, img_id, value = row
            #     new_db.update_values(
            #         algorithm=new_db.all_algorithms_ids[alg_id],
            #         metrics=metric_name,
            #         img_id=img_id,
            #         value=value
            #     ) # Toooo slow...
            bulk_data = []
            for row in fusion_metrics:
                alg_id, img_id, value = row
                alg_name = self.all_algorithms_ids[alg_id]
                bulk_data.append((new_db.all_algorithms_names[alg_name], metric_id, img_id, value))

            # 4. Insert the fetched metrics into the new database using bulk insert
            new_db.cursor.executemany('''
            INSERT OR REPLACE INTO fusion_metrics (algorithm_id, metric_id, image_id, value)
            VALUES (?, ?, ?, ?)
            ''', bulk_data)

            # 4. Commit changes to the new database
            new_db.commit()

        print("splited by metrics completed")

    def split_by_algorithm(self, output_dir: str) -> None:
        """
        Split the current database into separate databases for each algorithm.
        Each new database will contain only the fusion metrics for a single algorithm.
        """
        # Ensure the output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Iterate over each algorithm
        for algorithm_name, algorithm_id in self.all_algorithms_names.items():
            # 1. Create a new Database instance for the algorithm
            output_db_path = Path(output_dir) / f"{algorithm_name}.db"
            new_db = Database(
                output_dir, output_db_path.name, 
                metrics=self.all_metrics_names.keys(),
                algorithms=[algorithm_name,],
                mode='compute'
            )

            # 2. Fetch all fusion metrics for the current algorithm from the current database
            fusion_metrics = self.select_values(
                algorithm=algorithm_name,
                return_metric_id=True,
                return_img_id=True,
            )

            # 3. Prepare data for bulk insertion
            bulk_data = []
            for row in fusion_metrics:
                metric_id, img_id, value = row
                metric_name = self.all_metrics_ids[metric_id]
                bulk_data.append((algorithm_id, new_db.all_metrics_names[metric_name], img_id, value))

            # 4. Insert the fetched metrics into the new database using bulk insert
            new_db.cursor.executemany('''
            INSERT OR REPLACE INTO fusion_metrics (algorithm_id, metric_id, image_id, value)
            VALUES (?, ?, ?, ?)
            ''', bulk_data)

            # 5. Commit changes to the new database
            new_db.commit()

        print("Split by algorithms completed")