# database.py
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json

class CanteenDatabase:
    def __init__(self, db_path='data/canteen.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                window_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                payment_time_gap REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_diet (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                date TEXT NOT NULL,
                window_id INTEGER NOT NULL,
                meal_type TEXT NOT NULL,
                nutrition_score REAL,
                UNIQUE(student_id, date, meal_type)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS windows (
                window_id INTEGER PRIMARY KEY,
                window_name TEXT NOT NULL,
                window_type TEXT NOT NULL,
                description TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crowd_density (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                time_slot TEXT NOT NULL,
                window_id INTEGER NOT NULL,
                avg_time_gap REAL,
                density_level TEXT,
                order_count INTEGER
            )
        ''')
        self._init_windows(cursor)
        conn.commit()
        conn.close()
    def _init_windows(self, cursor):
        windows = [
            (1, '主食窗口', 'staple', '米饭、面条等主食'),
            (2, '小吃窗口', 'snack', '小吃、点心'),
            (3, '汤品窗口', 'soup', '汤、饮品'),
            (4, '快餐窗口', 'fastfood', '快餐、套餐'),
            (5, '特色窗口', 'special', '特色菜品')
        ]
        cursor.executemany('''
            INSERT OR IGNORE INTO windows (window_id, window_name, window_type, description)
            VALUES (?, ?, ?, ?)
        ''', windows)
    
    def insert_order(self, date, time, window_id, student_id, quantity, payment_time_gap=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO orders (date, time, window_id, student_id, quantity, payment_time_gap)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (date, time, window_id, student_id, quantity, payment_time_gap))
        
        conn.commit()
        conn.close()
    
    def insert_orders_batch(self, orders_data):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT INTO orders (date, time, window_id, student_id, quantity, payment_time_gap)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', orders_data)
        conn.commit()
        conn.close()
    
    def get_student_recent_diet(self, student_id, days=7):
        conn = sqlite3.connect(self.db_path)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        query = '''
            SELECT o.date, o.window_id, w.window_type, w.window_name,
                   COUNT(*) as visit_count, SUM(o.quantity) as total_quantity
            FROM orders o
            JOIN windows w ON o.window_id = w.window_id
            WHERE o.student_id = ? AND o.date BETWEEN ? AND ?
            GROUP BY o.date, o.window_id
            ORDER BY o.date DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(student_id, start_date, end_date))
        conn.close()
        
        return df
    
    def get_window_daily_stats(self, window_id=None, days=7):
        conn = sqlite3.connect(self.db_path)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        if window_id:
            query = '''
                SELECT date, window_id, 
                       COUNT(*) as order_count,
                       SUM(quantity) as total_quantity,
                       AVG(payment_time_gap) as avg_time_gap
                FROM orders
                WHERE window_id = ? AND date BETWEEN ? AND ?
                GROUP BY date
                ORDER BY date
            '''
            df = pd.read_sql_query(query, conn, params=(window_id, start_date, end_date))
        else:
            query = '''
                SELECT date, window_id,
                       COUNT(*) as order_count,
                       SUM(quantity) as total_quantity,
                       AVG(payment_time_gap) as avg_time_gap
                FROM orders
                WHERE date BETWEEN ? AND ?
                GROUP BY date, window_id
                ORDER BY date, window_id
            '''
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        return df
    
    def get_crowd_density_by_time(self, date, window_id=None):
        conn = sqlite3.connect(self.db_path)
        if window_id:
            query = '''
                SELECT time, payment_time_gap, quantity
                FROM orders
                WHERE date = ? AND window_id = ?
                ORDER BY time
            '''
            df = pd.read_sql_query(query, conn, params=(date, window_id))
        else:
            query = '''
                SELECT time, payment_time_gap, quantity, window_id
                FROM orders
                WHERE date = ?
                ORDER BY time
            '''
            df = pd.read_sql_query(query, conn, params=(date,))
        conn.close()
        return df
    
    def get_orders_by_date(self, date):
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT * FROM orders
            WHERE date = ?
            ORDER BY time
        '''
        df = pd.read_sql_query(query, conn, params=(date,))
        conn.close()
        return df
    
    def get_all_orders_as_dataframe(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM orders', conn)
        conn.close()
        return df
    
    def get_student_list(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT student_id FROM orders')
        students = [row[0] for row in cursor.fetchall()]
        conn.close()
        return students
    
    def get_date_range(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT MIN(date), MAX(date) FROM orders')
        result = cursor.fetchone()
        conn.close()
        return result[0], result[1]
    
    def clear_old_data(self, days_to_keep=30):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
        cursor.execute('DELETE FROM orders WHERE date < ?', (cutoff_date,))
        cursor.execute('DELETE FROM student_diet WHERE date < ?', (cutoff_date,))
        conn.commit()
        conn.close()

db = CanteenDatabase()

if __name__ == "__main__":
    print("数据库初始化成功")
    print(f"学生列表: {db.get_student_list()}")
