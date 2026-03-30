# api_data_import.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from database import db
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
class APIDataImporter:
    def __init__(self, config=None):
        self.config = config or {}
        self.default_columns = {
            'date': 'date',
            'time': 'time',
            'window_id': 'window_id',
            'student_id': 'student_id',
            'quantity': 'quantity'
        }
    
    def import_from_csv(self, filepath, column_mapping=None):
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                return {'success': False, 'error': 'CSV文件为空'}
            if column_mapping:
                df = df.rename(columns=column_mapping)
            required_cols = ['date', 'time', 'window_id', 'student_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return {'success': False, 'error': f'缺少必要列: {missing_cols}'}
            df = self._clean_data(df)
            df = self._calculate_payment_gaps(df)
            result = self._import_to_database(df)
            return {
                'success': True,
                'imported_count': result['count'],
                'date_range': result['date_range'],
                'windows': result['windows'],
                'students': result['students']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def import_from_json(self, json_data, column_mapping=None):
        try:
            # 解析JSON
            if isinstance(json_data, str):
                data = json.loads(json_data)
            else:
                data = json_data
            df = pd.DataFrame(data)
            if df.empty:
                return {'success': False, 'error': 'JSON数据为空'}
            if column_mapping:
                df = df.rename(columns=column_mapping)
            df = self._clean_data(df)
            df = self._calculate_payment_gaps(df)
            result = self._import_to_database(df)
            return {
                'success': True,
                'imported_count': result['count'],
                'date_range': result['date_range'],
                'windows': result['windows'],
                'students': result['students']
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def import_from_api(self, api_url, headers=None, params=None, column_mapping=None):
        if not HAS_REQUESTS:
            return {'success': False, 'error': 'requests模块未安装，无法使用API导入功能'}
        
        try:
            response = requests.get(api_url, headers=headers, params=params, timeout=10)
            
            if response.status_code != 200:
                return {'success': False, 'error': f'API请求失败: {response.status_code}'}
            data = response.json()
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            return self.import_from_json(data, column_mapping)
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def import_pos_data(self, pos_data):
        try:
            processed_data = []
            
            for record in pos_data:
                timestamp = record.get('timestamp') or record.get('time')
                if timestamp:
                    if isinstance(timestamp, str):
                        dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    else:
                        dt = timestamp
                    
                    date = dt.strftime('%Y-%m-%d')
                    time = dt.strftime('%H:%M')
                else:
                    continue
                processed_data.append({
                    'date': date,
                    'time': time,
                    'window_id': int(record.get('window_id', 1)),
                    'student_id': str(record.get('student_id', '')),
                    'quantity': int(record.get('quantity', 1))
                })
            if not processed_data:
                return {'success': False, 'error': '没有有效的POS数据'}
            df = pd.DataFrame(processed_data)
            df = self._calculate_payment_gaps(df)
            result = self._import_to_database(df)
            
            return {
                'success': True,
                'imported_count': result['count'],
                'date_range': result['date_range'],
                'windows': result['windows'],
                'students': result['students']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _clean_data(self, df):
        df = df.dropna(subset=['date', 'time', 'window_id', 'student_id'])
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['time'] = df['time'].apply(self._standardize_time)
        df['window_id'] = df['window_id'].astype(int)
        df['student_id'] = df['student_id'].astype(str)
        df['quantity'] = df.get('quantity', 1).astype(int)
        df = df.drop_duplicates()
        
        return df
    
    def _standardize_time(self, time_val):
        try:
            if isinstance(time_val, str):
                for fmt in ['%H:%M:%S', '%H:%M', '%I:%M %p', '%I:%M:%S %p']:
                    try:
                        t = datetime.strptime(time_val.strip(), fmt)
                        return t.strftime('%H:%M')
                    except:
                        continue
            elif isinstance(time_val, datetime):
                return time_val.strftime('%H:%M')
        except:
            pass
        
        return '12:00'  # 默认时间
    
    def _calculate_payment_gaps(self, df):
        df = df.sort_values(['date', 'window_id', 'time'])
        df['payment_time_gap'] = None
        for (date, window), group in df.groupby(['date', 'window_id']):
            if len(group) < 2:
                continue
            times = []
            for _, row in group.iterrows():
                try:
                    h, m = map(int, row['time'].split(':'))
                    times.append(h * 3600 + m * 60)
                except:
                    times.append(0)
            gaps = [0] 
            for i in range(1, len(times)):
                gap = times[i] - times[i-1]
                if gap > 600:
                    gap = 0
                gaps.append(gap)
            df.loc[group.index, 'payment_time_gap'] = gaps
        
        return df
    
    def _import_to_database(self, df):
        records = []
        for _, row in df.iterrows():
            records.append((
                row['date'],
                row['time'],
                row['window_id'],
                row['student_id'],
                row['quantity'],
                row.get('payment_time_gap', 0) or 0
            ))
        db.insert_orders_batch(records)
        
        return {
            'count': len(records),
            'date_range': (df['date'].min(), df['date'].max()),
            'windows': df['window_id'].unique().tolist(),
            'students': df['student_id'].unique().tolist()
        }
    
    def validate_data_quality(self, days=7):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = db.get_all_orders_as_dataframe()
        
        if df.empty:
            return {
                'status': 'error',
                'message': '数据库中没有数据',
                'total_records': 0
            }
        
        df['date'] = pd.to_datetime(df['date'])
        recent_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        total_records = len(df)
        recent_records = len(recent_df)
        daily_counts = recent_df.groupby(recent_df['date'].dt.date).size()
        window_counts = recent_df['window_id'].value_counts().to_dict()
        unique_students = recent_df['student_id'].nunique()
        missing_time_gaps = recent_df['payment_time_gap'].isna().sum()
        
        return {
            'status': 'ok',
            'total_records': total_records,
            'recent_records': recent_records,
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'daily_average': round(daily_counts.mean(), 1) if len(daily_counts) > 0 else 0,
            'window_distribution': window_counts,
            'unique_students': unique_students,
            'data_quality': {
                'missing_time_gaps': int(missing_time_gaps),
                'completeness_rate': round((1 - missing_time_gaps / len(recent_df)) * 100, 2) if len(recent_df) > 0 else 0
            }
        }
    
    def export_data(self, start_date=None, end_date=None, format='csv', filepath=None):
        df = db.get_all_orders_as_dataframe()
        
        if df.empty:
            return {'success': False, 'error': '没有数据可导出'}
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        if df.empty:
            return {'success': False, 'error': '指定日期范围内没有数据'}
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'data/export_{timestamp}.{format}'
        try:
            if format == 'csv':
                df.to_csv(filepath, index=False)
            elif format == 'json':
                df.to_json(filepath, orient='records', force_ascii=False, indent=2)
            elif format == 'excel':
                df.to_excel(filepath, index=False)
            else:
                return {'success': False, 'error': f'不支持的格式: {format}'}
            
            return {
                'success': True,
                'filepath': filepath,
                'record_count': len(df),
                'format': format
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
api_importer = APIDataImporter()

if __name__ == "__main__":
    print("API数据导入模块加载成功")
    quality = api_importer.validate_data_quality()
    print("\n 数据质量报告:")
    print(json.dumps(quality, indent=2, ensure_ascii=False))
