# canteen_system.py
import os
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("psutil模块未安装，无法监控系统资源")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from database import db
from universal_predictor import UniversalCanteenPredictor
from smart_recommendation import SmartDietRecommendation
from crowd_analyzer import CrowdDensityAnalyzer
from weather_calendar_api import weather_calendar_api
from api_data_import import api_importer
from data_generator import CanteenDataGenerator
from visualization import CanteenVisualization

def get_system_resources():
    if HAS_PSUTIL:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_used_mb = round(memory.used / 1024 / 1024, 2)
        memory_total_mb = round(memory.total / 1024 / 1024, 2)
        memory_percent = memory.percent
        return {
            'memory_used_mb': memory_used_mb,
            'memory_total_mb': memory_total_mb,
            'memory_percent': memory_percent,
            'cpu_percent': cpu_percent,
            'system_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    else:
        return {
            'memory_used_mb': 0,
            'memory_total_mb': 0,
            'memory_percent': 0,
            'cpu_percent': 0,
            'system_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

class SmartCanteenSystem:
    
    def __init__(self):
        self.predictor = UniversalCanteenPredictor()
        self.recommender = SmartDietRecommendation()
        self.crowd_analyzer = CrowdDensityAnalyzer()
        self.data_generator = CanteenDataGenerator()
        
        self.is_model_trained = False
        self.last_training_date = None
    
    def initialize_with_sample_data(self, days=30):
        """使用模拟数据初始化系统"""
        print("="*60)
        print("初始化食堂智能系统（作者：刘昊航、胡璟宇。指导老师：黄晓露）")
        print("="*60)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        print("生成模拟数据...")
        data = self.data_generator.generate_all_data(start_date, days)
        print("导入数据到数据库...")
        import sqlite3
        conn = sqlite3.connect('data/canteen.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM orders')
        conn.commit()
        conn.close()
        print("已清空旧订单数据")
        records = []
        for _, row in data['orders'].iterrows():
            records.append((
                row['date'], row['time'], row['window_id'],
                row['student_id'], row['quantity'], 0
            ))
        db.insert_orders_batch(records)
        print(f"导入 {len(records)} 条订单记录")
        print("训练预测模型...")
        self.train_prediction_model()
        print("系统初始化完成！")
        resources = get_system_resources()
        print(f"系统资源使用: 内存 {resources['memory_used_mb']}MB/{resources['memory_total_mb']}MB ({resources['memory_percent']}%), CPU {resources['cpu_percent']}%")
        return {
            'orders_count': len(records),
            'date_range': (data['orders']['date'].min(), data['orders']['date'].max()),
            'students': data['orders']['student_id'].nunique(),
            'windows': data['orders']['window_id'].nunique()
        }
    
    def train_prediction_model(self):
        """训练订单预测模型"""
        daily_orders = db.get_window_daily_stats(days=30)
        if daily_orders.empty:
            print("没有数据可供训练")
            return False
        daily_orders = daily_orders.rename(columns={
            'total_quantity': 'total_orders',
            'avg_time_gap': 'avg_time_gap'
        })

        date_range = daily_orders['date'].unique()
        if len(date_range) < 7:
            print(f"数据不足7天（当前{len(date_range)}天），无法训练模型")
            return False
        start_date = min(date_range)
        end_date = max(date_range)
        try:
            weather_df = weather_calendar_api.get_weather_forecast(days=len(date_range))
            holiday_df = weather_calendar_api.get_holiday_range(start_date, end_date)
        except Exception as e:
            print(f"API调用失败，使用模拟数据: {str(e)}")
            weather_df = self.data_generator.generate_weather_data(start_date, len(date_range))
            holiday_df = self.data_generator.generate_holiday_data(start_date, len(date_range))
        print("准备特征数据...")
        prepared_data = self.predictor.prepare_features(
            daily_orders, weather_df, holiday_df
        )
        print("训练模型...")
        results = self.predictor.train(prepared_data)
        self.is_model_trained = True
        self.last_training_date = datetime.now()
        print("模型训练完成")
        print("训练结果:")
        for window_id, stats in results.items():
            print(f"  窗口 {window_id}: MAPE={stats['mape_mean']:.2f}%, 样本数={stats['samples']}")
        return True
    
    def predict_next_day_orders(self, prediction_date=None):
        """预测下一天各窗口订单量"""
        if not self.is_model_trained:
            print("模型尚未训练，请先训练模型")
            return None
        
        if prediction_date is None:
            prediction_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        weather_info = weather_calendar_api.get_weather_forecast(days=3)
        holiday_info = weather_calendar_api.get_holiday_info(prediction_date)
        orders_df = db.get_all_orders_as_dataframe()
        daily_orders = orders_df.groupby(['date', 'window_id']).agg({
            'quantity': 'sum'
        }).reset_index()
        daily_orders.columns = ['date', 'window_id', 'total_orders']
        pred_dt = datetime.strptime(prediction_date, '%Y-%m-%d')
        
        predictions = {}
        
        for window_id in [1, 2, 3, 4, 5]:
            window_data = daily_orders[daily_orders['window_id'] == window_id]
            if len(window_data) < 7:
                predictions[window_id] = {
                    'predicted_orders': None,
                    'confidence': 0,
                    'reason': '历史数据不足'
                }
                continue
            recent_orders = window_data.sort_values('date').tail(7)['total_orders'].tolist()
            weather_row = weather_info[weather_info['date'] == prediction_date]
            if not weather_row.empty:
                weather = weather_row.iloc[0]['weather']
                temperature = weather_row.iloc[0]['temperature']
            else:
                weather = '晴'
                temperature = 25.0
            features = {}
            features['day_of_week'] = pred_dt.weekday()
            features['is_weekend'] = 1 if pred_dt.weekday() >= 5 else 0
            features['day_of_month'] = pred_dt.day
            features['month'] = pred_dt.month
            features['day_of_week_sin'] = np.sin(2 * np.pi * pred_dt.weekday() / 7)
            features['day_of_week_cos'] = np.cos(2 * np.pi * pred_dt.weekday() / 7)
            features['month_sin'] = np.sin(2 * np.pi * pred_dt.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * pred_dt.month / 12)
            features['is_holiday'] = holiday_info.get('is_holiday', 0)
            features['avg_7d'] = np.mean(recent_orders)
            features['std_7d'] = np.std(recent_orders)
            features['max_7d'] = max(recent_orders)
            features['min_7d'] = min(recent_orders)
            features['median_7d'] = np.median(recent_orders)
            features['avg_3d'] = np.mean(recent_orders[-3:])
            features['avg_14d'] = np.mean(recent_orders) if len(window_data) >= 14 else np.mean(recent_orders)
            features['trend_3d'] = recent_orders[-1] - recent_orders[-4] if len(recent_orders) >= 4 else 0
            dow_orders = [o for d, o in zip(window_data['date'], window_data['total_orders']) 
                         if pd.to_datetime(d).weekday() == pred_dt.weekday()]
            features['dow_avg'] = np.mean(dow_orders) if dow_orders else np.mean(recent_orders)
            weather_map = {'晴': 0, '多云': 1, '阴': 2, '小雨': 3, '中雨': 4, '大雨': 5}
            features['weather_encoded'] = weather_map.get(weather, 0)
            features['temperature'] = temperature
            features['temp_normalized'] = 0
            features['temp_high'] = 1 if temperature > 28 else 0
            features['temp_low'] = 1 if temperature < 10 else 0
            features['avg_time_gap'] = 0
            features['crowd_density'] = 0
            features['yoy_growth'] = 0
            try:
                pred_orders = self.predictor.predict(prediction_date, window_id, features)
                
                predictions[window_id] = {
                    'predicted_orders': pred_orders,
                    'confidence': max(0, 100 - self.predictor.training_stats.get(window_id, {}).get('mape_mean', 20)),
                    'weather': weather,
                    'temperature': temperature,
                    'is_holiday': holiday_info.get('is_holiday', 0)
                }
            except Exception as e:
                predictions[window_id] = {
                    'predicted_orders': None,
                    'confidence': 0,
                    'reason': str(e)
                }
        return {
            'prediction_date': prediction_date,
            'predictions': predictions,
            'weather': weather,
            'holiday_info': holiday_info
        }
    
    def get_student_recommendation(self, student_id, weather=None, temperature=None):
        """获取学生饮食推荐"""
        return self.recommender.get_recommendation(student_id, weather, temperature)
    
    def get_crowd_analysis(self, date=None, window_id=None):
        """获取人流密度分析"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        return self.crowd_analyzer.analyze_crowd_density(date, window_id)
    
    def get_preparation_plan(self, date=None):
        """获取备餐计划"""
        if date is None:
            date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        prediction = self.predict_next_day_orders(date)

        crowd_analysis = self.crowd_analyzer.predict_crowd_for_preparation(date)
        
        preparation_plan = {}
        
        for window_id in [1, 2, 3, 4, 5]:
            pred_data = prediction['predictions'].get(window_id, {})
            crowd_data = crowd_analysis
            
            predicted_orders = pred_data.get('predicted_orders', 0)
            
            preparation_plan[window_id] = {
                'predicted_orders': predicted_orders,
                'suggested_preparation': int(predicted_orders * 1.1),
                'confidence': pred_data.get('confidence', 0)
            }
        return {
            'date': date,
            'weather': prediction.get('weather'),
            'holiday': prediction.get('holiday_info', {}).get('is_holiday', 0),
            'preparation_plan': preparation_plan
        }
    
    def import_data(self, source_type, **kwargs):
        """导入数据"""
        if source_type == 'csv':
            return api_importer.import_from_csv(**kwargs)
        elif source_type == 'json':
            return api_importer.import_from_json(**kwargs)
        elif source_type == 'api':
            return api_importer.import_from_api(**kwargs)
        elif source_type == 'pos':
            return api_importer.import_pos_data(**kwargs)
        else:
            return {'success': False, 'error': f'不支持的数据源类型: {source_type}'}
    
    def get_system_status(self):
        """获取系统状态"""
        data_quality = api_importer.validate_data_quality()

        model_status = {
            'is_trained': self.is_model_trained,
            'last_training': self.last_training_date.strftime('%Y-%m-%d %H:%M') if self.last_training_date else None,
            'training_stats': self.predictor.training_stats if self.is_model_trained else {}
        }

        system_resources = get_system_resources()
        
        return {
            'data_quality': data_quality,
            'model_status': model_status,
            'system_resources': system_resources,
            'system_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def generate_report(self, days=7):
        """生成系统运行报告"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        orders_df = db.get_all_orders_as_dataframe()
        
        if orders_df.empty:
            return {'error': '没有数据'}
        
        orders_df['date'] = pd.to_datetime(orders_df['date'])
        recent_orders = orders_df[
            (orders_df['date'] >= start_date) & 
            (orders_df['date'] <= end_date)
        ]

        total_orders = len(recent_orders)
        unique_students = recent_orders['student_id'].nunique()

        window_stats = recent_orders.groupby('window_id').agg({
            'quantity': 'sum',
            'student_id': 'nunique'
        }).to_dict('index')

        daily_stats = recent_orders.groupby(recent_orders['date'].dt.date).agg({
            'quantity': 'sum'
        }).to_dict()['quantity']
        
        return {
            'report_period': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            },
            'summary': {
                'total_orders': total_orders,
                'unique_students': unique_students,
                'avg_daily_orders': round(total_orders / days, 1)
            },
            'window_statistics': window_stats,
            'daily_statistics': daily_stats
        }

def main():
    """主程序演示"""
    print("\n" + "="*60)
    print("校园食堂智能化系统 - 演示")
    print("="*60)
    print("系统初始化中...")
    resources = get_system_resources()
    print(f"初始系统资源: 内存 {resources['memory_used_mb']}MB, CPU {resources['cpu_percent']}%")
    system = SmartCanteenSystem()
    print("\n开始初始化数据...")
    init_result = system.initialize_with_sample_data(days=30)
    print("初始化数据:")
    print(f"  订单数: {init_result['orders_count']}")
    print(f"  学生数: {init_result['students']}")
    print(f"  窗口数: {init_result['windows']}")
    print(f"  日期范围: {init_result['date_range'][0]} 至 {init_result['date_range'][1]}")
    print("\n" + "-"*60)
    print("预测下一天订单量")
    print("-"*60)
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"预测日期: {tomorrow}")
    prediction = system.predict_next_day_orders(tomorrow)
    weather_info = weather_calendar_api.get_weather_forecast(days=3)
    print(f"天气: {prediction['weather']}")
    if not weather_info.empty and 'temperature' in weather_info.columns:
        print(f"温度: {weather_info.iloc[0]['temperature']}°C")
    print(f"节假日: {'是' if prediction['holiday_info'].get('is_holiday') else '否'}")
    print("各窗口预测订单量:")
    for window_id, pred in prediction['predictions'].items():
        if pred.get('predicted_orders'):
            print(f"  窗口 {window_id}: {pred['predicted_orders']} 份 (置信度: {pred['confidence']:.1f}%)")
        else:
            print(f"  窗口 {window_id}: 无法预测 - {pred.get('reason', '未知原因')}")
        print("\n" + "-"*60)
        print("学生饮食推荐示例")
        print("-"*60)
        sample_students = ['S001', 'S010', 'S050']
        for student_id in sample_students:
            print(f"\n学生 {student_id}:")
            temp = None
            if 'temperature' in weather_info.iloc[0]:
                temp = weather_info.iloc[0]['temperature']
            rec = system.get_student_recommendation(student_id, weather=prediction['weather'], temperature=temp)
            print(f"  营养评分: {rec['nutrition_score']}/100")
            print(f"  推荐窗口: {rec['recommended_windows'][0]['window_name']} (原因: {', '.join(rec['recommended_windows'][0]['reasons'])})")
            print(f"  健康建议: {rec['health_tips'][0]}")
    print("\n" + "-"*60)
    print("今日人流密度分析")
    print("-"*60)
    today = datetime.now().strftime('%Y-%m-%d')
    crowd = system.get_crowd_analysis(today)
    print(f"日期: {crowd['date']}")
    print(f"总订单数: {crowd['total_orders']}")
    print("各时段人流密度:")
    for meal_type, data in crowd['density_by_time'].items():
        print(f"  {meal_type}: {data['density_level']} (订单数: {data['order_count']}, 平均间隔: {data['avg_time_gap']}秒)")
    print("\n" + "-"*60)
    print("明日备餐计划")
    print("-"*60)
    prep = system.get_preparation_plan(tomorrow)
    print(f"日期: {prep['date']}")
    print("各窗口建议备餐量:")
    for window_id, plan in prep['preparation_plan'].items():
        if plan['predicted_orders'] > 0:
            print(f"  窗口 {window_id}: 预测 {plan['predicted_orders']} 份 → 建议备餐 {plan['suggested_preparation']} 份")
    print("\n" + "-"*60)
    print("系统状态")
    print("-"*60)
    status = system.get_system_status()
    print(f"数据状态: {status['data_quality']['status']}")
    print(f"总记录数: {status['data_quality']['total_records']}")
    print(f"模型训练状态: {'已训练' if status['model_status']['is_trained'] else '未训练'}")
    print(f"系统资源使用: 内存 {status['system_resources']['memory_used_mb']}MB/{status['system_resources']['memory_total_mb']}MB ({status['system_resources']['memory_percent']}%), CPU {status['system_resources']['cpu_percent']}%")
    print("\n" + "="*60)
    print("生成系统分析报告")
    print("="*60)
    training_stats = system.predictor.training_stats if system.is_model_trained else None
    CanteenVisualization.generate_all_analysis(training_stats)
    print("\n" + "="*60)
    print("演示完成！小路牛逼！")
    print("="*60)
if __name__ == "__main__":
    main()
