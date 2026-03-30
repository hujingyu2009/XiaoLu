# crowd_analyzer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from database import db

class CrowdDensityAnalyzer:
    def __init__(self):
        self.time_slots = {
            '早餐': ('06:00', '09:00'),
            '午餐': ('11:00', '14:00'),
            '晚餐': ('17:00', '20:00')
        }
        self.density_levels = {
            '低': 0,
            '中': 1,
            '高': 2,
            '极高': 3
        }
    
    def analyze_crowd_density(self, date, window_id=None):
        orders_df = db.get_orders_by_date(date)
        
        if orders_df.empty:
            return {
                'date': date,
                'total_orders': 0,
                'density_by_time': {},
                'average_time_gap': 0
            }
        if window_id is not None:
            orders_df = orders_df[orders_df['window_id'] == window_id]
        orders_df = orders_df.sort_values('time')
        total_orders = len(orders_df)
        def time_to_seconds(time_str):
            try:
                h, m = map(int, time_str.split(':'))
                return h * 3600 + m * 60
            except:
                return 0
        
        if total_orders > 1:
            orders_df['time_seconds'] = orders_df['time'].apply(time_to_seconds)
            time_gaps = []
            for i in range(1, len(orders_df)):
                gap = orders_df['time_seconds'].iloc[i] - orders_df['time_seconds'].iloc[i-1]
                if gap > 0: 
                    time_gaps.append(gap)
            avg_time_gap = np.mean(time_gaps) if time_gaps else 0
        else:
            avg_time_gap = 0
        density_by_time = {}
        
        for meal_type, (start_time, end_time) in self.time_slots.items():
            meal_orders = self._filter_orders_by_time(orders_df, start_time, end_time)
            if not meal_orders.empty:
                order_count = len(meal_orders)
                if order_count > 1:
                    meal_orders['time_seconds'] = meal_orders['time'].apply(time_to_seconds)
                    meal_time_gaps = []
                    for i in range(1, len(meal_orders)):
                        gap = meal_orders['time_seconds'].iloc[i] - meal_orders['time_seconds'].iloc[i-1]
                        if gap > 0:
                            meal_time_gaps.append(gap)
                    time_gap = np.mean(meal_time_gaps) if meal_time_gaps else 0
                else:
                    time_gap = 0
                density_level = self._calculate_density_level(order_count, time_gap)
                
                density_by_time[meal_type] = {
                    'order_count': order_count,
                    'avg_time_gap': round(time_gap, 2),
                    'density_level': density_level
                }
            else:
                density_by_time[meal_type] = {
                    'order_count': 0,
                    'avg_time_gap': 0,
                    'density_level': '低'
                }
        
        return {
            'date': date,
            'total_orders': total_orders,
            'average_time_gap': round(avg_time_gap, 2),
            'density_by_time': density_by_time,
            'window_id': window_id
        }
    
    def predict_crowd_for_preparation(self, date):
        history_days = 7
        end_date = datetime.strptime(date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=history_days)
        
        historical_data = []
        
        print(f"计算历史人流数据（最近{history_days}天）...")
        for i in range(history_days):
            hist_date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
            analysis = self.analyze_crowd_density(hist_date)
            historical_data.append(analysis)
            print(f"  {hist_date}: 订单数={analysis['total_orders']}, 平均间隔={analysis['average_time_gap']}秒")
        if historical_data:
            avg_orders = np.mean([d['total_orders'] for d in historical_data])
            avg_time_gap = np.mean([d['average_time_gap'] for d in historical_data])
            time_slot_avgs = {}
            for meal_type in self.time_slots:
                meal_orders = [d['density_by_time'].get(meal_type, {}).get('order_count', 0) for d in historical_data]
                time_slot_avgs[meal_type] = np.mean(meal_orders)
        else:
            avg_orders = 0
            avg_time_gap = 0
            time_slot_avgs = {meal: 0 for meal in self.time_slots}
        print(f"预测结果: 总订单数={round(avg_orders)}, 平均间隔={round(avg_time_gap, 2)}秒")
        return {
            'predicted_date': date,
            'predicted_total_orders': round(avg_orders),
            'predicted_avg_time_gap': round(avg_time_gap, 2),
            'predicted_by_time': time_slot_avgs
        }
    
    def get_crowd_trend(self, days=7):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        trend_data = []
        for i in range(days):
            date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
            analysis = self.analyze_crowd_density(date)
            trend_data.append({
                'date': date,
                'total_orders': analysis['total_orders'],
                'average_time_gap': analysis['average_time_gap'],
                'density_breakdown': analysis['density_by_time']
            })
        
        return {
            'trend_data': trend_data,
            'date_range': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            }
        }
    
    def generate_crowd_features(self, date, window_id):
        analysis = self.analyze_crowd_density(date, window_id)
        history_features = self._calculate_history_features(window_id, days=3)
        features = {
            'avg_time_gap': analysis['average_time_gap'],
            'total_orders': analysis['total_orders'],
            'crowd_density': self._get_density_score(analysis),
            **history_features
        }
        return features
    def _filter_orders_by_time(self, df, start_time, end_time):
        def time_to_minutes(time_str):
            try:
                h, m = map(int, time_str.split(':'))
                return h * 60 + m
            except:
                return 0
        start_min = time_to_minutes(start_time)
        end_min = time_to_minutes(end_time)
        df = df.copy()
        df['time_minutes'] = df['time'].apply(time_to_minutes)
        return df[(df['time_minutes'] >= start_min) & (df['time_minutes'] <= end_min)]
    def _calculate_density_level(self, order_count, time_gap):
        if order_count == 0:
            return '低'
        density_score = order_count / (time_gap + 1)
        if density_score < 0.1:
            return '低'
        elif density_score < 0.3:
            return '中'
        elif density_score < 0.6:
            return '高'
        else:
            return '极高'
    def _calculate_history_features(self, window_id, days=3):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        history_data = []
        for i in range(days):
            date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
            analysis = self.analyze_crowd_density(date, window_id)
            history_data.append(analysis)
        if history_data:
            return {
                'avg_orders_3d': np.mean([d['total_orders'] for d in history_data]),
                'avg_time_gap_3d': np.mean([d['average_time_gap'] for d in history_data]),
                'max_orders_3d': np.max([d['total_orders'] for d in history_data])
            }
        else:
            return {
                'avg_orders_3d': 0,
                'avg_time_gap_3d': 0,
                'max_orders_3d': 0
            }
    def _get_density_score(self, analysis):
        total_orders = analysis['total_orders']
        avg_gap = analysis['average_time_gap']
        if total_orders == 0:
            return 0
        return total_orders / (avg_gap + 1)


if __name__ == "__main__":
    analyzer = CrowdDensityAnalyzer()
    today = datetime.now().strftime('%Y-%m-%d')
    print("人流密度分析器模块加载成功")
    print(f"分析 {today} 的人流密度:")
    analysis = analyzer.analyze_crowd_density(today)
    print(f"总订单数: {analysis['total_orders']}")
    print(f"平均付款间隔: {analysis['average_time_gap']}秒")
    print("各时段人流密度:")
    for meal_type, data in analysis['density_by_time'].items():
        print(f"  {meal_type}: {data['density_level']} (订单数: {data['order_count']}, 平均间隔: {data['avg_time_gap']}秒)")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    prediction = analyzer.predict_crowd_for_preparation(tomorrow)
    print(f"预测 {tomorrow} 的人流:")
    print(f"预计总订单数: {prediction['predicted_total_orders']}")
