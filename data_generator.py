# data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class CanteenDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.windows = {
            1: {'name': '主食窗口', 'type': 'staple', 'base_orders': 150},
            2: {'name': '小吃窗口', 'type': 'snack', 'base_orders': 80},
            3: {'name': '汤品窗口', 'type': 'soup', 'base_orders': 60},
            4: {'name': '快餐窗口', 'type': 'fastfood', 'base_orders': 120},
            5: {'name': '特色窗口', 'type': 'special', 'base_orders': 90}
        }

        self.student_pool = [f'S{str(i).zfill(3)}' for i in range(1, 201)]
        self.weather_types = ['晴', '多云', '阴', '小雨', '中雨', '大雨']
        self.arma_params = {
            'ar_coef': [0.6, 0.3],  # 自回归系数 phi_1, phi_2
            'ma_coef': [0.4],       # 移动平均系数 theta_1
            'sigma': 0.15           # 白噪声标准差
        }

        self.window_correlation = np.array([
            [1.00, -0.15,  0.05, -0.35, -0.10],  # 主食窗口
            [-0.15,  1.00,  0.10, -0.10, -0.25],  # 小吃窗口
            [0.05,  0.10,  1.00,  0.05,  0.05],  # 汤品窗口
            [-0.35, -0.10,  0.05,  1.00, -0.20],  # 快餐窗口
            [-0.10, -0.25,  0.05, -0.20,  1.00]   # 特色窗口
        ])
        self.order_history = {wid: [] for wid in self.windows.keys()}
        self.error_history = {wid: [] for wid in self.windows.keys()} 
    
    def generate_orders(self, start_date, days=30, noise_level=0.4):
        orders = []
        start = datetime.strptime(start_date, '%Y-%m-%d')
        self._initialize_history()
        growth_factor = 1.0
        growth_rate = 0.05 
        
        for day_offset in range(days):
            current_date = start + timedelta(days=day_offset)
            date_str = current_date.strftime('%Y-%m-%d')
            weekday = current_date.weekday()
            is_weekend = weekday >= 5
            weather = self._generate_weather(current_date)
            temperature = self._generate_temperature(current_date, weather)
            base_factors = self._calculate_base_factors(is_weekend, weather, temperature, noise_level)
            for window_id in base_factors:
                base_factors[window_id] *= growth_factor
            
            final_orders_dict = self._apply_arma_and_correlation(base_factors)
            for window_id, final_orders in final_orders_dict.items():
                min_orders = int(self.windows[window_id]['base_orders'] * 0.8)
                final_orders = max(min_orders, int(final_orders))  
                self.order_history[window_id].append(final_orders)
                daily_orders = self._generate_daily_orders(
                    date_str, window_id, final_orders, is_weekend, weekday
                )
                orders.extend(daily_orders)
            growth_factor *= (1 + growth_rate)
            if is_weekend:
                growth_factor *= 0.95
        
        df = pd.DataFrame(orders)
        return df
    
    def _initialize_history(self):
        for window_id in self.windows.keys():
            base = self.windows[window_id]['base_orders']
            self.order_history[window_id] = [
                int(base * np.random.uniform(0.8, 1.2)) for _ in range(7)
            ]
            self.error_history[window_id] = [0.0] * 7 
    
    def _calculate_base_factors(self, is_weekend, weather, temperature, noise_level):
        base_factors = {}
        for window_id, config in self.windows.items():
            base_orders = config['base_orders']
            if is_weekend:
                weekend_factor = np.random.uniform(0.5, 0.9)
            else:
                weekend_factor = np.random.uniform(0.8, 1.2)
        
            weather_factor = self._get_weather_factor(weather)
            temp_factor = self._get_temperature_factor(temperature)
            noise = np.random.normal(1, noise_level)
            special_event_factor = 1.0
            if np.random.random() < 0.1:
                special_event_factor = np.random.uniform(0.7, 1.5)
            base_factors[window_id] = base_orders * weekend_factor * weather_factor * temp_factor * noise * special_event_factor
        
        return base_factors
    
    def _apply_arma_and_correlation(self, base_factors):
        window_ids = list(self.windows.keys())
        n_windows = len(window_ids)
        arma_adjusted = {}
        for window_id in window_ids:
            history = self.order_history[window_id]
            errors = self.error_history[window_id]
            ar_component = 0
            ar_coef = self.arma_params['ar_coef']
            for i, phi in enumerate(ar_coef):
                if len(history) > i:
                    if history[-(i+1)] > 0:
                        relative_change = (history[-1] - history[-(i+1)]) / history[-(i+1)]
                        ar_component += phi * relative_change
            ma_component = 0
            ma_coef = self.arma_params['ma_coef']
            for i, theta in enumerate(ma_coef):
                if len(errors) > i:
                    ma_component += theta * errors[-(i+1)]
            white_noise = np.random.normal(0, self.arma_params['sigma'])
            arma_factor = 1 + ar_component + ma_component + white_noise
            arma_factor = max(0.5, min(1.5, arma_factor)) 
            arma_adjusted[window_id] = base_factors[window_id] * arma_factor
            current_error = white_noise
            self.error_history[window_id].append(current_error)
        window_orders = np.array([arma_adjusted[wid] for wid in window_ids])
        mean_orders = np.mean(window_orders)
        relative_orders = (window_orders - mean_orders) / mean_orders 
        final_orders = {}
        for i, window_id in enumerate(window_ids):
            competition_effect = 0
            for j, other_id in enumerate(window_ids):
                if i != j:
                    correlation = self.window_correlation[i, j]
                    competition_effect += correlation * relative_orders[j]
            competition_factor = 1 + competition_effect * 0.3  
            competition_factor = max(0.7, min(1.3, competition_factor))
            
            final_orders[window_id] = arma_adjusted[window_id] * competition_factor
        
        return final_orders
    
    def _generate_weather(self, date):
        day_of_year = date.timetuple().tm_yday
        if 60 <= day_of_year <= 150:  # 春季
            probs = [0.25, 0.35, 0.20, 0.15, 0.04, 0.01]
        elif 151 <= day_of_year <= 240:  # 夏季
            probs = [0.40, 0.30, 0.15, 0.10, 0.04, 0.01]
        elif 241 <= day_of_year <= 330:  # 秋季
            probs = [0.35, 0.35, 0.15, 0.10, 0.04, 0.01]
        else:  # 冬季
            probs = [0.30, 0.25, 0.25, 0.10, 0.08, 0.02]
        return np.random.choice(self.weather_types, p=probs)
    
    def _get_weather_description(self, temperature):
        if temperature < 10:
            return '冷'
        elif temperature < 15:
            return '较冷'
        elif temperature < 25:
            return '舒适'
        elif temperature < 30:
            return '较热'
        else:
            return '热'
    
    def _generate_temperature(self, date, weather):
        day_of_year = date.timetuple().tm_yday
        if 60 <= day_of_year <= 150:  # 春季
            base_temp = 18
        elif 151 <= day_of_year <= 240:  # 夏季
            base_temp = 28
        elif 241 <= day_of_year <= 330:  # 秋季
            base_temp = 16
        else:  # 冬季
            base_temp = 3
        
        weather_adjust = {
            '晴': 3, '多云': 1, '阴': -1, '小雨': -3, '中雨': -5, '大雨': -7
        }
        temp = base_temp + weather_adjust.get(weather, 0) + np.random.normal(0, 2)
        return round(temp, 1)
    
    def _get_weather_factor(self, weather):
        factors = {
            '晴': 1.1,
            '多云': 1.0,
            '阴': 0.95,
            '小雨': 0.85,
            '中雨': 0.75,
            '大雨': 0.65
        }
        return factors.get(weather, 1.0)
    
    def _get_temperature_factor(self, temperature):
        if temperature < 0:
            return 0.9
        elif temperature < 10:
            return 0.95
        elif temperature < 25:
            return 1.0
        elif temperature < 35:
            return 0.95
        else:
            return 0.85
    
    def _generate_daily_orders(self, date, window_id, num_orders, is_weekend, weekday):
        orders = []
        if is_weekend:
            peak_hours = [
                (7, 9, np.random.uniform(0.15, 0.25)),    # 早餐
                (11, 13, np.random.uniform(0.30, 0.40)), # 午餐
                (17, 19, np.random.uniform(0.25, 0.35)), # 晚餐
                (20, 21, np.random.uniform(0.10, 0.20))  # 夜宵
            ]
        else:
            peak_hours = [
                (7, 8, np.random.uniform(0.10, 0.20)),   # 早餐
                (11, 13, np.random.uniform(0.40, 0.50)), # 午餐
                (17, 19, np.random.uniform(0.30, 0.40)), # 晚餐
                (20, 21, np.random.uniform(0.03, 0.07))  # 夜宵
            ]
        
        total_ratio = sum(ratio for _, _, ratio in peak_hours)
        peak_hours = [(start, end, ratio/total_ratio) for start, end, ratio in peak_hours]
        for start_hour, end_hour, ratio in peak_hours:
            num_in_slot = int(num_orders * ratio)
            for _ in range(num_in_slot):
                hour = random.randint(start_hour, end_hour)
                minute = random.randint(0, 59)
                time_str = f"{hour:02d}:{minute:02d}"
                if random.random() < 0.3:  # 30%的概率随机选择学生，不考虑偏好
                    student_id = random.choice(self.student_pool)
                else:
                    student_id = self._select_student_with_preference(window_id)
                quantity = np.random.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.1, 0.05])
                
                orders.append({
                    'date': date,
                    'time': time_str,
                    'window_id': window_id,
                    'student_id': student_id,
                    'quantity': quantity
                })
        if random.random() < 0.2:  
            extra_orders = random.randint(1, 5)
            for _ in range(extra_orders):
                hour = random.randint(9, 20)
                minute = random.randint(0, 59)
                time_str = f"{hour:02d}:{minute:02d}"
                student_id = random.choice(self.student_pool)
                quantity = np.random.choice([1, 2], p=[0.8, 0.2])
                orders.append({
                    'date': date,
                    'time': time_str,
                    'window_id': window_id,
                    'student_id': student_id,
                    'quantity': quantity
                })
        
        return orders
    
    def _select_student_with_preference(self, window_id):
        if not hasattr(self, 'student_preferences'):
            self.student_preferences = {}
            for student in self.student_pool:
                if random.random() < 0.8:
                    prefs = random.sample(list(self.windows.keys()), k=random.randint(1, 2))
                    self.student_preferences[student] = prefs
                else:
                    self.student_preferences[student] = list(self.windows.keys())
        preferred_students = [
            s for s, prefs in self.student_preferences.items()
            if window_id in prefs
        ]
        if preferred_students and random.random() < 0.7:
            return random.choice(preferred_students)
        else:
            return random.choice(self.student_pool)
    
    def generate_weather_data(self, start_date, days=30):
        start = datetime.strptime(start_date, '%Y-%m-%d')
        weather_data = []
        for day_offset in range(days):
            current_date = start + timedelta(days=day_offset)
            date_str = current_date.strftime('%Y-%m-%d')
            weather = self._generate_weather(current_date)
            temperature = self._generate_temperature(current_date, weather)
            weather_desc = self._get_weather_description(temperature)
            
            weather_data.append({
                'date': date_str,
                'weather': f"{weather} {weather_desc}", 
                'temperature': temperature,
                'weather_desc': weather_desc 
            })
        
        return pd.DataFrame(weather_data)
    
    def generate_holiday_data(self, start_date, days=30):
        start = datetime.strptime(start_date, '%Y-%m-%d')
        holiday_dates = {
            '01-01': '元旦',
            '05-01': '劳动节',
            '06-01': '儿童节',
            '10-01': '国庆节',
            '10-02': '国庆节',
            '10-03': '国庆节',
        }
        holiday_data = []
        for day_offset in range(days):
            current_date = start + timedelta(days=day_offset)
            date_str = current_date.strftime('%Y-%m-%d')
            month_day = date_str[5:]
            weekday = current_date.weekday()
            is_holiday = 0
            if month_day in holiday_dates:
                is_holiday = 1
            elif weekday >= 5: 
                is_holiday = 1
            holiday_data.append({
                'date': date_str,
                'is_holiday': is_holiday
            })
        
        return pd.DataFrame(holiday_data)
    
    def generate_all_data(self, start_date, days=30):
        print(f"正在生成 {days} 天的模拟数据...")
        orders_df = self.generate_orders(start_date, days)
        print(f"生成 {len(orders_df)} 条订单记录")
        weather_df = self.generate_weather_data(start_date, days)
        print(f"生成 {len(weather_df)} 条天气记录")
        holiday_df = self.generate_holiday_data(start_date, days)
        print(f"生成 {len(holiday_df)} 条节假日记录")
        return {
            'orders': orders_df,
            'weather': weather_df,
            'holidays': holiday_df
        }
    
    def save_to_files(self, data, output_dir='data'):
        import os
        os.makedirs(output_dir, exist_ok=True)
        orders_file = f'{output_dir}/sales.csv'
        data['orders'].to_csv(orders_file, index=False)
        print(f"订单数据已保存到 {orders_file}")
        weather_file = f'{output_dir}/weather.csv'
        data['weather'].to_csv(weather_file, index=False)
        print(f"天气数据已保存到 {weather_file}")
        holiday_file = f'{output_dir}/holidays.csv'
        data['holidays'].to_csv(holiday_file, index=False)
        print(f"节假日数据已保存到 {holiday_file}")
    
    def analyze_generated_data(self, data):
        orders_df = data['orders']
        print("\n" + "="*60)
        print("生成数据统计分析报告")
        print("="*60)
        print("\n【各窗口订单量统计】")
        window_stats = orders_df.groupby('window_id').agg({
            'student_id': 'count',
            'quantity': ['mean', 'std']
        }).round(2)
        window_stats.columns = ['订单数', '平均数量', '数量标准差']
        print(window_stats)
        print("\n【窗口间订单量相关性矩阵】")
        daily_window_orders = orders_df.groupby(['date', 'window_id']).size().unstack(fill_value=0)
        correlation_matrix = daily_window_orders.corr().round(3)
        print(correlation_matrix)
        print("\n【时间序列自相关性分析】")
        for window_id in self.windows.keys():
            window_orders = daily_window_orders[window_id].values
            if len(window_orders) > 2:
                autocorr = np.corrcoef(window_orders[:-1], window_orders[1:])[0, 1]
                print(f"窗口{window_id} ({self.windows[window_id]['name']}): 一阶自相关系数 = {autocorr:.3f}")
        print("\n【星期分布统计】")
        orders_df['datetime'] = pd.to_datetime(orders_df['date'])
        orders_df['weekday'] = orders_df['datetime'].dt.day_name()
        weekday_stats = orders_df.groupby('weekday').size()
        print(weekday_stats)
        print("\n" + "="*60)
        return {
            'window_stats': window_stats,
            'correlation_matrix': correlation_matrix,
            'daily_window_orders': daily_window_orders
        }
if __name__ == "__main__":
    generator = CanteenDataGenerator(seed=42)
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    data = generator.generate_all_data(start_date, days=60)
    generator.save_to_files(data)
    analysis_results = generator.analyze_generated_data(data)
    print("\n数据生成完成！")
    print(f"订单数据样本:")
    print(data['orders'].head())
