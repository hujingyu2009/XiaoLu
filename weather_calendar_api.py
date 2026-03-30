# weather_calendar_api.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print(" requests模块未安装，使用模拟天气数据模式")
class WeatherCalendarAPI:
    def __init__(self):
        self.weather_api_key = 'dfee3525647f4bb583789302a8f89839'
        self.weather_base_url = 'https://pv4nmughuu.re.qweatherapi.com/v7/weather/7d'
        self.holiday_api_url = 'https://api.jiejiariapi.com/v1/is_holiday'
        self.holiday_api_key = 'jjr_hcrvtepo_ocmaleh4koji6vopfhosdv7ptzf7f3raxkqqwcncqt7keaacni2a'
        self.weather_cache = {}
        self.holiday_cache = {}
    
    def get_weather_forecast(self, location='101240511', days=3):
        if not HAS_REQUESTS:
            print("没有安装requests模块，使用模拟天气数据")
            df = self._generate_mock_weather(days)
            df['data_source'] = 'mock'
            return df
        try:
            url = f"{self.weather_base_url}?location={location}"
            print(f"正在调用天气API: {url}")
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'X-QW-Api-Key': self.weather_api_key
            }
            response = requests.get(url, headers=headers, timeout=5)
            print(f"API响应状态码: {response.status_code}")
            print(f"API响应内容: {response.text}")
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200':
                    daily = data.get('daily', [])
                    weather_data = []
                    for day in daily[:days]:
                        weather_data.append({
                            'date': day['fxDate'],
                            'weather': day['textDay'],
                            'temperature': (float(day['tempMax']) + float(day['tempMin'])) / 2,
                            'temp_max': float(day['tempMax']),
                            'temp_min': float(day['tempMin']),
                            'humidity': float(day.get('humidity', 50)),
                            'wind_speed': float(day.get('windSpeedDay', 10)),
                            'precip': float(day.get('precip', 0))
                        })
                    df = pd.DataFrame(weather_data)
                    df['data_source'] = 'api'
                    for _, row in df.iterrows():
                        self.weather_cache[row['date']] = row.to_dict()
                    print(f"成功获取{len(df)}天天气数据 (API)")
                    return df
            print("API调用失败，使用模拟天气数据")
            df = self._generate_mock_weather(days)
            df['data_source'] = 'mock'
            return df
            
        except Exception as e:
            print(f"获取天气数据失败: {e}")
            df = self._generate_mock_weather(days)
            df['data_source'] = 'mock'
            return df
    def _generate_mock_weather(self, days):
        weather_types = ['晴', '多云', '阴', '小雨', '中雨']
        weather_data = []
        base_date = datetime.now()
        for i in range(days):
            date = (base_date + timedelta(days=i)).strftime('%Y-%m-%d')
            day_of_year = (base_date + timedelta(days=i)).timetuple().tm_yday
            if 60 <= day_of_year <= 150:  # 春季
                base_temp = 20
                weather_probs = [0.3, 0.4, 0.2, 0.1, 0.0]
            elif 151 <= day_of_year <= 240:  # 夏季
                base_temp = 30
                weather_probs = [0.4, 0.3, 0.1, 0.15, 0.05]
            elif 241 <= day_of_year <= 330:  # 秋季
                base_temp = 18
                weather_probs = [0.35, 0.35, 0.2, 0.1, 0.0]
            else:  # 冬季
                base_temp = 5
                weather_probs = [0.4, 0.3, 0.2, 0.05, 0.05]
            
            weather = np.random.choice(weather_types, p=weather_probs)
            temp_variation = np.random.normal(0, 3)
            temperature = base_temp + temp_variation
            weather_data.append({
                'date': date,
                'weather': weather,
                'temperature': round(temperature, 1),
                'temp_max': round(temperature + 3, 1),
                'temp_min': round(temperature - 3, 1),
                'humidity': round(np.random.uniform(40, 80), 1),
                'wind_speed': round(np.random.uniform(5, 20), 1),
                'precip': round(np.random.uniform(0, 10), 1) if '雨' in weather else 0
            })
        return pd.DataFrame(weather_data)
    def get_holiday_info(self, date_str):
        if date_str in self.holiday_cache:
            return self.holiday_cache[date_str]
        if not HAS_REQUESTS:
            return self._calculate_holiday_local(date_str)
        try:
            url = f"{self.holiday_api_url}?date={date_str}&key={self.holiday_api_key}"
            print(f"正在调用节假日API: {url}")
            response = requests.get(url, timeout=5)
            print(f"API响应状态码: {response.status_code}")
            print(f"API响应内容: {response.text}")
            if response.status_code == 200:
                data = response.json()
                is_holiday = data.get('is_holiday', False)
                holiday_name = data.get('holiday', '')
                holiday_info = {
                    'date': date_str,
                    'is_holiday': 1 if is_holiday else 0,
                    'holiday_name': holiday_name,
                    'holiday_type': 2 if is_holiday else 0,
                    'is_weekend': False
                }
                date = datetime.strptime(date_str, '%Y-%m-%d')
                holiday_info['is_weekend'] = date.weekday() >= 5
                if holiday_info['is_weekend'] and not holiday_info['is_holiday']:
                    holiday_info['is_holiday'] = 1
                    holiday_info['holiday_type'] = 1  # 1=周末
                    holiday_info['holiday_name'] = '周末'
                
                self.holiday_cache[date_str] = holiday_info
                print(f"成功获取节假日信息: {holiday_info}")
                return holiday_info
        
        except Exception as e:
            print(f"获取节假日信息失败: {e}")
        return self._calculate_holiday_local(date_str)
    def _calculate_holiday_local(self, date_str):
        date = datetime.strptime(date_str, '%Y-%m-%d')
        weekday = date.weekday()
        is_weekend = weekday >= 5
        holidays = {
            '01-01': '元旦',
            '05-01': '劳动节',
            '06-01': '儿童节',
            '10-01': '国庆节',
            '10-02': '国庆节',
            '10-03': '国庆节',
        }
        month_day = date_str[5:]
        holiday_name = holidays.get(month_day, '')
        is_holiday = 1 if (is_weekend or holiday_name) else 0
        holiday_info = {
            'date': date_str,
            'is_holiday': is_holiday,
            'holiday_name': holiday_name,
            'holiday_type': 0 if holiday_name else (3 if is_weekend else -1),
            'is_weekend': is_weekend
        }
        
        self.holiday_cache[date_str] = holiday_info
        return holiday_info
    
    def get_holiday_range(self, start_date, end_date):
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        holidays = []
        current = start
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            holiday_info = self.get_holiday_info(date_str)
            holidays.append(holiday_info)
            current += timedelta(days=1)
        return pd.DataFrame(holidays)
    
    def get_combined_data(self, start_date, end_date, location='101240511'):
        days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
        weather_df = self.get_weather_forecast(location, days)
        holiday_df = self.get_holiday_range(start_date, end_date)
        combined = pd.merge(weather_df, holiday_df, on='date', how='outer')
        return combined
    
    def save_to_csv(self, data, filepath='data/weather_holiday.csv'):
        data.to_csv(filepath, index=False)
        print(f"天气节假日数据已保存到 {filepath}")

weather_calendar_api = WeatherCalendarAPI()

if __name__ == "__main__":
    api = WeatherCalendarAPI()
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    weather = api.get_weather_forecast(days=3)
    print("\n天气预报:")
    print(weather)
    holiday = api.get_holiday_info(tomorrow)
    print(f"\n节假日信息 ({tomorrow}):")
    print(holiday)
