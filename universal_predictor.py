# universal_predictor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class UniversalCanteenPredictor:
    """
    此程序由指导老师大力帮助
    """
    def __init__(self):
        self.models = {} 
        self.scalers = {} 
        self.feature_columns = None
        self.is_trained = False
        self.training_stats = {}
        
    def _extract_time_features(self, df):
        """提取时间相关特征 - 完全基于输入数据"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=周一, 6=周日
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df
    
    def _extract_weather_features(self, df, weather_data):
        df = df.copy()
        
        if weather_data is not None and not weather_data.empty:
            weather_data = weather_data.copy()
            weather_data['date'] = pd.to_datetime(weather_data['date']).dt.strftime('%Y-%m-%d')
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = pd.merge(df, weather_data, on='date', how='left')
            if 'weather' in df.columns:
                unique_weathers = df['weather'].dropna().unique()
                weather_map = {w: i for i, w in enumerate(unique_weathers)}
                df['weather_encoded'] = df['weather'].map(weather_map).fillna(0)
            if 'temperature' in df.columns:
                df['temp_normalized'] = (df['temperature'] - df['temperature'].mean()) / (df['temperature'].std() + 1e-8)
                df['temp_high'] = (df['temperature'] > df['temperature'].quantile(0.75)).astype(int)
                df['temp_low'] = (df['temperature'] < df['temperature'].quantile(0.25)).astype(int)
        
        return df
    
    def _extract_holiday_features(self, df, holiday_data):
        df = df.copy()
        if holiday_data is not None and not holiday_data.empty:
            holiday_data = holiday_data.copy()
            holiday_data['date'] = pd.to_datetime(holiday_data['date']).dt.strftime('%Y-%m-%d')
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = pd.merge(df, holiday_data, on='date', how='left')
            if 'is_holiday' in df.columns:
                df['is_holiday'] = df['is_holiday'].fillna(0)
        
        return df
    
    def _calculate_historical_features(self, df, window_col='window_id', target_col='total_orders'):
        df = df.copy()
        df = df.sort_values(['date'])
        for window_id in df[window_col].unique():
            mask = df[window_col] == window_id
            window_data = df[mask].copy()
            if len(window_data) < 7:
                continue
            df.loc[mask, 'avg_7d'] = window_data[target_col].rolling(window=7, min_periods=1).mean().shift(1).values
            df.loc[mask, 'std_7d'] = window_data[target_col].rolling(window=7, min_periods=1).std().shift(1).values
            df.loc[mask, 'max_7d'] = window_data[target_col].rolling(window=7, min_periods=1).max().shift(1).values
            df.loc[mask, 'min_7d'] = window_data[target_col].rolling(window=7, min_periods=1).min().shift(1).values
            df.loc[mask, 'median_7d'] = window_data[target_col].rolling(window=7, min_periods=1).median().shift(1).values
            df.loc[mask, 'avg_3d'] = window_data[target_col].rolling(window=3, min_periods=1).mean().shift(1).values
            df.loc[mask, 'avg_14d'] = window_data[target_col].rolling(window=min(14, len(window_data)), min_periods=1).mean().shift(1).values
            if len(window_data) >= 14:
                df.loc[mask, 'yoy_growth'] = (
                    (window_data[target_col] - window_data[target_col].shift(7)) / 
                    (window_data[target_col].shift(7) + 1e-8)
                ).shift(1).values
            df.loc[mask, 'trend_3d'] = (
                window_data[target_col].shift(1) - window_data[target_col].shift(4)
            ).values
            for day in range(7):
                day_mask = window_data['day_of_week'] == day
                if day_mask.any():
                    day_avg = window_data[day_mask][target_col].expanding().mean().shift(1).values
                    df.loc[mask & (df['day_of_week'] == day), 'dow_avg'] = day_avg
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != target_col and col != window_col:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _extract_crowd_features(self, df, crowd_data):
        df = df.copy()
        if crowd_data is not None and not crowd_data.empty:
            crowd_daily = crowd_data.groupby(['date', 'window_id']).agg({
                'avg_time_gap': 'mean',
                'order_count': 'sum'
            }).reset_index()
            
            df = pd.merge(df, crowd_daily, on=['date', 'window_id'], how='left')
            if 'avg_time_gap' in df.columns:
                df['crowd_density'] = 1 / (df['avg_time_gap'] + 1) 
        
        return df
    
    def prepare_features(self, order_data, weather_data=None, holiday_data=None, crowd_data=None):
        df = order_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = self._extract_time_features(df)
        df = self._extract_weather_features(df, weather_data)
        df = self._extract_holiday_features(df, holiday_data)
        df = self._calculate_historical_features(df)
        df = self._extract_crowd_features(df, crowd_data)
        exclude_cols = ['date', 'weather', 'holiday_name']
        for col in exclude_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'total_orders' in numeric_cols:
            numeric_cols.remove('total_orders')
        
        self.feature_columns = numeric_cols
        for col in self.feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    
    def train(self, prepared_data, window_col='window_id', target_col='total_orders'):
        if self.feature_columns is None:
            raise ValueError("请先调用prepare_features准备特征数据")
        
        results = {}
        for window_id in prepared_data[window_col].unique():
            window_data = prepared_data[prepared_data[window_col] == window_id].copy()
            
            if len(window_data) < 7:
                print(f"窗口 {window_id} 数据不足7天，跳过训练")
                continue
            X = window_data[self.feature_columns].fillna(0)
            y = window_data[target_col]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            tscv = TimeSeriesSplit(n_splits=min(3, len(X)//7))
            mape_scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                # 避免除以很小的数，真实值小于100的按100计算
                y_val_adj = np.maximum(y_val, 100)
                mape = np.mean(np.abs((y_val - y_pred) / y_val_adj)) * 100
                mape_scores.append(mape)
            final_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            )
            final_model.fit(X_scaled, y)
            self.models[window_id] = final_model
            self.scalers[window_id] = scaler
            results[window_id] = {
                'mape_mean': np.mean(mape_scores),
                'mape_std': np.std(mape_scores),
                'samples': len(window_data),
                'features': len(self.feature_columns)
            }
        
        self.is_trained = True
        self.training_stats = results
        
        return results
    
    def predict(self, prediction_date, window_id, input_features):
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        if window_id not in self.models:
            raise ValueError(f"窗口 {window_id} 没有训练好的模型")
        X = pd.DataFrame([input_features])
        feature_dict = {}
        for col in self.feature_columns:
            feature_dict[col] = input_features.get(col, 0)
        
        X = pd.DataFrame([feature_dict])
        X = X[self.feature_columns].fillna(0)
        X_scaled = self.scalers[window_id].transform(X)
        prediction = self.models[window_id].predict(X_scaled)[0]
        return max(0, round(prediction))
    
    def predict_all_windows(self, prediction_date, features_by_window):
        predictions = {}
        for window_id, features in features_by_window.items():
            try:
                pred = self.predict(prediction_date, window_id, features)
                predictions[window_id] = pred
            except Exception as e:
                print(f"窗口 {window_id} 预测失败: {e}")
                predictions[window_id] = None
        return predictions
    
    def get_feature_importance(self, window_id):
        if window_id not in self.models:
            return None
        importance = self.models[window_id].feature_importances_
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
    def save_model(self, filepath='models/universal_predictor.pkl'):
        joblib.dump({
            'models': self.models,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'training_stats': self.training_stats
        }, filepath)
        print(f"模型已保存到 {filepath}")
    def load_model(self, filepath='models/universal_predictor.pkl'):
        """加载模型"""
        data = joblib.load(filepath)
        self.models = data['models']
        self.scalers = data['scalers']
        self.feature_columns = data['feature_columns']
        self.training_stats = data['training_stats']
        self.is_trained = True
        print(f"模型已从 {filepath} 加载")
if __name__ == "__main__":
    print("通用预测器模块加载成功")
