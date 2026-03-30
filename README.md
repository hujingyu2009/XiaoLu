# 由于我们没有从互联网以及学校获取到任何有关于食堂订单的数据，所以并没有在项目程序中添加有关于导入外部数据的功能，但我们提供了一种方法（未经测试/无法测试），以下是我们提供的教程，可自行通过此方法进行测试或使用。  
## 由""小鹿""指导，""刘昊航""、""胡璟宇""进行工作


## **1. 数据准备**

### **1.1 数据格式要求**
确保真实数据包含以下字段（与系统数据库结构对应）：
- `date`: 日期（格式：YYYY-MM-DD，如2026-03-30）
- `time`: 时间（格式：HH:MM，如12:30）
- `window_id`: 窗口ID（1-5，对应主食、小吃、汤品、快餐、特色窗口）
- `student_id`: 学生ID（字符串，如S001）
- `quantity`: 订单数量（整数，默认1）


### **1.2 数据文件准备**
将真实数据保存为CSV文件，例如 `real_canteen_data.csv`，格式如下：

| date       | time  | window_id | student_id | quantity |
|------------|-------|-----------|------------|----------|
| 2026-03-20 | 12:05 | 1         | S001       | 1        |
| 2026-03-20 | 12:06 | 2         | S002       | 1        |
| 2026-03-21 | 11:50 | 1         | S003       | 1        |


## **2. 数据导入**

### **2.1 方法一：使用API导入模块**
创建一个导入脚本 `import_real_data.py`：

```python
# import_real_data.py
from api_data_import import api_importer

# 导入CSV文件
result = api_importer.import_from_csv(
    filepath='real_canteen_data.csv',
    column_mapping={}  # 若列名与要求一致，无需映射
)

print("导入结果:")
print(result)
```

### **2.2 方法二：直接操作数据库**
若数据格式特殊，可编写自定义导入脚本：

```python
# custom_import.py
import pandas as pd
from database import db

# 读取数据
df = pd.read_csv('real_canteen_data.csv')

# 数据清洗
df = df.dropna(subset=['date', 'time', 'window_id', 'student_id'])
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
df['time'] = df['time'].apply(lambda x: x[:5])  # 确保格式为HH:MM
df['window_id'] = df['window_id'].astype(int)
df['student_id'] = df['student_id'].astype(str)
df['quantity'] = df.get('quantity', 1).astype(int)

# 计算支付时间间隔（可选）
df = df.sort_values(['date', 'window_id', 'time'])
df['payment_time_gap'] = 0

# 导入数据库
records = []
for _, row in df.iterrows():
    records.append((
        row['date'], row['time'], row['window_id'],
        row['student_id'], row['quantity'], row.get('payment_time_gap', 0)
    ))

db.insert_orders_batch(records)
print(f"导入 {len(records)} 条真实数据")
```


## **3. 验证数据导入**

运行以下脚本检查导入的数据：

```python
# check_imported_data.py
from database import db

# 获取数据统计
status = db.get_all_orders_as_dataframe()
print(f"总订单数: {len(status)}")
print(f"日期范围: {status['date'].min()} 至 {status['date'].max()}")
print(f"学生数: {status['student_id'].nunique()}")
print(f"窗口分布:\n{status['window_id'].value_counts()}")
```


## **4. 训练模型**

### **4.1 使用主程序训练**
运行主程序 `canteen_system.py`，系统会自动训练模型：

```bash
python canteen_system.py
```

**注意**：主程序默认会清空旧数据并生成模拟数据，需要修改 `initialize_with_sample_data` 方法，跳过数据生成步骤。


### **4.2 自定义训练脚本**
创建 `train_model.py` 脚本，直接使用真实数据训练：

```python
# train_model.py
from database import db
from universal_predictor import UniversalCanteenPredictor
from data_generator import CanteenDataGenerator
from weather_calendar_api import weather_calendar_api
from datetime import datetime, timedelta

# 初始化组件
predictor = UniversalCanteenPredictor()
data_generator = CanteenDataGenerator()

# 获取历史订单数据
daily_orders = db.get_window_daily_stats(days=30)
if daily_orders.empty:
    print("没有数据可供训练")
    exit()

# 重命名列
daily_orders = daily_orders.rename(columns={
    'total_quantity': 'total_orders',
    'avg_time_gap': 'avg_time_gap'
})

# 检查数据量
date_range = daily_orders['date'].unique()
if len(date_range) < 7:
    print(f"数据不足7天（当前{len(date_range)}天），无法训练模型")
    exit()

start_date = min(date_range)
end_date = max(date_range)

# 获取天气和节假日数据
try:
    weather_df = weather_calendar_api.get_weather_forecast(days=len(date_range))
    holiday_df = weather_calendar_api.get_holiday_range(start_date, end_date)
except Exception as e:
    print(f"API调用失败，使用模拟数据: {str(e)}")
    weather_df = data_generator.generate_weather_data(start_date, len(date_range))
    holiday_df = data_generator.generate_holiday_data(start_date, len(date_range))

# 准备特征
print("准备特征数据...")
prepared_data = predictor.prepare_features(
    daily_orders, weather_df, holiday_df
)

# 训练模型
print("训练模型...")
results = predictor.train(prepared_data)

# 保存模型
predictor.save_model('models/real_data_model.pkl')

# 打印训练结果
print("训练结果:")
for window_id, stats in results.items():
    print(f"  窗口 {window_id}: MAPE={stats['mape_mean']:.2f}%, 样本数={stats['samples']}")
```


## **5. 预测使用**

### **5.1 预测下一天订单量**
创建 `predict_next_day.py` 脚本：

```python
# predict_next_day.py
from database import db
from universal_predictor import UniversalCanteenPredictor
from weather_calendar_api import weather_calendar_api
from datetime import datetime, timedelta
import numpy as np

# 加载训练好的模型
predictor = UniversalCanteenPredictor()
predictor.load_model('models/real_data_model.pkl')

# 预测日期
prediction_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

# 获取天气和节假日信息
weather_info = weather_calendar_api.get_weather_forecast(days=3)
holiday_info = weather_calendar_api.get_holiday_info(prediction_date)

# 获取历史订单数据
orders_df = db.get_all_orders_as_dataframe()
daily_orders = orders_df.groupby(['date', 'window_id']).agg({
    'quantity': 'sum'
}).reset_index()
daily_orders.columns = ['date', 'window_id', 'total_orders']

pred_dt = datetime.strptime(prediction_date, '%Y-%m-%d')

# 预测各窗口订单量
predictions = {}
for window_id in [1, 2, 3, 4, 5]:
    window_data = daily_orders[daily_orders['window_id'] == window_id]
    if len(window_data) < 7:
        predictions[window_id] = {
            'predicted_orders': None,
            'reason': '历史数据不足'
        }
        continue
    
    # 提取最近7天数据
    recent_orders = window_data.sort_values('date').tail(7)['total_orders'].tolist()
    
    # 构建特征
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
    
    # 天气特征
    weather_row = weather_info[weather_info['date'] == prediction_date]
    if not weather_row.empty:
        weather = weather_row.iloc[0]['weather']
        temperature = weather_row.iloc[0]['temperature']
    else:
        weather = '晴'
        temperature = 25.0
    
    weather_map = {'晴': 0, '多云': 1, '阴': 2, '小雨': 3, '中雨': 4, '大雨': 5}
    features['weather_encoded'] = weather_map.get(weather, 0)
    features['temperature'] = temperature
    features['temp_normalized'] = 0
    features['temp_high'] = 1 if temperature > 28 else 0
    features['temp_low'] = 1 if temperature < 10 else 0
    
    # 人流特征
    features['avg_time_gap'] = 0
    features['crowd_density'] = 0
    features['yoy_growth'] = 0
    
    # 预测
    try:
        pred_orders = predictor.predict(prediction_date, window_id, features)
        predictions[window_id] = {
            'predicted_orders': pred_orders,
            'weather': weather,
            'temperature': temperature
        }
    except Exception as e:
        predictions[window_id] = {
            'predicted_orders': None,
            'reason': str(e)
        }

# 打印预测结果
print(f"预测日期: {prediction_date}")
print("各窗口预测订单量:")
for window_id, pred in predictions.items():
    if pred.get('predicted_orders'):
        print(f"  窗口 {window_id}: {pred['predicted_orders']} 份")
    else:
        print(f"  窗口 {window_id}: 无法预测 - {pred.get('reason', '未知原因')}")
```


## **6. 模型评估**

### **6.1 查看MAPE值**
训练完成后，系统会输出各窗口的MAPE值。理想情况下：
- **MAPE < 20%**：模型表现优秀
- **MAPE 20%-30%**：模型表现良好
- **MAPE > 30%**：模型需要优化


### **6.2 生成分析报告**
运行 `canteen_system.py` 后，系统会生成分析报告和图表，位于 `analysis` 文件夹中，包括：
- 预测值与实际值对比图
- MAPE分布直方图
- 系统拓扑图
- 数据库ER图


## **7. 优化建议**

### **7.1 数据质量优化**
- **数据量**：确保至少有7天以上的历史数据
- **数据完整性**：填充缺失值，处理异常值
- **时间分布**：确保数据覆盖早中晚各时段


### **7.2 模型参数优化**
修改 `universal_predictor.py` 中的模型参数：
- `n_estimators`：增加树的数量（如200）
- `max_depth`：调整树的深度（如6）
- `learning_rate`：调整学习率（如0.05）


### **7.3 特征工程优化**
- **添加新特征**：如季节因素、特殊事件（考试周、运动会等）
- **特征选择**：使用 `get_feature_importance` 方法查看特征重要性，保留重要特征


## **8. 常见问题解决**

### **8.1 数据导入失败**
- 检查CSV文件格式是否正确
- 确保所有必要字段都存在
- 检查日期时间格式是否符合要求


### **8.2 模型训练失败**
- 确保数据量足够（至少7天）
- 检查数据中是否有异常值
- 尝试调整模型参数


### **8.3 预测结果不准确**
- 增加历史数据量
- 优化特征工程
- 考虑添加更多外部因素（如学校活动安排）


## **9. 示例工作流**

1. **准备数据**：整理真实食堂订单数据为CSV格式
2. **导入数据**：运行 `import_real_data.py` 导入数据
3. **验证数据**：运行 `check_imported_data.py` 检查数据
4. **训练模型**：运行 `train_model.py` 训练模型
5. **预测使用**：运行 `predict_next_day.py` 预测下一天订单量
6. **分析结果**：查看 `analysis` 文件夹中的分析报告


通过以上步骤，你可以使用真实的食堂数据训练模型并进行预测
