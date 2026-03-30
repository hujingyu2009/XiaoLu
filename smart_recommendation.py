# smart_recommendation.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from database import db

class SmartDietRecommendation:
    WINDOW_NUTRITION = {
        'staple': {'carbs': 0.8, 'protein': 0.3, 'fat': 0.2, 'fiber': 0.1, 'vitamins': 0.1},
        'snack': {'carbs': 0.4, 'protein': 0.2, 'fat': 0.5, 'fiber': 0.2, 'vitamins': 0.2},
        'soup': {'carbs': 0.1, 'protein': 0.3, 'fat': 0.1, 'fiber': 0.3, 'vitamins': 0.4},
        'fastfood': {'carbs': 0.5, 'protein': 0.4, 'fat': 0.5, 'fiber': 0.1, 'vitamins': 0.1},
        'special': {'carbs': 0.4, 'protein': 0.5, 'fat': 0.3, 'fiber': 0.3, 'vitamins': 0.3}
    }
    IDEAL_NUTRITION = {
        'carbs': 0.50,
        'protein': 0.20,
        'fat': 0.25,
        'fiber': 0.03,
        'vitamins': 0.02
    }
    def __init__(self):
        self.window_info = self._load_window_info()
    def _load_window_info(self):
        import sqlite3
        conn = sqlite3.connect('data/canteen.db')
        df = pd.read_sql_query('SELECT * FROM windows', conn)
        conn.close()
        return df.set_index('window_id').to_dict('index')
    def analyze_student_diet(self, student_id, days=7):
        diet_df = db.get_student_recent_diet(student_id, days)
        if diet_df.empty:
            return {
                'student_id': student_id,
                'days_analyzed': 0,
                'total_meals': 0,
                'nutrition_score': 50,
                'window_preference': {},
                'nutrition_breakdown': {},
                'recommendation': '数据不足，建议均衡饮食'
            }
        window_counts = diet_df.groupby('window_id').agg({
            'visit_count': 'sum',
            'window_type': 'first',
            'window_name': 'first'
        }).to_dict('index')
        total_visits = sum(w['visit_count'] for w in window_counts.values())
        nutrition_total = defaultdict(float)
        for window_id, data in window_counts.items():
            window_type = data['window_type']
            weight = data['visit_count'] / total_visits if total_visits > 0 else 0
            
            if window_type in self.WINDOW_NUTRITION:
                for nutrient, value in self.WINDOW_NUTRITION[window_type].items():
                    nutrition_total[nutrient] += value * weight
        nutrition_score = self._calculate_nutrition_score(dict(nutrition_total))
        nutrition_gaps = self._identify_nutrition_gaps(dict(nutrition_total))
        return {
            'student_id': student_id,
            'days_analyzed': days,
            'total_meals': total_visits,
            'nutrition_score': nutrition_score,
            'window_preference': {
                wid: {
                    'name': data['window_name'],
                    'type': data['window_type'],
                    'visits': data['visit_count'],
                    'percentage': round(data['visit_count'] / total_visits * 100, 1) if total_visits > 0 else 0
                }
                for wid, data in window_counts.items()
            },
            'nutrition_breakdown': dict(nutrition_total),
            'nutrition_gaps': nutrition_gaps
        }
    def _calculate_nutrition_score(self, nutrition_actual):
        if not nutrition_actual:
            return 50
        total = sum(nutrition_actual.values())
        if total == 0:
            return 50
        normalized = {k: v/total for k, v in nutrition_actual.items()}
        differences = []
        for nutrient in self.IDEAL_NUTRITION:
            actual = normalized.get(nutrient, 0)
            ideal = self.IDEAL_NUTRITION[nutrient]
            diff = abs(actual - ideal) / ideal if ideal > 0 else 0
            differences.append(min(diff, 1.0)) 
        avg_diff = np.mean(differences)
        score = max(0, min(100, 100 - avg_diff * 100))
        return round(score, 1)
    def _identify_nutrition_gaps(self, nutrition_actual):
        gaps = []
        total = sum(nutrition_actual.values())
        if total == 0:
            return ['所有营养素']
        normalized = {k: v/total for k, v in nutrition_actual.items()}
        for nutrient, ideal_ratio in self.IDEAL_NUTRITION.items():
            actual_ratio = normalized.get(nutrient, 0)
            if actual_ratio < ideal_ratio * 0.7: 
                gaps.append(nutrient)
        return gaps
    def get_recommendation(self, student_id, current_weather=None, temperature=None, meal_type='lunch'):
        print(f"分析学生 {student_id} 的饮食习惯...")
        analysis = self.analyze_student_diet(student_id)
        gaps = analysis.get('nutrition_gaps', [])
        print(f"营养缺口: {gaps}")
        weather_factor = self._analyze_weather_factor(current_weather, temperature)
        print(f"天气因素: {weather_factor['factor']} - {weather_factor['recommendation']}")
        recommended_windows = self._recommend_windows(
            gaps, 
            analysis['window_preference'],
            weather_factor,
            meal_type
        )
        health_tips = self._generate_health_tips(analysis, gaps)
        return {
            'student_id': student_id,
            'nutrition_score': analysis['nutrition_score'],
            'recommended_windows': recommended_windows,
            'health_tips': health_tips,
            'weather_factor': weather_factor,
            'nutrition_gaps': gaps,
            'analysis_summary': self._generate_summary(analysis)
        }
    def _analyze_weather_factor(self, weather, temperature=None):
        if weather is None and temperature is None:
            return {'factor': 'normal', 'recommendation': '正常饮食'}
        if temperature is not None:
            if temperature < 10:
                return {
                    'factor': 'cold',
                    'recommendation': '推荐热食和高热量食物',
                    'preferred_types': ['staple', 'fastfood', 'soup'],
                    'avoid_types': ['snack']
                }
            elif temperature < 15:
                return {
                    'factor': 'cool',
                    'recommendation': '推荐热食',
                    'preferred_types': ['staple', 'soup'],
                    'avoid_types': ['snack']
                }
            elif temperature > 30:
                return {
                    'factor': 'hot',
                    'recommendation': '推荐清淡和汤品',
                    'preferred_types': ['soup', 'special'],
                    'avoid_types': ['fastfood']
                }
            elif temperature > 25:
                return {
                    'factor': 'warm',
                    'recommendation': '推荐清淡饮食',
                    'preferred_types': ['soup', 'special'],
                    'avoid_types': ['fastfood']
                }
        if weather is not None:
            weather = str(weather).lower()
            if '雨' in weather or 'rain' in weather:
                return {
                    'factor': 'rainy',
                    'recommendation': '推荐热食和汤品',
                    'preferred_types': ['soup', 'staple'],
                    'avoid_types': ['snack']
                }
            elif 'hot' in weather or '热' in weather or '晴' in weather:
                return {
                    'factor': 'hot',
                    'recommendation': '推荐清淡和汤品',
                    'preferred_types': ['soup', 'special'],
                    'avoid_types': ['fastfood']
                }
            elif 'cold' in weather or '冷' in weather:
                return {
                    'factor': 'cold',
                    'recommendation': '推荐高热量主食',
                    'preferred_types': ['staple', 'fastfood'],
                    'avoid_types': ['soup']
                }
            elif 'cool' in weather or '较冷' in weather:
                return {
                    'factor': 'cool',
                    'recommendation': '推荐热食',
                    'preferred_types': ['staple', 'soup'],
                    'avoid_types': ['snack']
                }
        return {
            'factor': 'normal',
            'recommendation': '正常饮食',
            'preferred_types': [],
            'avoid_types': []
        }
    def _recommend_windows(self, gaps, preferences, weather_factor, meal_type):
        recommendations = []
        needed_types = set()
        for gap in gaps:
            if gap == 'protein':
                needed_types.update(['special', 'fastfood'])
            elif gap == 'fiber' or gap == 'vitamins':
                needed_types.update(['soup', 'special'])
            elif gap == 'carbs':
                needed_types.add('staple')
            elif gap == 'fat':
                needed_types.add('snack')
        if not needed_types:
            needed_types = {'staple', 'soup', 'special'}
        preferred_weather = set(weather_factor.get('preferred_types', []))
        avoid_weather = set(weather_factor.get('avoid_types', []))
        for window_id, info in self.window_info.items():
            window_type = info['window_type']
            score = 0
            reasons = []
            if window_type in needed_types:
                score += 30
                reasons.append(f'补充{self._get_nutrient_name(window_type)}')
            if window_type in preferred_weather:
                score += 20
                reasons.append('适合当前天气')
            if window_type in avoid_weather:
                score -= 15
            if window_id in preferences:
                visit_pct = preferences[window_id]['percentage']
                if visit_pct > 50:
                    score -= 20
                    reasons.append('建议换换口味')
            score += 10
            recommendations.append({
                'window_id': window_id,
                'window_name': info['window_name'],
                'window_type': window_type,
                'score': max(0, score),
                'reasons': reasons
            })
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        print("推荐窗口详情:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec['window_name']} (分数: {rec['score']})")
            print(f"    原因: {', '.join(rec['reasons'])}")
        return recommendations[:3] 
    
    def _get_nutrient_name(self, window_type):
        mapping = {
            'staple': '碳水化合物',
            'snack': '脂肪能量',
            'soup': '维生素和纤维',
            'fastfood': '蛋白质',
            'special': '均衡营养'
        }
        return mapping.get(window_type, '营养')
    def _generate_health_tips(self, analysis, gaps):
        tips = []
        score = analysis['nutrition_score']
        if score >= 80:
            tips.append('您的饮食习惯非常健康，请继续保持！')
        elif score >= 60:
            tips.append('您的饮食习惯良好，还有提升空间。')
        else:
            tips.append('建议调整饮食结构，注意营养均衡。')
        gap_tips = {
            'carbs': '建议适当增加主食摄入，保证能量供应',
            'protein': '建议增加蛋白质摄入，如肉类、豆制品',
            'fat': '建议适量摄入健康脂肪',
            'fiber': '建议多吃蔬菜水果，补充膳食纤维',
            'vitamins': '建议增加新鲜蔬果摄入，补充维生素'
        }
        for gap in gaps:
            if gap in gap_tips:
                tips.append(gap_tips[gap])
        return tips
    def _generate_summary(self, analysis):
        total = analysis['total_meals']
        if total == 0:
            return '暂无饮食记录'
        pref = analysis['window_preference']
        if pref:
            top_window = max(pref.items(), key=lambda x: x[1]['visits'])
            return f"最近{analysis['days_analyzed']}天共就餐{total}次，最常去{top_window[1]['name']}"
        return f"最近{analysis['days_analyzed']}天共就餐{total}次"
    def get_all_students_recommendations(self, weather=None, temperature=None):
        students = db.get_student_list()
        recommendations = {}
        for student_id in students:
            recommendations[student_id] = self.get_recommendation(student_id, weather, temperature)
        return recommendations
if __name__ == "__main__":
    recommender = SmartDietRecommendation()
    print("智能推荐系统模块加载成功")
