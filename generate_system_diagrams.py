import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import os
os.makedirs('analysis', exist_ok=True)

def generate_system_topology():
    plt.figure(figsize=(12, 6))
    modules = {
        'data_generator': (0.1, 0.7),
        'database': (0.3, 0.7),
        'universal_predictor': (0.6, 0.7),
        'canteen_system': (0.8, 0.7),
        'crowd_analyzer': (0.5, 0.3),
        'smart_recommendation': (0.7, 0.3),
        'weather_calendar_api': (0.3, 0.3),
        'api_data_import': (0.1, 0.3)
    }
    for module, (x, y) in modules.items():
        plt.text(x, y, module, 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='blue'),
                 ha='center', va='center', fontsize=10, fontweight='bold')
    arrows = [
        ('data_generator', 'database'),
        ('database', 'universal_predictor'),
        ('universal_predictor', 'canteen_system'),
        ('database', 'crowd_analyzer'),
        ('database', 'smart_recommendation'),
        ('weather_calendar_api', 'universal_predictor'),
        ('api_data_import', 'database'),
        ('crowd_analyzer', 'canteen_system'),
        ('smart_recommendation', 'canteen_system')
    ]
    
    for start, end in arrows:
        start_x, start_y = modules[start]
        end_x, end_y = modules[end]
        dx = end_x - start_x
        dy = end_y - start_y
        arrow = FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            connectionstyle="arc3,rad=0.1",
            arrowstyle="->",
            color="black",
            linewidth=1.5
        )
        plt.gca().add_patch(arrow)
    plt.text(0.5, 0.95, '系统模块拓扑图', ha='center', fontsize=14, fontweight='bold')
    plt.text(0.5, 0.05, '数据流向: data_generator → database → universal_predictor → canteen_system', 
             ha='center', fontsize=10, style='italic')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('analysis/system_topology.png', dpi=300, bbox_inches='tight')
    print('系统模块拓扑图已生成: analysis/system_topology.png')

def generate_database_er_diagram():
    """生成数据库E-R图"""
    plt.figure(figsize=(14, 8))
    tables = {
        'orders': (0.3, 0.7),
        'student_diet': (0.7, 0.7),
        'windows': (0.2, 0.3),
        'crowd_density': (0.6, 0.3)
    }
    table_structures = {
        'orders': [
            'id (PK)',
            'date',
            'time',
            'window_id (FK)',
            'student_id',
            'quantity',
            'payment_time_gap'
        ],
        'student_diet': [
            'id (PK)',
            'student_id',
            'date',
            'window_id (FK)',
            'meal_type',
            'nutrition_score'
        ],
        'windows': [
            'window_id (PK)',
            'window_name',
            'window_type',
            'description'
        ],
        'crowd_density': [
            'id (PK)',
            'date',
            'time_slot',
            'window_id (FK)',
            'avg_time_gap',
            'density_level',
            'order_count'
        ]
    }
    for table, (x, y) in tables.items():
        plt.text(x, y, table, 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='green'),
                 ha='center', va='center', fontsize=12, fontweight='bold')
        for i, field in enumerate(table_structures[table]):
            plt.text(x, y - 0.05 - i * 0.05, field, 
                     ha='center', va='center', fontsize=8)
    relationships = [
        ('windows', 'orders', '1:N'),
        ('windows', 'student_diet', '1:N'),
        ('windows', 'crowd_density', '1:N'),
        ('orders', 'student_diet', 'N:1')
    ]
    for start, end, cardinality in relationships:
        start_x, start_y = tables[start]
        end_x, end_y = tables[end]
        dx = end_x - start_x
        dy = end_y - start_y
        start_x_adjusted = start_x + 0.05 if dx > 0 else start_x - 0.05
        start_y_adjusted = start_y - 0.1 if dy < 0 else start_y
        end_x_adjusted = end_x - 0.05 if dx > 0 else end_x + 0.05
        end_y_adjusted = end_y + 0.15 if dy < 0 else end_y
        arrow = FancyArrowPatch(
            (start_x_adjusted, start_y_adjusted), (end_x_adjusted, end_y_adjusted),
            connectionstyle="arc3,rad=0.1",
            arrowstyle="->",
            color="black",
            linewidth=1.5
        )
        plt.gca().add_patch(arrow)
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        plt.text(mid_x, mid_y + 0.02, cardinality, 
                 ha='center', va='center', fontsize=8, 
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black'))
    plt.text(0.5, 0.95, '数据库表结构关系图', ha='center', fontsize=14, fontweight='bold')
    plt.text(0.5, 0.05, 'PK: 主键, FK: 外键', ha='center', fontsize=10, style='italic')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('analysis/database_er_diagram.png', dpi=300, bbox_inches='tight')
    print('数据库E-R图已生成: analysis/database_er_diagram.png')

if __name__ == "__main__":
    print("生成系统图表...")
    generate_system_topology()
    generate_database_er_diagram()
    print("图表生成完成！")
