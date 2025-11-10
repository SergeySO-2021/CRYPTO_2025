"""
Скрипт 18: Создание упрощенной классификации
Разбивает сегменты на небольшое количество классов на основе:
- Диапазоны значений slope
- Комбинированный коэффициент из других характеристик сегмента
"""

import sys
import io
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Исправление кодировки для Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# Импорт Trading Classifier
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
classifier_path = os.path.join(project_root, 'indicators', 'trading_classifier_iziceros', 'src')
sys.path.insert(0, classifier_path)

try:
    from trend_classifier import Segmenter, Config
    from trend_classifier.models import Metrics
except ImportError as e:
    print(f"[ERROR] Не удалось импортировать Trading Classifier: {e}")
    sys.exit(1)

# Импорт функций классификации
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))
from zone_based_optimization.classification_functions import create_market_zones_matrix

def calculate_combined_coefficient(zone_row):
    """
    Вычисляет комбинированный коэффициент из характеристик сегмента
    
    Использует:
    - trend_stability (стабильность тренда)
    - volatility_std (стандартное отклонение волатильности)
    - volume_level (уровень объема)
    - regime_confidence (уверенность в режиме)
    
    Returns:
        float: комбинированный коэффициент (0-1)
    """
    # Нормализуем каждую характеристику
    trend_stability = zone_row.get('trend_stability', 0.5)
    if pd.isna(trend_stability) or trend_stability < 0:
        trend_stability = 0.5
    
    # Волатильность (инвертируем - чем меньше std, тем выше коэффициент)
    volatility_std = zone_row.get('volatility_std', 0)
    if pd.isna(volatility_std):
        volatility_std = 0
    # Нормализуем: если std < 100, то коэффициент высокий
    volatility_coef = max(0, 1 - (volatility_std / 500))  # Нормализация
    
    # Объемы (преобразуем в числовое значение)
    volume_level = zone_row.get('volume_level', 'medium')
    volume_map = {'very_low': 0.2, 'low': 0.4, 'medium': 0.6, 'high': 0.8, 'very_high': 1.0}
    volume_coef = volume_map.get(volume_level, 0.6)
    
    # Режим (trending лучше, чем ranging)
    regime = zone_row.get('regime', 'ranging')
    regime_coef = 1.0 if regime == 'trending' else 0.7 if regime == 'ranging' else 0.5
    
    # Комбинированный коэффициент (взвешенная сумма)
    combined = (
        trend_stability * 0.3 +      # 30% - стабильность тренда
        volatility_coef * 0.2 +       # 20% - низкая волатильность
        volume_coef * 0.3 +           # 30% - высокий объем
        regime_coef * 0.2              # 20% - трендовый режим
    )
    
    return combined

def classify_slope_range(slope):
    """
    Классифицирует slope на диапазоны
    
    Returns:
        str: категория slope
    """
    if slope > 2.0:
        return 'strong_up'
    elif slope > 0.5:
        return 'moderate_up'
    elif slope > 0.001:
        return 'weak_up'
    elif slope > -0.001:
        return 'flat'
    elif slope > -0.5:
        return 'weak_down'
    elif slope > -2.0:
        return 'moderate_down'
    else:
        return 'strong_down'

def classify_combined_coefficient(coef):
    """
    Классифицирует комбинированный коэффициент на диапазоны
    
    Returns:
        str: категория коэффициента
    """
    if coef >= 0.8:
        return 'very_high'
    elif coef >= 0.65:
        return 'high'
    elif coef >= 0.5:
        return 'medium'
    elif coef >= 0.35:
        return 'low'
    else:
        return 'very_low'

def create_simple_classes(zones_df):
    """
    Создает упрощенную классификацию на основе slope и комбинированного коэффициента
    
    Args:
        zones_df: DataFrame с зонами
    
    Returns:
        DataFrame с добавленными колонками для упрощенной классификации
    """
    print(f"\n[CLASSIFY] Создание упрощенной классификации...")
    
    # Вычисляем комбинированный коэффициент для каждой зоны
    zones_df = zones_df.copy()
    zones_df['combined_coefficient'] = zones_df.apply(calculate_combined_coefficient, axis=1)
    
    # Классифицируем slope
    zones_df['slope_category'] = zones_df['trend_slope'].apply(classify_slope_range)
    
    # Классифицируем комбинированный коэффициент
    zones_df['coefficient_category'] = zones_df['combined_coefficient'].apply(classify_combined_coefficient)
    
    # Создаем упрощенный класс (комбинация slope и коэффициента)
    zones_df['simple_class'] = zones_df.apply(
        lambda row: f"{row['slope_category']}_{row['coefficient_category']}", 
        axis=1
    )
    
    # Статистика
    print(f"   [OK] Создано {zones_df['simple_class'].nunique()} уникальных классов")
    print(f"\n   [СТАТИСТИКА] Распределение классов:")
    class_counts = zones_df['simple_class'].value_counts()
    for class_name, count in class_counts.items():
        pct = (count / len(zones_df)) * 100
        print(f"      {class_name}: {count} ({pct:.1f}%)")
    
    # Статистика по slope категориям
    print(f"\n   [SLOPE] Распределение по категориям slope:")
    slope_counts = zones_df['slope_category'].value_counts()
    for slope_cat, count in slope_counts.items():
        pct = (count / len(zones_df)) * 100
        print(f"      {slope_cat}: {count} ({pct:.1f}%)")
    
    # Статистика по коэффициентам
    print(f"\n   [COEFFICIENT] Распределение по категориям коэффициента:")
    coef_counts = zones_df['coefficient_category'].value_counts()
    for coef_cat, count in coef_counts.items():
        pct = (count / len(zones_df)) * 100
        print(f"      {coef_cat}: {count} ({pct:.1f}%)")
    
    return zones_df

def main():
    print("=" * 80)
    print("СОЗДАНИЕ УПРОЩЕННОЙ КЛАССИФИКАЦИИ")
    print("На основе slope и комбинированного коэффициента")
    print("=" * 80)
    
    # Загрузка оптимальных параметров
    params_file = Path('results/optimization_15m_adaptive.json')
    if not params_file.exists():
        print(f"\n[ERROR] Файл с оптимальными параметрами не найден: {params_file}")
        return
    
    with open(params_file, 'r', encoding='utf-8') as f:
        optimization_result = json.load(f)
    
    best_config = optimization_result['best_config']
    print(f"\n[LOAD] Оптимальные параметры:")
    print(f"   N={best_config['N']}, overlap={best_config['overlap_ratio']}, "
          f"alpha={best_config['alpha']}, beta={best_config['beta']}")
    
    # Загрузка данных
    train_file = Path('data/train/df_btc_15m_train.csv')
    if not train_file.exists():
        train_file = Path('data/prepared/df_btc_15m_prepared.csv')
        if not train_file.exists():
            print(f"\n[ERROR] Данные не найдены")
            return
    
    print(f"\n[LOAD] Загрузка данных...")
    df = pd.read_csv(train_file, index_col=0, parse_dates=True)
    print(f"   [OK] Загружено {len(df)} записей")
    
    # Создание сегментов
    print(f"\n[SEGMENT] Создание сегментов...")
    config = Config(
        N=best_config['N'],
        overlap_ratio=best_config['overlap_ratio'],
        alpha=best_config['alpha'],
        beta=best_config['beta'],
        metrics_for_alpha=Metrics.RELATIVE_ABSOLUTE_ERROR,
        metrics_for_beta=Metrics.RELATIVE_ABSOLUTE_ERROR
    )
    
    seg = Segmenter(df=df, column="close", config=config)
    seg.calculate_segments()
    print(f"   [OK] Создано {len(seg.segments)} сегментов")
    
    # Создание матрицы зон
    print(f"\n[ZONES] Создание матрицы зон...")
    zones_df = create_market_zones_matrix(df, seg)
    print(f"   [OK] Создано {len(zones_df)} зон")
    
    # Создание упрощенной классификации
    zones_df_with_classes = create_simple_classes(zones_df)
    
    # Сохранение результатов
    output_file = Path('results/zones_with_simple_classes.csv')
    output_file.parent.mkdir(exist_ok=True)
    zones_df_with_classes.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n[SAVE] Результаты сохранены: {output_file}")
    
    # Сохранение метаданных
    metadata = {
        'total_zones': len(zones_df_with_classes),
        'unique_classes': int(zones_df_with_classes['simple_class'].nunique()),
        'class_distribution': zones_df_with_classes['simple_class'].value_counts().to_dict(),
        'slope_categories': zones_df_with_classes['slope_category'].value_counts().to_dict(),
        'coefficient_categories': zones_df_with_classes['coefficient_category'].value_counts().to_dict(),
        'slope_ranges': {
            'strong_up': 'slope > 2.0',
            'moderate_up': '0.5 < slope <= 2.0',
            'weak_up': '0.001 < slope <= 0.5',
            'flat': '-0.001 <= slope <= 0.001',
            'weak_down': '-0.5 <= slope < -0.001',
            'moderate_down': '-2.0 <= slope < -0.5',
            'strong_down': 'slope < -2.0'
        },
        'coefficient_ranges': {
            'very_high': 'coef >= 0.8',
            'high': '0.65 <= coef < 0.8',
            'medium': '0.5 <= coef < 0.65',
            'low': '0.35 <= coef < 0.5',
            'very_low': 'coef < 0.35'
        }
    }
    
    metadata_file = Path('results/simple_classes_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"   Метаданные: {metadata_file}")
    
    print(f"\n" + "=" * 80)
    print("[SUCCESS] УПРОЩЕННАЯ КЛАССИФИКАЦИЯ СОЗДАНА!")
    print("=" * 80)
    
    print(f"\nСледующие шаги:")
    print(f"   1. Создать матрицы индикаторов для упрощенных классов")
    print(f"   2. Проверить точность индикаторов в определении классов")

if __name__ == "__main__":
    main()

