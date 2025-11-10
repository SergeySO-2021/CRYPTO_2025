"""
Скрипт 17: Проверка качества Trend Classifier
Сравнивает знак slope (тренд) с реальным изменением цены BTC
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

def validate_trend_classifier_quality():
    """
    Проверяет качество Trend Classifier:
    1. Сравнивает знак slope с реальным изменением цены
    2. Вычисляет процент правильных определений тренда
    """
    print("=" * 80)
    print("ПРОВЕРКА КАЧЕСТВА TREND CLASSIFIER")
    print("Сравнение slope с реальным изменением цены BTC")
    print("=" * 80)
    
    # Загрузка оптимальных параметров
    params_file = Path('results/optimization_15m_adaptive.json')
    if not params_file.exists():
        print(f"\n[ERROR] Файл с оптимальными параметрами не найден: {params_file}")
        return None
    
    with open(params_file, 'r', encoding='utf-8') as f:
        optimization_result = json.load(f)
    
    best_config = optimization_result['best_config']
    print(f"\n[LOAD] Оптимальные параметры:")
    print(f"   N={best_config['N']}, overlap={best_config['overlap_ratio']}, "
          f"alpha={best_config['alpha']}, beta={best_config['beta']}")
    
    # Загрузка данных (train set)
    train_file = Path('data/train/df_btc_15m_train.csv')
    if not train_file.exists():
        # Fallback на prepared data
        train_file = Path('data/prepared/df_btc_15m_prepared.csv')
        if not train_file.exists():
            print(f"\n[ERROR] Данные не найдены")
            return None
    
    print(f"\n[LOAD] Загрузка данных...")
    df = pd.read_csv(train_file, index_col=0, parse_dates=True)
    print(f"   [OK] Загружено {len(df)} записей")
    print(f"   Период: {df.index[0]} - {df.index[-1]}")
    
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
    
    # Создание матрицы зон для получения полной информации
    print(f"\n[ZONES] Создание матрицы зон...")
    zones_df = create_market_zones_matrix(df, seg)
    print(f"   [OK] Создано {len(zones_df)} зон")
    
    # Проверка качества: сравнение slope с реальным изменением цены
    print(f"\n[VALIDATE] Проверка качества Trend Classifier...")
    
    validation_results = []
    
    for _, zone_row in zones_df.iterrows():
        start_idx = int(zone_row['start_idx'])
        stop_idx = int(zone_row['stop_idx'])
        
        if start_idx >= len(df) or stop_idx > len(df) or start_idx >= stop_idx:
            continue
        
        # Данные сегмента
        segment_data = df.iloc[start_idx:stop_idx]
        
        # Реальное изменение цены
        price_start = segment_data['close'].iloc[0]
        price_end = segment_data['close'].iloc[-1]
        price_change = price_end - price_start
        price_change_pct = ((price_end - price_start) / price_start) * 100
        
        # Тренд по Trend Classifier
        trend_slope = zone_row['trend_slope']
        trend_direction = zone_row['trend_direction']  # up/down/flat
        
        # Определяем реальное направление по изменению цены
        if price_change_pct > 0.1:  # Порог для роста (0.1%)
            real_direction = 'up'
        elif price_change_pct < -0.1:  # Порог для падения
            real_direction = 'down'
        else:
            real_direction = 'flat'
        
        # Проверка совпадения
        direction_match = (trend_direction == real_direction)
        
        # Дополнительная проверка: знак slope должен совпадать со знаком изменения цены
        slope_sign = 1 if trend_slope > 0.001 else (-1 if trend_slope < -0.001 else 0)
        price_sign = 1 if price_change > 0 else (-1 if price_change < 0 else 0)
        sign_match = (slope_sign == price_sign) or (slope_sign == 0 and abs(price_change_pct) < 0.1)
        
        validation_results.append({
            'zone_id': zone_row['zone_id'],
            'zone_number': zone_row['zone_number'],
            'start_time': zone_row['start_time'],
            'stop_time': zone_row['stop_time'],
            'duration_candles': zone_row['duration_candles'],
            'trend_slope': trend_slope,
            'trend_direction': trend_direction,
            'price_start': price_start,
            'price_end': price_end,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'real_direction': real_direction,
            'direction_match': direction_match,
            'slope_sign': slope_sign,
            'price_sign': price_sign,
            'sign_match': sign_match,
            'volatility': zone_row['volatility'],
            'regime': zone_row['regime'],
            'volume_level': zone_row['volume_level']
        })
    
    results_df = pd.DataFrame(validation_results)
    
    # Статистика
    print(f"\n[RESULTS] Результаты проверки:")
    print(f"   Всего сегментов: {len(results_df)}")
    
    # Точность по направлениям
    direction_accuracy = results_df['direction_match'].sum() / len(results_df)
    print(f"\n   [ТОЧНОСТЬ] По направлениям (up/down/flat):")
    print(f"      Правильных: {results_df['direction_match'].sum()}/{len(results_df)}")
    print(f"      Точность: {direction_accuracy:.2%}")
    
    # Точность по знакам
    sign_accuracy = results_df['sign_match'].sum() / len(results_df)
    print(f"\n   [ТОЧНОСТЬ] По знакам (slope vs price change):")
    print(f"      Правильных: {results_df['sign_match'].sum()}/{len(results_df)}")
    print(f"      Точность: {sign_accuracy:.2%}")
    
    # Детальная статистика по направлениям
    print(f"\n   [ДЕТАЛИ] Статистика по направлениям:")
    for direction in ['up', 'down', 'flat']:
        direction_data = results_df[results_df['trend_direction'] == direction]
        if len(direction_data) > 0:
            correct = direction_data['direction_match'].sum()
            total = len(direction_data)
            accuracy = correct / total if total > 0 else 0
            print(f"      {direction.upper()}: {correct}/{total} ({accuracy:.2%})")
    
    # Анализ ошибок
    errors = results_df[~results_df['direction_match']]
    if len(errors) > 0:
        print(f"\n   [ОШИБКИ] Примеры несовпадений:")
        for _, error in errors.head(10).iterrows():
            print(f"      Зона {error['zone_number']}:")
            print(f"         Trend Classifier: {error['trend_direction']} (slope={error['trend_slope']:.6f})")
            print(f"         Реальное: {error['real_direction']} (изменение={error['price_change_pct']:.2f}%)")
            print(f"         Период: {error['start_time']} - {error['stop_time']}")
    
    # Корреляция между slope и изменением цены
    correlation = results_df['trend_slope'].corr(results_df['price_change_pct'])
    print(f"\n   [КОРРЕЛЯЦИЯ] Slope vs Price Change %:")
    print(f"      Корреляция: {correlation:.4f}")
    
    # Статистика по величине изменений
    print(f"\n   [СТАТИСТИКА] Распределение изменений цены:")
    print(f"      Среднее изменение: {results_df['price_change_pct'].mean():.2f}%")
    print(f"      Медианное изменение: {results_df['price_change_pct'].median():.2f}%")
    print(f"      Min изменение: {results_df['price_change_pct'].min():.2f}%")
    print(f"      Max изменение: {results_df['price_change_pct'].max():.2f}%")
    print(f"      Std изменение: {results_df['price_change_pct'].std():.2f}%")
    
    # Сохранение результатов
    output_file = Path('results/trend_classifier_validation.json')
    output_file.parent.mkdir(exist_ok=True)
    
    summary = {
        'total_segments': len(results_df),
        'direction_accuracy': float(direction_accuracy),
        'sign_accuracy': float(sign_accuracy),
        'correlation_slope_price': float(correlation),
        'direction_stats': {
            direction: {
                'total': int(len(results_df[results_df['trend_direction'] == direction])),
                'correct': int(results_df[results_df['trend_direction'] == direction]['direction_match'].sum()),
                'accuracy': float(results_df[results_df['trend_direction'] == direction]['direction_match'].sum() / 
                               len(results_df[results_df['trend_direction'] == direction]))
                if len(results_df[results_df['trend_direction'] == direction]) > 0 else 0.0
            }
            for direction in ['up', 'down', 'flat']
        },
        'price_change_stats': {
            'mean': float(results_df['price_change_pct'].mean()),
            'median': float(results_df['price_change_pct'].median()),
            'min': float(results_df['price_change_pct'].min()),
            'max': float(results_df['price_change_pct'].max()),
            'std': float(results_df['price_change_pct'].std())
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    
    # Сохранение детальных результатов
    results_csv = Path('results/trend_classifier_validation_details.csv')
    results_df.to_csv(results_csv, index=False, encoding='utf-8')
    
    print(f"\n[SAVE] Результаты сохранены:")
    print(f"   Сводка: {output_file}")
    print(f"   Детали: {results_csv}")
    
    # Оценка
    print(f"\n" + "=" * 80)
    print("ОЦЕНКА КАЧЕСТВА TREND CLASSIFIER")
    print("=" * 80)
    
    if direction_accuracy >= 0.9:
        print(f"✅ Отличное качество: {direction_accuracy:.2%}")
        print(f"   Trend Classifier очень точно определяет направления трендов")
    elif direction_accuracy >= 0.8:
        print(f"✅ Хорошее качество: {direction_accuracy:.2%}")
        print(f"   Trend Classifier хорошо определяет направления трендов")
    elif direction_accuracy >= 0.7:
        print(f"⚠️  Приемлемое качество: {direction_accuracy:.2%}")
        print(f"   Trend Classifier определяет направления, но есть ошибки")
    else:
        print(f"❌ Низкое качество: {direction_accuracy:.2%}")
        print(f"   Требуется пересмотр параметров Trend Classifier")
    
    if correlation > 0.7:
        print(f"\n✅ Сильная корреляция slope и изменения цены: {correlation:.4f}")
    elif correlation > 0.5:
        print(f"\n⚠️  Умеренная корреляция: {correlation:.4f}")
    else:
        print(f"\n❌ Слабая корреляция: {correlation:.4f}")
    
    print(f"\n" + "=" * 80)
    print("[SUCCESS] ПРОВЕРКА ЗАВЕРШЕНА!")
    print("=" * 80)
    
    return summary

def main():
    try:
        results = validate_trend_classifier_quality()
        
        if results:
            print(f"\nРекомендации:")
            if results['direction_accuracy'] < 0.7:
                print(f"   1. Пересмотреть параметры Trend Classifier")
                print(f"   2. Увеличить пороги alpha/beta для более стабильных сегментов")
            elif results['direction_accuracy'] < 0.8:
                print(f"   1. Рассмотреть улучшение параметров")
                print(f"   2. Проверить качество данных")
            else:
                print(f"   ✅ Trend Classifier работает хорошо, можно продолжать")
        
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

