"""
Скрипт 20: Проверка точности индикаторов в определении упрощенных классов
Сравнивает предсказания индикаторов с реальными классами от Trend Classifier
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

# Функции вычисления индикаторов (копируем из предыдущего скрипта)
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    return bb_position

def calculate_atr_ratio(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    close_prev = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - close_prev)
    tr3 = abs(low - close_prev)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr / close

def calculate_volume_ratio(volume):
    avg_volume = volume.rolling(window=100).mean()
    return volume / avg_volume

def calculate_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx

def calculate_stochastic(df, period=14, smooth_k=3, smooth_d=3):
    high = df['high']
    low = df['low']
    close = df['close']
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    k_smooth = k_percent.rolling(window=smooth_k).mean()
    d_smooth = k_smooth.rolling(window=smooth_d).mean()
    return k_smooth, d_smooth

def calculate_cci(df, period=20):
    high = df['high']
    low = df['low']
    close = df['close']
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean()))
    )
    cci = (typical_price - sma) / (0.015 * mad)
    return cci

class SimpleClassClassifier:
    """Классификатор упрощенных классов на основе индикаторов"""
    
    def __init__(self, indicators_matrix_path):
        with open(indicators_matrix_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.indicators_matrix = data['indicators_by_class']
        self.indicator_names = ['rsi', 'macd_diff', 'bb_position', 'atr_ratio', 'volume_ratio',
                               'adx', 'stochastic_k', 'stochastic_d', 'cci']
    
    def calculate_current_indicators(self, df, current_time=None):
        """Вычисляет текущие значения индикаторов"""
        if current_time is not None:
            available_data = df[df.index <= current_time].copy()
        else:
            available_data = df.copy()
        
        if len(available_data) < 100:
            return None
        
        close = available_data['close']
        current_indicators = {}
        
        if len(close) >= 14:
            rsi = calculate_rsi(close, period=14)
            if not rsi.isna().iloc[-1]:
                current_indicators['rsi'] = float(rsi.iloc[-1])
        
        if len(close) >= 26:
            macd_diff = calculate_macd(close)
            if not macd_diff.isna().iloc[-1]:
                current_indicators['macd_diff'] = float(macd_diff.iloc[-1])
        
        if len(close) >= 20:
            bb_position = calculate_bollinger_bands(close)
            if not bb_position.isna().iloc[-1]:
                current_indicators['bb_position'] = float(bb_position.iloc[-1])
        
        if len(available_data) >= 14:
            atr_ratio = calculate_atr_ratio(available_data)
            if not atr_ratio.isna().iloc[-1]:
                current_indicators['atr_ratio'] = float(atr_ratio.iloc[-1])
        
        if len(available_data) >= 100 and 'volume' in available_data.columns:
            volume_ratio = calculate_volume_ratio(available_data['volume'])
            if not volume_ratio.isna().iloc[-1]:
                current_indicators['volume_ratio'] = float(volume_ratio.iloc[-1])
        
        if len(available_data) >= 14:
            adx = calculate_adx(available_data, period=14)
            if not adx.isna().iloc[-1]:
                current_indicators['adx'] = float(adx.iloc[-1])
        
        if len(available_data) >= 14:
            stochastic_k, stochastic_d = calculate_stochastic(available_data, period=14)
            if not stochastic_k.isna().iloc[-1]:
                current_indicators['stochastic_k'] = float(stochastic_k.iloc[-1])
            if not stochastic_d.isna().iloc[-1]:
                current_indicators['stochastic_d'] = float(stochastic_d.iloc[-1])
        
        if len(available_data) >= 20:
            cci = calculate_cci(available_data, period=20)
            if not cci.isna().iloc[-1]:
                current_indicators['cci'] = float(cci.iloc[-1])
        
        return current_indicators if current_indicators else None
    
    def match_indicators_to_class(self, current_indicators):
        """Сопоставляет индикаторы с классами"""
        matches = []
        
        for class_name, class_indicators in self.indicators_matrix.items():
            score = 0
            total_weight = 0
            matched_indicators = 0
            
            for ind_name in self.indicator_names:
                if ind_name not in current_indicators or ind_name not in class_indicators:
                    continue
                
                current_value = current_indicators[ind_name]
                class_stats = class_indicators[ind_name]
                
                min_val = class_stats.get('min', float('-inf'))
                max_val = class_stats.get('max', float('inf'))
                percentile_25 = class_stats.get('percentile_25')
                percentile_75 = class_stats.get('percentile_75')
                median = class_stats.get('median', 0)
                
                # Оценка по percentiles (более точная)
                if percentile_25 is not None and percentile_75 is not None:
                    if percentile_25 <= current_value <= percentile_75:
                        indicator_score = 1.0  # Идеальное совпадение
                    elif min_val <= current_value <= max_val:
                        indicator_score = 0.7  # Хорошее совпадение
                    else:
                        indicator_score = 0.0
                        continue
                else:
                    # Fallback на min/max
                    if min_val <= current_value <= max_val:
                        indicator_score = 0.7
                    else:
                        indicator_score = 0.0
                        continue
                
                matched_indicators += 1
                
                # Бонус за близость к median
                if median != 0 and class_stats.get('std', 0) > 0:
                    z_score = abs(current_value - median) / class_stats.get('std', 1)
                    proximity_bonus = max(0, 1 - z_score / 2)
                    indicator_score = (indicator_score + proximity_bonus) / 2
                
                # Веса индикаторов
                weight = 1.5 if ind_name in ['rsi', 'macd_diff'] else 1.2 if ind_name in ['adx', 'stochastic_k'] else 1.0
                score += indicator_score * weight
                total_weight += weight
            
            if matched_indicators > 0:
                normalized_score = score / total_weight if total_weight > 0 else 0
                confidence = matched_indicators / len(self.indicator_names)
                matches.append((class_name, normalized_score, confidence))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def classify(self, df, current_time=None):
        """Классифицирует текущую ситуацию на класс"""
        if current_time is None:
            current_time = df.index[-1]
        
        current_indicators = self.calculate_current_indicators(df, current_time=current_time)
        if not current_indicators:
            return {'class': None, 'confidence': 0.0, 'score': 0.0, 'error': 'Недостаточно данных'}
        
        matches = self.match_indicators_to_class(current_indicators)
        if not matches:
            return {'class': None, 'confidence': 0.0, 'score': 0.0, 'error': 'Не найдено совпадений'}
        
        best_match = matches[0]
        class_name, score, confidence = best_match
        
        return {
            'class': class_name,
            'score': score,
            'confidence': confidence,
            'current_indicators': current_indicators,
            'top_3_matches': [{'class': c, 'score': s, 'confidence': conf} for c, s, conf in matches[:3]]
        }

def validate_indicators_accuracy():
    """Проверяет точность индикаторов в определении упрощенных классов"""
    print("=" * 80)
    print("ПРОВЕРКА ТОЧНОСТИ ИНДИКАТОРОВ")
    print("Сравнение предсказаний индикаторов с классами Trend Classifier")
    print("=" * 80)
    
    # Загрузка данных
    train_file = Path('data/train/df_btc_15m_train.csv')
    if not train_file.exists():
        train_file = Path('data/prepared/df_btc_15m_prepared.csv')
        if not train_file.exists():
            print(f"\n[ERROR] Данные не найдены")
            return None
    
    print(f"\n[LOAD] Загрузка данных...")
    df = pd.read_csv(train_file, index_col=0, parse_dates=True)
    print(f"   [OK] Загружено {len(df)} записей")
    
    # Загрузка зон с классами
    zones_file = Path('results/zones_with_simple_classes.csv')
    if not zones_file.exists():
        print(f"\n[ERROR] Файл с классами не найден: {zones_file}")
        return None
    
    print(f"\n[LOAD] Загрузка классов...")
    zones_df = pd.read_csv(zones_file)
    print(f"   [OK] Загружено {len(zones_df)} зон")
    print(f"   Уникальных классов: {zones_df['simple_class'].nunique()}")
    
    # Загрузка классификатора
    matrix_file = Path('results/indicators_matrix_simple_classes.json')
    if not matrix_file.exists():
        print(f"\n[ERROR] Файл матрицы индикаторов не найден: {matrix_file}")
        return None
    
    print(f"\n[LOAD] Загрузка классификатора...")
    classifier = SimpleClassClassifier(matrix_file)
    print(f"   [OK] Загружено {len(classifier.indicators_matrix)} классов")
    
    # Создание словаря для быстрого поиска класса по времени
    class_by_time = {}
    for _, zone_row in zones_df.iterrows():
        start_time = pd.to_datetime(zone_row['start_time'])
        stop_time = pd.to_datetime(zone_row['stop_time'])
        class_name = zone_row['simple_class']
        
        zone_data = df[(df.index >= start_time) & (df.index <= stop_time)]
        for time_point in zone_data.index:
            class_by_time[time_point] = class_name
    
    # Проверка точности
    print(f"\n[VALIDATE] Проверка точности на {len(class_by_time)} точках...")
    
    comparison_results = []
    sample_points = list(class_by_time.keys())[100::100]  # Каждые 100 точек
    
    for i, current_time in enumerate(sample_points):
        if current_time not in class_by_time:
            continue
        
        true_class = class_by_time[current_time]
        
        # Предсказание классификатора
        result = classifier.classify(df, current_time=current_time)
        predicted_class = result.get('class')
        
        if predicted_class:
            comparison_results.append({
                'time': current_time,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'match': true_class == predicted_class,
                'score': result.get('score', 0),
                'confidence': result.get('confidence', 0)
            })
        
        if (i + 1) % 10 == 0:
            print(f"   Обработано: {i + 1}/{len(sample_points)} точек")
    
    if not comparison_results:
        print(f"\n[WARNING] Не удалось получить результаты")
        return None
    
    results_df = pd.DataFrame(comparison_results)
    
    # Статистика
    print(f"\n[RESULTS] Результаты проверки:")
    print(f"   Всего сравнений: {len(results_df)}")
    
    # Точность (exact match)
    exact_matches = results_df['match'].sum()
    accuracy_exact = exact_matches / len(results_df)
    print(f"\n   [ТОЧНОСТЬ] Exact match:")
    print(f"      Совпадений: {exact_matches}/{len(results_df)}")
    print(f"      Точность: {accuracy_exact:.2%}")
    
    # Точность по slope категориям (упрощенная проверка)
    def extract_slope_category(class_name):
        if class_name:
            parts = class_name.split('_')
            if len(parts) >= 2:
                return '_'.join(parts[:2])  # strong_up, moderate_down и т.д.
        return None
    
    results_df['true_slope'] = results_df['true_class'].apply(extract_slope_category)
    results_df['predicted_slope'] = results_df['predicted_class'].apply(extract_slope_category)
    results_df['slope_match'] = results_df['true_slope'] == results_df['predicted_slope']
    
    slope_matches = results_df['slope_match'].sum()
    accuracy_slope = slope_matches / len(results_df)
    print(f"\n   [ТОЧНОСТЬ] По slope категориям:")
    print(f"      Совпадений: {slope_matches}/{len(results_df)}")
    print(f"      Точность: {accuracy_slope:.2%}")
    
    # Confidence
    print(f"\n   [CONFIDENCE] Распределение:")
    print(f"      Средняя confidence: {results_df['confidence'].mean():.2%}")
    print(f"      Медианная confidence: {results_df['confidence'].median():.2%}")
    
    # Точность в зависимости от confidence
    print(f"\n   [АНАЛИЗ] Точность в зависимости от confidence:")
    for conf_threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        high_conf = results_df[results_df['confidence'] >= conf_threshold]
        if len(high_conf) > 0:
            high_conf_accuracy = high_conf['match'].sum() / len(high_conf)
            print(f"      Confidence >= {conf_threshold:.1f}: {high_conf_accuracy:.2%} "
                  f"({len(high_conf)} случаев)")
    
    # Статистика по классам
    print(f"\n   [КЛАССЫ] Статистика:")
    print(f"      Уникальных классов (true): {results_df['true_class'].nunique()}")
    print(f"      Уникальных классов (predicted): {results_df['predicted_class'].nunique()}")
    
    # Топ-5 наиболее частых классов
    print(f"\n   [ТОП-5] Наиболее частые классы (true):")
    top_true_classes = results_df['true_class'].value_counts().head(5)
    for class_name, count in top_true_classes.items():
        class_matches = results_df[results_df['true_class'] == class_name]['match'].sum()
        class_total = count
        class_accuracy = class_matches / class_total if class_total > 0 else 0
        print(f"      {class_name}: {count} раз, точность: {class_accuracy:.2%}")
    
    # Сохранение результатов
    output_file = Path('results/indicators_accuracy_validation.json')
    output_file.parent.mkdir(exist_ok=True)
    
    summary = {
        'total_comparisons': len(results_df),
        'accuracy_exact': float(accuracy_exact),
        'accuracy_slope': float(accuracy_slope),
        'avg_confidence': float(results_df['confidence'].mean()),
        'avg_score': float(results_df['score'].mean()),
        'unique_classes_true': int(results_df['true_class'].nunique()),
        'unique_classes_predicted': int(results_df['predicted_class'].nunique()),
        'confidence_analysis': {
            f'conf_{int(t*100)}': {
                'count': int(len(results_df[results_df['confidence'] >= t])),
                'accuracy': float(results_df[results_df['confidence'] >= t]['match'].sum() / 
                                len(results_df[results_df['confidence'] >= t]))
                if len(results_df[results_df['confidence'] >= t]) > 0 else 0.0
            }
            for t in [0.5, 0.6, 0.7, 0.8, 0.9]
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n[SAVE] Результаты сохранены: {output_file}")
    
    # Оценка
    print(f"\n" + "=" * 80)
    print("ОЦЕНКА ТОЧНОСТИ ИНДИКАТОРОВ")
    print("=" * 80)
    
    if accuracy_exact >= 0.7:
        print(f"✅ Хорошая точность: {accuracy_exact:.2%}")
        print(f"   Индикаторы хорошо определяют упрощенные классы")
    elif accuracy_exact >= 0.5:
        print(f"⚠️  Приемлемая точность: {accuracy_exact:.2%}")
        print(f"   Индикаторы определяют классы, но есть место для улучшения")
    else:
        print(f"❌ Низкая точность: {accuracy_exact:.2%}")
        print(f"   Требуется улучшение логики классификации")
    
    if accuracy_slope >= 0.7:
        print(f"\n✅ Хорошая точность по slope: {accuracy_slope:.2%}")
        print(f"   Индикаторы правильно определяют направление и силу тренда")
    
    print(f"\n" + "=" * 80)
    print("[SUCCESS] ПРОВЕРКА ЗАВЕРШЕНА!")
    print("=" * 80)
    
    return summary

def main():
    try:
        results = validate_indicators_accuracy()
        
        if results:
            print(f"\nРекомендации:")
            if results['accuracy_exact'] < 0.5:
                print(f"   1. Упростить классификацию еще больше")
                print(f"   2. Улучшить логику сопоставления")
                print(f"   3. Добавить больше индикаторов")
            elif results['accuracy_exact'] < 0.7:
                print(f"   1. Рассмотреть улучшение логики")
                print(f"   2. Использовать иерархическую классификацию")
            else:
                print(f"   ✅ Точность приемлема, можно использовать в продакшене")
        
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
