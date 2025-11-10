"""
Скрипт 19: Создание матриц индикаторов для упрощенных классов
Создает матрицы диапазонов индикаторов для каждого упрощенного класса
Индикаторы: RSI, MACD, BB, ATR, Volume, ADX, Stochastic, CCI
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

# Импорт функций классификации (не используется в этом скрипте, но оставлено для совместимости)

def calculate_rsi(prices, period=14):
    """Вычисляет RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Вычисляет MACD и возвращает разницу с сигналом"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Вычисляет позицию цены в полосах Боллинджера"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    return bb_position

def calculate_atr_ratio(df, period=14):
    """Вычисляет ATR ratio"""
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
    """Вычисляет отношение объема к среднему"""
    avg_volume = volume.rolling(window=100).mean()
    return volume / avg_volume

def calculate_adx(df, period=14):
    """
    Вычисляет ADX (Average Directional Index)
    ADX показывает силу тренда (0-100)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Smoothing
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_stochastic(df, period=14, smooth_k=3, smooth_d=3):
    """
    Вычисляет Stochastic Oscillator
    Возвращает %K и %D
    """
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
    """
    Вычисляет CCI (Commodity Channel Index)
    """
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

def create_indicators_matrix_simple(df, zones_df):
    """
    Создает матрицу диапазонов индикаторов для каждого упрощенного класса
    
    Args:
        df: DataFrame с данными
        zones_df: DataFrame с зонами и упрощенными классами
    
    Returns:
        dict: матрица индикаторов по классам
    """
    print(f"\n[INDICATORS] Вычисление индикаторов...")
    
    # Вычисляем все индикаторы
    close = df['close']
    
    rsi = calculate_rsi(close, period=14)
    macd_diff = calculate_macd(close)
    bb_position = calculate_bollinger_bands(close)
    atr_ratio = calculate_atr_ratio(df)
    volume_ratio = calculate_volume_ratio(df['volume'])
    adx = calculate_adx(df, period=14)
    stochastic_k, stochastic_d = calculate_stochastic(df, period=14)
    cci = calculate_cci(df, period=20)
    
    print(f"   [OK] Все индикаторы вычислены")
    
    # Создаем матрицу для каждого класса
    print(f"\n[MATRIX] Создание матрицы индикаторов для классов...")
    
    indicators_matrix = {}
    unique_classes = zones_df['simple_class'].unique()
    
    for class_name in unique_classes:
        # Находим все зоны этого класса
        class_zones = zones_df[zones_df['simple_class'] == class_name]
        
        if len(class_zones) == 0:
            continue
        
        # Собираем все данные индикаторов для этого класса
        all_indicators_data = []
        
        for _, zone_row in class_zones.iterrows():
            start_idx = int(zone_row['start_idx'])
            stop_idx = int(zone_row['stop_idx'])
            
            if start_idx >= len(df) or stop_idx > len(df) or start_idx >= stop_idx:
                continue
            
            zone_data = df.iloc[start_idx:stop_idx]
            
            # Собираем индикаторы для этой зоны
            zone_indicators = pd.DataFrame({
                'rsi': rsi.iloc[start_idx:stop_idx],
                'macd_diff': macd_diff.iloc[start_idx:stop_idx],
                'bb_position': bb_position.iloc[start_idx:stop_idx],
                'atr_ratio': atr_ratio.iloc[start_idx:stop_idx],
                'volume_ratio': volume_ratio.iloc[start_idx:stop_idx],
                'adx': adx.iloc[start_idx:stop_idx],
                'stochastic_k': stochastic_k.iloc[start_idx:stop_idx],
                'stochastic_d': stochastic_d.iloc[start_idx:stop_idx],
                'cci': cci.iloc[start_idx:stop_idx]
            })
            
            all_indicators_data.append(zone_indicators)
        
        if not all_indicators_data:
            continue
        
        # Объединяем все данные
        all_class_data = pd.concat(all_indicators_data, ignore_index=True)
        
        # Вычисляем статистику для каждого индикатора
        indicators_ranges = {}
        
        for indicator_name in ['rsi', 'macd_diff', 'bb_position', 'atr_ratio', 'volume_ratio', 
                              'adx', 'stochastic_k', 'stochastic_d', 'cci']:
            if indicator_name not in all_class_data.columns:
                continue
            
            indicator_data = all_class_data[indicator_name].dropna()
            
            if len(indicator_data) == 0:
                continue
            
            indicators_ranges[indicator_name] = {
                'min': float(indicator_data.min()),
                'max': float(indicator_data.max()),
                'median': float(indicator_data.median()),
                'mean': float(indicator_data.mean()),
                'std': float(indicator_data.std()),
                'percentile_25': float(indicator_data.quantile(0.25)),
                'percentile_75': float(indicator_data.quantile(0.75)),
                'count': int(len(indicator_data))
            }
        
        indicators_matrix[class_name] = indicators_ranges
        
        print(f"   [OK] Класс '{class_name}': {len(class_zones)} зон, {len(all_class_data)} точек данных")
    
    return indicators_matrix

def main():
    print("=" * 80)
    print("СОЗДАНИЕ МАТРИЦЫ ИНДИКАТОРОВ ДЛЯ УПРОЩЕННЫХ КЛАССОВ")
    print("Индикаторы: RSI, MACD, BB, ATR, Volume, ADX, Stochastic, CCI")
    print("=" * 80)
    
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
    
    # Загрузка зон с упрощенными классами
    zones_file = Path('results/zones_with_simple_classes.csv')
    if not zones_file.exists():
        print(f"\n[ERROR] Файл с упрощенными классами не найден: {zones_file}")
        print(f"   Запустите сначала scripts/18_create_simple_classes.py")
        return
    
    print(f"\n[LOAD] Загрузка упрощенных классов...")
    zones_df = pd.read_csv(zones_file)
    print(f"   [OK] Загружено {len(zones_df)} зон")
    print(f"   Уникальных классов: {zones_df['simple_class'].nunique()}")
    
    # Создание матрицы индикаторов
    indicators_matrix = create_indicators_matrix_simple(df, zones_df)
    
    # Сохранение результатов
    output_file = Path('results/indicators_matrix_simple_classes.json')
    output_file.parent.mkdir(exist_ok=True)
    
    metadata = {
        'metadata': {
            'total_classes': len(indicators_matrix),
            'indicators': ['rsi', 'macd_diff', 'bb_position', 'atr_ratio', 'volume_ratio', 
                          'adx', 'stochastic_k', 'stochastic_d', 'cci'],
            'created_at': pd.Timestamp.now().isoformat()
        },
        'indicators_by_class': indicators_matrix
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n[SAVE] Матрица индикаторов сохранена: {output_file}")
    
    # Статистика
    print(f"\n[STATS] Статистика по классам:")
    for class_name, indicators in indicators_matrix.items():
        print(f"   {class_name}:")
        for ind_name, stats in indicators.items():
            print(f"      {ind_name}: min={stats['min']:.3f}, max={stats['max']:.3f}, "
                  f"median={stats['median']:.3f}, count={stats['count']}")
    
    print(f"\n" + "=" * 80)
    print("[SUCCESS] МАТРИЦА ИНДИКАТОРОВ СОЗДАНА!")
    print("=" * 80)
    
    print(f"\nСледующий шаг:")
    print(f"   Проверить точность индикаторов в определении упрощенных классов")

if __name__ == "__main__":
    main()

