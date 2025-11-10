"""
Функции классификации рыночных зон
Все функции для расширенной классификации зон
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# ============================================================================
# ВОЛАТИЛЬНОСТЬ
# ============================================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Вычисляет ATR (Average True Range)"""
    high = df['high']
    low = df['low']
    close = df['close']
    close_prev = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close_prev)
    tr3 = abs(low - close_prev)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr

def classify_volatility_atr(df: pd.DataFrame, segment, atr_period: int = 14) -> str:
    """Классифицирует волатильность по ATR"""
    segment_data = df.iloc[segment.start:segment.stop]
    atr = calculate_atr(df, period=atr_period)
    segment_atr = atr.iloc[segment.start:segment.stop].mean()
    baseline_atr = atr.rolling(100).median().iloc[-1]
    
    ratio = segment_atr / baseline_atr if baseline_atr > 0 else 0
    
    if ratio < 0.3:
        return 'very_low'
    elif ratio < 0.6:
        return 'low'
    elif ratio < 1.4:
        return 'medium'
    elif ratio < 2.0:
        return 'high'
    else:
        return 'very_high'

def classify_volatility_std(df: pd.DataFrame, segment, seg) -> str:
    """Классифицирует волатильность по стандартному отклонению"""
    segment_std = segment.std
    all_stds = [s.std for s in seg.segments]
    baseline_std = np.median(all_stds)
    
    ratio = segment_std / baseline_std if baseline_std > 0 else 0
    
    if ratio < 0.5:
        return 'low'
    elif ratio < 1.5:
        return 'medium'
    else:
        return 'high'

def classify_volatility_range(df: pd.DataFrame, segment, seg) -> str:
    """Классифицирует волатильность по размаху цен"""
    span = segment.span
    all_spans = [s.span for s in seg.segments]
    baseline_span = np.median(all_spans)
    
    ratio = span / baseline_span if baseline_span > 0 else 0
    
    if ratio < 0.5:
        return 'low'
    elif ratio < 1.5:
        return 'medium'
    else:
        return 'high'

def classify_volatility_combined(df: pd.DataFrame, segment, seg) -> str:
    """Комбинированная классификация волатильности"""
    vol_atr = classify_volatility_atr(df, segment)
    vol_std = classify_volatility_std(df, segment, seg)
    vol_range = classify_volatility_range(df, segment, seg)
    
    vol_scores = {
        'very_low': 0, 'low': 1, 'medium': 2, 'high': 3, 'very_high': 4
    }
    
    vol_atr_score = vol_scores.get(vol_atr, 2)
    vol_std_score = {'low': 1, 'medium': 2, 'high': 3}.get(vol_std, 2)
    vol_range_score = {'low': 1, 'medium': 2, 'high': 3}.get(vol_range, 2)
    
    avg_score = (vol_atr_score + vol_std_score + vol_range_score) / 3
    
    if avg_score < 1.0:
        return 'low'
    elif avg_score < 2.0:
        return 'medium'
    else:
        return 'high'

# ============================================================================
# ОБЪЕМЫ
# ============================================================================

def classify_volume_average(df: pd.DataFrame, segment) -> str:
    """Классифицирует объемы по среднему значению"""
    segment_data = df.iloc[segment.start:segment.stop]
    avg_volume = segment_data['volume'].mean()
    volume_median = df['volume'].rolling(100).median().iloc[-1]
    
    ratio = avg_volume / volume_median if volume_median > 0 else 0
    
    if ratio < 0.5:
        return 'very_low'
    elif ratio < 0.8:
        return 'low'
    elif ratio < 1.2:
        return 'medium'
    elif ratio < 1.5:
        return 'high'
    else:
        return 'very_high'

def classify_volume_pressure(df: pd.DataFrame, segment) -> str:
    """Классифицирует давление объемов (buy vs sell)"""
    segment_data = df.iloc[segment.start:segment.stop]
    
    if 'trades_buy_volume' not in segment_data.columns:
        return 'unknown'
    
    buy_vol = segment_data['trades_buy_volume'].sum()
    sell_vol = segment_data['trades_sell_volume'].sum()
    
    if buy_vol + sell_vol == 0:
        return 'balanced'
    
    buy_ratio = buy_vol / (buy_vol + sell_vol)
    
    if buy_ratio > 0.6:
        return 'strong_buy_pressure'
    elif buy_ratio > 0.55:
        return 'buy_pressure'
    elif buy_ratio < 0.4:
        return 'strong_sell_pressure'
    elif buy_ratio < 0.45:
        return 'sell_pressure'
    else:
        return 'balanced'

def classify_volume_trend(df: pd.DataFrame, segment) -> str:
    """Определяет тренд объемов"""
    segment_data = df.iloc[segment.start:segment.stop]
    x = np.arange(len(segment_data))
    y = segment_data['volume'].values
    
    if len(y) < 2:
        return 'unknown'
    
    slope = np.polyfit(x, y, 1)[0]
    
    if slope > 0.1:
        return 'increasing'
    elif slope < -0.1:
        return 'decreasing'
    else:
        return 'stable'

def classify_volume_combined(df: pd.DataFrame, segment) -> Dict:
    """Комбинированная классификация объемов"""
    vol_avg = classify_volume_average(df, segment)
    vol_pressure = classify_volume_pressure(df, segment)
    vol_trend = classify_volume_trend(df, segment)
    
    return {
        'level': vol_avg,
        'pressure': vol_pressure,
        'trend': vol_trend,
        'combined': f"{vol_avg}_{vol_pressure}_{vol_trend}"
    }

# ============================================================================
# РЫНОЧНЫЙ РЕЖИМ
# ============================================================================

def classify_regime_slope(segment) -> str:
    """Определяет режим по наклону тренда"""
    slope = segment.slope
    slope_std = segment.slopes_std
    
    slope_threshold = 0.001
    stability_threshold = 0.5
    
    is_stable = slope_std < abs(slope) * stability_threshold
    
    if abs(slope) > slope_threshold and is_stable:
        return 'trending_up' if slope > 0 else 'trending_down'
    else:
        return 'ranging'

def calculate_adx(df: pd.DataFrame, period: int = 14):
    """Вычисляет ADX (Average Directional Index)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm_smooth = plus_dm.rolling(window=period).mean()
    minus_dm_smooth = minus_dm.rolling(window=period).mean()
    tr_smooth = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
    adx = dx.rolling(window=period).mean()
    
    direction = pd.Series(index=df.index, dtype=str)
    direction[plus_di > minus_di] = 'up'
    direction[plus_di < minus_di] = 'down'
    direction[plus_di == minus_di] = 'neutral'
    
    return adx, direction

def classify_regime_adx(df: pd.DataFrame, segment, adx_period: int = 14) -> str:
    """Определяет режим по ADX"""
    adx, direction = calculate_adx(df, period=adx_period)
    
    segment_adx = adx.iloc[segment.start:segment.stop].mean()
    segment_direction = direction.iloc[segment.start:segment.stop].mode()
    
    if len(segment_direction) == 0:
        return 'ranging'
    
    segment_direction = segment_direction[0]
    
    if segment_adx > 25:
        return 'trending_up' if segment_direction == 'up' else 'trending_down'
    else:
        return 'ranging'

def classify_regime_breakout(df: pd.DataFrame, segment, breakout_threshold: float = 0.9) -> str:
    """Определяет режим по позиции цены"""
    segment_data = df.iloc[segment.start:segment.stop]
    
    price_high = segment_data['high'].max()
    price_low = segment_data['low'].min()
    price_range = price_high - price_low
    
    if price_range == 0:
        return 'ranging'
    
    current_price = segment_data['close'].iloc[-1]
    price_position = (current_price - price_low) / price_range
    
    if price_position > breakout_threshold:
        return 'breakout_up'
    elif price_position < (1 - breakout_threshold):
        return 'breakout_down'
    else:
        return 'ranging'

def classify_regime_cv(df: pd.DataFrame, segment) -> str:
    """Определяет режим по коэффициенту вариации"""
    segment_data = df.iloc[segment.start:segment.stop]
    
    price_mean = segment_data['close'].mean()
    price_std = segment_data['close'].std()
    
    if price_mean == 0:
        return 'ranging'
    
    cv = price_std / price_mean
    
    if cv < 0.01:
        return 'ranging'
    elif cv < 0.02:
        slope = segment.slope
        return 'trending_up' if slope > 0 else 'trending_down'
    else:
        return 'ranging'

def classify_regime_combined(df: pd.DataFrame, segment, seg) -> Dict:
    """Комбинированное определение рыночного режима"""
    regime_slope = classify_regime_slope(segment)
    regime_adx = classify_regime_adx(df, segment)
    regime_breakout = classify_regime_breakout(df, segment)
    regime_cv = classify_regime_cv(df, segment)
    
    scores = {'trending_up': 0, 'trending_down': 0, 'ranging': 0}
    
    # Подсчет голосов
    if regime_slope == 'trending_up':
        scores['trending_up'] += 2
    elif regime_slope == 'trending_down':
        scores['trending_down'] += 2
    else:
        scores['ranging'] += 2
    
    if regime_adx == 'trending_up':
        scores['trending_up'] += 2
    elif regime_adx == 'trending_down':
        scores['trending_down'] += 2
    else:
        scores['ranging'] += 2
    
    if regime_breakout == 'breakout_up':
        scores['trending_up'] += 1
    elif regime_breakout == 'breakout_down':
        scores['trending_down'] += 1
    else:
        scores['ranging'] += 1
    
    if regime_cv == 'trending_up':
        scores['trending_up'] += 1
    elif regime_cv == 'trending_down':
        scores['trending_down'] += 1
    else:
        scores['ranging'] += 1
    
    best_regime = max(scores, key=scores.get)
    
    return {
        'regime': best_regime,
        'confidence': scores[best_regime] / sum(scores.values()) if sum(scores.values()) > 0 else 0,
        'details': {
            'slope': regime_slope,
            'adx': regime_adx,
            'breakout': regime_breakout,
            'cv': regime_cv
        }
    }

# ============================================================================
# СТАДИИ ВОЛАТИЛЬНОСТИ
# ============================================================================

def calculate_volatility_stages(df: pd.DataFrame, slow_period: int = 25, 
                                fast_period: int = 7, deviation: float = 0.5):
    """
    Вычисляет стадии волатильности (аналог VolatilityStagesAW)
    
    Returns:
        stages, fast_sma, slow_sma, upper_bound, lower_bound
    """
    # Вычисляем ATR
    atr = calculate_atr(df, period=14)
    
    # Быстрая и медленная SMA
    fast_sma = atr.rolling(window=fast_period).mean()
    slow_sma = atr.rolling(window=slow_period).mean()
    
    # Границы
    upper_bound = slow_sma * (1 + deviation)
    lower_bound = slow_sma * (1 - deviation)
    
    # Стадии
    stages = pd.Series(index=df.index, dtype=int)
    
    stages[fast_sma < lower_bound] = 1
    stages[(fast_sma >= lower_bound) & (fast_sma <= upper_bound)] = 2
    stages[(fast_sma > upper_bound) & (fast_sma <= upper_bound * 2)] = 3
    stages[fast_sma > upper_bound * 2] = 4
    
    stages = stages.ffill().fillna(2)
    
    return stages, fast_sma, slow_sma, upper_bound, lower_bound

# ============================================================================
# СОЗДАНИЕ МАТРИЦЫ ЗОН
# ============================================================================

def create_market_zones_matrix(df: pd.DataFrame, seg) -> pd.DataFrame:
    """
    Создает полную матрицу рыночных зон
    
    Args:
        df: DataFrame с OHLCV данными
        seg: Segmenter объект с вычисленными сегментами
    
    Returns:
        DataFrame с полной информацией о зонах
    """
    zones = []
    
    # Вычисляем стадии волатильности для всего датафрейма
    stages, _, _, _, _ = calculate_volatility_stages(df)
    
    for i, segment in enumerate(seg.segments):
        # Проверка границ (segment.stop может быть равен len(df), что выходит за границы)
        start_idx = segment.start
        stop_idx = min(segment.stop, len(df))  # Ограничиваем максимальным индексом
        
        # Пропускаем сегменты с некорректными индексами
        if start_idx >= len(df) or stop_idx <= start_idx:
            continue
        
        # Базовые характеристики
        trend_slope = segment.slope
        trend_direction = 'up' if trend_slope > 0.001 else 'down' if trend_slope < -0.001 else 'flat'
        
        # Классификации
        volatility = classify_volatility_combined(df, segment, seg)
        volume_info = classify_volume_combined(df, segment)
        regime_info = classify_regime_combined(df, segment, seg)
        
        # Доминирующая стадия волатильности
        segment_stages = stages.iloc[start_idx:stop_idx]
        dominant_stage = segment_stages.mode()[0] if len(segment_stages.mode()) > 0 else 2
        
        # Уникальный ID зоны
        zone_id = f"{trend_direction}_{volatility}_{regime_info['regime']}_{volume_info['level']}_stage{dominant_stage}"
        
        # Дополнительные метрики
        segment_data = df.iloc[start_idx:stop_idx]
        
        zone = {
            'zone_id': zone_id,
            'zone_number': i + 1,
            'start_idx': start_idx,
            'stop_idx': stop_idx,
            'start_time': df.index[start_idx],
            'stop_time': df.index[stop_idx - 1],  # Используем stop_idx - 1 для корректного индекса
            'duration_candles': stop_idx - start_idx,
            'duration_hours': (df.index[stop_idx - 1] - df.index[start_idx]).total_seconds() / 3600,
            
            # Тренд
            'trend_slope': trend_slope,
            'trend_direction': trend_direction,
            'trend_stability': 1 - (segment.slopes_std / abs(segment.slope)) if segment.slope != 0 else 0,
            
            # Волатильность
            'volatility': volatility,
            'volatility_stage': dominant_stage,
            'volatility_std': segment.std,
            'volatility_span': segment.span,
            
            # Объемы
            'volume_level': volume_info['level'],
            'volume_pressure': volume_info['pressure'],
            'volume_trend': volume_info['trend'],
            'avg_volume': segment_data['volume'].mean(),
            
            # Режим
            'regime': regime_info['regime'],
            'regime_confidence': regime_info['confidence'],
            
            # Цены
            'price_start': segment_data['close'].iloc[0],
            'price_end': segment_data['close'].iloc[-1],
            'price_change': segment_data['close'].iloc[-1] - segment_data['close'].iloc[0],
            'price_change_pct': ((segment_data['close'].iloc[-1] - segment_data['close'].iloc[0]) / 
                                segment_data['close'].iloc[0]) * 100,
            'price_high': segment_data['high'].max(),
            'price_low': segment_data['low'].min(),
        }
        
        zones.append(zone)
    
    return pd.DataFrame(zones)

