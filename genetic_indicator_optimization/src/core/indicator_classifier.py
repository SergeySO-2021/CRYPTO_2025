"""
Система классификации индикаторов по типам и характеристикам
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


class IndicatorType(Enum):
    """Типы индикаторов"""
    TREND_FOLLOWING = "trend_following"
    OSCILLATOR = "oscillator"
    VOLUME_BASED = "volume_based"
    VOLATILITY = "volatility"
    LEVELS_ZONES = "levels_zones"
    SYSTEMIC = "systemic"


@dataclass
class IndicatorCharacteristics:
    """Характеристики класса индикаторов"""
    best_market_conditions: List[str]
    worst_market_conditions: List[str]
    signal_type: str
    typical_holding_period: str
    risk_profile: str
    optimal_timeframe: List[str]


class IndicatorClassifier:
    """
    Классификатор индикаторов по типам и характеристикам
    
    Классифицирует 43+ индикатора из IndicatorEngine на 6 классов:
    - Trend Following (14 индикаторов)
    - Oscillators (11 индикаторов)
    - Volume Based (5 индикаторов)
    - Volatility (4 индикатора)
    - Levels Zones (6 индикаторов)
    - Systemic (3 индикатора)
    """
    
    # Классификация индикаторов по типам
    INDICATOR_CLASSIFICATION = {
        # ТРЕНДОВЫЕ ИНДИКАТОРЫ (14)
        'SuperTrend': {
            'type': IndicatorType.TREND_FOLLOWING,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['strong_trend', 'breakout'],
                worst_market_conditions=['ranging', 'choppy'],
                signal_type='trend_following',
                typical_holding_period='medium_term',
                risk_profile='medium',
                optimal_timeframe=['15m', '1h', '4h']
            )
        },
        'MACD': {
            'type': IndicatorType.TREND_FOLLOWING,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'momentum'],
                worst_market_conditions=['sideways', 'reversing'],
                signal_type='trend_momentum',
                typical_holding_period='medium_term',
                risk_profile='medium',
                optimal_timeframe=['1h', '4h', '1d']
            )
        },
        'Bollinger Bands': {
            'type': IndicatorType.TREND_FOLLOWING,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['volatile_trend', 'breakout'],
                worst_market_conditions=['low_volatility', 'ranging'],
                signal_type='mean_reversion_trend',
                typical_holding_period='short_term',
                risk_profile='medium',
                optimal_timeframe=['15m', '1h', '4h']
            )
        },
        'EMA Cross': {
            'type': IndicatorType.TREND_FOLLOWING,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'momentum'],
                worst_market_conditions=['sideways', 'choppy'],
                signal_type='trend_crossover',
                typical_holding_period='medium_term',
                risk_profile='low',
                optimal_timeframe=['15m', '1h', '4h']
            )
        },
        'Triple EMA Cross': {
            'type': IndicatorType.TREND_FOLLOWING,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['strong_trend'],
                worst_market_conditions=['ranging', 'reversing'],
                signal_type='trend_crossover',
                typical_holding_period='medium_term',
                risk_profile='low',
                optimal_timeframe=['1h', '4h', '1d']
            )
        },
        'Parabolic SAR': {
            'type': IndicatorType.TREND_FOLLOWING,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'momentum'],
                worst_market_conditions=['sideways', 'choppy'],
                signal_type='trend_following',
                typical_holding_period='medium_term',
                risk_profile='medium',
                optimal_timeframe=['15m', '1h', '4h']
            )
        },
        'Half Trend': {
            'type': IndicatorType.TREND_FOLLOWING,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending'],
                worst_market_conditions=['ranging', 'choppy'],
                signal_type='trend_following',
                typical_holding_period='medium_term',
                risk_profile='medium',
                optimal_timeframe=['15m', '1h', '4h']
            )
        },
        'Ichimoku Cloud': {
            'type': IndicatorType.TREND_FOLLOWING,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'breakout'],
                worst_market_conditions=['sideways', 'choppy'],
                signal_type='trend_following',
                typical_holding_period='medium_term',
                risk_profile='medium',
                optimal_timeframe=['1h', '4h', '1d']
            )
        },
        'VWAP': {
            'type': IndicatorType.TREND_FOLLOWING,  # Также может быть volume_based
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['all_conditions', 'institutional_flow'],
                worst_market_conditions=['low_volume', 'holidays'],
                signal_type='support_resistance',
                typical_holding_period='intraday',
                risk_profile='low',
                optimal_timeframe=['5m', '15m', '1h']
            )
        },
        'B-Xtrender': {
            'type': IndicatorType.TREND_FOLLOWING,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending'],
                worst_market_conditions=['ranging'],
                signal_type='trend_following',
                typical_holding_period='medium_term',
                risk_profile='medium',
                optimal_timeframe=['15m', '1h', '4h']
            )
        },
        'Bull Bear Power Trend': {
            'type': IndicatorType.TREND_FOLLOWING,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'momentum'],
                worst_market_conditions=['sideways'],
                signal_type='trend_momentum',
                typical_holding_period='short_term',
                risk_profile='medium',
                optimal_timeframe=['15m', '1h']
            )
        },
        'Conditional Sampling EMA': {
            'type': IndicatorType.TREND_FOLLOWING,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending'],
                worst_market_conditions=['choppy'],
                signal_type='trend_following',
                typical_holding_period='medium_term',
                risk_profile='low',
                optimal_timeframe=['15m', '1h', '4h']
            )
        },
        'Rational Quadratic Kernel': {
            'type': IndicatorType.TREND_FOLLOWING,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending'],
                worst_market_conditions=['noisy'],
                signal_type='trend_smoothing',
                typical_holding_period='medium_term',
                risk_profile='low',
                optimal_timeframe=['1h', '4h', '1d']
            )
        },
        'PVSRA': {
            'type': IndicatorType.TREND_FOLLOWING,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'volume_confirmation'],
                worst_market_conditions=['low_volume'],
                signal_type='trend_volume',
                typical_holding_period='medium_term',
                risk_profile='medium',
                optimal_timeframe=['15m', '1h', '4h']
            )
        },
        
        # ОСЦИЛЛЯТОРЫ (11)
        'RSI': {
            'type': IndicatorType.OSCILLATOR,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['ranging', 'overbought_oversold'],
                worst_market_conditions=['strong_trend', 'breakaway_gap'],
                signal_type='mean_reversion',
                typical_holding_period='short_term',
                risk_profile='low',
                optimal_timeframe=['15m', '1h', '4h']
            )
        },
        'Stochastic Oscillator': {
            'type': IndicatorType.OSCILLATOR,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['ranging', 'consolidation'],
                worst_market_conditions=['strong_trend', 'gap_moves'],
                signal_type='momentum_reversal',
                typical_holding_period='short_term',
                risk_profile='medium',
                optimal_timeframe=['15m', '1h', '4h']
            )
        },
        'Williams %R': {
            'type': IndicatorType.OSCILLATOR,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['ranging', 'overbought_oversold'],
                worst_market_conditions=['strong_trend'],
                signal_type='momentum_reversal',
                typical_holding_period='short_term',
                risk_profile='medium',
                optimal_timeframe=['15m', '1h', '4h']
            )
        },
        'Commodity Channel Index': {
            'type': IndicatorType.OSCILLATOR,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['ranging', 'momentum'],
                worst_market_conditions=['strong_trend'],
                signal_type='momentum',
                typical_holding_period='short_term',
                risk_profile='medium',
                optimal_timeframe=['15m', '1h', '4h']
            )
        },
        'Momentum Oscillator': {
            'type': IndicatorType.OSCILLATOR,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'momentum'],
                worst_market_conditions=['sideways'],
                signal_type='momentum',
                typical_holding_period='short_term',
                risk_profile='medium',
                optimal_timeframe=['15m', '1h']
            )
        },
        'Rate of Change': {
            'type': IndicatorType.OSCILLATOR,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'momentum'],
                worst_market_conditions=['sideways'],
                signal_type='momentum',
                typical_holding_period='short_term',
                risk_profile='medium',
                optimal_timeframe=['15m', '1h']
            )
        },
        'True Strength Index': {
            'type': IndicatorType.OSCILLATOR,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'momentum'],
                worst_market_conditions=['choppy'],
                signal_type='momentum',
                typical_holding_period='medium_term',
                risk_profile='low',
                optimal_timeframe=['1h', '4h', '1d']
            )
        },
        'Detrended Price Oscillator': {
            'type': IndicatorType.OSCILLATOR,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending'],
                worst_market_conditions=['sideways'],
                signal_type='trend',
                typical_holding_period='medium_term',
                risk_profile='low',
                optimal_timeframe=['1h', '4h']
            )
        },
        'Choppiness Index': {
            'type': IndicatorType.OSCILLATOR,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['ranging'],
                worst_market_conditions=['trending'],
                signal_type='volatility',
                typical_holding_period='short_term',
                risk_profile='low',
                optimal_timeframe=['15m', '1h', '4h']
            )
        },
        'BB Oscillator': {
            'type': IndicatorType.OSCILLATOR,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['ranging', 'mean_reversion'],
                worst_market_conditions=['strong_trend'],
                signal_type='mean_reversion',
                typical_holding_period='short_term',
                risk_profile='medium',
                optimal_timeframe=['15m', '1h']
            )
        },
        'QQE Mod': {
            'type': IndicatorType.OSCILLATOR,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'momentum'],
                worst_market_conditions=['choppy'],
                signal_type='trend',
                typical_holding_period='medium_term',
                risk_profile='low',
                optimal_timeframe=['1h', '4h']
            )
        },
        
        # ОБЪЕМНЫЕ ИНДИКАТОРЫ (5)
        'Chaikin Money Flow': {
            'type': IndicatorType.VOLUME_BASED,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trend_confirmation', 'breakout'],
                worst_market_conditions=['low_volume', 'indecisive'],
                signal_type='volume_confirmation',
                typical_holding_period='medium_term',
                risk_profile='low',
                optimal_timeframe=['1h', '4h', '1d']
            )
        },
        'Volume Oscillator': {
            'type': IndicatorType.VOLUME_BASED,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'volume_confirmation'],
                worst_market_conditions=['low_volume'],
                signal_type='trend',
                typical_holding_period='short_term',
                risk_profile='low',
                optimal_timeframe=['15m', '1h']
            )
        },
        'On Balance Volume': {
            'type': IndicatorType.VOLUME_BASED,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'volume_confirmation'],
                worst_market_conditions=['low_volume'],
                signal_type='trend',
                typical_holding_period='medium_term',
                risk_profile='low',
                optimal_timeframe=['1h', '4h', '1d']
            )
        },
        'Accumulation/Distribution': {
            'type': IndicatorType.VOLUME_BASED,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'volume_confirmation'],
                worst_market_conditions=['low_volume'],
                signal_type='trend',
                typical_holding_period='medium_term',
                risk_profile='low',
                optimal_timeframe=['1h', '4h', '1d']
            )
        },
        'Volume Price Trend': {
            'type': IndicatorType.VOLUME_BASED,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'volume_confirmation'],
                worst_market_conditions=['low_volume'],
                signal_type='trend',
                typical_holding_period='medium_term',
                risk_profile='low',
                optimal_timeframe=['1h', '4h', '1d']
            )
        },
        
        # ВОЛАТИЛЬНОСТЬ (4)
        'Range Filter': {
            'type': IndicatorType.VOLATILITY,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['high_volatility', 'breakout'],
                worst_market_conditions=['low_volatility', 'consolidation'],
                signal_type='volatility_breakout',
                typical_holding_period='short_term',
                risk_profile='high',
                optimal_timeframe=['15m', '1h']
            )
        },
        'Range Filter Type 2': {
            'type': IndicatorType.VOLATILITY,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['high_volatility', 'breakout'],
                worst_market_conditions=['low_volatility', 'consolidation'],
                signal_type='volatility_breakout',
                typical_holding_period='short_term',
                risk_profile='high',
                optimal_timeframe=['15m', '1h']
            )
        },
        'Waddah Attar Explosion': {
            'type': IndicatorType.VOLATILITY,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['high_volatility', 'explosive_moves'],
                worst_market_conditions=['low_volatility'],
                signal_type='volatility_breakout',
                typical_holding_period='short_term',
                risk_profile='high',
                optimal_timeframe=['15m', '1h']
            )
        },
        'Chandelier Exit': {
            'type': IndicatorType.VOLATILITY,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'volatile'],
                worst_market_conditions=['sideways'],
                signal_type='trend',
                typical_holding_period='medium_term',
                risk_profile='medium',
                optimal_timeframe=['1h', '4h', '1d']
            )
        },
        'ATR': {
            'type': IndicatorType.VOLATILITY,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['all_conditions'],
                worst_market_conditions=[],
                signal_type='volatility_measure',
                typical_holding_period='short_term',
                risk_profile='low',
                optimal_timeframe=['15m', '1h', '4h']
            )
        },
        'Damiani Volatmeter': {
            'type': IndicatorType.VOLATILITY,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['volatile'],
                worst_market_conditions=['low_volatility'],
                signal_type='volatility_measure',
                typical_holding_period='short_term',
                risk_profile='low',
                optimal_timeframe=['15m', '1h']
            )
        },
        
        # УРОВНИ И ЗОНЫ (6)
        'Fibonacci Retracement': {
            'type': IndicatorType.LEVELS_ZONES,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['pullback', 'retracement'],
                worst_market_conditions=['strong_trend_no_pullback', 'chaotic'],
                signal_type='support_resistance',
                typical_holding_period='medium_term',
                risk_profile='medium',
                optimal_timeframe=['1h', '4h', '1d']
            )
        },
        'Pivot Levels': {
            'type': IndicatorType.LEVELS_ZONES,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['ranging', 'support_resistance'],
                worst_market_conditions=['strong_trend'],
                signal_type='support_resistance',
                typical_holding_period='short_term',
                risk_profile='low',
                optimal_timeframe=['15m', '1h']
            )
        },
        'Fair Value Gap': {
            'type': IndicatorType.LEVELS_ZONES,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['gap_moves', 'breakout'],
                worst_market_conditions=['sideways'],
                signal_type='support_resistance',
                typical_holding_period='short_term',
                risk_profile='medium',
                optimal_timeframe=['15m', '1h']
            )
        },
        'William Fractals': {
            'type': IndicatorType.LEVELS_ZONES,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'reversals'],
                worst_market_conditions=['sideways'],
                signal_type='support_resistance',
                typical_holding_period='medium_term',
                risk_profile='medium',
                optimal_timeframe=['1h', '4h']
            )
        },
        'Supply/Demand Zones': {
            'type': IndicatorType.LEVELS_ZONES,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['reversals', 'support_resistance'],
                worst_market_conditions=['strong_trend'],
                signal_type='support_resistance',
                typical_holding_period='medium_term',
                risk_profile='medium',
                optimal_timeframe=['1h', '4h', '1d']
            )
        },
        'Liquidity Zone': {
            'type': IndicatorType.LEVELS_ZONES,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['support_resistance', 'volume_confirmation'],
                worst_market_conditions=['low_volume'],
                signal_type='support_resistance',
                typical_holding_period='short_term',
                risk_profile='low',
                optimal_timeframe=['15m', '1h']
            )
        },
        
        # СИСТЕМНЫЕ (3)
        'Market Sessions': {
            'type': IndicatorType.SYSTEMIC,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['session_volatility', 'overlap'],
                worst_market_conditions=['overnight', 'holidays'],
                signal_type='time_filter',
                typical_holding_period='intraday',
                risk_profile='low',
                optimal_timeframe=['15m', '1h']
            )
        },
        'ZigZag': {
            'type': IndicatorType.SYSTEMIC,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending', 'swings'],
                worst_market_conditions=['choppy'],
                signal_type='trend_filter',
                typical_holding_period='medium_term',
                risk_profile='low',
                optimal_timeframe=['1h', '4h', '1d']
            )
        },
        'Heiken-Ashi Candlestick Oscillator': {
            'type': IndicatorType.SYSTEMIC,
            'characteristics': IndicatorCharacteristics(
                best_market_conditions=['trending'],
                worst_market_conditions=['choppy'],
                signal_type='trend_filter',
                typical_holding_period='medium_term',
                risk_profile='low',
                optimal_timeframe=['1h', '4h']
            )
        }
    }
    
    @classmethod
    def get_indicator_type(cls, indicator_name: str) -> IndicatorType:
        """
        Получить тип индикатора
        
        Args:
            indicator_name: Название индикатора
            
        Returns:
            IndicatorType: Тип индикатора
        """
        if indicator_name in cls.INDICATOR_CLASSIFICATION:
            return cls.INDICATOR_CLASSIFICATION[indicator_name]['type']
        # По умолчанию возвращаем trend_following
        return IndicatorType.TREND_FOLLOWING
    
    @classmethod
    def get_characteristics(cls, indicator_name: str) -> Optional[IndicatorCharacteristics]:
        """
        Получить характеристики индикатора
        
        Args:
            indicator_name: Название индикатора
            
        Returns:
            IndicatorCharacteristics или None
        """
        if indicator_name in cls.INDICATOR_CLASSIFICATION:
            return cls.INDICATOR_CLASSIFICATION[indicator_name]['characteristics']
        return None
    
    @classmethod
    def get_indicators_by_type(cls, indicator_type: IndicatorType) -> List[str]:
        """
        Получить все индикаторы заданного типа
        
        Args:
            indicator_type: Тип индикатора
            
        Returns:
            List[str]: Список названий индикаторов
        """
        return [
            name for name, info in cls.INDICATOR_CLASSIFICATION.items()
            if info['type'] == indicator_type
        ]
    
    @classmethod
    def get_all_indicators(cls) -> List[str]:
        """
        Получить список всех классифицированных индикаторов
        
        Returns:
            List[str]: Список всех индикаторов
        """
        return list(cls.INDICATOR_CLASSIFICATION.keys())
    
    @classmethod
    def get_classification_summary(cls) -> Dict[str, int]:
        """
        Получить сводку по классификации
        
        Returns:
            Dict: Количество индикаторов по типам
        """
        summary = {}
        for indicator_type in IndicatorType:
            summary[indicator_type.value] = len(cls.get_indicators_by_type(indicator_type))
        return summary

