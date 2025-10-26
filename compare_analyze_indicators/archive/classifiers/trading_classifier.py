"""
Реализация Trading Classifier (полосы тренда)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from base_classifier import BaseMarketZoneClassifier
import warnings
warnings.filterwarnings('ignore')


class TradingClassifier(BaseMarketZoneClassifier):
    """
    Trading Classifier - полосы тренда (адаптация trading_classifier.pine)
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Инициализация Trading Classifier
        
        Args:
            parameters: Параметры классификатора
        """
        default_params = {
            'length': 10  # период для SMEMA
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Trading Classifier", default_params)
    
    def fit(self, data: pd.DataFrame) -> 'TradingClassifier':
        """
        Обучение Trading Classifier
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            self
        """
        self.validate_data(data, require_volume=True)
        
        # Сохраняем данные для использования в методах
        self.data = data.copy()
        
        # Вычисляем все индикаторы
        self._calculate_indicators(data)
        
        self.is_fitted = True
        self.classes_ = [-1, 0, 1]  # Bearish, Sideways, Bullish
        
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Предсказание рыночных зон
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            Массив предсказаний
        """
        if not self.is_fitted:
            raise ValueError("Классификатор не обучен. Вызовите fit() сначала.")
        
        self.validate_data(data, require_volume=True)
        
        # Сохраняем данные для использования в методах
        self.data = data.copy()
        
        # Вычисляем индикаторы для новых данных
        self._calculate_indicators(data)
        
        # Получаем предсказания
        predictions = self._calculate_trend_strength(data, self.bands)
        
        return predictions
    
    def _calculate_indicators(self, data: pd.DataFrame) -> None:
        """Вычисление всех технических индикаторов"""
        
        # Вычисляем SMEMA
        self.smema_line = self._calculate_smema(data)
        
        # Вычисляем полосы тренда
        self.bands = self._calculate_trend_bands(data)
    
    def _calculate_smema(self, data: pd.DataFrame) -> pd.Series:
        """
        Smooth EMA - комбинация SMA и EMA
        
        Args:
            data: DataFrame с данными
            
        Returns:
            Series с SMEMA
        """
        length = self.parameters['length']
        
        # Сначала EMA
        ema_val = data['close'].ewm(span=length).mean()
        
        # Затем SMA от EMA (Smooth EMA)
        smema_val = ema_val.rolling(window=length).mean()
        
        return smema_val
    
    def _calculate_trend_bands(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Вычисление полос тренда
        
        Args:
            data: DataFrame с данными
            
        Returns:
            Словарь с полосами тренда
        """
        # Вычисляем step (адаптивный размер полос)
        step = self._calculate_step(data)
        
        # Создаем полосы
        bands = {
            'up3': self.smema_line + step * 3,
            'up2': self.smema_line + step * 2,
            'up1': self.smema_line + step,
            'dn1': self.smema_line - step,
            'dn2': self.smema_line - step * 2,
            'dn3': self.smema_line - step * 3
        }
        
        return bands
    
    def _calculate_step(self, data: pd.DataFrame) -> pd.Series:
        """
        Вычисление адаптивного размера полос
        
        Args:
            data: DataFrame с данными
            
        Returns:
            Series с размерами полос
        """
        # Используем разность high-low для определения размера полос
        price_range = data['high'] - data['low']
        
        # Сглаживаем разность
        step = price_range.rolling(window=100).mean()
        
        return step
    
    def _calculate_trend_strength(self, data: pd.DataFrame, bands: Dict[str, pd.Series]) -> np.ndarray:
        """
        ВЫЧИСЛЕНИЕ СИЛЫ ТРЕНДА - ИСПРАВЛЕННАЯ ВЕРСИЯ
        
        Args:
            data: DataFrame с данными
            bands: Словарь с полосами тренда
            
        Returns:
            Массив предсказаний
        """
        close = data['close']
        
        # Определяем пересечения с полосами
        above3 = close > bands['up3']
        above2 = close > bands['up2']
        above1 = close > bands['up1']
        
        below1 = close < bands['dn1']
        below2 = close < bands['dn2']
        below3 = close < bands['dn3']
        
        # Вычисляем силу бычьего и медвежьего тренда
        bull_strength = (above1.astype(int) + above2.astype(int) + above3.astype(int))
        bear_strength = (below1.astype(int) + below2.astype(int) + below3.astype(int))
        
        # Определяем направление тренда
        trend = self.smema_line > self.smema_line.shift(1)
        
        # Классификация
        predictions = np.where(
            trend & (bull_strength >= 1), 1,      # Бычий
            np.where(~trend & (bear_strength >= 1), -1, 0)  # Медвежий, Боковой
        )
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Получение важности признаков
        
        Returns:
            Словарь с важностью каждого признака
        """
        if not self.is_fitted:
            return {}
        
        # Для Trading Classifier важность определяется по полосам
        importance = {
            'smema_line': 0.4,
            'trend_bands': 0.4,
            'trend_direction': 0.2
        }
        
        return importance
    
    def get_detailed_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Получение детального анализа
        
        Args:
            data: DataFrame с данными
            
        Returns:
            DataFrame с детальным анализом
        """
        if not self.is_fitted:
            raise ValueError("Классификатор не обучен")
        
        # Сохраняем данные для использования в методах
        self.data = data.copy()
        
        self._calculate_indicators(data)
        
        # Получаем предсказания
        predictions = self._calculate_trend_strength(data, self.bands)
        
        # Создаем DataFrame с результатами
        results = pd.DataFrame({
            'smema': self.smema_line,
            'bull_strength': self._calculate_bull_strength(data, self.bands),
            'bear_strength': self._calculate_bear_strength(data, self.bands),
            'predictions': predictions
        }, index=data.index)
        
        return results
    
    def _calculate_bull_strength(self, data: pd.DataFrame, bands: Dict[str, pd.Series]) -> np.ndarray:
        """Вычисление силы бычьего тренда"""
        close = data['close']
        
        above1 = close > bands['up1']
        above2 = close > bands['up2']
        above3 = close > bands['up3']
        
        return (above1.astype(int) + above2.astype(int) + above3.astype(int))
    
    def _calculate_bear_strength(self, data: pd.DataFrame, bands: Dict[str, pd.Series]) -> np.ndarray:
        """Вычисление силы медвежьего тренда"""
        close = data['close']
        
        below1 = close < bands['dn1']
        below2 = close < bands['dn2']
        below3 = close < bands['dn3']
        
        return (below1.astype(int) + below2.astype(int) + below3.astype(int))
