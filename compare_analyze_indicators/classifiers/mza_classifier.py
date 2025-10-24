"""
Реализация Market Zone Analyzer (MZA) классификатора
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from base_classifier import BaseMarketZoneClassifier
import warnings
warnings.filterwarnings('ignore')


class MZAClassifier(BaseMarketZoneClassifier):
    """
    Market Zone Analyzer - четырехуровневая система анализа рыночных зон
    
    Компоненты:
    1. Trend Strength (Сила тренда): ADX/DMI, Moving Averages, Ichimoku
    2. Momentum (Импульс): RSI, Stochastic, MACD
    3. Price Action (Поведение цены): HH/LL, Heikin-Ashi, Candle Range
    4. Market Activity (Активность рынка): Bollinger Bands, ATR, Volume
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Инициализация MZA классификатора
        
        Args:
            parameters: Параметры классификатора
        """
        default_params = {
            # Trend Indicators
            'adx_length': 21,
            'adx_threshold': 18,
            'fast_ma_length': 21,
            'slow_ma_length': 55,
            
            # Momentum Indicators
            'rsi_length': 21,
            'stoch_length': 21,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Price Action Indicators
            'hh_ll_range': 21,
            'ha_doji_range': 8,
            'candle_range_length': 13,
            
            # Market Activity Indicators
            'bb_length': 20,
            'bb_multiplier': 2.2,
            'atr_length': 14,
            'volume_ma_length': 20,
            
            # Weights (адаптивные веса)
            'trend_weight': 0.4,
            'momentum_weight': 0.3,
            'price_action_weight': 0.3,
            
            # Volatility thresholds
            'high_volatility_threshold': 0.02,
            'low_volatility_threshold': 0.01
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Market Zone Analyzer", default_params)
        self.feature_importance_ = {}
    
    def fit(self, data: pd.DataFrame) -> 'MZAClassifier':
        """
        Обучение MZA классификатора
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            self
        """
        self.validate_data(data, require_volume=True)
        
        # Сохраняем данные для обучения
        self.data = data.copy()
        
        # Вычисляем все индикаторы
        self._calculate_indicators(self.data)
        
        # Определяем адаптивные веса на основе волатильности
        self._calculate_adaptive_weights(self.data)
        
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
        
        # Создаем временную копию данных для предсказания
        temp_data = data.copy()
        
        # Проверяем размеры данных для отладки
        if len(data) != len(self.data):
            print(f"Warning: data length {len(data)} != self.data length {len(self.data)}")
        
        # Сохраняем оригинальные данные и заменяем на временные
        original_data = self.data
        self.data = temp_data
        
        # Вычисляем индикаторы для новых данных
        self._calculate_indicators(temp_data)
        
        # Получаем оценки для каждой категории (упрощенные версии)
        trend_scores = self._calculate_simple_trend_scores(temp_data)
        momentum_scores = self._calculate_simple_momentum_scores(temp_data)
        price_action_scores = self._calculate_simple_price_action_scores(temp_data)
        market_activity_scores = self._calculate_simple_market_activity_scores(temp_data)
        
        # Проверяем размеры массивов для отладки
        if len(trend_scores) != len(data):
            print(f"Warning: trend_scores length {len(trend_scores)} != data length {len(data)}")
        if len(momentum_scores) != len(data):
            print(f"Warning: momentum_scores length {len(momentum_scores)} != data length {len(data)}")
        if len(price_action_scores) != len(data):
            print(f"Warning: price_action_scores length {len(price_action_scores)} != data length {len(data)}")
        if len(market_activity_scores) != len(data):
            print(f"Warning: market_activity_scores length {len(market_activity_scores)} != data length {len(data)}")
        
        # Вычисляем общий скор с фиксированными весами
        overall_scores = (
            trend_scores * 0.4 +  # Trend weight
            momentum_scores * 0.3 +  # Momentum weight
            price_action_scores * 0.3 +  # Price Action weight
            market_activity_scores * 0.1  # Market Activity как дополнительный фактор
        )
        
        # Классификация на основе общего скора
        predictions = np.where(overall_scores > 2, 1,  # Bullish
                              np.where(overall_scores < -2, -1, 0))  # Bearish, Sideways
        
        # Восстанавливаем оригинальные данные
        self.data = original_data
        
        return predictions
    
    def _calculate_indicators(self, data: pd.DataFrame) -> None:
        """Вычисление всех технических индикаторов"""
        
        # Trend Indicators
        self._calculate_adx_dmi(data)
        self._calculate_moving_averages(data)
        
        # Momentum Indicators
        self._calculate_rsi(data)
        self._calculate_stochastic(data)
        self._calculate_macd(data)
        
        # Price Action Indicators
        self._calculate_hh_ll(data)
        self._calculate_heikin_ashi(data)
        self._calculate_candle_range(data)
        
        # Market Activity Indicators
        self._calculate_bollinger_bands(data)
        self._calculate_atr(data)
        self._calculate_volume_indicators(data)
    
    def _calculate_adx_dmi(self, data: pd.DataFrame) -> None:
        """Вычисление ADX и DMI"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
        dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
        
        # Smoothed values
        length = self.parameters['adx_length']
        tr_smooth = tr.rolling(window=length).mean()
        dm_plus_smooth = pd.Series(dm_plus, index=data.index).rolling(window=length).mean()
        dm_minus_smooth = pd.Series(dm_minus, index=data.index).rolling(window=length).mean()
        
        # DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=length).mean()
        
        self.data['di_plus'] = di_plus
        self.data['di_minus'] = di_minus
        self.data['adx'] = adx
    
    def _calculate_moving_averages(self, data: pd.DataFrame) -> None:
        """Вычисление скользящих средних"""
        fast_length = self.parameters['fast_ma_length']
        slow_length = self.parameters['slow_ma_length']
        
        self.data['fast_ma'] = data['close'].rolling(window=fast_length).mean()
        self.data['slow_ma'] = data['close'].rolling(window=slow_length).mean()
        self.data['ma_slope'] = self.data['fast_ma'].diff()
    
    def _calculate_rsi(self, data: pd.DataFrame) -> None:
        """Вычисление RSI"""
        length = self.parameters['rsi_length']
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        self.data['rsi'] = rsi
    
    def _calculate_stochastic(self, data: pd.DataFrame) -> None:
        """Вычисление Stochastic"""
        length = self.parameters['stoch_length']
        lowest_low = data['low'].rolling(window=length).min()
        highest_high = data['high'].rolling(window=length).max()
        
        k_percent = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
        
        self.data['stoch_k'] = k_percent
    
    def _calculate_macd(self, data: pd.DataFrame) -> None:
        """Вычисление MACD"""
        fast = self.parameters['macd_fast']
        slow = self.parameters['macd_slow']
        signal = self.parameters['macd_signal']
        
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        self.data['macd'] = macd_line
        self.data['macd_signal'] = signal_line
        self.data['macd_histogram'] = histogram
    
    def _calculate_hh_ll(self, data: pd.DataFrame) -> None:
        """Вычисление Higher Highs / Lower Lows"""
        range_period = self.parameters['hh_ll_range']
        
        # Higher Highs
        hh = data['high'].rolling(window=range_period).max() == data['high']
        # Lower Lows
        ll = data['low'].rolling(window=range_period).min() == data['low']
        
        self.data['hh'] = hh.astype(int)
        self.data['ll'] = ll.astype(int)
    
    def _calculate_heikin_ashi(self, data: pd.DataFrame) -> None:
        """Вычисление Heikin-Ashi"""
        ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        ha_open = (data['open'].shift(1) + data['close'].shift(1)) / 2
        
        # Doji detection
        doji_range = self.parameters['ha_doji_range']
        body_size = abs(ha_close - ha_open)
        total_range = data['high'] - data['low']
        
        self.data['ha_doji'] = (body_size / total_range < doji_range / 100).astype(int)
    
    def _calculate_candle_range(self, data: pd.DataFrame) -> None:
        """Вычисление размера свечей"""
        length = self.parameters['candle_range_length']
        candle_range = data['high'] - data['low']
        avg_range = candle_range.rolling(window=length).mean()
        
        self.data['candle_range_ratio'] = candle_range / avg_range
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame) -> None:
        """Вычисление полос Боллинджера"""
        length = self.parameters['bb_length']
        multiplier = self.parameters['bb_multiplier']
        
        sma = data['close'].rolling(window=length).mean()
        std = data['close'].rolling(window=length).std()
        
        upper_band = sma + (std * multiplier)
        lower_band = sma - (std * multiplier)
        band_width = (upper_band - lower_band) / sma
        
        self.data['bb_upper'] = upper_band
        self.data['bb_lower'] = lower_band
        self.data['bb_width'] = band_width
    
    def _calculate_atr(self, data: pd.DataFrame) -> None:
        """Вычисление ATR"""
        length = self.parameters['atr_length']
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        self.data['atr'] = tr.rolling(window=length).mean()
    
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> None:
        """Вычисление объемных индикаторов"""
        length = self.parameters['volume_ma_length']
        
        volume_ma = data['volume'].rolling(window=length).mean()
        volume_ratio = data['volume'] / volume_ma
        
        self.data['volume_ratio'] = volume_ratio
    
    def _calculate_adaptive_weights(self, data: pd.DataFrame) -> None:
        """Вычисление адаптивных весов на основе волатильности"""
        volatility = self.calculate_volatility(data, window=20)
        
        # Используем фиксированные пороги для упрощения
        high_vol_threshold = 0.02
        low_vol_threshold = 0.01
        
        # Адаптивные веса в зависимости от волатильности
        high_vol_mask = volatility > high_vol_threshold
        low_vol_mask = volatility < low_vol_threshold
        
        # Обновляем веса
        self.parameters['trend_weight'] = np.where(high_vol_mask, 0.5, 
                                                   np.where(low_vol_mask, 0.25, 0.4))
        self.parameters['momentum_weight'] = np.where(high_vol_mask, 0.35,
                                                       np.where(low_vol_mask, 0.20, 0.30))
        self.parameters['price_action_weight'] = np.where(high_vol_mask, 0.15,
                                                          np.where(low_vol_mask, 0.55, 0.30))
    
    def _calculate_trend_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Вычисление скоров для Trend Strength"""
        scores = np.zeros(len(data))
        
        # ADX/DMI
        if 'adx' in self.data and 'di_plus' in self.data and 'di_minus' in self.data:
            adx_strong = self.data['adx'] > self.parameters['adx_threshold']
            di_bullish = self.data['di_plus'] > self.data['di_minus']
            di_bearish = self.data['di_minus'] > self.data['di_plus']
            
            scores += np.where(adx_strong & di_bullish, 1, 
                              np.where(adx_strong & di_bearish, -1, 0))
        
        # Moving Averages
        if 'fast_ma' in self.data and 'slow_ma' in self.data:
            ma_bullish = (self.data['fast_ma'] > self.data['slow_ma']) & (self.data['ma_slope'] > 0)
            ma_bearish = (self.data['fast_ma'] < self.data['slow_ma']) & (self.data['ma_slope'] < 0)
            
            scores += np.where(ma_bullish, 1, np.where(ma_bearish, -1, 0))
        
        return scores
    
    def _calculate_trend_scores_with_data(self, data: pd.DataFrame) -> np.ndarray:
        """Вычисление скоров для Trend Strength с переданными данными"""
        scores = np.zeros(len(data))
        
        # ADX/DMI
        if 'adx' in data and 'di_plus' in data and 'di_minus' in data:
            adx_strong = data['adx'] > 25  # Используем фиксированный порог
            di_bullish = data['di_plus'] > data['di_minus']
            di_bearish = data['di_minus'] > data['di_plus']
            
            scores += np.where(adx_strong & di_bullish, 1, 
                              np.where(adx_strong & di_bearish, -1, 0))
        
        # Moving Averages
        if 'fast_ma' in data and 'slow_ma' in data:
            # Простая проверка без ma_slope для избежания проблем с размерами
            ma_bullish = data['fast_ma'] > data['slow_ma']
            ma_bearish = data['fast_ma'] < data['slow_ma']
            
            scores += np.where(ma_bullish, 1, np.where(ma_bearish, -1, 0))
        
        return scores
    
    def _calculate_momentum_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Вычисление скоров для Momentum"""
        scores = np.zeros(len(data))
        
        # RSI
        if 'rsi' in self.data:
            rsi_bullish = (self.data['rsi'] > 50) & (self.data['rsi'] < 70)
            rsi_bearish = (self.data['rsi'] < 50) & (self.data['rsi'] > 30)
            
            scores += np.where(rsi_bullish, 1, np.where(rsi_bearish, -1, 0))
        
        # Stochastic
        if 'stoch_k' in self.data:
            stoch_bullish = (self.data['stoch_k'] > 50) & (self.data['stoch_k'] < 80)
            stoch_bearish = (self.data['stoch_k'] < 50) & (self.data['stoch_k'] > 20)
            
            scores += np.where(stoch_bullish, 1, np.where(stoch_bearish, -1, 0))
        
        # MACD
        if 'macd' in self.data and 'macd_signal' in self.data:
            macd_bullish = (self.data['macd'] > self.data['macd_signal']) & (self.data['macd_histogram'] > 0)
            macd_bearish = (self.data['macd'] < self.data['macd_signal']) & (self.data['macd_histogram'] < 0)
            
            scores += np.where(macd_bullish, 1, np.where(macd_bearish, -1, 0))
        
        return scores
    
    def _calculate_momentum_scores_with_data(self, data: pd.DataFrame) -> np.ndarray:
        """Вычисление скоров для Momentum с переданными данными"""
        scores = np.zeros(len(data))
        
        # RSI
        if 'rsi' in data:
            rsi_overbought = data['rsi'] > 70
            rsi_oversold = data['rsi'] < 30
            
            scores += np.where(rsi_overbought, -0.5, np.where(rsi_oversold, 0.5, 0))
        
        # Stochastic
        if 'stoch_k' in data:
            stoch_overbought = data['stoch_k'] > 80
            stoch_oversold = data['stoch_k'] < 20
            
            scores += np.where(stoch_overbought, -0.5, np.where(stoch_oversold, 0.5, 0))
        
        # MACD
        if 'macd' in data and 'macd_signal' in data:
            # Простая проверка без histogram для избежания проблем с размерами
            macd_bullish = data['macd'] > data['macd_signal']
            macd_bearish = data['macd'] < data['macd_signal']
            
            scores += np.where(macd_bullish, 0.5, np.where(macd_bearish, -0.5, 0))
        
        return scores
    
    def _calculate_price_action_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Вычисление скоров для Price Action"""
        scores = np.zeros(len(data))
        
        # HH/LL
        if 'hh' in self.data and 'll' in self.data:
            scores += np.where(self.data['hh'], 1, np.where(self.data['ll'], -1, 0))
        
        # Heikin-Ashi Doji (нерешительность)
        if 'ha_doji' in self.data:
            scores -= self.data['ha_doji'] * 0.5  # Снижаем скор при нерешительности
        
        # Candle Range
        if 'candle_range_ratio' in self.data:
            large_candles = self.data['candle_range_ratio'] > 1.5
            small_candles = self.data['candle_range_ratio'] < 0.5
            
            scores += np.where(large_candles, 0.5, np.where(small_candles, -0.5, 0))
        
        return scores
    
    def _calculate_price_action_scores_with_data(self, data: pd.DataFrame) -> np.ndarray:
        """Вычисление скоров для Price Action с переданными данными"""
        scores = np.zeros(len(data))
        
        # HH/LL
        if 'hh' in data and 'll' in data:
            hh_pattern = data['hh'] == 1
            ll_pattern = data['ll'] == 1
            
            scores += np.where(hh_pattern, 0.5, np.where(ll_pattern, -0.5, 0))
        
        # Heikin-Ashi
        if 'ha_trend' in data:
            ha_bullish = data['ha_trend'] == 1
            ha_bearish = data['ha_trend'] == -1
            
            scores += np.where(ha_bullish, 0.5, np.where(ha_bearish, -0.5, 0))
        
        # Candle Range
        if 'candle_range_ratio' in data:
            large_candles = data['candle_range_ratio'] > 1.5
            small_candles = data['candle_range_ratio'] < 0.5
            
            scores += np.where(large_candles, 0.5, np.where(small_candles, -0.5, 0))
        
        return scores
    
    def _calculate_market_activity_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Вычисление скоров для Market Activity"""
        scores = np.zeros(len(data))
        
        # Bollinger Bands
        if 'bb_width' in self.data:
            high_volatility = self.data['bb_width'] > self.data['bb_width'].rolling(20).quantile(0.8)
            low_volatility = self.data['bb_width'] < self.data['bb_width'].rolling(20).quantile(0.2)
            
            scores += np.where(high_volatility, 0.5, np.where(low_volatility, -0.5, 0))
        
        # Volume
        if 'volume_ratio' in self.data:
            high_volume = self.data['volume_ratio'] > 1.5
            low_volume = self.data['volume_ratio'] < 0.5
            
            scores += np.where(high_volume, 0.5, np.where(low_volume, -0.5, 0))
        
        return scores
    
    def _calculate_market_activity_scores_with_data(self, data: pd.DataFrame) -> np.ndarray:
        """Вычисление скоров для Market Activity с переданными данными"""
        scores = np.zeros(len(data))
        
        # Bollinger Bands
        if 'bb_width' in data:
            # Используем простые пороги вместо rolling quantile для избежания проблем с размерами
            bb_median = data['bb_width'].median()
            high_volatility = data['bb_width'] > bb_median * 1.2
            low_volatility = data['bb_width'] < bb_median * 0.8
            
            scores += np.where(high_volatility, 0.5, np.where(low_volatility, -0.5, 0))
        
        # Volume
        if 'volume_ratio' in data:
            high_volume = data['volume_ratio'] > 1.5
            low_volume = data['volume_ratio'] < 0.5
            
            scores += np.where(high_volume, 0.5, np.where(low_volume, -0.5, 0))
        
        return scores
    
    def _calculate_simple_trend_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Упрощенное вычисление скоров для Trend Strength"""
        scores = np.zeros(len(data))
        
        # Простая проверка тренда на основе скользящих средних
        if 'fast_ma' in data and 'slow_ma' in data:
            ma_bullish = data['fast_ma'] > data['slow_ma']
            ma_bearish = data['fast_ma'] < data['slow_ma']
            scores += np.where(ma_bullish, 1, np.where(ma_bearish, -1, 0))
        
        return scores
    
    def _calculate_simple_momentum_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Упрощенное вычисление скоров для Momentum"""
        scores = np.zeros(len(data))
        
        # RSI
        if 'rsi' in data:
            rsi_overbought = data['rsi'] > 70
            rsi_oversold = data['rsi'] < 30
            scores += np.where(rsi_overbought, -0.5, np.where(rsi_oversold, 0.5, 0))
        
        return scores
    
    def _calculate_simple_price_action_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Упрощенное вычисление скоров для Price Action"""
        scores = np.zeros(len(data))
        
        # Простая проверка на основе цен
        if 'close' in data and 'open' in data:
            bullish_candles = data['close'] > data['open']
            bearish_candles = data['close'] < data['open']
            scores += np.where(bullish_candles, 0.5, np.where(bearish_candles, -0.5, 0))
        
        return scores
    
    def _calculate_simple_market_activity_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Упрощенное вычисление скоров для Market Activity"""
        scores = np.zeros(len(data))
        
        # Простая проверка объема
        if 'volume_ratio' in data:
            high_volume = data['volume_ratio'] > 1.5
            low_volume = data['volume_ratio'] < 0.5
            scores += np.where(high_volume, 0.5, np.where(low_volume, -0.5, 0))
        
        return scores
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Получение важности признаков
        
        Returns:
            Словарь с важностью каждого признака
        """
        if not self.is_fitted:
            return {}
        
        # Базовые веса компонентов
        importance = {
            'trend_strength': self.parameters['trend_weight'],
            'momentum': self.parameters['momentum_weight'],
            'price_action': self.parameters['price_action_weight'],
            'market_activity': 0.1
        }
        
        return importance
    
    def get_detailed_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Получение детальных скоров по всем компонентам
        
        Args:
            data: DataFrame с данными
            
        Returns:
            DataFrame с детальными скорами
        """
        if not self.is_fitted:
            raise ValueError("Классификатор не обучен")
        
        # Сохраняем данные для использования в методах
        self.data = data.copy()
        
        self._calculate_indicators(data)
        
        trend_scores = self._calculate_trend_scores(data)
        momentum_scores = self._calculate_momentum_scores(data)
        price_action_scores = self._calculate_price_action_scores(data)
        market_activity_scores = self._calculate_market_activity_scores(data)
        
        # Общий скор
        overall_scores = (
            trend_scores * self.parameters['trend_weight'] +
            momentum_scores * self.parameters['momentum_weight'] +
            price_action_scores * self.parameters['price_action_weight'] +
            market_activity_scores * 0.1
        )
        
        # Создаем DataFrame с результатами
        results = pd.DataFrame({
            'trend_score': trend_scores,
            'momentum_score': momentum_scores,
            'price_action_score': price_action_scores,
            'market_activity_score': market_activity_scores,
            'overall_score': overall_scores
        }, index=data.index)
        
        return results
