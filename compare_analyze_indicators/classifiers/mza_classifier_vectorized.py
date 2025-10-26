"""
Полная векторизованная версия Market Zone Analyzer[BullByte]
Сохраняет всю сложность оригинального алгоритма с векторизованными вычислениями
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class VectorizedMZAClassifier:
    """
    Полная векторизованная версия Market Zone Analyzer[BullByte]
    
    Особенности:
    - Все 12+ индикаторов оригинального MZA
    - Динамические границы RSI и Stochastic
    - Ichimoku Cloud анализ
    - Heikin-Ashi паттерны
    - Volume анализ
    - ADX/DMI
    - Векторизованные вычисления для скорости
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Инициализация VectorizedMZAClassifier
        
        Args:
            parameters: Параметры классификатора
        """
        default_params = {
            # Trend Indicators
            'adx_length': 14,
            'adx_threshold': 20,
            'fast_ma_length': 20,
            'slow_ma_length': 50,
            
            # Momentum Indicators
            'rsi_length': 14,
            'stoch_length': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Price Action Indicators
            'hh_ll_range': 20,
            'ha_doji_range': 5,
            'candle_range_length': 8,
            
            # Market Activity Indicators
            'bb_length': 20,
            'bb_multiplier': 2.0,
            'atr_length': 14,
            'kc_length': 20,
            'kc_multiplier': 1.5,
            'volume_ma_length': 20,
            
            # Scoring and Weights
            'trend_weight': 0.40,
            'momentum_weight': 0.30,
            'price_action_weight': 0.30,
            'bullish_threshold': 2.0,
            'bearish_threshold': -2.0,
            'smoothing_length': 2,
            'hysteresis_bars': 2,
        }
        self.parameters = {**default_params, **(parameters or {})}
        self.is_fitted = False
        self.data = None
        self.indicators = pd.DataFrame()
        
        print("✅ VectorizedMZAClassifier инициализирован!")
        print("🚀 Полная версия MZA с векторизованными вычислениями")
    
    def fit(self, data: pd.DataFrame) -> 'VectorizedMZAClassifier':
        """
        Обучение классификатора (вычисление всех индикаторов)
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            self
        """
        self.validate_data(data, require_volume=True)
        
        # Сохраняем данные для обучения
        self.data = data.copy()
        
        # Вычисляем все индикаторы векторизованно
        self._calculate_all_indicators_vectorized(data)
        
        self.is_fitted = True
        print("✅ VectorizedMZAClassifier обучен!")
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Предсказание рыночных зон с векторизованными вычислениями
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            Массив предсказаний
        """
        if not self.is_fitted:
            raise ValueError("Классификатор не обучен. Вызовите fit() сначала.")
        
        self.validate_data(data, require_volume=True)
        
        # Вычисляем все индикаторы для предсказания
        indicators = self._calculate_all_indicators_vectorized(data)
        
        # Векторизованное вычисление скоров
        trend_scores = self._calculate_trend_scores_vectorized(indicators)
        momentum_scores = self._calculate_momentum_scores_vectorized(indicators)
        price_action_scores = self._calculate_price_action_scores_vectorized(indicators)
        market_activity_scores = self._calculate_market_activity_scores_vectorized(indicators)
        
        # Определяем Market Activity State векторизованно
        market_activity_states = self._determine_market_activity_states_vectorized(market_activity_scores)
        
        # Вычисляем динамические веса векторизованно
        trend_weights, momentum_weights, price_action_weights = self._calculate_dynamic_weights_vectorized(market_activity_states)
        
        # Вычисляем net scores векторизованно
        net_scores = (
            trend_scores * trend_weights +
            momentum_scores * momentum_weights +
            price_action_scores * price_action_weights
        )
        
        # Применяем сглаживание векторизованно
        if self.parameters['smoothing_length'] > 1:
            net_scores = self._apply_smoothing_vectorized(net_scores)
        
        # Определяем зоны векторизованно
        predictions = self._determine_zones_vectorized(net_scores)
        
        return predictions
    
    def fit_predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Обучает классификатор и сразу делает предсказания.
        """
        self.fit(data)
        return self.predict(data)
    
    def _calculate_all_indicators_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Вычисление всех индикаторов векторизованно"""
        indicators = pd.DataFrame(index=data.index)
        
        # ===== TREND INDICATORS =====
        
        # ADX/DMI
        indicators['adx'], indicators['plus_di'], indicators['minus_di'] = self._calculate_adx_vectorized(data)
        
        # Moving Average Slope
        indicators['ma_fast'] = data['close'].rolling(self.parameters['fast_ma_length']).mean()
        indicators['ma_slow'] = data['close'].rolling(self.parameters['slow_ma_length']).mean()
        indicators['ma_slope'] = (indicators['ma_fast'] - indicators['ma_slow']) / indicators['ma_slow']
        
        # Ichimoku Cloud
        indicators['ichimoku_a'], indicators['ichimoku_b'] = self._calculate_ichimoku_vectorized(data)
        indicators['ichimoku_differential'] = indicators['ichimoku_a'] - indicators['ichimoku_b']
        
        # ===== MOMENTUM INDICATORS =====
        
        # RSI с динамическими границами
        indicators['rsi'] = self._calculate_rsi_vectorized(data)
        indicators['rsi_upper'], indicators['rsi_lower'] = self._calculate_dynamic_rsi_bounds_vectorized(indicators['rsi'])
        
        # Stochastic с динамическими границами
        indicators['stoch_k'] = self._calculate_stochastic_vectorized(data)
        indicators['stoch_upper'], indicators['stoch_lower'] = self._calculate_dynamic_stoch_bounds_vectorized(indicators['stoch_k'])
        
        # MACD Histogram
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self._calculate_macd_vectorized(data)
        
        # ===== PRICE ACTION INDICATORS =====
        
        # Highest High / Lowest Low
        indicators['hh_ll_score'] = self._calculate_hh_ll_vectorized(data)
        
        # Heikin-Ashi
        indicators['ha_open'], indicators['ha_close'], indicators['ha_high'], indicators['ha_low'] = self._calculate_heikin_ashi_vectorized(data)
        indicators['ha_doji_score'] = self._calculate_ha_doji_vectorized(indicators)
        
        # Candle Range
        indicators['candle_range_score'] = self._calculate_candle_range_vectorized(data)
        
        # ===== MARKET ACTIVITY INDICATORS =====
        
        # Bollinger Bands Width
        indicators['bb_width'] = self._calculate_bb_width_vectorized(data)
        
        # ATR
        indicators['atr'] = self._calculate_atr_vectorized(data)
        
        # Keltner Channels Width
        indicators['kc_width'] = self._calculate_kc_width_vectorized(data)
        
        # Volume
        indicators['volume_ma'] = data['volume'].rolling(self.parameters['volume_ma_length']).mean()
        indicators['volume_std'] = data['volume'].rolling(self.parameters['volume_ma_length']).std()
        indicators['volume_score'] = self._calculate_volume_score_vectorized(data, indicators)
        
        return indicators
    
    def _calculate_adx_vectorized(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Векторизованный расчет ADX/DMI"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        plus_dm = np.where(
            (high.diff() > low.diff().abs()) & (high.diff() > 0),
            high.diff(), 0
        )
        minus_dm = np.where(
            (low.diff().abs() > high.diff()) & (low.diff() < 0),
            low.diff().abs(), 0
        )
        
        # Smoothing
        period = self.parameters['adx_length']
        plus_dm_smooth = pd.Series(plus_dm, index=data.index).rolling(period).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=data.index).rolling(period).mean()
        tr_smooth = true_range.rolling(period).mean()
        
        # DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # ADX
        adx = dx.rolling(period).mean()
        
        return adx, plus_di, minus_di
    
    def _calculate_rsi_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Векторизованный расчет RSI"""
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.parameters['rsi_length']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.parameters['rsi_length']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_dynamic_rsi_bounds_vectorized(self, rsi: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Векторизованный расчет динамических границ RSI"""
        window = 20
        rsi_ma = rsi.rolling(window).mean()
        rsi_std = rsi.rolling(window).std()
        
        upper_bound = rsi_ma + rsi_std
        lower_bound = rsi_ma - rsi_std
        
        return upper_bound, lower_bound
    
    def _calculate_stochastic_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Векторизованный расчет Stochastic %K"""
        period = self.parameters['stoch_length']
        lowest_low = data['low'].rolling(period).min()
        highest_high = data['high'].rolling(period).max()
        
        stoch_k = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
        return stoch_k
    
    def _calculate_dynamic_stoch_bounds_vectorized(self, stoch: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Векторизованный расчет динамических границ Stochastic"""
        window = 20
        stoch_ma = stoch.rolling(window).mean()
        stoch_std = stoch.rolling(window).std()
        
        upper_bound = stoch_ma + stoch_std
        lower_bound = stoch_ma - stoch_std
        
        return upper_bound, lower_bound
    
    def _calculate_macd_vectorized(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Векторизованный расчет MACD"""
        ema_fast = data['close'].ewm(span=self.parameters['macd_fast']).mean()
        ema_slow = data['close'].ewm(span=self.parameters['macd_slow']).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.parameters['macd_signal']).mean()
        macd_hist = macd_line - signal_line
        
        return macd_line, signal_line, macd_hist
    
    def _calculate_ichimoku_vectorized(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Векторизованный расчет Ichimoku Cloud"""
        high_9 = data['high'].rolling(9).max()
        low_9 = data['low'].rolling(9).min()
        high_26 = data['high'].rolling(26).max()
        low_26 = data['low'].rolling(26).min()
        
        senkou_a = (high_9 + low_9) / 2
        senkou_b = (high_26 + low_26) / 2
        
        return senkou_a, senkou_b
    
    def _calculate_hh_ll_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Векторизованный расчет Highest High / Lowest Low"""
        period = self.parameters['hh_ll_range']
        recent_high = data['high'].rolling(period).max()
        recent_low = data['low'].rolling(period).min()
        
        # Позиция цены относительно диапазона
        price_position = (data['close'] - recent_low) / (recent_high - recent_low)
        
        # Scoring
        hh_ll_score = np.where(
            price_position > 0.8, 1,  # Near highs
            np.where(price_position < 0.2, -1, 0)  # Near lows
        )
        
        return pd.Series(hh_ll_score, index=data.index)
    
    def _calculate_heikin_ashi_vectorized(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Векторизованный расчет Heikin-Ashi"""
        ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        ha_open = np.zeros(len(data))
        ha_open[0] = (data['open'].iloc[0] + data['close'].iloc[0]) / 2
        
        for i in range(1, len(data)):
            ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
        
        ha_high = np.maximum(data['high'], np.maximum(ha_open, ha_close))
        ha_low = np.minimum(data['low'], np.minimum(ha_open, ha_close))
        
        return (pd.Series(ha_open, index=data.index),
                pd.Series(ha_close, index=data.index),
                pd.Series(ha_high, index=data.index),
                pd.Series(ha_low, index=data.index))
    
    def _calculate_ha_doji_vectorized(self, indicators: pd.DataFrame) -> pd.Series:
        """Векторизованный расчет Heikin-Ashi Doji"""
        ha_body = np.abs(indicators['ha_close'] - indicators['ha_open'])
        ha_range = indicators['ha_high'] - indicators['ha_low']
        
        # Doji detection
        doji_ratio = ha_body / ha_range
        doji_threshold = 0.1  # 10% от диапазона
        
        ha_doji_score = np.where(
            doji_ratio < doji_threshold, 0,  # Doji
            np.where(indicators['ha_close'] > indicators['ha_open'], 1, -1)  # Bullish/Bearish
        )
        
        return pd.Series(ha_doji_score, index=indicators.index)
    
    def _calculate_candle_range_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Векторизованный расчет Candle Range"""
        period = self.parameters['candle_range_length']
        candle_range = data['high'] - data['low']
        range_ma = candle_range.rolling(period).mean()
        range_std = candle_range.rolling(period).std()
        
        # Scoring
        upper_threshold = range_ma + range_std
        lower_threshold = range_ma - range_std
        
        candle_range_score = np.where(
            candle_range > upper_threshold, 1,  # Large range
            np.where(candle_range < lower_threshold, -1, 0)  # Small range
        )
        
        return pd.Series(candle_range_score, index=data.index)
    
    def _calculate_bb_width_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Векторизованный расчет Bollinger Bands Width"""
        period = self.parameters['bb_length']
        multiplier = self.parameters['bb_multiplier']
        
        bb_middle = data['close'].rolling(period).mean()
        bb_std = data['close'].rolling(period).std()
        bb_upper = bb_middle + (bb_std * multiplier)
        bb_lower = bb_middle - (bb_std * multiplier)
        
        bb_width = (bb_upper - bb_lower) / bb_middle
        return bb_width
    
    def _calculate_atr_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Векторизованный расчет ATR"""
        period = self.parameters['atr_length']
        
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = pd.Series(true_range, index=data.index).rolling(period).mean()
        
        return atr
    
    def _calculate_kc_width_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Векторизованный расчет Keltner Channels Width"""
        period = self.parameters['kc_length']
        multiplier = self.parameters['kc_multiplier']
        
        kc_middle = data['close'].ewm(span=period).mean()
        atr = self._calculate_atr_vectorized(data)
        kc_upper = kc_middle + (atr * multiplier)
        kc_lower = kc_middle - (atr * multiplier)
        
        kc_width = (kc_upper - kc_lower) / kc_middle
        return kc_width
    
    def _calculate_volume_score_vectorized(self, data: pd.DataFrame, indicators: pd.DataFrame) -> pd.Series:
        """Векторизованный расчет Volume Score"""
        volume = data['volume']
        volume_ma = indicators['volume_ma']
        volume_std = indicators['volume_std']
        
        upper_threshold = volume_ma + volume_std
        lower_threshold = volume_ma - volume_std
        
        volume_score = np.where(
            volume > upper_threshold, 1,  # High volume
            np.where(volume < lower_threshold, -1, 0)  # Low volume
        )
        
        return pd.Series(volume_score, index=data.index)
    
    def _calculate_trend_scores_vectorized(self, indicators: pd.DataFrame) -> pd.Series:
        """Векторизованный расчет Trend Scores"""
        # ADX Score
        adx_score = np.where(
            (indicators['adx'] >= self.parameters['adx_threshold']) & (indicators['plus_di'] > indicators['minus_di']), 1,
            np.where(
                (indicators['adx'] >= self.parameters['adx_threshold']) & (indicators['minus_di'] > indicators['plus_di']), -1, 0
            )
        )
        
        # MA Slope Score
        ma_slope_score = np.where(indicators['ma_slope'] > 0, 1, -1)
        
        # Ichimoku Score
        ichimoku_score = np.where(indicators['ichimoku_differential'] > 0, 1, -1)
        
        # Combined Trend Score
        trend_score = adx_score + ma_slope_score + ichimoku_score
        
        return pd.Series(trend_score, index=indicators.index)
    
    def _calculate_momentum_scores_vectorized(self, indicators: pd.DataFrame) -> pd.Series:
        """Векторизованный расчет Momentum Scores"""
        # RSI Score
        rsi_score = np.where(
            indicators['rsi'] > indicators['rsi_upper'], 1,
            np.where(indicators['rsi'] < indicators['rsi_lower'], -1, 0)
        )
        
        # Stochastic Score
        stoch_score = np.where(
            indicators['stoch_k'] > indicators['stoch_upper'], 1,
            np.where(indicators['stoch_k'] < indicators['stoch_lower'], -1, 0)
        )
        
        # MACD Score
        macd_score = np.where(indicators['macd_hist'] > 0, 1, -1)
        
        # Combined Momentum Score
        momentum_score = rsi_score + stoch_score + macd_score
        
        return pd.Series(momentum_score, index=indicators.index)
    
    def _calculate_price_action_scores_vectorized(self, indicators: pd.DataFrame) -> pd.Series:
        """Векторизованный расчет Price Action Scores"""
        # HH/LL Score уже вычислен
        hh_ll_score = indicators['hh_ll_score']
        
        # Heikin-Ashi Score уже вычислен
        ha_doji_score = indicators['ha_doji_score']
        
        # Candle Range Score уже вычислен
        candle_range_score = indicators['candle_range_score']
        
        # Combined Price Action Score
        price_action_score = hh_ll_score + ha_doji_score + candle_range_score
        
        return pd.Series(price_action_score, index=indicators.index)
    
    def _calculate_market_activity_scores_vectorized(self, indicators: pd.DataFrame) -> pd.Series:
        """Векторизованный расчет Market Activity Scores"""
        # BB Width Score
        bb_width_ma = indicators['bb_width'].rolling(20).mean()
        bb_width_std = indicators['bb_width'].rolling(20).std()
        bb_width_score = np.where(
            indicators['bb_width'] > bb_width_ma + bb_width_std, 1,
            np.where(indicators['bb_width'] < bb_width_ma - bb_width_std, -1, 0)
        )
        
        # ATR Score
        atr_ma = indicators['atr'].rolling(20).mean()
        atr_std = indicators['atr'].rolling(20).std()
        atr_score = np.where(
            indicators['atr'] > atr_ma + atr_std, 1,
            np.where(indicators['atr'] < atr_ma - atr_std, -1, 0)
        )
        
        # KC Width Score
        kc_width_ma = indicators['kc_width'].rolling(20).mean()
        kc_width_std = indicators['kc_width'].rolling(20).std()
        kc_width_score = np.where(
            indicators['kc_width'] > kc_width_ma + kc_width_std, 1,
            np.where(indicators['kc_width'] < kc_width_ma - kc_width_std, -1, 0)
        )
        
        # Volume Score уже вычислен
        volume_score = indicators['volume_score']
        
        # Combined Market Activity Score
        market_activity_score = bb_width_score + atr_score + kc_width_score + volume_score
        
        return pd.Series(market_activity_score, index=indicators.index)
    
    def _determine_market_activity_states_vectorized(self, market_activity_scores: pd.Series) -> pd.Series:
        """Векторизованное определение Market Activity States"""
        states = np.where(
            market_activity_scores >= 2, 'High',
            np.where(market_activity_scores <= -2, 'Low', 'Medium')
        )
        
        return pd.Series(states, index=market_activity_scores.index)
    
    def _calculate_dynamic_weights_vectorized(self, market_activity_states: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Векторизованный расчет динамических весов"""
        trend_weights = np.where(
            market_activity_states == 'High', 0.5,
            np.where(market_activity_states == 'Low', 0.25, self.parameters['trend_weight'])
        )
        
        momentum_weights = np.where(
            market_activity_states == 'High', 0.35,
            np.where(market_activity_states == 'Low', 0.20, self.parameters['momentum_weight'])
        )
        
        price_action_weights = np.where(
            market_activity_states == 'High', 0.15,
            np.where(market_activity_states == 'Low', 0.55, self.parameters['price_action_weight'])
        )
        
        return (pd.Series(trend_weights, index=market_activity_states.index),
                pd.Series(momentum_weights, index=market_activity_states.index),
                pd.Series(price_action_weights, index=market_activity_states.index))
    
    def _apply_smoothing_vectorized(self, net_scores: pd.Series) -> pd.Series:
        """Векторизованное сглаживание"""
        return net_scores.rolling(self.parameters['smoothing_length']).mean()
    
    def _determine_zones_vectorized(self, net_scores: pd.Series) -> np.ndarray:
        """Векторизованное определение зон"""
        predictions = np.where(
            net_scores >= self.parameters['bullish_threshold'], 1,
            np.where(net_scores <= self.parameters['bearish_threshold'], -1, 0)
        )
        
        # Первые бары = 0 (недостаточно данных)
        min_period = max(self.parameters['slow_ma_length'], self.parameters['volume_ma_length'], self.parameters['atr_length'])
        predictions[:min_period] = 0
        
        return predictions
    
    def validate_data(self, data: pd.DataFrame, require_volume: bool = True) -> bool:
        """
        Валидация входных данных
        
        Args:
            data: DataFrame для проверки
            require_volume: Требовать ли колонку volume
            
        Returns:
            True если данные валидны, False иначе
        """
        required_columns = ['open', 'high', 'low', 'close']
        if require_volume:
            required_columns.append('volume')
        
        # Проверяем наличие всех необходимых колонок
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            raise ValueError(f"Отсутствуют колонки: {missing_cols}")
        
        # Проверяем на пропуски
        if data[required_columns].isnull().any().any():
            raise ValueError("В данных есть пропуски")
        
        # Проверяем логичность данных
        if not (data['high'] >= data['low']).all():
            raise ValueError("Некорректные данные: high < low")
        
        if not (data['high'] >= data['open']).all():
            raise ValueError("Некорректные данные: high < open")
        
        if not (data['high'] >= data['close']).all():
            raise ValueError("Некорректные данные: high < close")
        
        if not (data['low'] <= data['open']).all():
            raise ValueError("Некорректные данные: low > open")
        
        if not (data['low'] <= data['close']).all():
            raise ValueError("Некорректные данные: low > close")
        
        if require_volume and not (data['volume'] >= 0).all():
            raise ValueError("Некорректные данные: volume < 0")
        
        return True
