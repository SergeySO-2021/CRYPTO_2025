"""
Правильная реализация Market Zone Analyzer (MZA) на основе оригинального описания
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ProperMZAClassifier:
    """
    Правильная реализация Market Zone Analyzer[BullByte]
    Основана на оригинальном описании с 4 категориями анализа
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Инициализация MZA Classifier
        
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
            
            # Weights
            'trend_weight': 0.40,
            'momentum_weight': 0.30,
            'price_action_weight': 0.30,
            
            # Stability
            'smoothing_periods': 2,
            'use_hysteresis': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        self.params = default_params
        self.data = None
        self.net_scores = []
        self.zones = []
        
        print(f"✅ ProperMZAClassifier инициализирован!")
        print(f"📊 Параметры: ADX={self.params['adx_length']}, RSI={self.params['rsi_length']}")
    
    def fit(self, data: pd.DataFrame) -> 'ProperMZAClassifier':
        """
        Обучение MZA Classifier
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            self
        """
        self.data = data.copy()
        print("🤖 Обучение ProperMZAClassifier...")
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Предсказание рыночных зон
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            Массив предсказаний (-1, 0, 1)
        """
        print("🔮 Предсказание рыночных зон...")
        
        predictions = []
        net_scores = []
        
        for i in range(len(data)):
            if i < max(self.params['slow_ma_length'], self.params['adx_length'] * 2):
                predictions.append(0)  # Недостаточно данных
                net_scores.append(0)
                continue
            
            # Извлекаем данные для текущего момента
            current_data = data.iloc[:i+1]
            
            # Вычисляем все компоненты
            trend_score = self._calculate_trend_score(current_data, i)
            momentum_score = self._calculate_momentum_score(current_data, i)
            price_action_score = self._calculate_price_action_score(current_data, i)
            market_activity_score = self._calculate_market_activity_score(current_data, i)
            
            # Определяем Market Activity State
            activity_state = self._determine_activity_state(market_activity_score)
            
            # Применяем динамические веса
            weights = self._get_dynamic_weights(activity_state)
            
            # Вычисляем net score
            net_score = (
                trend_score * weights['trend'] +
                momentum_score * weights['momentum'] +
                price_action_score * weights['price_action']
            )
            
            # Применяем сглаживание
            if len(net_scores) > 0:
                net_score = (net_score + net_scores[-1]) / 2
            
            net_scores.append(net_score)
            
            # Определяем зону
            if net_score >= 2:
                predictions.append(1)  # Bullish
            elif net_score <= -2:
                predictions.append(-1)  # Bearish
            else:
                predictions.append(0)  # Sideways
        
        self.net_scores = net_scores
        self.zones = predictions
        
        print(f"✅ Предсказано {len(predictions)} режимов")
        return np.array(predictions)
    
    def _calculate_trend_score(self, data: pd.DataFrame, index: int) -> int:
        """Вычисление Trend Score"""
        score = 0
        
        # 1. ADX/DMI
        adx_score = self._calculate_adx_score(data, index)
        score += adx_score
        
        # 2. Moving Average Slope
        ma_score = self._calculate_ma_slope_score(data, index)
        score += ma_score
        
        # 3. Ichimoku Cloud Differential
        ichimoku_score = self._calculate_ichimoku_score(data, index)
        score += ichimoku_score
        
        return score
    
    def _calculate_momentum_score(self, data: pd.DataFrame, index: int) -> int:
        """Вычисление Momentum Score"""
        score = 0
        
        # 1. RSI
        rsi_score = self._calculate_rsi_score(data, index)
        score += rsi_score
        
        # 2. Stochastic %K
        stoch_score = self._calculate_stochastic_score(data, index)
        score += stoch_score
        
        # 3. MACD Histogram
        macd_score = self._calculate_macd_score(data, index)
        score += macd_score
        
        return score
    
    def _calculate_price_action_score(self, data: pd.DataFrame, index: int) -> int:
        """Вычисление Price Action Score"""
        score = 0
        
        # 1. HH/LL Range
        hh_ll_score = self._calculate_hh_ll_score(data, index)
        score += hh_ll_score
        
        # 2. Heikin-Ashi Doji
        ha_score = self._calculate_heikin_ashi_score(data, index)
        score += ha_score
        
        # 3. Candle Range
        candle_score = self._calculate_candle_range_score(data, index)
        score += candle_score
        
        return score
    
    def _calculate_market_activity_score(self, data: pd.DataFrame, index: int) -> int:
        """Вычисление Market Activity Score"""
        score = 0
        
        # 1. Bollinger Bands Width
        bb_score = self._calculate_bb_width_score(data, index)
        score += bb_score
        
        # 2. ATR
        atr_score = self._calculate_atr_score(data, index)
        score += atr_score
        
        # 3. Keltner Channels Width
        kc_score = self._calculate_kc_width_score(data, index)
        score += kc_score
        
        # 4. Volume
        volume_score = self._calculate_volume_score(data, index)
        score += volume_score
        
        return score
    
    def _calculate_adx_score(self, data: pd.DataFrame, index: int) -> int:
        """ADX/DMI Score"""
        try:
            # Упрощенный расчет ADX
            high = data['high'].iloc[index]
            low = data['low'].iloc[index]
            close = data['close'].iloc[index]
            
            if index == 0:
                return 0
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - data['close'].iloc[index-1])
            tr3 = abs(low - data['close'].iloc[index-1])
            tr = max(tr1, tr2, tr3)
            
            # Directional Movement
            dm_plus = max(0, high - data['high'].iloc[index-1]) if high - data['high'].iloc[index-1] > data['low'].iloc[index-1] - low else 0
            dm_minus = max(0, data['low'].iloc[index-1] - low) if data['low'].iloc[index-1] - low > high - data['high'].iloc[index-1] else 0
            
            # Упрощенная логика
            if tr > 0:
                if dm_plus > dm_minus:
                    return 1  # Bullish
                elif dm_minus > dm_plus:
                    return -1  # Bearish
            
            return 0
        except:
            return 0
    
    def _calculate_ma_slope_score(self, data: pd.DataFrame, index: int) -> int:
        """Moving Average Slope Score"""
        try:
            fast_ma = data['close'].rolling(self.params['fast_ma_length']).mean().iloc[index]
            slow_ma = data['close'].rolling(self.params['slow_ma_length']).mean().iloc[index]
            
            if fast_ma > slow_ma:
                return 1  # Bullish
            elif fast_ma < slow_ma:
                return -1  # Bearish
            else:
                return 0
        except:
            return 0
    
    def _calculate_ichimoku_score(self, data: pd.DataFrame, index: int) -> int:
        """Ichimoku Cloud Score"""
        try:
            # Упрощенный расчет Ichimoku
            high_9 = data['high'].rolling(9).max().iloc[index]
            low_9 = data['low'].rolling(9).min().iloc[index]
            high_26 = data['high'].rolling(26).max().iloc[index]
            low_26 = data['low'].rolling(26).min().iloc[index]
            
            senkou_a = (high_9 + low_9) / 2
            senkou_b = (high_26 + low_26) / 2
            
            if senkou_a > senkou_b:
                return 1  # Bullish
            elif senkou_a < senkou_b:
                return -1  # Bearish
            else:
                return 0
        except:
            return 0
    
    def _calculate_rsi_score(self, data: pd.DataFrame, index: int) -> int:
        """RSI Score"""
        try:
            rsi = self._calculate_rsi(data['close'], self.params['rsi_length'])
            rsi_value = rsi.iloc[index]
            
            # Динамические границы
            rsi_ma = rsi.rolling(20).mean().iloc[index]
            rsi_std = rsi.rolling(20).std().iloc[index]
            upper_bound = rsi_ma + rsi_std
            lower_bound = rsi_ma - rsi_std
            
            if rsi_value > upper_bound:
                return 1  # Bullish
            elif rsi_value < lower_bound:
                return -1  # Bearish
            else:
                return 0
        except:
            return 0
    
    def _calculate_stochastic_score(self, data: pd.DataFrame, index: int) -> int:
        """Stochastic %K Score"""
        try:
            k_percent = self._calculate_stochastic_k(data, self.params['stoch_length'])
            k_value = k_percent.iloc[index]
            
            # Динамические границы
            k_ma = k_percent.rolling(20).mean().iloc[index]
            k_std = k_percent.rolling(20).std().iloc[index]
            upper_bound = k_ma + k_std
            lower_bound = k_ma - k_std
            
            if k_value > upper_bound:
                return 1  # Bullish
            elif k_value < lower_bound:
                return -1  # Bearish
            else:
                return 0
        except:
            return 0
    
    def _calculate_macd_score(self, data: pd.DataFrame, index: int) -> int:
        """MACD Histogram Score"""
        try:
            macd_line, signal_line, histogram = self._calculate_macd(data['close'])
            hist_value = histogram.iloc[index]
            
            if hist_value > 0:
                return 1  # Bullish
            elif hist_value < 0:
                return -1  # Bearish
            else:
                return 0
        except:
            return 0
    
    def _calculate_hh_ll_score(self, data: pd.DataFrame, index: int) -> int:
        """HH/LL Range Score"""
        try:
            range_period = self.params['hh_ll_range']
            if index < range_period:
                return 0
            
            recent_high = data['high'].iloc[index-range_period:index+1].max()
            recent_low = data['low'].iloc[index-range_period:index+1].min()
            current_close = data['close'].iloc[index]
            
            # Проверяем близость к экстремумам
            high_distance = abs(current_close - recent_high) / recent_high
            low_distance = abs(current_close - recent_low) / recent_low
            
            if high_distance < 0.01:  # В пределах 1% от максимума
                return 1  # Bullish
            elif low_distance < 0.01:  # В пределах 1% от минимума
                return -1  # Bearish
            else:
                return 0
        except:
            return 0
    
    def _calculate_heikin_ashi_score(self, data: pd.DataFrame, index: int) -> int:
        """Heikin-Ashi Score"""
        try:
            ha_close, ha_open = self._calculate_heikin_ashi(data)
            ha_close_val = ha_close.iloc[index]
            ha_open_val = ha_open.iloc[index]
            
            if ha_close_val > ha_open_val:
                return 1  # Bullish
            elif ha_close_val < ha_open_val:
                return -1  # Bearish
            else:
                return 0
        except:
            return 0
    
    def _calculate_candle_range_score(self, data: pd.DataFrame, index: int) -> int:
        """Candle Range Score"""
        try:
            range_length = self.params['candle_range_length']
            if index < range_length:
                return 0
            
            current_range = data['high'].iloc[index] - data['low'].iloc[index]
            avg_range = data['high'].rolling(range_length).max().iloc[index] - data['low'].rolling(range_length).min().iloc[index]
            range_std = (data['high'] - data['low']).rolling(range_length).std().iloc[index]
            
            upper_bound = avg_range + range_std
            lower_bound = avg_range - range_std
            
            if current_range > upper_bound:
                return 1  # Bullish
            elif current_range < lower_bound:
                return -1  # Bearish
            else:
                return 0
        except:
            return 0
    
    def _calculate_bb_width_score(self, data: pd.DataFrame, index: int) -> int:
        """Bollinger Bands Width Score"""
        try:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['close'])
            bb_width = (bb_upper.iloc[index] - bb_lower.iloc[index]) / bb_middle.iloc[index]
            
            bb_width_ma = bb_width.rolling(20).mean().iloc[index]
            bb_width_std = bb_width.rolling(20).std().iloc[index]
            
            upper_threshold = bb_width_ma + bb_width_std
            lower_threshold = bb_width_ma - bb_width_std
            
            if bb_width > upper_threshold:
                return 1  # High volatility
            elif bb_width < lower_threshold:
                return -1  # Low volatility
            else:
                return 0
        except:
            return 0
    
    def _calculate_atr_score(self, data: pd.DataFrame, index: int) -> int:
        """ATR Score"""
        try:
            atr = self._calculate_atr(data)
            atr_value = atr.iloc[index]
            atr_ma = atr.rolling(20).mean().iloc[index]
            atr_std = atr.rolling(20).std().iloc[index]
            
            upper_threshold = atr_ma + atr_std
            lower_threshold = atr_ma - atr_std
            
            if atr_value > upper_threshold:
                return 1  # High volatility
            elif atr_value < lower_threshold:
                return -1  # Low volatility
            else:
                return 0
        except:
            return 0
    
    def _calculate_kc_width_score(self, data: pd.DataFrame, index: int) -> int:
        """Keltner Channels Width Score"""
        try:
            kc_upper, kc_middle, kc_lower = self._calculate_keltner_channels(data)
            kc_width = (kc_upper.iloc[index] - kc_lower.iloc[index]) / kc_middle.iloc[index]
            
            kc_width_ma = kc_width.rolling(20).mean().iloc[index]
            kc_width_std = kc_width.rolling(20).std().iloc[index]
            
            upper_threshold = kc_width_ma + kc_width_std
            lower_threshold = kc_width_ma - kc_width_std
            
            if kc_width > upper_threshold:
                return 1  # High volatility
            elif kc_width < lower_threshold:
                return -1  # Low volatility
            else:
                return 0
        except:
            return 0
    
    def _calculate_volume_score(self, data: pd.DataFrame, index: int) -> int:
        """Volume Score"""
        try:
            if 'volume' not in data.columns:
                return 0
            
            volume = data['volume'].iloc[index]
            volume_ma = data['volume'].rolling(self.params['volume_ma_length']).mean().iloc[index]
            volume_std = data['volume'].rolling(self.params['volume_ma_length']).std().iloc[index]
            
            upper_threshold = volume_ma + volume_std
            lower_threshold = volume_ma - volume_std
            
            if volume > upper_threshold:
                return 1  # High volume
            elif volume < lower_threshold:
                return -1  # Low volume
            else:
                return 0
        except:
            return 0
    
    def _determine_activity_state(self, market_activity_score: int) -> str:
        """Определение Market Activity State"""
        if market_activity_score >= 2:
            return "High"
        elif market_activity_score <= -2:
            return "Low"
        else:
            return "Medium"
    
    def _get_dynamic_weights(self, activity_state: str) -> Dict[str, float]:
        """Получение динамических весов"""
        if activity_state == "High":
            return {
                'trend': 0.50,
                'momentum': 0.35,
                'price_action': 0.15
            }
        elif activity_state == "Low":
            return {
                'trend': 0.25,
                'momentum': 0.20,
                'price_action': 0.55
            }
        else:  # Medium
            return {
                'trend': self.params['trend_weight'],
                'momentum': self.params['momentum_weight'],
                'price_action': self.params['price_action_weight']
            }
    
    # Вспомогательные методы для расчета индикаторов
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic_k(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Расчет Stochastic %K"""
        lowest_low = data['low'].rolling(period).min()
        highest_high = data['high'].rolling(period).max()
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        return k_percent
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Расчет MACD"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_heikin_ashi(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Расчет Heikin-Ashi"""
        ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        ha_open = pd.Series(index=data.index, dtype=float)
        ha_open.iloc[0] = (data['open'].iloc[0] + data['close'].iloc[0]) / 2
        
        for i in range(1, len(data)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
        
        return ha_close, ha_open
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Расчет Bollinger Bands"""
        middle = prices.rolling(self.params['bb_length']).mean()
        std = prices.rolling(self.params['bb_length']).std()
        upper = middle + (std * self.params['bb_multiplier'])
        lower = middle - (std * self.params['bb_multiplier'])
        return upper, middle, lower
    
    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Расчет ATR"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(self.params['atr_length']).mean()
        return atr
    
    def _calculate_keltner_channels(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Расчет Keltner Channels"""
        middle = data['close'].ewm(span=self.params['kc_length']).mean()
        atr = self._calculate_atr(data)
        upper = middle + (atr * self.params['kc_multiplier'])
        lower = middle - (atr * self.params['kc_multiplier'])
        return upper, middle, lower
