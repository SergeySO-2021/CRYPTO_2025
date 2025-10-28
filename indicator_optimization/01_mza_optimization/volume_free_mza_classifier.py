#!/usr/bin/env python3
"""
MZA без Volume - адаптированная версия для работы с данными без Volume
Использует альтернативные методы определения рыночной активности
"""

import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, List, Tuple, Optional

class VolumeFreeMZAClassifier:
    """
    MZA классификатор, адаптированный для работы без данных Volume
    Использует альтернативные методы определения рыночной активности
    """
    
    def __init__(self, parameters: Dict):
        self.params = parameters
        self.required_columns = ['open', 'high', 'low', 'close']  # Без volume
        
    def calculate_market_activity_without_volume(self, data: pd.DataFrame) -> Dict:
        """
        Определение рыночной активности без Volume
        Использует альтернативные индикаторы активности
        """
        
        # 1. Волатильность как замена Volume
        price_range = data['high'] - data['low']
        volatility = price_range.rolling(window=self.params['atrLength']).std()
        volatility_ma = volatility.rolling(window=self.params['atrLength']).mean()
        volatility_std = volatility.rolling(window=self.params['atrLength']).std()
        
        # 2. ATR как индикатор активности
        atr = ta.ATR(data['high'], data['low'], data['close'], 
                    timeperiod=self.params['atrLength'])
        atr_ma = ta.SMA(atr, timeperiod=self.params['atrLength'])
        atr_std = ta.STDDEV(atr, timeperiod=self.params['atrLength'])
        
        # 3. Bollinger Bands ширина как индикатор активности
        bb_upper, bb_middle, bb_lower = ta.BBANDS(data['close'],
                                                timeperiod=self.params['bbLength'],
                                                nbdevup=self.params['bbMultiplier'],
                                                nbdevdn=self.params['bbMultiplier'])
        bb_width = bb_upper - bb_lower
        bb_width_ma = ta.SMA(bb_width, timeperiod=self.params['bbLength'])
        bb_width_std = ta.STDDEV(bb_width, timeperiod=self.params['bbLength'])
        
        # 4. Количество значимых движений
        significant_moves = (price_range > price_range.rolling(window=20).quantile(0.7)).astype(int)
        move_frequency = significant_moves.rolling(window=20).mean()
        
        return {
            'volatility': volatility,
            'volatility_ma': volatility_ma,
            'volatility_std': volatility_std,
            'atr': atr,
            'atr_ma': atr_ma,
            'atr_std': atr_std,
            'bb_width': bb_width,
            'bb_width_ma': bb_width_ma,
            'bb_width_std': bb_width_std,
            'move_frequency': move_frequency
        }
    
    def determine_market_activity_state(self, market_data: Dict, i: int) -> str:
        """
        Определение состояния рыночной активности без Volume
        """
        try:
            # Комбинированный индикатор активности
            volatility = market_data['volatility'].iloc[i]
            volatility_ma = market_data['volatility_ma'].iloc[i]
            volatility_std = market_data['volatility_std'].iloc[i]
            
            atr = market_data['atr'].iloc[i]
            atr_ma = market_data['atr_ma'].iloc[i]
            atr_std = market_data['atr_std'].iloc[i]
            
            bb_width = market_data['bb_width'].iloc[i]
            bb_width_ma = market_data['bb_width_ma'].iloc[i]
            bb_width_std = market_data['bb_width_std'].iloc[i]
            
            move_frequency = market_data['move_frequency'].iloc[i]
            
            # Композитный скор активности
            activity_score = 0
            
            # Волатильность выше нормы
            if volatility > volatility_ma + volatility_std:
                activity_score += 2
            elif volatility > volatility_ma:
                activity_score += 1
            
            # ATR выше нормы
            if atr > atr_ma + atr_std:
                activity_score += 2
            elif atr > atr_ma:
                activity_score += 1
            
            # Bollinger Bands шире нормы
            if bb_width > bb_width_ma + bb_width_std:
                activity_score += 2
            elif bb_width > bb_width_ma:
                activity_score += 1
            
            # Частые значимые движения
            if move_frequency > 0.3:
                activity_score += 1
            
            # Определение состояния
            if activity_score >= 5:
                return "High"
            elif activity_score >= 2:
                return "Medium"
            else:
                return "Low"
                
        except Exception as e:
            return "Medium"  # По умолчанию
    
    def calculate_adaptive_weights_without_volume(self, market_activity: str, params: Dict) -> Dict:
        """
        Адаптивные веса без использования Volume
        """
        base_weights = {
            'trend': params['trendWeightBase'],
            'momentum': params['momentumWeightBase'],
            'price_action': params['priceActionWeightBase']
        }
        
        # Нормализация весов
        total = sum(base_weights.values())
        base_weights = {k: v/total for k, v in base_weights.items()}
        
        # Адаптация под активность рынка
        if market_activity == "High":
            # Высокая активность - больше доверия к тренду и моментуму
            weights = {
                'trend': base_weights['trend'] * 1.3,
                'momentum': base_weights['momentum'] * 1.2,
                'price_action': base_weights['price_action'] * 0.7
            }
        elif market_activity == "Low":
            # Низкая активность - больше доверия к price action
            weights = {
                'trend': base_weights['trend'] * 0.8,
                'momentum': base_weights['momentum'] * 0.9,
                'price_action': base_weights['price_action'] * 1.4
            }
        else:  # Medium
            weights = base_weights.copy()
        
        # Нормализация
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Основная функция предсказания зон без Volume
        """
        try:
            if len(data) < max(self.params['slowMALength'], self.params['adxLength']):
                return np.zeros(len(data))
            
            # Расчет всех индикаторов
            trend_data = self.calculate_trend_indicators(data)
            momentum_data = self.calculate_momentum_indicators(data)
            price_action_data = self.calculate_price_action_indicators(data)
            market_activity_data = self.calculate_market_activity_without_volume(data)
            
            zones = np.zeros(len(data))
            
            # Инициализация состояний
            last_zone = "SIDEWAYS"
            zone_counter = 0
            
            for i in range(len(data)):
                if i < max(self.params['slowMALength'], self.params['adxLength']):
                    zones[i] = 0
                    continue
                
                # Расчет скоринга
                trend_score = self.calculate_trend_score(trend_data, i)
                momentum_score = self.calculate_momentum_score(momentum_data, i)
                price_action_score = self.calculate_price_action_score(price_action_data, trend_score, i)
                
                # Определение активности рынка
                market_activity = self.determine_market_activity_state(market_activity_data, i)
                
                # Адаптивные веса
                weights = self.calculate_adaptive_weights_without_volume(market_activity, self.params)
                
                # Финальный скор
                net_score = (trend_score * weights['trend'] + 
                           momentum_score * weights['momentum'] + 
                           price_action_score * weights['price_action'])
                
                # Определение зоны
                if net_score > 0.6:
                    current_zone = "BULLISH"
                elif net_score < -0.6:
                    current_zone = "BEARISH"
                else:
                    current_zone = "SIDEWAYS"
                
                # Применение гистерезиса
                if self.params.get('useHysteresis', True):
                    current_zone = self.apply_hysteresis(current_zone, last_zone, zone_counter)
                
                # Конвертация в числовой формат
                if current_zone == "BULLISH":
                    zones[i] = 1
                elif current_zone == "BEARISH":
                    zones[i] = -1
                else:
                    zones[i] = 0
                
                # Обновление состояний
                if current_zone != last_zone:
                    zone_counter = 0
                else:
                    zone_counter += 1
                last_zone = current_zone
            
            return zones
            
        except Exception as e:
            print(f"❌ Ошибка в predict: {e}")
            return np.zeros(len(data))
    
    # Остальные методы остаются такими же, как в оригинальном AccurateMZAClassifier
    def calculate_trend_indicators(self, data: pd.DataFrame) -> Dict:
        """Расчет трендовых индикаторов"""
        try:
            # ADX/DMI
            adx = ta.ADX(data['high'], data['low'], data['close'], 
                        timeperiod=self.params['adxLength'])
            plus_di = ta.PLUS_DI(data['high'], data['low'], data['close'],
                               timeperiod=self.params['adxLength'])
            minus_di = ta.MINUS_DI(data['high'], data['low'], data['close'],
                                 timeperiod=self.params['adxLength'])
            
            # Moving Averages
            fast_ma = ta.SMA(data['close'], timeperiod=self.params['fastMALength'])
            slow_ma = ta.SMA(data['close'], timeperiod=self.params['slowMALength'])
            ma_slope = fast_ma - slow_ma
            
            return {
                'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di,
                'fast_ma': fast_ma, 'slow_ma': slow_ma, 'ma_slope': ma_slope
            }
        except Exception as e:
            print(f"❌ Ошибка в calculate_trend_indicators: {e}")
            return {}
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict:
        """Расчет индикаторов моментума"""
        try:
            # RSI
            rsi = ta.RSI(data['close'], timeperiod=self.params['rsiLength'])
            
            # Stochastic
            stoch_k = ta.STOCH(data['high'], data['low'], data['close'],
                             fastk_period=self.params['stochKLength'],
                             slowk_period=3, slowk_matype=0,
                             slowd_period=3, slowd_matype=0)[0]
            
            # MACD
            macd, macd_signal, macd_hist = ta.MACD(data['close'],
                                                  fastperiod=self.params['macdFast'],
                                                  slowperiod=self.params['macdSlow'],
                                                  signalperiod=self.params['macdSignal'])
            
            return {
                'rsi': rsi, 'stoch_k': stoch_k, 'macd_hist': macd_hist
            }
        except Exception as e:
            print(f"❌ Ошибка в calculate_momentum_indicators: {e}")
            return {}
    
    def calculate_price_action_indicators(self, data: pd.DataFrame) -> Dict:
        """Расчет индикаторов price action"""
        try:
            # HH/LL
            highest_high = ta.MAX(data['high'], timeperiod=self.params['hhllRange'])
            lowest_low = ta.MIN(data['low'], timeperiod=self.params['hhllRange'])
            price_range = highest_high - lowest_low
            
            # Candle Range
            candle_range = data['high'] - data['low']
            avg_candle_range = ta.SMA(candle_range, timeperiod=self.params['candleRangeLength'])
            
            return {
                'price_range': price_range, 'candle_range': candle_range, 
                'avg_candle_range': avg_candle_range
            }
        except Exception as e:
            print(f"❌ Ошибка в calculate_price_action_indicators: {e}")
            return {}
    
    def calculate_trend_score(self, trend_data: Dict, i: int) -> float:
        """Расчет трендового скора"""
        try:
            if not trend_data:
                return 0.0
            
            adx = trend_data['adx'].iloc[i]
            plus_di = trend_data['plus_di'].iloc[i]
            minus_di = trend_data['minus_di'].iloc[i]
            ma_slope = trend_data['ma_slope'].iloc[i]
            
            score = 0.0
            
            # ADX сила тренда
            if adx > self.params['adxThreshold']:
                if plus_di > minus_di:
                    score += 0.4
                else:
                    score -= 0.4
            
            # MA slope
            if ma_slope > 0:
                score += 0.3
            else:
                score -= 0.3
            
            return np.clip(score, -1.0, 1.0)
        except Exception as e:
            return 0.0
    
    def calculate_momentum_score(self, momentum_data: Dict, i: int) -> float:
        """Расчет скора моментума"""
        try:
            if not momentum_data:
                return 0.0
            
            rsi = momentum_data['rsi'].iloc[i]
            stoch_k = momentum_data['stoch_k'].iloc[i]
            macd_hist = momentum_data['macd_hist'].iloc[i]
            
            score = 0.0
            
            # RSI
            if rsi > 70:
                score -= 0.3
            elif rsi < 30:
                score += 0.3
            
            # Stochastic
            if stoch_k > 80:
                score -= 0.2
            elif stoch_k < 20:
                score += 0.2
            
            # MACD
            if macd_hist > 0:
                score += 0.2
            else:
                score -= 0.2
            
            return np.clip(score, -1.0, 1.0)
        except Exception as e:
            return 0.0
    
    def calculate_price_action_score(self, price_action_data: Dict, trend_score: float, i: int) -> float:
        """Расчет скора price action"""
        try:
            if not price_action_data:
                return 0.0
            
            candle_range = price_action_data['candle_range'].iloc[i]
            avg_candle_range = price_action_data['avg_candle_range'].iloc[i]
            
            score = 0.0
            
            # Размер свечи относительно среднего
            if candle_range > avg_candle_range * 1.5:
                score += trend_score * 0.3  # Усиливаем трендовый сигнал
            
            return np.clip(score, -1.0, 1.0)
        except Exception as e:
            return 0.0
    
    def apply_hysteresis(self, current_zone: str, last_zone: str, zone_counter: int) -> str:
        """Применение гистерезиса для стабильности"""
        if current_zone != last_zone and zone_counter < 3:
            return last_zone
        return current_zone
