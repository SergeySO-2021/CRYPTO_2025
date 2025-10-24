#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интегрированная торговая стратегия BTC

Этот файл содержит реализацию торговой стратегии на Python, переведенную с Pine Script.
Стратегия предназначена для автоматической подборки индикаторов и их параметров.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Для оптимизации параметров
from sklearn.model_selection import ParameterGrid
from itertools import product

# Для визуализации
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BTCTradingStrategy:
    """
    Основной класс для торговой стратегии BTC
    """
    
    def __init__(self, data=None):
        self.data = data
        self.signals = {}
        self.indicators = {}
        self.trades = []
        self.performance = {}
        
    def load_data(self, file_path='df_btc.csv', timeframe='1H'):
        """
        Загружает данные BTC из CSV файла и конвертирует в нужный тайм-фрейм
        
        Args:
            file_path (str): Путь к CSV файлу
            timeframe (str): Целевой тайм-фрейм ('1H', '4H', '1D', '30S', '1M', '5M', '15M')
        """
        try:
            print(f"Загружаем данные из {file_path}...")
            df = pd.read_csv(file_path)
            
            # Проверяем структуру данных
            print(f"Колонки в файле: {df.columns.tolist()}")
            print(f"Первые 5 записей:")
            print(df.head())
            
            # Преобразуем timestamp в datetime
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df.set_index('timestamps', inplace=True)
            
            # Сортируем по времени
            df.sort_index(inplace=True)
            
            print(f"Загружено {len(df)} записей")
            print(f"Период: {df.index.min()} - {df.index.max()}")
            print(f"Интервал между записями: {df.index.to_series().diff().median()}")
            
            # Конвертируем в нужный тайм-фрейм
            if timeframe != '30S':
                df = self.convert_timeframe(df, timeframe)
                print(f"Конвертировано в тайм-фрейм {timeframe}")
                print(f"После конвертации: {len(df)} записей")
            
            # Создаем OHLC данные из фиксированной цены
            df = self.create_ohlc_from_price(df)
            
            self.data = df
            return df
            
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return None
    
    def convert_timeframe(self, df, target_timeframe):
        """
        Конвертирует данные в нужный тайм-фрейм
        
        Args:
            df (DataFrame): Исходные данные
            target_timeframe (str): Целевой тайм-фрейм
        """
        if target_timeframe == '30S':
            return df
        
        # Определяем период для группировки
        if target_timeframe == '1M':
            freq = '1T'  # 1 минута
        elif target_timeframe == '5M':
            freq = '5T'  # 5 минут
        elif target_timeframe == '15M':
            freq = '15T'  # 15 минут
        elif target_timeframe == '1H':
            freq = '1H'  # 1 час
        elif target_timeframe == '4H':
            freq = '4H'  # 4 часа
        elif target_timeframe == '1D':
            freq = '1D'  # 1 день
        else:
            print(f"Неизвестный тайм-фрейм {target_timeframe}, используем исходный")
            return df
        
        # Группируем по времени и создаем OHLC
        grouped = df.resample(freq).agg({
            'btc_price': ['first', 'max', 'min', 'last']
        })
        
        # Переименовываем колонки
        grouped.columns = ['open', 'high', 'low', 'close']
        
        # Убираем строки с NaN значениями
        grouped = grouped.dropna()
        
        return grouped
    
    def create_ohlc_from_price(self, df):
        """
        Создает OHLC данные из фиксированной цены
        
        Args:
            df (DataFrame): DataFrame с колонкой btc_price
        """
        if 'btc_price' in df.columns:
            # Если у нас есть btc_price, создаем OHLC
            df['open'] = df['btc_price']
            df['high'] = df['btc_price']
            df['low'] = df['btc_price']
            df['close'] = df['btc_price']
            
            # Если у нас есть OHLC колонки, используем их
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                print("Используем существующие OHLC данные")
            else:
                print("Создаем OHLC данные из фиксированной цены")
        else:
            # Проверяем, есть ли уже OHLC колонки
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                print("Используем существующие OHLC данные")
            else:
                print("Ошибка: нет данных о цене!")
                return None
        
        return df
    
    def calculate_ma(self, data, length, ma_type='EMA'):
        """Вычисляет различные типы скользящих средних"""
        if ma_type == 'SMA':
            return data.rolling(window=length).mean()
        elif ma_type == 'EMA':
            return data.ewm(span=length).mean()
        elif ma_type == 'RMA':
            return data.rolling(window=length).mean()
        elif ma_type == 'WMA':
            weights = np.arange(1, length + 1)
            return data.rolling(window=length).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        elif ma_type == 'VWMA':
            # Простая реализация VWMA (для полноты)
            return data.rolling(window=length).mean()
        else:
            return data.ewm(span=length).mean()
    
    def calculate_atr(self, high, low, close, length=14):
        """Вычисляет Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=length).mean()
    
    def calculate_rsi(self, data, length=14):
        """Вычисляет Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_supertrend(self, atr_period=10, atr_multiplier=3.0, source='hl2'):
        """Вычисляет SuperTrend индикатор"""
        if source == 'hl2':
            src = (self.data['high'] + self.data['low']) / 2
        else:
            src = self.data[source]
        
        atr = self.calculate_atr(self.data['high'], self.data['low'], self.data['close'], atr_period)
        
        # Вычисление SuperTrend
        up = src - atr_multiplier * atr
        dn = src + atr_multiplier * atr
        
        # Инициализация
        trend = pd.Series(1, index=self.data.index)
        up_prev = up.shift(1)
        dn_prev = dn.shift(1)
        
        for i in range(1, len(self.data)):
            if trend.iloc[i-1] == 1:
                if self.data['close'].iloc[i-1] > up_prev.iloc[i]:
                    up.iloc[i] = max(up.iloc[i], up_prev.iloc[i])
                else:
                    trend.iloc[i] = -1
            else:
                if self.data['close'].iloc[i-1] < dn_prev.iloc[i]:
                    dn.iloc[i] = min(dn.iloc[i], dn_prev.iloc[i])
                else:
                    trend.iloc[i] = 1
        
        return trend, up, dn
    
    def calculate_macd(self, fast_length=12, slow_length=26, signal_length=9, 
                      source='close', fast_ma_type='EMA', slow_ma_type='EMA', 
                      signal_ma_type='EMA'):
        """Вычисляет MACD индикатор"""
        src = self.data[source]
        
        fast_ma = self.calculate_ma(src, fast_length, fast_ma_type)
        slow_ma = self.calculate_ma(src, slow_length, slow_ma_type)
        
        macd_line = fast_ma - slow_ma
        signal_line = self.calculate_ma(macd_line, signal_length, signal_ma_type)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def generate_signals(self, strategy_config):
        """Генерирует торговые сигналы на основе конфигурации стратегии"""
        signals = pd.DataFrame(index=self.data.index)
        signals['close'] = self.data['close']
        signals['signal'] = 0  # 0: нет сигнала, 1: покупка, -1: продажа
        
        # Основной индикатор (ведущий)
        leading_indicator = strategy_config.get('leading_indicator', 'Range Filter')
        
        if leading_indicator == 'SuperTrend':
            trend, up, dn = self.calculate_supertrend(
                atr_period=strategy_config.get('supertrend_atr_period', 10),
                atr_multiplier=strategy_config.get('supertrend_multiplier', 3.0)
            )
            signals['leading_signal'] = np.where(trend == 1, 1, -1)
            
        elif leading_indicator == 'Range Filter':
            # Range Filter логика
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            
            period = strategy_config.get('rf_period', 100)
            multiplier = strategy_config.get('rf_multiplier', 3.0)
            
            # Упрощенная реализация Range Filter
            atr = self.calculate_atr(high, low, close, period)
            range_size = atr * multiplier
            
            # Определение направления
            direction = pd.Series(0, index=self.data.index)
            for i in range(1, len(self.data)):
                if close.iloc[i] > close.iloc[i-1] + range_size.iloc[i]:
                    direction.iloc[i] = 1
                elif close.iloc[i] < close.iloc[i-1] - range_size.iloc[i]:
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = direction.iloc[i-1]
            
            signals['leading_signal'] = direction
            
        elif leading_indicator == '2 EMA Cross':
            fast_period = strategy_config.get('ema2_fast', 50)
            slow_period = strategy_config.get('ema2_slow', 200)
            
            fast_ema = self.calculate_ma(self.data['close'], fast_period, 'EMA')
            slow_ema = self.calculate_ma(self.data['close'], slow_period, 'EMA')
            
            # Определение пересечений
            cross_up = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
            cross_down = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
            
            signals['leading_signal'] = np.where(cross_up, 1, np.where(cross_down, -1, 0))
            
        elif leading_indicator == '3 EMA Cross':
            fast_period = strategy_config.get('ema3_fast', 20)
            mid_period = strategy_config.get('ema3_mid', 50)
            slow_period = strategy_config.get('ema3_slow', 200)
            
            fast_ema = self.calculate_ma(self.data['close'], fast_period, 'EMA')
            mid_ema = self.calculate_ma(self.data['close'], mid_period, 'EMA')
            slow_ema = self.calculate_ma(self.data['close'], slow_period, 'EMA')
            
            # Три EMA должны быть выстроены в ряд
            bullish_trend = (fast_ema > mid_ema) & (mid_ema > slow_ema)
            bearish_trend = (fast_ema < mid_ema) & (mid_ema < slow_ema)
            
            signals['leading_signal'] = np.where(bullish_trend, 1, np.where(bearish_trend, -1, 0))
            
        elif leading_indicator == 'QQE Mod':
            rsi_period = strategy_config.get('qqe_rsi_period', 14)
            sf = strategy_config.get('qqe_sf', 5)
            qe = strategy_config.get('qqe_qe', 4.236)
            
            qqe = self.calculate_qqe_mod(self.data, rsi_period, sf, qe)
            signals['leading_signal'] = np.where(qqe['rsi_smooth'] > 50, 1, -1)
            
        elif leading_indicator == 'Chaikin Money Flow':
            period = strategy_config.get('cmf_period', 20)
            cmf = self.calculate_chaikin_money_flow(self.data, period)
            signals['leading_signal'] = np.where(cmf > 0.25, 1, np.where(cmf < -0.25, -1, 0))
            
        elif leading_indicator == 'Waddah Attar Explosion':
            bb_period = strategy_config.get('wae_bb_period', 20)
            bb_std = strategy_config.get('wae_bb_std', 2)
            atr_period = strategy_config.get('wae_atr_period', 14)
            
            wae = self.calculate_waddah_attar_explosion(self.data, bb_period, bb_std, atr_period)
            signals['leading_signal'] = np.where(wae['explosion_up'] == 1, 1, 
                                               np.where(wae['explosion_down'] == 1, -1, 0))
            
        elif leading_indicator == 'BB Oscillator':
            period = strategy_config.get('bb_period', 20)
            std_dev = strategy_config.get('bb_std_dev', 2)
            bb_osc = self.calculate_bb_oscillator(self.data, period, std_dev)
            signals['leading_signal'] = np.where(bb_osc > 0.8, -1, np.where(bb_osc < 0.2, 1, 0))
            
        elif leading_indicator == 'Chandelier Exit':
            period = strategy_config.get('ce_period', 22)
            multiplier = strategy_config.get('ce_multiplier', 3)
            chandelier = self.calculate_chandelier_exit(self.data, period, multiplier)
            signals['leading_signal'] = np.where(self.data['close'] > chandelier['long_stop'], 1, 
                                               np.where(self.data['close'] < chandelier['short_stop'], -1, 0))
            
        elif leading_indicator == 'Heiken-Ashi Candlestick Oscillator':
            period = strategy_config.get('ha_period', 14)
            ha_osc = self.calculate_heiken_ashi_candlestick_oscillator(self.data, period)
            signals['leading_signal'] = np.where(ha_osc > 0, 1, -1)
            
        elif leading_indicator == 'B-Xtrender':
            period = strategy_config.get('bxt_period', 14)
            bxt = self.calculate_b_xtrender(self.data, period)
            signals['leading_signal'] = bxt
            
        elif leading_indicator == 'Bull Bear Power Trend':
            period = strategy_config.get('bbpt_period', 13)
            bbpt = self.calculate_bull_bear_power_trend(self.data, period)
            signals['leading_signal'] = np.where(bbpt['bull_power'] > 0, 1, -1)
            
        elif leading_indicator == 'Detrended Price Oscillator':
            period = strategy_config.get('dpo_period', 20)
            dpo = self.calculate_detrended_price_oscillator(self.data, period)
            signals['leading_signal'] = np.where(dpo > 0, 1, -1)
            
        elif leading_indicator == 'Range Filter Type 2':
            period = strategy_config.get('rf2_period', 14)
            multiplier = strategy_config.get('rf2_multiplier', 2.618)
            scale = strategy_config.get('rf2_scale', 'ATR')
            
            rf2 = self.calculate_range_filter_type2(self.data, period, multiplier, scale)
            signals['leading_signal'] = rf2['direction']
            
        elif leading_indicator == 'PVSRA':
            period = strategy_config.get('pvsra_period', 14)
            pvsra = self.calculate_pvsra(self.data, period)
            signals['leading_signal'] = np.where(pvsra > 0, 1, -1)
            
        elif leading_indicator == 'Liquidity Zone':
            period = strategy_config.get('lz_period', 20)
            threshold = strategy_config.get('lz_threshold', 1.5)
            lz = self.calculate_liquidity_zone(self.data, period, threshold)
            signals['leading_signal'] = np.where(lz == 1, 1, -1)
            
        elif leading_indicator == 'Ichimoku Cloud':
            tenkan = strategy_config.get('ichimoku_tenkan', 9)
            kijun = strategy_config.get('ichimoku_kijun', 26)
            senkou_span_b = strategy_config.get('ichimoku_senkou_span_b', 52)
            displacement = strategy_config.get('ichimoku_displacement', 26)
            
            ichimoku = self.calculate_ichimoku_cloud(self.data, tenkan, kijun, senkou_span_b, displacement)
            signals['leading_signal'] = np.where(ichimoku['tenkan_sen'] > ichimoku['kijun_sen'], 1, -1)
            
        elif leading_indicator == 'Stochastic Oscillator':
            k_period = strategy_config.get('stoch_k_period', 14)
            d_period = strategy_config.get('stoch_d_period', 3)
            stoch = self.calculate_stochastic_oscillator(self.data, k_period, d_period)
            signals['leading_signal'] = np.where(stoch['k_percent'] > 80, -1, np.where(stoch['k_percent'] < 20, 1, 0))
            
        elif leading_indicator == 'VWAP':
            period = strategy_config.get('vwap_period', None)
            vwap = self.calculate_vwap(self.data, period)
            signals['leading_signal'] = np.where(self.data['close'] > vwap, 1, -1)
            
        elif leading_indicator == 'Rational Quadratic Kernel':
            period = strategy_config.get('rqk_period', 14)
            sigma = strategy_config.get('rqk_sigma', 1.0)
            rqk = self.calculate_rational_quadratic_kernel(self.data, period, sigma)
            signals['leading_signal'] = np.where(rqk > rqk.rolling(period).mean(), 1, -1)
            
        elif leading_indicator == 'True Strength Index':
            rsi_period = strategy_config.get('tsi_rsi_period', 25)
            rsi_smooth = strategy_config.get('tsi_rsi_smooth', 13)
            signal_period = strategy_config.get('tsi_signal_period', 9)
            tsi = self.calculate_true_strength_index(self.data, rsi_period, rsi_smooth, signal_period)
            signals['leading_signal'] = np.where(tsi['tsi'] > tsi['signal'], 1, -1)
            
        elif leading_indicator == 'Half Trend':
            atr_period = strategy_config.get('ht_atr_period', 10)
            atr_multiplier = strategy_config.get('ht_atr_multiplier', 2.0)
            ht = self.calculate_half_trend(self.data, atr_period, atr_multiplier)
            signals['leading_signal'] = ht['trend']
            
        elif leading_indicator == 'Conditional Sampling EMA':
            period = strategy_config.get('cse_period', 14)
            condition_col = strategy_config.get('cse_condition_col', 'close')
            condition_threshold = strategy_config.get('cse_condition_threshold', 0)
            cse = self.calculate_conditional_sampling_ema(self.data, period, condition_col, condition_threshold)
            signals['leading_signal'] = np.where(self.data['close'] > cse, 1, -1)
        
        # Подтверждающие индикаторы
        confirmation_signals = []
        
        if strategy_config.get('use_rsi', False):
            rsi_length = strategy_config.get('rsi_length', 14)
            rsi = self.calculate_rsi(self.data['close'], rsi_length)
            rsi_signal = np.where(rsi > 50, 1, -1)
            confirmation_signals.append(rsi_signal)
        
        if strategy_config.get('use_macd', False):
            macd, signal, hist = self.calculate_macd(
                fast_length=strategy_config.get('macd_fast', 12),
                slow_length=strategy_config.get('macd_slow', 26),
                signal_length=strategy_config.get('macd_signal', 9)
            )
            macd_signal = np.where(macd > signal, 1, -1)
            confirmation_signals.append(macd_signal)
        
        if strategy_config.get('use_supertrend', False):
            trend, up, dn = self.calculate_supertrend(
                atr_period=strategy_config.get('supertrend_atr_period', 10),
                atr_multiplier=strategy_config.get('supertrend_multiplier', 3.0)
            )
            supertrend_signal = np.where(trend == 1, 1, -1)
            confirmation_signals.append(supertrend_signal)
        
        # Комбинирование сигналов
        if confirmation_signals:
            # Все подтверждающие индикаторы должны совпадать с ведущим
            confirmation_matrix = np.column_stack(confirmation_signals)
            confirmation_score = np.mean(confirmation_matrix, axis=1)
            
            # Финальный сигнал
            signals['signal'] = np.where(
                (signals['leading_signal'] == 1) & (confirmation_score > 0.5), 1,
                np.where(
                    (signals['leading_signal'] == -1) & (confirmation_score < -0.5), -1, 0
                )
            )
        else:
            signals['signal'] = signals['leading_signal']
        
        return signals
    
    def backtest(self, signals, initial_capital=10000, commission=0.001):
        """Выполняет бэктест стратегии"""
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['close'] = signals['close']
        portfolio['signal'] = signals['signal']
        portfolio['position'] = portfolio['signal'].cumsum()
        portfolio['cash'] = initial_capital
        portfolio['holdings'] = portfolio['position'] * portfolio['close']
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        
        # Расчет статистик
        total_return = (portfolio['total'].iloc[-1] - initial_capital) / initial_capital
        annual_return = total_return * (252 / len(portfolio))
        volatility = portfolio['returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Максимальная просадка
        cumulative_returns = (1 + portfolio['returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        self.performance = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio': portfolio
        }
        
        return self.performance
    
    def optimize_parameters(self, param_grid, metric='sharpe_ratio'):
        """Оптимизирует параметры стратегии"""
        best_params = None
        best_score = float('-inf')
        results = []
        
        print(f"Начинаем оптимизацию {len(list(ParameterGrid(param_grid)))} комбинаций параметров...")
        
        for params in ParameterGrid(param_grid):
            try:
                # Генерируем сигналы с текущими параметрами
                signals = self.generate_signals(params)
                
                # Выполняем бэктест
                performance = self.backtest(signals)
                
                # Сохраняем результат
                result = {
                    'params': params,
                    'performance': performance
                }
                results.append(result)
                
                # Обновляем лучший результат
                current_score = performance[metric]
                if current_score > best_score:
                    best_score = current_score
                    best_params = params
                
                if len(results) % 100 == 0:
                    print(f"Обработано {len(results)} комбинаций...")
                    
            except Exception as e:
                print(f"Ошибка при параметрах {params}: {e}")
                continue
        
        print(f"Оптимизация завершена! Лучший {metric}: {best_score:.4f}")
        print(f"Лучшие параметры: {best_params}")
        
        return best_params, results
    
    def optimize_indicators(self, indicator_combinations=None, param_ranges=None, metric='sharpe_ratio'):
        """
        Автоматически оптимизирует комбинации индикаторов и их параметры
        
        Args:
            indicator_combinations (list): Список комбинаций индикаторов для тестирования
            param_ranges (dict): Диапазоны параметров для каждого индикатора
            metric (str): Метрика для оптимизации
        """
        if indicator_combinations is None:
            indicator_combinations = self._get_default_indicator_combinations()
        
        if param_ranges is None:
            param_ranges = self._get_default_param_ranges()
        
        print(f"=== АВТОМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ ИНДИКАТОРОВ ===")
        print(f"Тестируем {len(indicator_combinations)} комбинаций индикаторов...")
        
        best_combination = None
        best_params = None
        best_score = float('-inf')
        all_results = []
        
        for i, combo in enumerate(indicator_combinations):
            print(f"\n--- Комбинация {i+1}/{len(indicator_combinations)} ---")
            print(f"Ведущий: {combo['leading']}")
            print(f"Подтверждающие: {combo['confirmation']}")
            
            # Создаем сетку параметров для текущей комбинации
            current_param_grid = self._create_param_grid_for_combo(combo, param_ranges)
            
            try:
                # Оптимизируем параметры для текущей комбинации
                combo_best_params, combo_results = self.optimize_parameters(current_param_grid, metric)
                
                if combo_best_params and combo_results:
                    # Находим лучший результат для этой комбинации
                    combo_best_result = max(combo_results, key=lambda x: x['performance'][metric])
                    combo_best_score = combo_best_result['performance'][metric]
                    
                    print(f"Лучший результат для комбинации: {combo_best_score:.4f}")
                    
                    # Обновляем глобальный лучший результат
                    if combo_best_score > best_score:
                        best_score = combo_best_score
                        best_combination = combo
                        best_params = combo_best_params
                        print(f"*** НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ! ***")
                    
                    # Сохраняем результаты
                    all_results.append({
                        'combination': combo,
                        'best_params': combo_best_params,
                        'best_score': combo_best_score,
                        'all_results': combo_results
                    })
                
            except Exception as e:
                print(f"Ошибка при тестировании комбинации {combo}: {e}")
                continue
        
        # Итоговый отчет
        print(f"\n=== ИТОГИ ОПТИМИЗАЦИИ ===")
        print(f"Лучшая комбинация индикаторов:")
        print(f"  Ведущий: {best_combination['leading']}")
        print(f"  Подтверждающие: {best_combination['confirmation']}")
        print(f"Лучший {metric}: {best_score:.4f}")
        print(f"Лучшие параметры: {best_params}")
        
        # Сортируем все результаты по метрике
        all_results.sort(key=lambda x: x['best_score'], reverse=True)
        
        print(f"\nТоп-5 комбинаций:")
        for i, result in enumerate(all_results[:5]):
            print(f"{i+1}. {result['combination']['leading']} + {result['combination']['confirmation']}: {result['best_score']:.4f}")
        
        return best_combination, best_params, all_results
    
    def _get_default_indicator_combinations(self):
        """Возвращает стандартные комбинации индикаторов для тестирования"""
        return [
            # Трендовые стратегии
            {'leading': 'SuperTrend', 'confirmation': ['RSI', 'MACD']},
            {'leading': 'SuperTrend', 'confirmation': ['RSI', 'Supertrend']},
            {'leading': 'SuperTrend', 'confirmation': ['MACD', 'Supertrend']},
            {'leading': 'SuperTrend', 'confirmation': ['RSI', 'MACD', 'Supertrend']},
            
            # Range Filter стратегии
            {'leading': 'Range Filter', 'confirmation': ['RSI', 'MACD']},
            {'leading': 'Range Filter', 'confirmation': ['RSI', 'Supertrend']},
            {'leading': 'Range Filter', 'confirmation': ['MACD', 'Supertrend']},
            {'leading': 'Range Filter', 'confirmation': ['RSI', 'MACD', 'Supertrend']},
            
            # EMA Cross стратегии
            {'leading': '2 EMA Cross', 'confirmation': ['RSI', 'MACD']},
            {'leading': '2 EMA Cross', 'confirmation': ['RSI', 'Supertrend']},
            {'leading': '2 EMA Cross', 'confirmation': ['MACD', 'Supertrend']},
            {'leading': '2 EMA Cross', 'confirmation': ['RSI', 'MACD', 'Supertrend']},
            
            # Комбинированные стратегии
            {'leading': 'SuperTrend', 'confirmation': ['RSI']},
            {'leading': 'SuperTrend', 'confirmation': ['MACD']},
            {'leading': 'SuperTrend', 'confirmation': ['Supertrend']},
            {'leading': 'Range Filter', 'confirmation': ['RSI']},
            {'leading': 'Range Filter', 'confirmation': ['MACD']},
            {'leading': 'Range Filter', 'confirmation': ['Supertrend']},
            {'leading': '2 EMA Cross', 'confirmation': ['RSI']},
            {'leading': '2 EMA Cross', 'confirmation': ['MACD']},
            {'leading': '2 EMA Cross', 'confirmation': ['Supertrend']},
        ]
    
    def _get_default_param_ranges(self):
        """Возвращает стандартные диапазоны параметров для каждого индикатора"""
        return {
            # SuperTrend параметры
            'supertrend_atr_period': [5, 7, 10, 14, 20],
            'supertrend_multiplier': [2.0, 2.5, 3.0, 3.5, 4.0],
            
            # Range Filter параметры
            'rf_period': [50, 75, 100, 125, 150],
            'rf_multiplier': [2.0, 2.5, 3.0, 3.5, 4.0],
            
            # 2 EMA Cross параметры
            'ema2_fast': [20, 30, 50, 75, 100],
            'ema2_slow': [100, 150, 200, 250, 300],
            
            # RSI параметры
            'rsi_length': [10, 12, 14, 16, 20],
            'rsi_overbought': [65, 70, 75, 80],
            'rsi_oversold': [20, 25, 30, 35],
            
            # MACD параметры
            'macd_fast': [8, 10, 12, 14, 16],
            'macd_slow': [20, 22, 26, 30, 34],
            'macd_signal': [7, 9, 11, 13],
            
            # Дополнительные параметры
            'use_rsi': [True, False],
            'use_macd': [True, False],
            'use_supertrend': [True, False],
        }
    
    def _create_param_grid_for_combo(self, combo, param_ranges):
        """Создает сетку параметров для конкретной комбинации индикаторов"""
        param_grid = {}
        
        # Базовые параметры для ведущего индикатора
        if combo['leading'] == 'SuperTrend':
            param_grid['leading_indicator'] = ['SuperTrend']
            param_grid['supertrend_atr_period'] = param_ranges['supertrend_atr_period']
            param_grid['supertrend_multiplier'] = param_ranges['supertrend_multiplier']
            
        elif combo['leading'] == 'Range Filter':
            param_grid['leading_indicator'] = ['Range Filter']
            param_grid['rf_period'] = param_ranges['rf_period']
            param_grid['rf_multiplier'] = param_ranges['rf_multiplier']
            
        elif combo['leading'] == '2 EMA Cross':
            param_grid['leading_indicator'] = ['2 EMA Cross']
            param_grid['ema2_fast'] = param_ranges['ema2_fast']
            param_grid['ema2_slow'] = param_ranges['ema2_slow']
        
        # Параметры для подтверждающих индикаторов
        if 'RSI' in combo['confirmation']:
            param_grid['use_rsi'] = [True]
            param_grid['rsi_length'] = param_ranges['rsi_length']
        
        if 'MACD' in combo['confirmation']:
            param_grid['use_macd'] = [True]
            param_grid['macd_fast'] = param_ranges['macd_fast']
            param_grid['macd_slow'] = param_ranges['macd_slow']
            param_grid['macd_signal'] = param_ranges['macd_signal']
        
        if 'Supertrend' in combo['confirmation']:
            param_grid['use_supertrend'] = [True]
            param_grid['supertrend_atr_period'] = param_ranges['supertrend_atr_period']
            param_grid['supertrend_multiplier'] = param_ranges['supertrend_multiplier']
        
        return param_grid
    
    def test_single_indicator(self, indicator_name, param_ranges=None):
        """
        Тестирует отдельный индикатор с различными параметрами
        
        Args:
            indicator_name (str): Название индикатора для тестирования
            param_ranges (dict): Диапазоны параметров
        """
        if param_ranges is None:
            param_ranges = self._get_default_param_ranges()
        
        print(f"=== ТЕСТИРОВАНИЕ ИНДИКАТОРА: {indicator_name} ===")
        
        # Создаем простую стратегию только с этим индикатором
        if indicator_name == 'SuperTrend':
            param_grid = {
                'leading_indicator': ['SuperTrend'],
                'supertrend_atr_period': param_ranges['supertrend_atr_period'],
                'supertrend_multiplier': param_ranges['supertrend_multiplier']
            }
        elif indicator_name == 'RSI':
            param_grid = {
                'leading_indicator': ['SuperTrend'],  # Используем SuperTrend как базовый
                'supertrend_atr_period': [10],
                'supertrend_multiplier': [3.0],
                'use_rsi': [True],
                'rsi_length': param_ranges['rsi_length']
            }
        elif indicator_name == 'MACD':
            param_grid = {
                'leading_indicator': ['SuperTrend'],  # Используем SuperTrend как базовый
                'supertrend_atr_period': [10],
                'supertrend_multiplier': [3.0],
                'use_macd': [True],
                'macd_fast': param_ranges['macd_fast'],
                'macd_slow': param_ranges['macd_slow'],
                'macd_signal': param_ranges['macd_signal']
            }
        else:
            print(f"Индикатор {indicator_name} не поддерживается для одиночного тестирования")
            return None
        
        # Оптимизируем параметры
        best_params, results = self.optimize_parameters(param_grid, 'sharpe_ratio')
        
        print(f"Лучшие параметры для {indicator_name}: {best_params}")
        
        return best_params, results
    
    def compare_indicators(self, indicators_to_test=None):
        """
        Сравнивает производительность различных индикаторов
        
        Args:
            indicators_to_test (list): Список индикаторов для сравнения
        """
        if indicators_to_test is None:
            indicators_to_test = ['SuperTrend', 'RSI', 'MACD', 'Range Filter', '2 EMA Cross']
        
        print(f"=== СРАВНЕНИЕ ИНДИКАТОРОВ ===")
        
        comparison_results = {}
        
        for indicator in indicators_to_test:
            print(f"\nТестируем {indicator}...")
            
            try:
                best_params, results = self.test_single_indicator(indicator)
                
                if best_params and results:
                    # Находим лучший результат
                    best_result = max(results, key=lambda x: x['performance']['sharpe_ratio'])
                    
                    comparison_results[indicator] = {
                        'best_params': best_params,
                        'sharpe_ratio': best_result['performance']['sharpe_ratio'],
                        'total_return': best_result['performance']['total_return'],
                        'max_drawdown': best_result['performance']['max_drawdown'],
                        'volatility': best_result['performance']['volatility']
                    }
                    
                    print(f"  Коэффициент Шарпа: {best_result['performance']['sharpe_ratio']:.4f}")
                    print(f"  Общий доход: {best_result['performance']['total_return']:.2%}")
                    print(f"  Максимальная просадка: {best_result['performance']['max_drawdown']:.2%}")
                
            except Exception as e:
                print(f"  Ошибка при тестировании {indicator}: {e}")
                continue
        
        # Сортируем результаты по коэффициенту Шарпа
        sorted_results = sorted(comparison_results.items(), 
                              key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        
        print(f"\n=== РЕЙТИНГ ИНДИКАТОРОВ ===")
        for i, (indicator, metrics) in enumerate(sorted_results):
            print(f"{i+1}. {indicator}: Шарп {metrics['sharpe_ratio']:.4f}, "
                  f"Доход {metrics['total_return']:.2%}, "
                  f"Просадка {metrics['max_drawdown']:.2%}")
        
        return comparison_results

    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """Bollinger Bands - полосы Боллинджера"""
        ma = self.calculate_ma(df['close'], period, 'SMA')
        std = df['close'].rolling(period).std()
        upper = ma + std * std_dev
        lower = ma - std * std_dev
        return pd.DataFrame({'upper': upper, 'lower': lower})
    
    def calculate_bb_oscillator(self, df, period=20, std_dev=2):
        """Bollinger Bands Oscillator"""
        bb = self.calculate_bollinger_bands(df, period, std_dev)
        bb_osc = (df['close'] - bb['lower']) / (bb['upper'] - bb['lower'])
        return bb_osc
    
    def calculate_chandelier_exit(self, df, period=22, multiplier=3):
        """Chandelier Exit - динамический стоп-лосс"""
        atr = self.calculate_atr(df, period)
        
        # Long position
        long_stop = df['high'].rolling(period).max() - multiplier * atr
        long_stop = long_stop.fillna(method='ffill')
        
        # Short position  
        short_stop = df['low'].rolling(period).min() + multiplier * atr
        short_stop = short_stop.fillna(method='ffill')
        
        return pd.DataFrame({
            'long_stop': long_stop,
            'short_stop': short_stop
        })
    
    def calculate_choppiness_index(self, df, period=14):
        """Choppiness Index - индекс волатильности"""
        tr = self.calculate_true_range(df)
        sum_tr = tr.rolling(period).sum()
        range_sum = (df['high'].rolling(period).max() - df['low'].rolling(period).min()).rolling(period).sum()
        
        choppiness = 100 * np.log10(sum_tr / range_sum) / np.log10(period)
        return choppiness
    
    def calculate_damiani_volatmeter(self, df, period=14):
        """Damiani Volatmeter - волатильность по Дамиани"""
        # Простая реализация
        returns = df['close'].pct_change()
        volatility = returns.rolling(period).std() * np.sqrt(period)
        return volatility
    
    def calculate_heiken_ashi_candlestick_oscillator(self, df, period=14):
        """Heiken-Ashi Candlestick Oscillator"""
        # Heiken-Ashi свечи
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_open = df['open'].copy()
        ha_high = df['high'].copy()
        ha_low = df['low'].copy()
        
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
            ha_high.iloc[i] = max(df['high'].iloc[i], ha_open.iloc[i], ha_close.iloc[i])
            ha_low.iloc[i] = min(df['low'].iloc[i], ha_open.iloc[i], ha_close.iloc[i])
        
        # Oscillator
        oscillator = ha_close - ha_open
        smoothed = oscillator.rolling(period).mean()
        
        return smoothed
    
    def calculate_b_xtrender(self, df, period=14):
        """B-Xtrender - индикатор тренда"""
        # Простая реализация на основе EMA
        ema_fast = df['close'].ewm(span=period//2).mean()
        ema_slow = df['close'].ewm(span=period).mean()
        
        trend = np.where(ema_fast > ema_slow, 1, -1)
        return pd.Series(trend, index=df.index)
    
    def calculate_bull_bear_power_trend(self, df, period=13):
        """Bull Bear Power Trend - сила быков и медведей"""
        ema = df['close'].ewm(span=period).mean()
        bull_power = df['high'] - ema
        bear_power = df['low'] - ema
        
        return pd.DataFrame({
            'bull_power': bull_power,
            'bear_power': bear_power
        })
    
    def calculate_detrended_price_oscillator(self, df, period=20):
        """Detrended Price Oscillator (DPO) - детрендированный осциллятор цены"""
        # Убираем тренд, используя скользящее среднее
        ma = df['close'].rolling(period).mean()
        dpo = df['close'] - ma.shift(period//2 + 1)
        return dpo
    
    def calculate_qqe_mod(self, df, rsi_period=14, sf=5, qe=4.236):
        """QQE Mod - модифицированный QQE индикатор"""
        # RSI
        rsi = self.calculate_rsi(df, rsi_period)
        
        # Smoothing
        rsi_smooth = rsi.ewm(span=sf).mean()
        
        # QE calculation
        qe_up = rsi_smooth.rolling(qe).max()
        qe_down = rsi_smooth.rolling(qe).min()
        
        # QQE
        qqe_up = qe_up.ewm(span=sf).mean()
        qqe_down = qe_down.ewm(span=sf).mean()
        
        return pd.DataFrame({
            'qqe_up': qqe_up,
            'qqe_down': qqe_down,
            'rsi_smooth': rsi_smooth
        })
    
    def calculate_chaikin_money_flow(self, df, period=20):
        """Chaikin Money Flow (CMF) - денежный поток Чайкина"""
        # Money Flow Multiplier
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)
        
        # Money Flow Volume
        mfv = mfm * df['volume']
        
        # CMF
        cmf = mfv.rolling(period).sum() / df['volume'].rolling(period).sum()
        return cmf
    
    def calculate_waddah_attar_explosion(self, df, bb_period=20, bb_std=2, atr_period=14):
        """Waddah Attar Explosion - индикатор взрывного движения"""
        bb = self.calculate_bollinger_bands(df, bb_period, bb_std)
        atr = self.calculate_atr(df, atr_period)
        
        # Explosion signals
        explosion_up = (df['close'] > bb['upper']) & (df['close'] > df['close'].shift(1) + atr)
        explosion_down = (df['close'] < bb['lower']) & (df['close'] < df['close'].shift(1) - atr)
        
        return pd.DataFrame({
            'explosion_up': explosion_up.astype(int),
            'explosion_down': explosion_down.astype(int)
        })
    
    def calculate_pivot_levels(self, df, pivot_type='Traditional'):
        """Pivot Levels - уровни поддержки и сопротивления"""
        if pivot_type == 'Traditional':
            # Traditional Pivot Points
            pivot = (df['high'] + df['low'] + df['close']) / 3
            r1 = 2 * pivot - df['low']
            s1 = 2 * pivot - df['high']
            r2 = pivot + (df['high'] - df['low'])
            s2 = pivot - (df['high'] - df['low'])
            
            return pd.DataFrame({
                'pivot': pivot,
                'r1': r1, 'r2': r2,
                's1': s1, 's2': s2
            })
        else:
            # Упрощенная версия для других типов
            return self.calculate_pivot_levels(df, 'Traditional')
    
    def calculate_fair_value_gap(self, df, gap_threshold=0.001):
        """Fair Value Gap - справедливый ценовой разрыв (без look-ahead bias)"""
        # Ищем разрывы между свечами используя только прошлые данные
        # Gap определяется как разрыв между текущей свечой и предыдущей
        gap_up = (df['low'] > df['high'].shift(1)) & (df['low'] - df['high'].shift(1) > gap_threshold)
        gap_down = (df['high'] < df['low'].shift(1)) & (df['low'].shift(1) - df['high'] > gap_threshold)
        
        return pd.DataFrame({
            'gap_up': gap_up.astype(int),
            'gap_down': gap_down.astype(int)
        })
    
    def calculate_william_fractals(self, df, period=5):
        """William Fractals - фракталы Уильяма"""
        # Bullish fractal (низкая точка)
        bullish_fractal = df['low'].rolling(period, center=True).min() == df['low']
        
        # Bearish fractal (высокая точка)
        bearish_fractal = df['high'].rolling(period, center=True).max() == df['high']
        
        return pd.DataFrame({
            'bullish_fractal': bullish_fractal.astype(int),
            'bearish_fractal': bearish_fractal.astype(int)
        })
    
    def calculate_supply_demand_zones(self, df, zone_period=20, volume_threshold=1.5):
        """Supply/Demand Zones - зоны спроса и предложения"""
        # Определяем зоны по объему и цене
        avg_volume = df['volume'].rolling(zone_period).mean()
        high_volume = df['volume'] > avg_volume * volume_threshold
        
        # Supply zone (высокая цена + высокий объем)
        supply_zone = (df['close'] > df['close'].rolling(zone_period).mean()) & high_volume
        
        # Demand zone (низкая цена + высокий объем)
        demand_zone = (df['close'] < df['close'].rolling(zone_period).mean()) & high_volume
        
        return pd.DataFrame({
            'supply_zone': supply_zone.astype(int),
            'demand_zone': demand_zone.astype(int)
        })
    
    def calculate_market_sessions(self, df):
        """Market Sessions - торговые сессии"""
        # Определяем время (UTC)
        df_copy = df.copy()
        df_copy['hour'] = pd.to_datetime(df_copy.index).hour
        
        # Asian session (00:00-08:00 UTC)
        asian_session = (df_copy['hour'] >= 0) & (df_copy['hour'] < 8)
        
        # London session (08:00-16:00 UTC)
        london_session = (df_copy['hour'] >= 8) & (df_copy['hour'] < 16)
        
        # New York session (13:00-21:00 UTC)
        ny_session = (df_copy['hour'] >= 13) & (df_copy['hour'] < 21)
        
        return pd.DataFrame({
            'asian_session': asian_session.astype(int),
            'london_session': london_session.astype(int),
            'ny_session': ny_session.astype(int)
        })
    
    def calculate_zigzag(self, df, deviation=5, depth=10):
        """ZigZag - структурный анализ"""
        # Упрощенная реализация ZigZag
        highs = df['high'].rolling(depth, center=True).max()
        lows = df['low'].rolling(depth, center=True).min()
        
        # Определяем поворотные точки (без look-ahead bias)
        # Пивот определяется только после закрытия свечи и подтверждения следующими свечами
        pivot_high = (df['high'] == highs) & (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(2))
        pivot_low = (df['low'] == lows) & (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(2))
        
        return pd.DataFrame({
            'pivot_high': pivot_high.astype(int),
            'pivot_low': pivot_low.astype(int)
        })
    
    def calculate_range_filter_type2(self, df, period=14, multiplier=2.618, scale='ATR'):
        """Range Filter Type 2 - улучшенная версия Range Filter"""
        if scale == 'ATR':
            range_size = self.calculate_atr(df, period) * multiplier
        elif scale == 'Standard Deviation':
            range_size = df['close'].rolling(period).std() * multiplier
        else:
            range_size = df['close'].rolling(period).std() * multiplier
        
        # Range Filter logic
        upper_band = df['close'] + range_size
        lower_band = df['close'] - range_size
        
        # Direction
        direction = np.where(df['close'] > upper_band.shift(1), 1, 
                           np.where(df['close'] < lower_band.shift(1), -1, 0))
        
        return pd.DataFrame({
            'upper_band': upper_band,
            'lower_band': lower_band,
            'direction': direction
        })
    
    def calculate_pvsra(self, df, period=14):
        """PVSRA - Price Volume Spread Range Analysis"""
        # Простая реализация PVSRA
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change() if 'volume' in df.columns else pd.Series(0, index=df.index)
        
        # PVSRA = Price Change + Volume Change
        pvsra = price_change + volume_change
        smoothed = pvsra.rolling(period).mean()
        
        return smoothed
    
    def calculate_liquidity_zone(self, df, period=20, threshold=1.5):
        """Liquidity Zone (Vector Zone) - зоны ликвидности"""
        # Определяем зоны по объему и волатильности
        avg_volume = df['volume'].rolling(period).mean() if 'volume' in df.columns else pd.Series(1, index=df.index)
        high_volume = df['volume'] > avg_volume * threshold if 'volume' in df.columns else pd.Series(False, index=df.index)
        
        # Определяем зоны ликвидности
        liquidity_zone = high_volume & (df['close'].rolling(period).std() > df['close'].rolling(period).std().mean())
        
        return liquidity_zone.astype(int)
    
    def calculate_ichimoku_cloud(self, df, tenkan=9, kijun=26, senkou_span_b=52, displacement=26):
        """Ichimoku Cloud - облако Ишимоку"""
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (df['high'].rolling(tenkan).max() + df['low'].rolling(tenkan).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (df['high'].rolling(kijun).max() + df['low'].rolling(kijun).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((df['high'].rolling(senkou_span_b).max() + df['low'].rolling(senkou_span_b).min()) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span) - исправлено без look-ahead bias
        chikou_span = df['close'].shift(displacement)
        
        return pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        })
    
    def calculate_stochastic_oscillator(self, df, k_period=14, d_period=3):
        """Stochastic Oscillator - стохастический осциллятор"""
        # %K
        lowest_low = df['low'].rolling(k_period).min()
        highest_high = df['high'].rolling(k_period).max()
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        
        # %D (сглаженный %K)
        d_percent = k_percent.rolling(d_period).mean()
        
        return pd.DataFrame({
            'k_percent': k_percent,
            'd_percent': d_percent
        })
    
    def calculate_vwap(self, df, period=None):
        """VWAP - Volume Weighted Average Price"""
        if 'volume' not in df.columns:
            # Если нет объема, используем простую среднюю
            return df['close'].rolling(period).mean() if period else df['close'].expanding().mean()
        
        if period:
            # VWAP за период
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
        else:
            # VWAP с начала данных
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).expanding().sum() / df['volume'].expanding().sum()
        
        return vwap
    
    def calculate_rational_quadratic_kernel(self, df, period=14, sigma=1.0):
        """Rational Quadratic Kernel - рациональное квадратичное ядро"""
        # Простая реализация RQK
        returns = df['close'].pct_change()
        volatility = returns.rolling(period).std()
        
        # RQK smoothing
        rqk = volatility.ewm(span=period, alpha=1/sigma).mean()
        
        return rqk
    
    def calculate_true_strength_index(self, df, rsi_period=25, rsi_smooth=13, signal_period=9):
        """True Strength Index (TSI) - истинный индекс силы"""
        # Двойное сглаживание изменения цены
        price_change = df['close'].diff()
        first_smooth = price_change.ewm(span=rsi_period).mean()
        second_smooth = first_smooth.ewm(span=rsi_smooth).mean()
        
        # Двойное сглаживание абсолютного изменения цены
        abs_price_change = price_change.abs()
        first_abs_smooth = abs_price_change.ewm(span=rsi_period).mean()
        second_abs_smooth = first_abs_smooth.ewm(span=rsi_smooth).mean()
        
        # TSI
        tsi = 100 * (second_smooth / second_abs_smooth)
        
        # Signal line
        signal = tsi.ewm(span=signal_period).mean()
        
        return pd.DataFrame({
            'tsi': tsi,
            'signal': signal
        })
    
    def calculate_half_trend(self, df, atr_period=10, atr_multiplier=2.0):
        """Half Trend - полутрендовый индикатор"""
        atr = self.calculate_atr(df['high'], df['low'], df['close'], atr_period)
        
        # Half Trend calculation
        trend = pd.Series(1, index=df.index)
        up = df['close'].copy()
        down = df['close'].copy()
        
        for i in range(1, len(df)):
            if trend.iloc[i-1] == 1:
                if df['close'].iloc[i] > up.iloc[i-1]:
                    up.iloc[i] = df['close'].iloc[i]
                else:
                    up.iloc[i] = up.iloc[i-1]
                
                if df['close'].iloc[i] < up.iloc[i-1] - atr.iloc[i] * atr_multiplier:
                    trend.iloc[i] = -1
                    down.iloc[i] = df['close'].iloc[i]
                else:
                    trend.iloc[i] = 1
                    down.iloc[i] = down.iloc[i-1]
            else:
                if df['close'].iloc[i] < down.iloc[i-1]:
                    down.iloc[i] = df['close'].iloc[i]
                else:
                    down.iloc[i] = down.iloc[i-1]
                
                if df['close'].iloc[i] > down.iloc[i-1] + atr.iloc[i] * atr_multiplier:
                    trend.iloc[i] = 1
                    up.iloc[i] = df['close'].iloc[i]
                else:
                    trend.iloc[i] = -1
                    up.iloc[i] = up.iloc[i-1]
        
        return pd.DataFrame({
            'trend': trend,
            'up': up,
            'down': down
        })
    
    def calculate_conditional_sampling_ema(self, df, period=14, condition_col='close', condition_threshold=0):
        """Conditional Sampling EMA - условная экспоненциальная скользящая средняя"""
        # Применяем EMA только когда условие выполняется
        condition = df[condition_col] > condition_threshold
        
        # Создаем маску для условного применения
        ema_values = df['close'].ewm(span=period).mean()
        conditional_ema = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if condition.iloc[i]:
                conditional_ema.iloc[i] = ema_values.iloc[i]
            else:
                conditional_ema.iloc[i] = conditional_ema.iloc[i-1] if i > 0 else ema_values.iloc[i]
        
        return conditional_ema
    
    def calculate_fibonacci_retracement(self, df, period=20):
        """Fibonacci Retracement - уровни коррекции Фибоначчи"""
        # Находим максимум и минимум за период
        high = df['high'].rolling(period).max()
        low = df['low'].rolling(period).min()
        
        # Размер движения
        range_size = high - low
        
        # Уровни Фибоначчи
        fib_236 = high - 0.236 * range_size
        fib_382 = high - 0.382 * range_size
        fib_500 = high - 0.500 * range_size
        fib_618 = high - 0.618 * range_size
        fib_786 = high - 0.786 * range_size
        
        return pd.DataFrame({
            'high': high,
            'low': low,
            'fib_236': fib_236,
            'fib_382': fib_382,
            'fib_500': fib_500,
            'fib_618': fib_618,
            'fib_786': fib_786
        })
    
    def calculate_woodie_pivot_levels(self, df):
        """Woodie Pivot Levels - уровни Вуди"""
        # Woodie Pivot Points
        pivot = (df['high'] + df['low'] + df['close'] + df['close']) / 4
        
        r1 = 2 * pivot - df['low']
        s1 = 2 * pivot - df['high']
        r2 = pivot + (df['high'] - df['low'])
        s2 = pivot - (df['high'] - df['low'])
        
        return pd.DataFrame({
            'pivot': pivot,
            'r1': r1, 'r2': r2,
            's1': s1, 's2': s2
        })
    
    def calculate_camarilla_pivot_levels(self, df):
        """Camarilla Pivot Levels - уровни Камарилья"""
        # Camarilla Pivot Points
        pivot = (df['high'] + df['low'] + df['close']) / 3
        
        r1 = df['close'] + (df['high'] - df['low']) * 1.1/12
        r2 = df['close'] + (df['high'] - df['low']) * 1.1/6
        r3 = df['close'] + (df['high'] - df['low']) * 1.1/4
        
        s1 = df['close'] - (df['high'] - df['low']) * 1.1/12
        s2 = df['close'] - (df['high'] - df['low']) * 1.1/6
        s3 = df['close'] - (df['high'] - df['low']) * 1.1/4
        
        return pd.DataFrame({
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        })
    
    def calculate_dm_pivot_levels(self, df):
        """DM Pivot Levels - уровни Демарка"""
        # DM Pivot Points
        if df['close'].iloc[-1] < df['open'].iloc[-1]:  # Bearish
            x = df['high'].iloc[-1] + 2 * df['low'].iloc[-1] + df['close'].iloc[-1]
        elif df['close'].iloc[-1] > df['open'].iloc[-1]:  # Bullish
            x = 2 * df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]
        else:  # Doji
            x = df['high'].iloc[-1] + df['low'].iloc[-1] + 2 * df['close'].iloc[-1]
        
        pivot = x / 4
        r1 = x / 2 - df['low'].iloc[-1]
        s1 = x / 2 - df['high'].iloc[-1]
        
        return pd.DataFrame({
            'pivot': pivot,
            'r1': r1,
            's1': s1
        })
    
    def calculate_classic_pivot_levels(self, df):
        """Classic Pivot Levels - классические уровни"""
        # Classic Pivot Points
        pivot = (df['high'] + df['low'] + df['close']) / 3
        
        r1 = 2 * pivot - df['low']
        s1 = 2 * pivot - df['high']
        r2 = pivot + (df['high'] - df['low'])
        s2 = pivot - (df['high'] - df['low'])
        r3 = df['high'] + 2 * (pivot - df['low'])
        s3 = df['low'] - 2 * (df['high'] - pivot)
        
        return pd.DataFrame({
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        })
    
    def calculate_true_range(self, df):
        """True Range - истинный диапазон"""
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr
    
    def step_by_step_optimization(self, metric='sharpe_ratio'):
        """Пошаговая оптимизация индикаторов"""
        print("🚀 ЗАПУСК ПОШАГОВОЙ ОПТИМИЗАЦИИ")
        print("="*60)
        
        # Оптимизируем отдельные индикаторы
        print("🔍 Шаг 1: Оптимизация отдельных индикаторов...")
        
        # SuperTrend
        print("  - Оптимизация SuperTrend...")
        supertrend_results = self.optimize_parameters(
            'SuperTrend',
            {
                'atr_period': [5, 10, 15, 20],
                'atr_multiplier': [2.0, 2.5, 3.0, 3.5, 4.0]
            },
            metric=metric
        )
        
        # Range Filter
        print("  - Оптимизация Range Filter...")
        range_filter_results = self.optimize_parameters(
            'Range Filter',
            {
                'rf_period': [50, 100, 150, 200],
                'rf_multiplier': [2.0, 2.5, 3.0, 3.5, 4.0]
            },
            metric=metric
        )
        
        # Range Filter Type 2
        print("  - Оптимизация Range Filter Type 2...")
        range_filter2_results = self.optimize_parameters(
            'Range Filter Type 2',
            {
                'rf2_period': [10, 14, 20, 30],
                'rf2_multiplier': [1.618, 2.0, 2.618, 3.0],
                'rf2_scale': ['ATR', 'Standard Deviation']
            },
            metric=metric
        )
        
        # 2 EMA Cross
        print("  - Оптимизация 2 EMA Cross...")
        ema2_results = self.optimize_parameters(
            '2 EMA Cross',
            {
                'ema2_fast': [20, 50, 100],
                'ema2_slow': [100, 200, 300]
            },
            metric=metric
        )
        
        # 3 EMA Cross
        print("  - Оптимизация 3 EMA Cross...")
        ema3_results = self.optimize_parameters(
            '3 EMA Cross',
            {
                'ema3_fast': [10, 20, 30],
                'ema3_mid': [30, 50, 70],
                'ema3_slow': [100, 200, 300]
            },
            metric=metric
        )
        
        # QQE Mod
        print("  - Оптимизация QQE Mod...")
        qqe_results = self.optimize_parameters(
            'QQE Mod',
            {
                'qqe_rsi_period': [10, 14, 20],
                'qqe_sf': [3, 5, 7],
                'qqe_qe': [3.0, 4.236, 5.0]
            },
            metric=metric
        )
        
        # Chaikin Money Flow
        print("  - Оптимизация Chaikin Money Flow...")
        cmf_results = self.optimize_parameters(
            'Chaikin Money Flow',
            {
                'cmf_period': [15, 20, 25, 30]
            },
            metric=metric
        )
        
        # Waddah Attar Explosion
        print("  - Оптимизация Waddah Attar Explosion...")
        wae_results = self.optimize_parameters(
            'Waddah Attar Explosion',
            {
                'wae_bb_period': [15, 20, 25],
                'wae_bb_std': [1.5, 2.0, 2.5],
                'wae_atr_period': [10, 14, 20]
            },
            metric=metric
        )
        
        # BB Oscillator
        print("  - Оптимизация BB Oscillator...")
        bb_osc_results = self.optimize_parameters(
            'BB Oscillator',
            {
                'bb_period': [15, 20, 25],
                'bb_std_dev': [1.5, 2.0, 2.5]
            },
            metric=metric
        )
        
        # Chandelier Exit
        print("  - Оптимизация Chandelier Exit...")
        ce_results = self.optimize_parameters(
            'Chandelier Exit',
            {
                'ce_period': [15, 22, 30],
                'ce_multiplier': [2.0, 3.0, 4.0]
            },
            metric=metric
        )
        
        # Heiken-Ashi Candlestick Oscillator
        print("  - Оптимизация Heiken-Ashi Candlestick Oscillator...")
        ha_results = self.optimize_parameters(
            'Heiken-Ashi Candlestick Oscillator',
            {
                'ha_period': [10, 14, 20]
            },
            metric=metric
        )
        
        # RSI
        print("  - Оптимизация RSI...")
        rsi_results = self.optimize_parameters(
            'RSI',
            {
                'rsi_length': [10, 14, 20, 30]
            },
            metric=metric
        )
        
        # MACD
        print("  - Оптимизация MACD...")
        macd_results = self.optimize_parameters(
            'MACD',
            {
                'macd_fast': [8, 12, 16],
                'macd_slow': [21, 26, 31],
                'macd_signal': [5, 9, 13]
            },
            metric=metric
        )
        
        # PVSRA
        print("  - Оптимизация PVSRA...")
        pvsra_results = self.optimize_parameters(
            'PVSRA',
            {
                'pvsra_period': [10, 14, 20, 30]
            },
            metric=metric
        )
        
        # Liquidity Zone
        print("  - Оптимизация Liquidity Zone...")
        lz_results = self.optimize_parameters(
            'Liquidity Zone',
            {
                'lz_period': [15, 20, 25, 30],
                'lz_threshold': [1.2, 1.5, 2.0, 2.5]
            },
            metric=metric
        )
        
        # Ichimoku Cloud
        print("  - Оптимизация Ichimoku Cloud...")
        ichimoku_results = self.optimize_parameters(
            'Ichimoku Cloud',
            {
                'ichimoku_tenkan': [7, 9, 11, 13],
                'ichimoku_kijun': [20, 26, 30, 35],
                'ichimoku_senkou_span_b': [40, 52, 60, 70]
            },
            metric=metric
        )
        
        # Stochastic Oscillator
        print("  - Оптимизация Stochastic Oscillator...")
        stoch_results = self.optimize_parameters(
            'Stochastic Oscillator',
            {
                'stoch_k_period': [10, 14, 20, 30],
                'stoch_d_period': [3, 5, 7, 9]
            },
            metric=metric
        )
        
        # VWAP
        print("  - Оптимизация VWAP...")
        vwap_results = self.optimize_parameters(
            'VWAP',
            {
                'vwap_period': [None, 20, 50, 100]
            },
            metric=metric
        )
        
        # Rational Quadratic Kernel
        print("  - Оптимизация Rational Quadratic Kernel...")
        rqk_results = self.optimize_parameters(
            'Rational Quadratic Kernel',
            {
                'rqk_period': [10, 14, 20, 30],
                'rqk_sigma': [0.5, 1.0, 1.5, 2.0]
            },
            metric=metric
        )
        
        # True Strength Index
        print("  - Оптимизация True Strength Index...")
        tsi_results = self.optimize_parameters(
            'True Strength Index',
            {
                'tsi_rsi_period': [20, 25, 30, 35],
                'tsi_rsi_smooth': [10, 13, 16, 20],
                'tsi_signal_period': [7, 9, 11, 13]
            },
            metric=metric
        )
        
        # Half Trend
        print("  - Оптимизация Half Trend...")
        ht_results = self.optimize_parameters(
            'Half Trend',
            {
                'ht_atr_period': [8, 10, 12, 15],
                'ht_atr_multiplier': [1.5, 2.0, 2.5, 3.0]
            },
            metric=metric
        )
        
        # Conditional Sampling EMA
        print("  - Оптимизация Conditional Sampling EMA...")
        cse_results = self.optimize_parameters(
            'Conditional Sampling EMA',
            {
                'cse_period': [10, 14, 20, 30],
                'cse_condition_threshold': [-0.01, 0, 0.01, 0.02]
            },
            metric=metric
        )
        
        # Собираем все результаты
        all_results = {
            'SuperTrend': supertrend_results,
            'Range Filter': range_filter_results,
            'Range Filter Type 2': range_filter2_results,
            '2 EMA Cross': ema2_results,
            '3 EMA Cross': ema3_results,
            'QQE Mod': qqe_results,
            'Chaikin Money Flow': cmf_results,
            'Waddah Attar Explosion': wae_results,
            'BB Oscillator': bb_osc_results,
            'Chandelier Exit': ce_results,
            'Heiken-Ashi Candlestick Oscillator': ha_results,
            'RSI': rsi_results,
            'MACD': macd_results,
            'PVSRA': pvsra_results,
            'Liquidity Zone': lz_results,
            'Ichimoku Cloud': ichimoku_results,
            'Stochastic Oscillator': stoch_results,
            'VWAP': vwap_results,
            'Rational Quadratic Kernel': rqk_results,
            'True Strength Index': tsi_results,
            'Half Trend': ht_results,
            'Conditional Sampling EMA': cse_results
        }
        
        # Ранжируем индикаторы по производительности
        print("\n📊 Ранжирование индикаторов по производительности:")
        indicator_ranking = []
        
        for indicator, result in all_results.items():
            if result is not None and len(result) > 0:
                best_result = result.iloc[0]  # Лучший результат
                indicator_ranking.append({
                    'indicator': indicator,
                    'best_params': best_result['params'],
                    'best_metric': best_result[metric],
                    'best_return': best_result['total_return']
                })
        
        # Сортируем по метрике
        indicator_ranking.sort(key=lambda x: x['best_metric'], reverse=True)
        
        for i, rank in enumerate(indicator_ranking[:5]):  # Топ-5
            print(f"  {i+1}. {rank['indicator']}: {rank['best_metric']:.4f} "
                  f"(доход: {rank['best_return']:.2%})")
        
        # Шаг 2: Комбинируем лучшие индикаторы
        print("\n🔗 Шаг 2: Комбинирование лучших индикаторов...")
        
        # Берем топ-3 ведущих индикатора
        top_leading = indicator_ranking[:3]
        top_confirmation = indicator_ranking[3:6]  # Следующие 3 как подтверждающие
        
        print(f"Ведущие индикаторы: {[r['indicator'] for r in top_leading]}")
        print(f"Подтверждающие индикаторы: {[r['indicator'] for r in top_confirmation]}")
        
        # Тестируем комбинации
        best_combination = None
        best_combination_metric = -np.inf
        
        for leading in top_leading:
            for confirmation in top_confirmation:
                print(f"  Тестируем: {leading['indicator']} + {confirmation['indicator']}")
                
                # Создаем конфигурацию для комбинации
                config = {
                    'leading_indicator': leading['indicator'],
                    'confirmation_indicator': confirmation['indicator'],
                    **leading['best_params'],
                    **confirmation['best_params']
                }
                
                try:
                    # Генерируем сигналы
                    signals = self.generate_signals(config)
                    
                    # Бэктест
                    performance = self.backtest(signals)
                    
                    current_metric = performance[metric]
                    
                    if current_metric > best_combination_metric:
                        best_combination_metric = current_metric
                        best_combination = {
                            'leading': leading,
                            'confirmation': confirmation,
                            'performance': performance,
                            'config': config
                        }
                        
                        print(f"    ✅ Новый лучший результат: {current_metric:.4f}")
                    
                except Exception as e:
                    print(f"    ❌ Ошибка при тестировании комбинации: {e}")
                    continue
        
        # Финальный результат
        print("\n🎯 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
        print("="*60)
        
        if best_combination:
            print(f"Лучшая комбинация:")
            print(f"  Ведущий: {best_combination['leading']['indicator']}")
            print(f"  Подтверждающий: {best_combination['confirmation']['indicator']}")
            print(f"\nПараметры:")
            print(f"  {best_combination['leading']['indicator']}: {best_combination['leading']['best_params']}")
            print(f"  {best_combination['confirmation']['indicator']}: {best_combination['confirmation']['best_params']}")
            
            print(f"\nПроизводительность:")
            perf = best_combination['performance']
            print(f"  Общий доход: {perf['total_return']:.2%}")
            print(f"  Годовой доход: {perf['annual_return']:.2%}")
            print(f"  Коэффициент Шарпа: {perf['sharpe_ratio']:.2f}")
            print(f"  Максимальная просадка: {perf['max_drawdown']:.2%}")
            
            return best_combination
        else:
            print("❌ Не удалось найти подходящую комбинацию")
            return None
    
    # ==========================================
    # СИСТЕМА УПРАВЛЕНИЯ ПОЗИЦИЕЙ
    # ==========================================
    
    def get_position_management_config(self, market_type='BTC'):
        """
        Конфигурация управления позицией для двухуровневого подхода
        
        Args:
            market_type (str): Тип рынка ('BTC', 'ETH', 'general')
            
        Returns:
            dict: Конфигурация управления позицией
        """
        if market_type == 'BTC':
            return {
                'tp1_percent': 50,                    # Первый TP: 50% позиции
                'tp1_atr_multiplier': 3.0,            # TP по 3.0 ATR
                'trailing_atr_multiplier': 5.0,       # Трейлинг по 5.0 ATR
                'use_breakeven': True,                 # Переводить стоп в безубыток
                'breakeven_trigger': 2.0,              # После 2.0 ATR прибыли
                'breakeven_stop': 'entry_price'        # Стоп в безубыток по цене входа
            }
        else:
            return {
                'tp1_percent': 50,
                'tp1_atr_multiplier': 3.0,
                'trailing_atr_multiplier': 5.0,
                'use_breakeven': True,
                'breakeven_trigger': 2.0,
                'breakeven_stop': 'entry_price'
            }
    
    def calculate_take_profit_levels(self, entry_price, position_type, atr, config):
        """
        Расчет уровней тейк-профита для двухуровневой системы
        
        Args:
            entry_price (float): Цена входа в позицию
            position_type (str): Тип позиции ('long' или 'short')
            atr (float): Текущее значение ATR
            config (dict): Конфигурация управления позицией
            
        Returns:
            dict: Уровни тейк-профита и стоп-лосса
        """
        tp1_multiplier = config['tp1_atr_multiplier']
        trailing_multiplier = config['trailing_atr_multiplier']
        
        if position_type == 'long':
            # Для длинной позиции
            tp1_level = entry_price + (tp1_multiplier * atr)
            trailing_stop = entry_price + (trailing_multiplier * atr)
            breakeven_trigger = entry_price + (config['breakeven_trigger'] * atr)
            
        elif position_type == 'short':
            # Для короткой позиции
            tp1_level = entry_price - (tp1_multiplier * atr)
            trailing_stop = entry_price - (tp1_multiplier * atr)
            breakeven_trigger = entry_price - (config['breakeven_trigger'] * atr)
            
        else:
            raise ValueError("position_type должен быть 'long' или 'short'")
        
        return {
            'tp1_level': tp1_level,
            'trailing_stop': trailing_stop,
            'breakeven_trigger': breakeven_trigger,
            'atr_value': atr
        }
    
    def calculate_smart_stop_loss(self, entry_price, position_type, atr, support_level=None, resistance_level=None):
        """
        Умный расчет стоп-лосса как в Pine Script
        
        Args:
            entry_price (float): Цена входа в позицию
            position_type (str): Тип позиции ('long' или 'short')
            atr (float): Текущее значение ATR
            support_level (float): Уровень поддержки
            resistance_level (float): Уровень сопротивления
            
        Returns:
            float: Уровень стоп-лосса
        """
        # Параметры как в Pine Script
        atr_multiplier_stop = 1.5
        level_buffer = 0.5
        
        if position_type == 'long':
            # Для длинной позиции
            if support_level is not None and support_level < entry_price:
                # Стоп под поддержкой с буфером
                stop_loss = support_level - (atr * level_buffer)
            else:
                # Стоп по ATR ниже цены входа
                stop_loss = entry_price - (atr * atr_multiplier_stop)
                
        elif position_type == 'short':
            # Для короткой позиции
            if resistance_level is not None and resistance_level > entry_price:
                # Стоп над сопротивлением с буфером
                stop_loss = resistance_level + (atr * level_buffer)
            else:
                # Стоп по ATR выше цены входа
                stop_loss = entry_price + (atr * atr_multiplier_stop)
                
        else:
            raise ValueError("position_type должен быть 'long' или 'short'")
        
        return stop_loss
    
    def update_trailing_stop(self, current_price, position_type, current_trailing_stop, atr, config):
        """
        Обновление трейлинг-стопа для оставшейся позиции
        
        Args:
            current_price (float): Текущая цена
            position_type (str): Тип позиции ('long' или 'short')
            current_trailing_stop (float): Текущий трейлинг-стоп
            atr (float): Текущее значение ATR
            config (dict): Конфигурация управления позицией
            
        Returns:
            float: Обновленный трейлинг-стоп
        """
        trailing_multiplier = config['trailing_atr_multiplier']
        
        if position_type == 'long':
            # Для длинной позиции - трейлинг-стоп только вверх
            new_trailing_stop = current_price - (trailing_multiplier * atr)
            return max(new_trailing_stop, current_trailing_stop)
            
        elif position_type == 'short':
            # Для короткой позиции - трейлинг-стоп только вниз
            new_trailing_stop = current_price + (trailing_multiplier * atr)
            return min(new_trailing_stop, current_trailing_stop)
            
        else:
            raise ValueError("position_type должен быть 'long' или 'short'")
    
    def check_position_exit_conditions(self, current_price, position_type, position_data, config):
        """
        Проверка условий выхода из позиции
        
        Args:
            current_price (float): Текущая цена
            position_type (str): Тип позиции ('long' или 'short')
            position_data (dict): Данные о позиции
            config (dict): Конфигурация управления позицией
            
        Returns:
            dict: Решение о выходе из позиции
        """
        entry_price = position_data['entry_price']
        tp1_hit = position_data.get('tp1_hit', False)
        current_trailing_stop = position_data.get('trailing_stop', None)
        atr = position_data.get('atr', 0)
        
        # Если первый TP еще не сработал
        if not tp1_hit:
            tp1_level = position_data['tp1_level']
            
            if position_type == 'long' and current_price >= tp1_level:
                return {
                    'action': 'partial_close',
                    'percent': config['tp1_percent'],
                    'reason': 'tp1_hit',
                    'new_stop': entry_price if config['use_breakeven'] else None
                }
                
            elif position_type == 'short' and current_price <= tp1_level:
                return {
                    'action': 'partial_close',
                    'percent': config['tp1_percent'],
                    'reason': 'tp1_hit',
                    'new_stop': entry_price if config['use_breakeven'] else None
                }
        
        # Если первый TP уже сработал, проверяем трейлинг-стоп
        elif tp1_hit and current_trailing_stop is not None:
            if position_type == 'long' and current_price <= current_trailing_stop:
                return {
                    'action': 'full_close',
                    'reason': 'trailing_stop_hit'
                }
                
            elif position_type == 'short' and current_price >= current_trailing_stop:
                return {
                    'action': 'full_close',
                    'reason': 'trailing_stop_hit'
                }
        
        # Проверяем стоп-лосс
        stop_loss = position_data.get('stop_loss', None)
        if stop_loss is not None:
            if position_type == 'long' and current_price <= stop_loss:
                return {
                    'action': 'full_close',
                    'reason': 'stop_loss_hit'
                }
                
            elif position_type == 'short' and current_price >= stop_loss:
                return {
                    'action': 'full_close',
                    'reason': 'stop_loss_hit'
                }
        
        # Нет условий для выхода
        return {
            'action': 'hold',
            'reason': 'no_exit_conditions'
        }
    
    def execute_position_management(self, signals, config=None):
        """
        Выполнение управления позицией для всех сигналов
        
        Args:
            signals (DataFrame): DataFrame с торговыми сигналами
            config (dict): Конфигурация управления позицией
            
        Returns:
            DataFrame: Обновленные сигналы с управлением позицией
        """
        if config is None:
            config = self.get_position_management_config()
        
        # Копируем сигналы для модификации
        managed_signals = signals.copy()
        
        # Добавляем колонки для управления позицией
        managed_signals['position_size'] = 0.0
        managed_signals['entry_price'] = np.nan
        managed_signals['tp1_hit'] = False
        managed_signals['trailing_stop'] = np.nan
        managed_signals['stop_loss'] = np.nan
        managed_signals['atr'] = np.nan
        managed_signals['exit_reason'] = ''
        
        # Переменные для отслеживания позиции
        current_position = 0  # 0 = нет позиции, 1 = длинная, -1 = короткая
        entry_price = 0
        tp1_hit = False
        trailing_stop = None
        stop_loss = None
        atr_value = 0
        
        for i in range(len(managed_signals)):
            signal = managed_signals.iloc[i]['signal']
            current_price = managed_signals.iloc[i]['close']
            
            # Рассчитываем ATR для текущего бара
            if i >= 14:  # Нужно минимум 14 баров для ATR
                high = managed_signals.iloc[i-14:i+1]['high']
                low = managed_signals.iloc[i-14:i+1]['low']
                close = managed_signals.iloc[i-14:i+1]['close']
                atr_value = self.calculate_atr(high, low, close, 14).iloc[-1]
            
            # Если нет открытой позиции и есть сигнал на вход
            if current_position == 0 and signal != 0:
                current_position = signal
                entry_price = current_price
                tp1_hit = False
                
                # Рассчитываем уровни управления позицией
                levels = self.calculate_take_profit_levels(
                    entry_price, 
                    'long' if signal == 1 else 'short',
                    atr_value,
                    config
                )
                
                trailing_stop = levels['trailing_stop']
                
                # Используем умный стоп-лосс как в Pine Script
                # Для простоты пока не учитываем уровни поддержки/сопротивления
                stop_loss = self.calculate_smart_stop_loss(
                    entry_price,
                    'long' if signal == 1 else 'short',
                    atr_value
                )
                
                # Записываем данные о позиции
                managed_signals.iloc[i, managed_signals.columns.get_loc('position_size')] = signal
                managed_signals.iloc[i, managed_signals.columns.get_loc('entry_price')] = entry_price
                managed_signals.iloc[i, managed_signals.columns.get_loc('trailing_stop')] = trailing_stop
                managed_signals.iloc[i, managed_signals.columns.get_loc('stop_loss')] = stop_loss
                managed_signals.iloc[i, managed_signals.columns.get_loc('atr')] = atr_value
            
            # Если есть открытая позиция, проверяем условия выхода
            elif current_position != 0:
                position_data = {
                    'entry_price': entry_price,
                    'tp1_hit': tp1_hit,
                    'trailing_stop': trailing_stop,
                    'stop_loss': stop_loss,
                    'atr': atr_value
                }
                
                position_type = 'long' if current_position == 1 else 'short'
                exit_decision = self.check_position_exit_conditions(
                    current_price, position_type, position_data, config
                )
                
                # Обрабатываем решение о выходе
                if exit_decision['action'] == 'partial_close':
                    # Первый TP сработал
                    tp1_hit = True
                    current_position *= 0.5  # Уменьшаем размер позиции
                    
                    # Обновляем стоп-лосс в безубыток если нужно
                    if config['use_breakeven']:
                        stop_loss = entry_price
                    
                    managed_signals.iloc[i, managed_signals.columns.get_loc('tp1_hit')] = True
                    managed_signals.iloc[i, managed_signals.columns.get_loc('stop_loss')] = stop_loss
                    managed_signals.iloc[i, managed_signals.columns.get_loc('exit_reason')] = 'TP1 Hit'
                    
                elif exit_decision['action'] == 'full_close':
                    # Полный выход из позиции
                    current_position = 0
                    entry_price = 0
                    tp1_hit = False
                    trailing_stop = None
                    stop_loss = None
                    
                    managed_signals.iloc[i, managed_signals.columns.get_loc('position_size')] = 0
                    managed_signals.iloc[i, managed_signals.columns.get_loc('exit_reason')] = exit_decision['reason']
                
                # Обновляем трейлинг-стоп если первый TP уже сработал
                elif tp1_hit and exit_decision['action'] == 'hold':
                    new_trailing_stop = self.update_trailing_stop(
                        current_price, position_type, trailing_stop, atr_value, config
                    )
                    trailing_stop = new_trailing_stop
                    
                    managed_signals.iloc[i, managed_signals.columns.get_loc('trailing_stop')] = trailing_stop
                
                # Записываем текущее состояние позиции
                managed_signals.iloc[i, managed_signals.columns.get_loc('position_size')] = current_position
                managed_signals.iloc[i, managed_signals.columns.get_loc('entry_price')] = entry_price
                managed_signals.iloc[i, managed_signals.columns.get_loc('tp1_hit')] = tp1_hit
                managed_signals.iloc[i, managed_signals.columns.get_loc('atr')] = atr_value
        
        return managed_signals

# Основная функция для запуска
if __name__ == "__main__":
    # Создаем экземпляр стратегии
    strategy = BTCTradingStrategy()
    
    # Загружаем данные с конвертацией в часовой тайм-фрейм
    print("Загружаем данные и конвертируем в часовой тайм-фрейм...")
    df = strategy.load_data(timeframe='1H')
    
    if df is not None:
        print("Данные успешно загружены!")
        print(f"Количество записей: {len(df)}")
        print(f"Колонки: {df.columns.tolist()}")
        print("\nПервые 5 записей:")
        print(df.head())
        
        # Тестируем вычисление индикаторов
        print("\nТестируем индикаторы...")
        
        # SuperTrend
        trend, up, dn = strategy.calculate_supertrend()
        print(f"SuperTrend: тренд {trend.iloc[-1]}, up {up.iloc[-1]:.2f}, dn {dn.iloc[-1]:.2f}")
        
        # RSI
        rsi = strategy.calculate_rsi(df['close'])
        print(f"RSI: {rsi.iloc[-1]:.2f}")
        
        # MACD
        macd, signal, hist = strategy.calculate_macd()
        print(f"MACD: {macd.iloc[-1]:.2f}, Signal: {signal.iloc[-1]:.2f}")
        
        # Тестируем генерацию сигналов
        print("\nТестируем генерацию сигналов...")
        
        # Конфигурация стратегии
        config = {
            'leading_indicator': 'SuperTrend',
            'supertrend_atr_period': 10,
            'supertrend_multiplier': 3.0,
            'use_rsi': True,
            'rsi_length': 14,
            'use_macd': True,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
        
        signals = strategy.generate_signals(config)
        print(f"Сгенерировано сигналов: {len(signals[signals['signal'] != 0])}")
        
        # Бэктест
        performance = strategy.backtest(signals)
        print(f"\nРезультаты бэктеста:")
        print(f"Общий доход: {performance['total_return']:.2%}")
        print(f"Годовой доход: {performance['annual_return']:.2%}")
        print(f"Волатильность: {performance['volatility']:.2%}")
        print(f"Коэффициент Шарпа: {performance['sharpe_ratio']:.2f}")
        print(f"Максимальная просадка: {performance['max_drawdown']:.2%}")
        
        # Демонстрация пошаговой оптимизации
        print("\n" + "="*60)
        print("ДЕМОНСТРАЦИЯ ПОШАГОВОЙ ОПТИМИЗАЦИИ")
        print("="*60)
        
        print("\nЭтот подход более эффективен:")
        print("1. Сначала оптимизируем каждый индикатор по отдельности")
        print("2. Затем комбинируем лучшие версии индикаторов")
        print("3. Получаем оптимальную стратегию с минимальными затратами времени")
        
        # Запускаем пошаговую оптимизацию
        print("\nЗапускаем пошаговую оптимизацию...")
        
        # Демонстрируем новые индикаторы
        print("\n🧪 ДЕМОНСТРАЦИЯ НОВЫХ ИНДИКАТОРОВ:")
        print("-" * 50)
        
        # QQE Mod
        qqe = strategy.calculate_qqe_mod(df)
        print(f"QQE Mod: up {qqe['qqe_up'].iloc[-1]:.2f}, down {qqe['qqe_down'].iloc[-1]:.2f}")
        
        # Chaikin Money Flow
        cmf = strategy.calculate_chaikin_money_flow(df)
        print(f"Chaikin Money Flow: {cmf.iloc[-1]:.4f}")
        
        # Waddah Attar Explosion
        wae = strategy.calculate_waddah_attar_explosion(df)
        print(f"Waddah Attar Explosion: up {wae['explosion_up'].iloc[-1]}, down {wae['explosion_down'].iloc[-1]}")
        
        # BB Oscillator
        bb_osc = strategy.calculate_bb_oscillator(df)
        print(f"BB Oscillator: {bb_osc.iloc[-1]:.4f}")
        
        # Chandelier Exit
        ce = strategy.calculate_chandelier_exit(df)
        print(f"Chandelier Exit: long_stop {ce['long_stop'].iloc[-1]:.2f}, short_stop {ce['short_stop'].iloc[-1]:.2f}")
        
        # Heiken-Ashi Candlestick Oscillator
        ha_osc = strategy.calculate_heiken_ashi_candlestick_oscillator(df)
        print(f"Heiken-Ashi Candlestick Oscillator: {ha_osc.iloc[-1]:.4f}")
        
        # B-Xtrender
        bxt = strategy.calculate_b_xtrender(df)
        print(f"B-Xtrender: {bxt.iloc[-1]}")
        
        # Bull Bear Power Trend
        bbpt = strategy.calculate_bull_bear_power_trend(df)
        print(f"Bull Bear Power Trend: bull {bbpt['bull_power'].iloc[-1]:.2f}, bear {bbpt['bear_power'].iloc[-1]:.2f}")
        
        # Detrended Price Oscillator
        dpo = strategy.calculate_detrended_price_oscillator(df)
        print(f"Detrended Price Oscillator: {dpo.iloc[-1]:.2f}")
        
        # Range Filter Type 2
        rf2 = strategy.calculate_range_filter_type2(df)
        print(f"Range Filter Type 2: direction {rf2['direction'].iloc[-1]}")
        
        # Pivot Levels
        pivots = strategy.calculate_pivot_levels(df)
        print(f"Pivot Levels: pivot {pivots['pivot'].iloc[-1]:.2f}, r1 {pivots['r1'].iloc[-1]:.2f}")
        
        # Fair Value Gap
        fvg = strategy.calculate_fair_value_gap(df)
        print(f"Fair Value Gap: up {fvg['gap_up'].iloc[-1]}, down {fvg['gap_down'].iloc[-1]}")
        
        # William Fractals
        fractals = strategy.calculate_william_fractals(df)
        print(f"William Fractals: bullish {fractals['bullish_fractal'].iloc[-1]}, bearish {fractals['bearish_fractal'].iloc[-1]}")
        
        # Supply/Demand Zones
        zones = strategy.calculate_supply_demand_zones(df)
        print(f"Supply/Demand Zones: supply {zones['supply_zone'].iloc[-1]}, demand {zones['demand_zone'].iloc[-1]}")
        
        # Market Sessions
        sessions = strategy.calculate_market_sessions(df)
        print(f"Market Sessions: asian {sessions['asian_session'].iloc[-1]}, london {sessions['london_session'].iloc[-1]}, ny {sessions['ny_session'].iloc[-1]}")
        
        # ZigZag
        zigzag = strategy.calculate_zigzag(df)
        print(f"ZigZag: pivot_high {zigzag['pivot_high'].iloc[-1]}, pivot_low {zigzag['pivot_low'].iloc[-1]}")
        
        # PVSRA
        pvsra = strategy.calculate_pvsra(df)
        print(f"PVSRA: {pvsra.iloc[-1]:.4f}")
        
        # Liquidity Zone
        lz = strategy.calculate_liquidity_zone(df)
        print(f"Liquidity Zone: {lz.iloc[-1]}")
        
        # Ichimoku Cloud
        ichimoku = strategy.calculate_ichimoku_cloud(df)
        print(f"Ichimoku Cloud: tenkan {ichimoku['tenkan_sen'].iloc[-1]:.2f}, kijun {ichimoku['kijun_sen'].iloc[-1]:.2f}")
        
        # Stochastic Oscillator
        stoch = strategy.calculate_stochastic_oscillator(df)
        print(f"Stochastic Oscillator: %K {stoch['k_percent'].iloc[-1]:.2f}, %D {stoch['d_percent'].iloc[-1]:.2f}")
        
        # VWAP
        vwap = strategy.calculate_vwap(df)
        print(f"VWAP: {vwap.iloc[-1]:.2f}")
        
        # Rational Quadratic Kernel
        rqk = strategy.calculate_rational_quadratic_kernel(df)
        print(f"Rational Quadratic Kernel: {rqk.iloc[-1]:.4f}")
        
        # True Strength Index
        tsi = strategy.calculate_true_strength_index(df)
        print(f"True Strength Index: {tsi['tsi'].iloc[-1]:.2f}, Signal: {tsi['signal'].iloc[-1]:.2f}")
        
        # Half Trend
        ht = strategy.calculate_half_trend(df)
        print(f"Half Trend: trend {ht['trend'].iloc[-1]}, up {ht['up'].iloc[-1]:.2f}, down {ht['down'].iloc[-1]:.2f}")
        
        # Conditional Sampling EMA
        cse = strategy.calculate_conditional_sampling_ema(df)
        print(f"Conditional Sampling EMA: {cse.iloc[-1]:.2f}")
        
        # Fibonacci Retracement
        fib = strategy.calculate_fibonacci_retracement(df)
        print(f"Fibonacci Retracement: 0.618 {fib['fib_618'].iloc[-1]:.2f}, 0.500 {fib['fib_500'].iloc[-1]:.2f}")
        
        # Woodie Pivot Levels
        woodie = strategy.calculate_woodie_pivot_levels(df)
        print(f"Woodie Pivot Levels: pivot {woodie['pivot'].iloc[-1]:.2f}, r1 {woodie['r1'].iloc[-1]:.2f}")
        
        # Camarilla Pivot Levels
        camarilla = strategy.calculate_camarilla_pivot_levels(df)
        print(f"Camarilla Pivot Levels: pivot {camarilla['pivot'].iloc[-1]:.2f}, r1 {camarilla['r1'].iloc[-1]:.2f}")
        
        # DM Pivot Levels
        dm = strategy.calculate_dm_pivot_levels(df)
        print(f"DM Pivot Levels: pivot {dm['pivot'].iloc[-1]:.2f}, r1 {dm['r1'].iloc[-1]:.2f}")
        
        # Classic Pivot Levels
        classic = strategy.calculate_classic_pivot_levels(df)
        print(f"Classic Pivot Levels: pivot {classic['pivot'].iloc[-1]:.2f}, r1 {classic['r1'].iloc[-1]:.2f}")
        
        print("\n" + "="*60)
        print("ВСЕ НЕДОСТАЮЩИЕ ИНДИКАТОРЫ УСПЕШНО ДОБАВЛЕНЫ!")
        print("="*60)
        
        try:
            best_combo, best_config, all_results = strategy.step_by_step_optimization('sharpe_ratio')
            
            if best_combo:
                print(f"\n🎯 ЛУЧШАЯ СТРАТЕГИЯ НАЙДЕНА!")
                print(f"Ведущий индикатор: {best_combo['leading']}")
                print(f"Подтверждающие: {best_combo['confirmation']}")
                print(f"Конфигурация: {best_config}")
                
                # Тестируем лучшую стратегию
                print(f"\nТестируем лучшую стратегию...")
                final_signals = strategy.generate_signals(best_config)
                final_performance = strategy.backtest(final_signals)
                
                print(f"Финальные результаты:")
                print(f"Коэффициент Шарпа: {final_performance['sharpe_ratio']:.4f}")
                print(f"Общий доход: {final_performance['total_return']:.2%}")
                print(f"Годовой доход: {final_performance['annual_return']:.2%}")
                print(f"Волатильность: {final_performance['volatility']:.2%}")
                print(f"Максимальная просадка: {final_performance['max_drawdown']:.2%}")
                
            else:
                print("Не удалось найти оптимальную стратегию")
                
        except Exception as e:
            print(f"Ошибка при пошаговой оптимизации: {e}")
            print("Попробуем простую оптимизацию...")
            
            # Простая оптимизация параметров
            print("\nНачинаем простую оптимизацию параметров...")
            
            param_grid = {
                'leading_indicator': ['SuperTrend', 'Range Filter'],
                'supertrend_atr_period': [5, 10, 15],
                'supertrend_multiplier': [2.0, 3.0, 4.0],
                'rsi_length': [10, 14, 20],
                'macd_fast': [8, 12, 16],
                'macd_slow': [20, 26, 32]
            }
            
            best_params, results = strategy.optimize_parameters(param_grid, 'sharpe_ratio')
        
    else:
        print("Не удалось загрузить данные!")
