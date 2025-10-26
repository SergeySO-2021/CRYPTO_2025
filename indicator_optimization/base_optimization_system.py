"""
Базовые классы системы оптимизации технических индикаторов
под различные рыночные зоны.

Автор: CRYPTO_2025 Project
Дата: 26.10.2025
Версия: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import sys
import os
from pathlib import Path

# Добавляем пути к существующим компонентам
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'compare_analyze_indicators' / 'classifiers'))
sys.path.append(str(Path(__file__).parent.parent / 'indicators' / 'trading_classifier_iziceros' / 'src'))

try:
    from compare_analyze_indicators.classifiers.mza_classifier_vectorized import VectorizedMZAClassifier
    from compare_analyze_indicators.classifiers.ml_classifier_optimized import OptimizedMarketRegimeMLClassifier
    from indicators.trading_classifier_iziceros.src.trend_classifier import Segmenter, Config, CONFIG_REL, CONFIG_ABS, CONFIG_REL_SLOPE_ONLY
except ImportError as e:
    print(f"Предупреждение: Не удалось импортировать некоторые компоненты: {e}")
    print("Будут использованы упрощенные версии")


class IndicatorOptimizationSystem:
    """
    Основной класс системы оптимизации индикаторов.
    
    Обеспечивает:
    - Загрузку и настройку классификаторов рыночных зон
    - Оптимизацию параметров индикаторов для разных зон
    - Интеграцию с существующими компонентами проекта
    """
    
    def __init__(self, classifier_type: str = 'mza', data_path: str = '../'):
        """
        Инициализация системы оптимизации.
        
        Args:
            classifier_type: Тип классификатора ('mza', 'trend_classifier', 'ml')
            data_path: Путь к данным проекта
        """
        self.classifier_type = classifier_type
        self.data_path = data_path
        self.classifier = None
        self.indicator_engine = None
        self.optimizer = None
        self.data = {}
        
        # Инициализация компонентов
        self._load_classifier()
        self._load_indicator_engine()
        self._load_optimizer()
        self._load_data()
        
    def _load_classifier(self):
        """Загрузка классификатора рыночных зон."""
        try:
            if self.classifier_type == 'mza':
                self.classifier = VectorizedMZAClassifier()
                print("✅ MZA классификатор загружен")
            elif self.classifier_type == 'ml':
                self.classifier = OptimizedMarketRegimeMLClassifier()
                print("✅ ML классификатор загружен")
            elif self.classifier_type == 'trend_classifier':
                self.classifier = Segmenter()
                print("✅ Trend Classifier загружен")
            else:
                raise ValueError(f"Неизвестный тип классификатора: {self.classifier_type}")
        except Exception as e:
            print(f"❌ Ошибка загрузки классификатора: {e}")
            self.classifier = None
            
    def _load_indicator_engine(self):
        """Загрузка движка индикаторов."""
        try:
            # Здесь будет интеграция с 08_indicator_engine_clean.ipynb
            # Пока создаем заглушку
            self.indicator_engine = SimpleIndicatorEngine()
            print("✅ Движок индикаторов загружен")
        except Exception as e:
            print(f"❌ Ошибка загрузки движка индикаторов: {e}")
            self.indicator_engine = None
            
    def _load_optimizer(self):
        """Загрузка оптимизатора параметров."""
        try:
            self.optimizer = GeneticOptimizer()
            print("✅ Оптимизатор загружен")
        except Exception as e:
            print(f"❌ Ошибка загрузки оптимизатора: {e}")
            self.optimizer = None
            
    def _load_data(self):
        """Загрузка данных BTC."""
        timeframes = ['15m', '30m', '1h', '4h', '1d']
        
        for tf in timeframes:
            try:
                file_path = f"{self.data_path}df_btc_{tf}.csv"
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df['timestamps'] = pd.to_datetime(df['timestamps'])
                    df.set_index('timestamps', inplace=True)
                    self.data[tf] = df
                    print(f"✅ Данные {tf} загружены: {len(df)} записей")
                else:
                    print(f"⚠️ Файл {file_path} не найден")
            except Exception as e:
                print(f"❌ Ошибка загрузки данных {tf}: {e}")
                
    def identify_zones(self, data: pd.DataFrame, timeframe: str = '1h') -> List[Dict]:
        """
        Идентификация рыночных зон с помощью классификатора.
        
        Args:
            data: Данные для анализа
            timeframe: Таймфрейм данных
            
        Returns:
            Список зон с метаданными
        """
        if self.classifier is None:
            raise ValueError("Классификатор не загружен")
            
        try:
            if self.classifier_type == 'mza':
                # MZA классификатор
                predictions = self.classifier.fit_predict(data)
                zones = self._convert_mza_predictions_to_zones(predictions, data)
                
            elif self.classifier_type == 'trend_classifier':
                # Trend Classifier
                segments = self.classifier.segment(data)
                zones = self._convert_segments_to_zones(segments, data)
                
            elif self.classifier_type == 'ml':
                # ML классификатор
                predictions = self.classifier.fit_predict(data)
                zones = self._convert_ml_predictions_to_zones(predictions, data)
                
            else:
                raise ValueError(f"Неизвестный тип классификатора: {self.classifier_type}")
                
            print(f"✅ Идентифицировано {len(zones)} зон для {timeframe}")
            return zones
            
        except Exception as e:
            print(f"❌ Ошибка идентификации зон: {e}")
            return []
            
    def _convert_mza_predictions_to_zones(self, predictions: np.ndarray, data: pd.DataFrame) -> List[Dict]:
        """Конвертация предсказаний MZA в зоны."""
        zones = []
        current_zone = None
        
        for i, prediction in enumerate(predictions):
            if current_zone is None or current_zone['type'] != prediction:
                if current_zone is not None:
                    current_zone['end'] = i - 1
                    zones.append(current_zone)
                
                current_zone = {
                    'start': i,
                    'end': len(predictions) - 1,
                    'type': prediction,
                    'classifier': 'mza'
                }
        
        if current_zone is not None:
            zones.append(current_zone)
            
        return zones
        
    def _convert_segments_to_zones(self, segments: List, data: pd.DataFrame) -> List[Dict]:
        """Конвертация сегментов Trend Classifier в зоны."""
        zones = []
        
        for segment in segments:
            zone_type = 'bull' if segment.slope > 0.1 else 'bear' if segment.slope < -0.1 else 'sideways'
            
            zone = {
                'start': segment.start,
                'end': segment.stop,
                'type': zone_type,
                'classifier': 'trend_classifier',
                'slope': segment.slope,
                'quality': getattr(segment, 'quality_score', 0.5)
            }
            zones.append(zone)
            
        return zones
        
    def _convert_ml_predictions_to_zones(self, predictions: np.ndarray, data: pd.DataFrame) -> List[Dict]:
        """Конвертация предсказаний ML в зоны."""
        zones = []
        current_zone = None
        
        for i, prediction in enumerate(predictions):
            if current_zone is None or current_zone['type'] != prediction:
                if current_zone is not None:
                    current_zone['end'] = i - 1
                    zones.append(current_zone)
                
                current_zone = {
                    'start': i,
                    'end': len(predictions) - 1,
                    'type': prediction,
                    'classifier': 'ml'
                }
        
        if current_zone is not None:
            zones.append(current_zone)
            
        return zones
        
    def optimize_for_zones(self, timeframe: str = '1h', indicators: List[str] = None) -> Dict:
        """
        Оптимизация индикаторов для всех зон на указанном таймфрейме.
        
        Args:
            timeframe: Таймфрейм для оптимизации
            indicators: Список индикаторов для оптимизации
            
        Returns:
            Словарь с оптимизированными параметрами для каждой зоны
        """
        if timeframe not in self.data:
            raise ValueError(f"Данные для таймфрейма {timeframe} не найдены")
            
        if indicators is None:
            indicators = ['rsi', 'macd', 'bollinger_bands', 'supertrend']
            
        data = self.data[timeframe]
        zones = self.identify_zones(data, timeframe)
        
        print(f"🎯 Начинаем оптимизацию для {len(zones)} зон на {timeframe}")
        
        optimized_params = {}
        
        for i, zone in enumerate(zones):
            print(f"📊 Оптимизация зоны {i+1}/{len(zones)}: {zone['type']}")
            
            zone_data = data.iloc[zone['start']:zone['end']+1]
            
            if len(zone_data) < 50:  # Минимальный размер зоны
                print(f"⚠️ Зона {i+1} слишком мала ({len(zone_data)} записей), пропускаем")
                continue
                
            zone_params = {}
            
            for indicator in indicators:
                try:
                    params = self.optimizer.optimize_indicator(
                        indicator, 
                        zone_data, 
                        zone_type=zone['type']
                    )
                    zone_params[indicator] = params
                    print(f"  ✅ {indicator}: {params}")
                    
                except Exception as e:
                    print(f"  ❌ Ошибка оптимизации {indicator}: {e}")
                    zone_params[indicator] = {}
                    
            optimized_params[f"zone_{i}_{zone['type']}"] = {
                'zone_info': zone,
                'parameters': zone_params
            }
            
        print(f"✅ Оптимизация завершена для {len(optimized_params)} зон")
        return optimized_params
        
    def optimize_all_timeframes(self, indicators: List[str] = None) -> Dict:
        """
        Оптимизация индикаторов для всех таймфреймов.
        
        Args:
            indicators: Список индикаторов для оптимизации
            
        Returns:
            Словарь с результатами оптимизации по таймфреймам
        """
        results = {}
        
        for timeframe in self.data.keys():
            print(f"\n🚀 Оптимизация для таймфрейма: {timeframe}")
            try:
                results[timeframe] = self.optimize_for_zones(timeframe, indicators)
            except Exception as e:
                print(f"❌ Ошибка оптимизации для {timeframe}: {e}")
                results[timeframe] = {}
                
        return results
        
    def get_adaptive_rules(self, optimized_params: Dict) -> Dict:
        """
        Создание адаптивных правил переключения параметров.
        
        Args:
            optimized_params: Результаты оптимизации
            
        Returns:
            Словарь с адаптивными правилами
        """
        rules = {
            'zone_detection': {
                'classifier': self.classifier_type,
                'method': 'real_time'
            },
            'parameter_switching': {},
            'fallback_rules': {}
        }
        
        # Анализ оптимизированных параметров
        for zone_name, zone_data in optimized_params.items():
            zone_type = zone_data['zone_info']['type']
            parameters = zone_data['parameters']
            
            rules['parameter_switching'][zone_type] = parameters
            
        # Создание fallback правил
        rules['fallback_rules'] = {
            'default_parameters': self._get_default_parameters(),
            'emergency_mode': True
        }
        
        return rules
        
    def _get_default_parameters(self) -> Dict:
        """Получение параметров по умолчанию."""
        return {
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'bollinger_bands': {'period': 20, 'std_dev': 2},
            'supertrend': {'atr_period': 10, 'atr_multiplier': 3}
        }


class SimpleIndicatorEngine:
    """Упрощенный движок индикаторов для тестирования."""
    
    def __init__(self):
        self.indicators = ['rsi', 'macd', 'bollinger_bands', 'supertrend']
        
    def calculate_indicator(self, indicator: str, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Расчет индикатора с заданными параметрами."""
        if indicator == 'rsi':
            return self._calculate_rsi(data, params)
        elif indicator == 'macd':
            return self._calculate_macd(data, params)
        elif indicator == 'bollinger_bands':
            return self._calculate_bollinger_bands(data, params)
        elif indicator == 'supertrend':
            return self._calculate_supertrend(data, params)
        else:
            raise ValueError(f"Неизвестный индикатор: {indicator}")
            
    def _calculate_rsi(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Расчет RSI."""
        period = params.get('period', 14)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_macd(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Расчет MACD."""
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        
        ema_fast = data['close'].ewm(span=fast_period).mean()
        ema_slow = data['close'].ewm(span=slow_period).mean()
        macd = ema_fast - ema_slow
        
        return macd
        
    def _calculate_bollinger_bands(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Расчет Bollinger Bands."""
        period = params.get('period', 20)
        std_dev = params.get('std_dev', 2)
        
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        bb_position = (data['close'] - lower_band) / (upper_band - lower_band)
        return bb_position
        
    def _calculate_supertrend(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Расчет SuperTrend."""
        atr_period = params.get('atr_period', 10)
        atr_multiplier = params.get('atr_multiplier', 3)
        
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=atr_period).mean()
        
        hl2 = (data['high'] + data['low']) / 2
        upper_band = hl2 + (atr_multiplier * atr)
        lower_band = hl2 - (atr_multiplier * atr)
        
        supertrend = np.where(data['close'] <= lower_band.shift(), upper_band, lower_band)
        return pd.Series(supertrend, index=data.index)


class GeneticOptimizer:
    """Генетический оптимизатор параметров индикаторов."""
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        
    def optimize_indicator(self, indicator: str, data: pd.DataFrame, zone_type: str) -> Dict:
        """
        Оптимизация параметров индикатора для конкретной зоны.
        
        Args:
            indicator: Название индикатора
            data: Данные зоны
            zone_type: Тип зоны (bull/bear/sideways)
            
        Returns:
            Оптимальные параметры
        """
        # Параметры для оптимизации
        param_ranges = self._get_param_ranges(indicator, zone_type)
        
        # Простая оптимизация (заглушка)
        # В реальной реализации здесь будет генетический алгоритм
        best_params = {}
        
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, tuple):
                # Числовой параметр
                if param_name in ['period', 'fast_period', 'slow_period', 'signal_period']:
                    best_params[param_name] = param_range[0] + (param_range[1] - param_range[0]) // 2
                elif param_name in ['std_dev', 'atr_multiplier']:
                    best_params[param_name] = param_range[0] + (param_range[1] - param_range[0]) / 2
                else:
                    best_params[param_name] = param_range[0]
            else:
                # Категориальный параметр
                best_params[param_name] = param_range[0] if param_range else None
                
        return best_params
        
    def _get_param_ranges(self, indicator: str, zone_type: str) -> Dict:
        """Получение диапазонов параметров для оптимизации."""
        
        base_ranges = {
            'rsi': {
                'period': (8, 30),
                'overbought': (60, 90),
                'oversold': (10, 40)
            },
            'macd': {
                'fast_period': (5, 30),
                'slow_period': (15, 70),
                'signal_period': (5, 30)
            },
            'bollinger_bands': {
                'period': (15, 40),
                'std_dev': (1.5, 4.0)
            },
            'supertrend': {
                'atr_period': (5, 30),
                'atr_multiplier': (1.5, 6.0)
            }
        }
        
        # Адаптация параметров под тип зоны
        ranges = base_ranges.get(indicator, {}).copy()
        
        if zone_type == 'bull':
            # Для бычьих зон - более агрессивные параметры
            if indicator == 'rsi':
                ranges['overbought'] = (70, 85)
                ranges['oversold'] = (25, 40)
        elif zone_type == 'bear':
            # Для медвежьих зон - более консервативные параметры
            if indicator == 'rsi':
                ranges['overbought'] = (65, 80)
                ranges['oversold'] = (20, 35)
        elif zone_type == 'sideways':
            # Для боковых зон - средние параметры
            if indicator == 'rsi':
                ranges['overbought'] = (70, 80)
                ranges['oversold'] = (20, 30)
                
        return ranges


# Пример использования
if __name__ == "__main__":
    # Создание системы оптимизации
    system = IndicatorOptimizationSystem(classifier_type='mza')
    
    # Оптимизация для одного таймфрейма
    results = system.optimize_for_zones('1h', ['rsi', 'macd'])
    
    # Создание адаптивных правил
    rules = system.get_adaptive_rules(results)
    
    print("🎯 Система оптимизации готова к работе!")
    print(f"📊 Результаты оптимизации: {len(results)} зон")
    print(f"🔧 Адаптивные правила: {len(rules['parameter_switching'])} типов зон")
