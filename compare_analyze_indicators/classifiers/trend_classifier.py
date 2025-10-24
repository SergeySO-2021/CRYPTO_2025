"""
Реализация Trend Classifier (сегментация временных рядов)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from base_classifier import BaseMarketZoneClassifier
import warnings
warnings.filterwarnings('ignore')


class TrendClassifier(BaseMarketZoneClassifier):
    """
    Trend Classifier - сегментация временных рядов (адаптация trend_classifier_iziceros)
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Инициализация Trend Classifier
        
        Args:
            parameters: Параметры классификатора
        """
        default_params = {
            'N': 24,           # размер окна
            'alpha': 2.0,       # порог изменения наклона
            'beta': 2.0,       # порог изменения смещения
            'overlap_ratio': 0.33  # коэффициент перекрытия
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Trend Classifier", default_params)
        self.segments = []
    
    def fit(self, data: pd.DataFrame) -> 'TrendClassifier':
        """
        Обучение Trend Classifier
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            self
        """
        self.validate_data(data, require_volume=False)
        
        # Сохраняем данные для использования в методах
        self.data = data.copy()
        
        # Вычисляем сегменты
        self._calculate_segments(data)
        
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
        
        self.validate_data(data, require_volume=False)
        
        # Сохраняем данные для использования в методах
        self.data = data.copy()
        
        # Вычисляем сегменты для новых данных
        self._calculate_segments(data)
        
        # Получаем предсказания на основе сегментов
        predictions = self._classify_segments(data)
        
        return predictions
    
    def _calculate_segments(self, data: pd.DataFrame) -> None:
        """
        Вычисление сегментов временного ряда
        
        Args:
            data: DataFrame с данными
        """
        n = self.parameters['N']
        overlap_ratio = self.parameters['overlap_ratio']
        alpha = self.parameters['alpha']
        beta = self.parameters['beta']
        
        offset = max(1, int(n * overlap_ratio))
        
        segments = []
        current_segment = {
            'start': 0,
            'slopes': [],
            'offsets': [],
            'starts': []
        }
        
        prev_fit = None
        
        for start in range(0, len(data) - n, offset):
            end = start + n
            
            x_window = np.array(range(start, end))
            y_window = np.array(data['close'][start:end])
            fit = np.polyfit(x_window, y_window, 1)
            
            current_segment['slopes'].append(fit[0])
            current_segment['offsets'].append(fit[1])
            current_segment['starts'].append(start)
            
            if prev_fit is not None:
                slope_change = abs(fit[0] - prev_fit[0])
                offset_change = abs(fit[1] - prev_fit[1])
                
                if slope_change > alpha or offset_change > beta:
                    # Создаем новый сегмент
                    segment = {
                        'start': current_segment['start'],
                        'stop': start + offset // 2,
                        'slope': np.mean(current_segment['slopes']),
                        'offset': np.mean(current_segment['offsets'])
                    }
                    segments.append(segment)
                    
                    current_segment = {
                        'start': start + offset // 2,
                        'slopes': [],
                        'offsets': [],
                        'starts': []
                    }
            
            prev_fit = fit
        
        # Добавляем последний сегмент
        if current_segment['slopes']:
            segment = {
                'start': current_segment['start'],
                'stop': len(data) - 1,
                'slope': np.mean(current_segment['slopes']),
                'offset': np.mean(current_segment['offsets'])
            }
            segments.append(segment)
        
        self.segments = segments
    
    def _classify_segments(self, data: pd.DataFrame) -> np.ndarray:
        """
        Классификация сегментов на рыночные зоны
        
        Args:
            data: DataFrame с данными
            
        Returns:
            Массив предсказаний
        """
        predictions = np.zeros(len(data))
        
        for segment in self.segments:
            start_idx = segment['start']
            stop_idx = min(segment['stop'], len(data) - 1)
            slope = segment['slope']
            
            # Классификация на основе наклона
            if slope > 0.1:  # Восходящий тренд
                regime = 1
            elif slope < -0.1:  # Нисходящий тренд
                regime = -1
            else:  # Боковое движение
                regime = 0
            
            # Применяем классификацию к сегменту
            predictions[start_idx:stop_idx + 1] = regime
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Получение важности признаков
        
        Returns:
            Словарь с важностью каждого признака
        """
        if not self.is_fitted:
            return {}
        
        # Для Trend Classifier важность определяется по параметрам
        importance = {
            'slope_threshold': 0.4,
            'offset_threshold': 0.3,
            'window_size': 0.2,
            'overlap_ratio': 0.1
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
        
        self._calculate_segments(data)
        
        # Получаем предсказания
        predictions = self._classify_segments(data)
        
        # Создаем DataFrame с результатами
        results = pd.DataFrame({
            'predictions': predictions,
            'segment_slopes': self._get_segment_slopes(data),
            'segment_quality': self._get_segment_quality(data)
        }, index=data.index)
        
        return results
    
    def _get_segment_slopes(self, data: pd.DataFrame) -> np.ndarray:
        """Получение наклонов сегментов для каждой точки"""
        slopes = np.zeros(len(data))
        
        for segment in self.segments:
            start_idx = segment['start']
            stop_idx = min(segment['stop'], len(data) - 1)
            slope = segment['slope']
            
            slopes[start_idx:stop_idx + 1] = slope
        
        return slopes
    
    def _get_segment_quality(self, data: pd.DataFrame) -> np.ndarray:
        """Получение качества сегментов для каждой точки"""
        quality = np.zeros(len(data))
        
        for segment in self.segments:
            start_idx = segment['start']
            stop_idx = min(segment['stop'], len(data) - 1)
            
            # Вычисляем R² для сегмента
            if stop_idx > start_idx:
                x_segment = np.array(range(start_idx, stop_idx + 1))
                y_segment = np.array(data['close'][start_idx:stop_idx + 1])
                
                if len(x_segment) > 1:
                    # Простое вычисление качества
                    segment_quality = 1.0 - abs(segment['slope']) / 100.0
                    quality[start_idx:stop_idx + 1] = max(0, segment_quality)
        
        return quality
