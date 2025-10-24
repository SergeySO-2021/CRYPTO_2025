"""
Базовый класс для всех классификаторов рыночных зон
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BaseMarketZoneClassifier(ABC):
    """
    Базовый класс для классификаторов рыночных зон
    
    Все классификаторы должны наследоваться от этого класса
    и реализовывать абстрактные методы
    """
    
    def __init__(self, name: str, parameters: Dict = None):
        """
        Инициализация классификатора
        
        Args:
            name: Название классификатора
            parameters: Параметры классификатора
        """
        self.name = name
        self.parameters = parameters or {}
        self.is_fitted = False
        self.classes_ = None
        
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseMarketZoneClassifier':
        """
        Обучение классификатора на данных
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Предсказание рыночных зон
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            Массив предсказаний (0 - боковое движение, 1 - бычий тренд, -1 - медвежий тренд)
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Получение важности признаков
        
        Returns:
            Словарь с важностью каждого признака
        """
        pass
    
    def get_parameters(self) -> Dict:
        """
        Получение параметров классификатора
        
        Returns:
            Словарь с параметрами
        """
        return self.parameters.copy()
    
    def set_parameters(self, parameters: Dict) -> None:
        """
        Установка параметров классификатора
        
        Args:
            parameters: Новые параметры
        """
        self.parameters.update(parameters)
        self.is_fitted = False
    
    def get_class_names(self) -> List[str]:
        """
        Получение названий классов
        
        Returns:
            Список названий классов
        """
        if self.classes_ is None:
            return ['Sideways', 'Bullish', 'Bearish']
        return [f'Class_{i}' for i in self.classes_]
    
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
        
        # Проверяем логичность данных (high >= low, high >= open, high >= close, etc.)
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
        
        return True
    
    def calculate_returns(self, data: pd.DataFrame, periods: int = 1) -> pd.Series:
        """
        Вычисление доходности
        
        Args:
            data: DataFrame с данными
            periods: Количество периодов для вычисления доходности
            
        Returns:
            Series с доходностью
        """
        return data['close'].pct_change(periods=periods)
    
    def calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Вычисление волатильности
        
        Args:
            data: DataFrame с данными
            window: Окно для вычисления волатильности
            
        Returns:
            Series с волатильностью
        """
        returns = self.calculate_returns(data)
        return returns.rolling(window=window).std()
    
    def get_market_regime_stats(self, predictions: np.ndarray) -> Dict[str, float]:
        """
        Получение статистики по рыночным режимам
        
        Args:
            predictions: Массив предсказаний
            
        Returns:
            Словарь со статистикой
        """
        unique, counts = np.unique(predictions, return_counts=True)
        total = len(predictions)
        
        stats = {}
        for regime, count in zip(unique, counts):
            regime_name = {0: 'Sideways', 1: 'Bullish', -1: 'Bearish'}.get(regime, f'Unknown_{regime}')
            stats[f'{regime_name}_count'] = count
            stats[f'{regime_name}_percentage'] = count / total * 100
        
        return stats
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"
    
    def __str__(self) -> str:
        return f"{self.name} classifier with parameters: {self.parameters}"
