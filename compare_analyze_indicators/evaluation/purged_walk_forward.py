"""
Purged Walk-Forward Validation для избежания look-ahead bias
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Generator
import warnings
warnings.filterwarnings('ignore')


class PurgedWalkForward:
    """
    УЛУЧШЕННАЯ Purged Walk-Forward Validation для избежания утечки данных
    """
    
    def __init__(self, n_splits: int = 5, purge_days: int = 2, embargo_days: int = 1):
        """
        Инициализация Purged Walk-Forward
        
        Args:
            n_splits: Количество разделений
            purge_days: Количество дней для очистки
            embargo_days: Количество дней эмбарго
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
    
    def split(self, data: pd.DataFrame) -> Generator[Tuple[List[int], List[int]], None, None]:
        """
        Разделение данных с учетом purge и embargo периодов
        
        Args:
            data: DataFrame с данными
            
        Yields:
            Tuple с индексами train и test
        """
        total_length = len(data)
        split_size = total_length // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = (i + 1) * split_size
            test_start = train_end + self.purge_days
            test_end = test_start + split_size - self.embargo_days
            
            if test_end <= total_length:
                train_indices = list(range(0, train_end))
                test_indices = list(range(test_start, test_end))
                
                yield train_indices, test_indices
    
    def evaluate_classifier(self, classifier, data: pd.DataFrame, 
                          metric_calculator) -> List[Dict[str, float]]:
        """
        Оценка классификатора с Purged Walk-Forward
        
        Args:
            classifier: Классификатор для оценки
            data: DataFrame с данными
            metric_calculator: Функция для вычисления метрик
            
        Returns:
            Список с результатами для каждого разделения
        """
        results = []
        
        for train_idx, test_idx in self.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Обучаем на train_data
            classifier.fit(train_data)
            
            # Предсказываем на test_data
            predictions = classifier.predict(test_data)
            
            # Вычисляем метрики
            metrics = metric_calculator(test_data, predictions)
            results.append(metrics)
        
        return results
    
    def calculate_stability_metrics(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Вычисление метрик стабильности
        
        Args:
            results: Список с результатами для каждого разделения
            
        Returns:
            Словарь с метриками стабильности
        """
        if not results:
            return {}
        
        # Преобразуем в DataFrame для удобства
        results_df = pd.DataFrame(results)
        
        stability_metrics = {}
        
        for column in results_df.columns:
            if column in ['bull_return', 'bear_return', 'return_spread', 'economic_value']:
                # Вычисляем стабильность для ключевых метрик
                values = results_df[column].dropna()
                
                if len(values) > 1:
                    # Коэффициент вариации (CV) - мера стабильности
                    cv = values.std() / abs(values.mean()) if values.mean() != 0 else 0
                    stability_metrics[f'{column}_stability'] = 1 / (1 + cv)  # Обратная зависимость
                    
                    # Тренд стабильности (положительный тренд = хорошая стабильность)
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    stability_metrics[f'{column}_trend'] = trend
        
        return stability_metrics


class TimeSeriesCrossValidation:
    """
    Временная кросс-валидация с учетом временной последовательности
    """
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        """
        Инициализация временной кросс-валидации
        
        Args:
            n_splits: Количество разделений
            test_size: Размер тестовой выборки
        """
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, data: pd.DataFrame) -> Generator[Tuple[List[int], List[int]], None, None]:
        """
        Разделение данных с учетом временной последовательности
        
        Args:
            data: DataFrame с данными
            
        Yields:
            Tuple с индексами train и test
        """
        total_length = len(data)
        test_length = int(total_length * self.test_size)
        
        for i in range(self.n_splits):
            # Вычисляем границы для train и test
            train_end = total_length - test_length - (self.n_splits - i - 1) * (test_length // self.n_splits)
            test_start = train_end
            test_end = test_start + test_length
            
            if test_end <= total_length:
                train_indices = list(range(0, train_end))
                test_indices = list(range(test_start, test_end))
                
                yield train_indices, test_indices
    
    def evaluate_classifier(self, classifier, data: pd.DataFrame, 
                          metric_calculator) -> List[Dict[str, float]]:
        """
        Оценка классификатора с временной кросс-валидацией
        
        Args:
            classifier: Классификатор для оценки
            data: DataFrame с данными
            metric_calculator: Функция для вычисления метрик
            
        Returns:
            Список с результатами для каждого разделения
        """
        results = []
        
        for train_idx, test_idx in self.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Обучаем на train_data
            classifier.fit(train_data)
            
            # Предсказываем на test_data
            predictions = classifier.predict(test_data)
            
            # Вычисляем метрики
            metrics = metric_calculator(test_data, predictions)
            results.append(metrics)
        
        return results


class MonteCarloValidation:
    """
    Монте-Карло валидация для проверки устойчивости
    """
    
    def __init__(self, n_iterations: int = 100, noise_level: float = 0.01):
        """
        Инициализация Монте-Карло валидации
        
        Args:
            n_iterations: Количество итераций
            noise_level: Уровень шума для тестирования
        """
        self.n_iterations = n_iterations
        self.noise_level = noise_level
    
    def add_noise(self, data: pd.DataFrame, noise_level: float) -> pd.DataFrame:
        """
        Добавление шума к данным
        
        Args:
            data: DataFrame с данными
            noise_level: Уровень шума
            
        Returns:
            DataFrame с добавленным шумом
        """
        noisy_data = data.copy()
        
        # Добавляем шум к ценам
        price_noise = np.random.normal(0, noise_level, len(data))
        noisy_data['close'] = noisy_data['close'] * (1 + price_noise)
        noisy_data['open'] = noisy_data['open'] * (1 + price_noise)
        noisy_data['high'] = noisy_data['high'] * (1 + price_noise)
        noisy_data['low'] = noisy_data['low'] * (1 + price_noise)
        
        return noisy_data
    
    def evaluate_classifier(self, classifier, data: pd.DataFrame, 
                          metric_calculator) -> List[Dict[str, float]]:
        """
        Оценка классификатора с Монте-Карло валидацией
        
        Args:
            classifier: Классификатор для оценки
            data: DataFrame с данными
            metric_calculator: Функция для вычисления метрик
            
        Returns:
            Список с результатами для каждой итерации
        """
        results = []
        
        for _ in range(self.n_iterations):
            # Добавляем шум к данным
            noisy_data = self.add_noise(data, self.noise_level)
            
            # Обучаем на зашумленных данных
            classifier.fit(noisy_data)
            
            # Предсказываем на оригинальных данных
            predictions = classifier.predict(data)
            
            # Вычисляем метрики
            metrics = metric_calculator(data, predictions)
            results.append(metrics)
        
        return results
    
    def calculate_robustness_metrics(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Вычисление метрик робастности
        
        Args:
            results: Список с результатами для каждой итерации
            
        Returns:
            Словарь с метриками робастности
        """
        if not results:
            return {}
        
        # Преобразуем в DataFrame для удобства
        results_df = pd.DataFrame(results)
        
        robustness_metrics = {}
        
        for column in results_df.columns:
            if column in ['bull_return', 'bear_return', 'return_spread', 'economic_value']:
                # Вычисляем робастность для ключевых метрик
                values = results_df[column].dropna()
                
                if len(values) > 1:
                    # Стандартное отклонение (меньше = более робастно)
                    std = values.std()
                    robustness_metrics[f'{column}_robustness'] = 1 / (1 + std)
                    
                    # Коэффициент вариации
                    cv = std / abs(values.mean()) if values.mean() != 0 else 0
                    robustness_metrics[f'{column}_cv'] = cv
        
        return robustness_metrics


class ComprehensiveValidation:
    """
    Комплексная валидация, объединяющая все методы
    """
    
    def __init__(self, n_splits: int = 5, purge_days: int = 2, embargo_days: int = 1,
                 n_iterations: int = 100, noise_level: float = 0.01):
        """
        Инициализация комплексной валидации
        
        Args:
            n_splits: Количество разделений
            purge_days: Количество дней для очистки
            embargo_days: Количество дней эмбарго
            n_iterations: Количество итераций для Монте-Карло
            noise_level: Уровень шума
        """
        self.purged_wf = PurgedWalkForward(n_splits, purge_days, embargo_days)
        self.ts_cv = TimeSeriesCrossValidation(n_splits)
        self.monte_carlo = MonteCarloValidation(n_iterations, noise_level)
    
    def evaluate_classifier(self, classifier, data: pd.DataFrame, 
                          metric_calculator) -> Dict[str, List[Dict[str, float]]]:
        """
        Комплексная оценка классификатора
        
        Args:
            classifier: Классификатор для оценки
            data: DataFrame с данными
            metric_calculator: Функция для вычисления метрик
            
        Returns:
            Словарь с результатами всех методов валидации
        """
        results = {}
        
        # Purged Walk-Forward
        results['purged_walk_forward'] = self.purged_wf.evaluate_classifier(
            classifier, data, metric_calculator
        )
        
        # Time Series Cross-Validation
        results['time_series_cv'] = self.ts_cv.evaluate_classifier(
            classifier, data, metric_calculator
        )
        
        # Monte Carlo Validation
        results['monte_carlo'] = self.monte_carlo.evaluate_classifier(
            classifier, data, metric_calculator
        )
        
        return results
    
    def calculate_comprehensive_metrics(self, results: Dict[str, List[Dict[str, float]]]) -> Dict[str, float]:
        """
        Вычисление комплексных метрик
        
        Args:
            results: Словарь с результатами всех методов валидации
            
        Returns:
            Словарь с комплексными метриками
        """
        comprehensive_metrics = {}
        
        # Метрики стабильности из Purged Walk-Forward
        if 'purged_walk_forward' in results:
            stability_metrics = self.purged_wf.calculate_stability_metrics(
                results['purged_walk_forward']
            )
            comprehensive_metrics.update(stability_metrics)
        
        # Метрики робастности из Монте-Карло
        if 'monte_carlo' in results:
            robustness_metrics = self.monte_carlo.calculate_robustness_metrics(
                results['monte_carlo']
            )
            comprehensive_metrics.update(robustness_metrics)
        
        # Общие метрики
        comprehensive_metrics['total_validations'] = sum(
            len(result_list) for result_list in results.values()
        )
        
        return comprehensive_metrics


def improved_look_ahead_analysis(data, classifier, n_splits=5):
    """
    УЛУЧШЕННЫЙ анализ look-ahead bias
    
    Args:
        data: DataFrame с данными
        classifier: Классификатор для анализа
        n_splits: Количество разделений
        
    Returns:
        Список с результатами анализа
    """
    pwf = PurgedWalkForward(n_splits=n_splits)
    
    results = []
    
    for train_idx, test_idx in pwf.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Обучаем на train_data
        classifier.fit(train_data)
        
        # Предсказываем на test_data
        predictions = classifier.predict(test_data)
        
        # Вычисляем экономические метрики
        from economic_metrics import EconomicMetrics
        economic_metrics = EconomicMetrics()
        metrics = economic_metrics.calculate_economic_metrics(test_data, predictions)
        results.append(metrics)
    
    return results


def select_best_classifier_improved(results_dict):
    """
    УЛУЧШЕННЫЙ выбор классификатора с экономическим фокусом
    
    Args:
        results_dict: Словарь с результатами классификаторов
        
    Returns:
        Tuple с лучшим классификатором, скором и всеми скорами
    """
    
    # ОБНОВЛЕННЫЕ ВЕСА с фокусом на экономическую полезность
    weights = {
        'economic_value': 0.35,      # Способность разделять режимы по доходности
        'temporal_stability': 0.25,  # Стабильность на разных периодах
        'parameter_robustness': 0.20, # Устойчивость к изменению параметров
        'computational_efficiency': 0.15, # Скорость вычислений
        'interpretability': 0.05     # Интерпретируемость результатов
    }
    
    scores = {}
    
    for classifier_name, results in results_dict.items():
        # Вычисляем economic_value из нескольких компонентов
        return_spread = min(abs(results['return_spread']) / 0.05, 1.0)  # Нормализуем к 5%
        trend_efficiency = results.get('trend_efficiency', 0.5)
        
        economic_value = (return_spread * 0.6 + trend_efficiency * 0.4)
        
        # Вычисляем общий скор
        score = (
            economic_value * weights['economic_value'] +
            results.get('temporal_stability', 0.5) * weights['temporal_stability'] +
            results.get('parameter_robustness', 0.5) * weights['parameter_robustness'] +
            results.get('computational_efficiency', 0.5) * weights['computational_efficiency'] +
            results.get('interpretability', 0.5) * weights['interpretability']
        )
        
        scores[classifier_name] = score
    
    best_classifier = max(scores, key=scores.get)
    return best_classifier, scores[best_classifier], scores
