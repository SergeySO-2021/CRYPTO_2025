"""
Экономические метрики для оценки классификаторов рыночных зон
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EconomicMetrics:
    """
    Класс для вычисления экономических метрик классификаторов
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_economic_metrics(self, data: pd.DataFrame, predictions: np.ndarray) -> Dict[str, float]:
        """
        УЛУЧШЕННЫЕ экономические метрики классификатора
        
        Args:
            data: DataFrame с рыночными данными
            predictions: Массив предсказаний классификатора
            
        Returns:
            Словарь с экономическими метриками
        """
        # Вычисляем доходность
        returns = data['close'].pct_change().dropna()
        
        # Выравниваем индексы
        aligned_returns = returns.iloc[1:]  # убираем первый NaN
        aligned_predictions = predictions[1:len(aligned_returns)+1]
        
        # Разделяем по предсказаниям
        bull_mask = aligned_predictions == 1
        bear_mask = aligned_predictions == -1
        sideways_mask = aligned_predictions == 0
        
        # 1. Стратифицированная доходность
        bull_returns = aligned_returns[bull_mask].mean() if bull_mask.any() else 0
        bear_returns = aligned_returns[bear_mask].mean() if bear_mask.any() else 0
        sideways_returns = aligned_returns[sideways_mask].mean() if sideways_mask.any() else 0
        
        # 2. Разделение волатильности
        bull_vol = aligned_returns[bull_mask].std() if bull_mask.any() else 0
        bear_vol = aligned_returns[bear_mask].std() if bear_mask.any() else 0
        sideways_vol = aligned_returns[sideways_mask].std() if sideways_mask.any() else 0
        
        # 3. Эффективность следования тренду (УЛУЧШЕННАЯ)
        trend_efficiency = self._calculate_trend_efficiency_improved(aligned_returns, aligned_predictions)
        
        # 4. Персистентность режимов (НОВАЯ МЕТРИКА)
        regime_persistence = self._calculate_regime_persistence(aligned_predictions)
        
        # 5. Матрица переходов между режимами (НОВАЯ МЕТРИКА)
        transition_matrix = self._calculate_transition_matrix(aligned_predictions)
        
        # 6. Экономическая ценность
        economic_value = self._calculate_economic_value(bull_returns, bear_returns, sideways_vol)
        
        # 7. Коэффициент Шарпа для каждого режима
        bull_sharpe = bull_returns / bull_vol if bull_vol > 0 else 0
        bear_sharpe = bear_returns / bear_vol if bear_vol > 0 else 0
        sideways_sharpe = sideways_returns / sideways_vol if sideways_vol > 0 else 0
        
        # 8. Максимальная просадка для каждого режима
        bull_drawdown = self._calculate_max_drawdown(aligned_returns[bull_mask]) if bull_mask.any() else 0
        bear_drawdown = self._calculate_max_drawdown(aligned_returns[bear_mask]) if bear_mask.any() else 0
        sideways_drawdown = self._calculate_max_drawdown(aligned_returns[sideways_mask]) if sideways_mask.any() else 0
        
        metrics = {
            'bull_return': bull_returns,
            'bear_return': bear_returns,
            'sideways_return': sideways_returns,
            'return_spread': bull_returns - bear_returns,
            'bull_volatility': bull_vol,
            'bear_volatility': bear_vol,
            'sideways_volatility': sideways_vol,
            'volatility_ratio': sideways_vol / (bull_vol + bear_vol) if (bull_vol + bear_vol) > 0 else 0,
            'trend_efficiency': trend_efficiency,
            'regime_persistence': regime_persistence,
            'transition_matrix': transition_matrix,
            'economic_value': economic_value,
            'bull_sharpe': bull_sharpe,
            'bear_sharpe': bear_sharpe,
            'sideways_sharpe': sideways_sharpe,
            'bull_drawdown': bull_drawdown,
            'bear_drawdown': bear_drawdown,
            'sideways_drawdown': sideways_drawdown,
            'regime_separation': self._calculate_regime_separation(bull_returns, bear_returns, sideways_returns)
        }
        
        # Сохраняем историю метрик
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_trend_efficiency_improved(self, returns: pd.Series, predictions: np.ndarray) -> float:
        """
        УЛУЧШЕННАЯ эффективность следования тренду
        
        Args:
            returns: Series с доходностью
            predictions: Массив предсказаний
            
        Returns:
            Эффективность следования тренду
        """
        # Если предсказан бычий тренд, а доходность положительная - эффективно
        bull_correct = ((predictions == 1) & (returns > 0)).sum()
        bear_correct = ((predictions == -1) & (returns < 0)).sum()
        total_trend_signals = (predictions != 0).sum()
        
        return (bull_correct + bear_correct) / total_trend_signals if total_trend_signals > 0 else 0
    
    def _calculate_regime_persistence(self, predictions: np.ndarray) -> float:
        """
        Вычисление персистентности режимов
        
        Args:
            predictions: Массив предсказаний
            
        Returns:
            Персистентность режимов
        """
        if len(predictions) < 2:
            return 0
        
        # Подсчитываем количество переходов между режимами
        transitions = (predictions[1:] != predictions[:-1]).sum()
        total_periods = len(predictions) - 1
        
        # Персистентность = 1 - (количество переходов / общее количество периодов)
        persistence = 1 - (transitions / total_periods) if total_periods > 0 else 0
        
        return persistence
    
    def _calculate_transition_matrix(self, predictions: np.ndarray) -> Dict[str, float]:
        """
        Вычисление матрицы переходов между режимами
        
        Args:
            predictions: Массив предсказаний
            
        Returns:
            Словарь с матрицей переходов
        """
        if len(predictions) < 2:
            return {}
        
        # Создаем матрицу переходов
        transition_counts = {}
        regimes = [-1, 0, 1]
        
        for from_regime in regimes:
            for to_regime in regimes:
                key = f"{from_regime}_to_{to_regime}"
                transition_counts[key] = 0
        
        # Подсчитываем переходы
        for i in range(1, len(predictions)):
            from_regime = predictions[i-1]
            to_regime = predictions[i]
            key = f"{from_regime}_to_{to_regime}"
            if key in transition_counts:
                transition_counts[key] += 1
        
        # Нормализуем по общему количеству переходов
        total_transitions = sum(transition_counts.values())
        if total_transitions > 0:
            for key in transition_counts:
                transition_counts[key] = transition_counts[key] / total_transitions
        
        return transition_counts
    
    def _calculate_economic_value(self, bull_return: float, bear_return: float, sideways_vol: float) -> float:
        """
        Вычисление экономической ценности классификатора
        
        Args:
            bull_return: Средняя доходность в бычьем режиме
            bear_return: Средняя доходность в медвежьем режиме
            sideways_vol: Волатильность в боковом режиме
            
        Returns:
            Экономическая ценность
        """
        # Экономическая ценность = способность разделять режимы * стабильность
        regime_separation = abs(bull_return - bear_return)
        stability = 1 / (1 + sideways_vol)  # Обратная зависимость от волатильности
        
        return regime_separation * stability
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Вычисление максимальной просадки
        
        Args:
            returns: Series с доходностью
            
        Returns:
            Максимальная просадка
        """
        if len(returns) == 0:
            return 0
        
        # Вычисляем кумулятивную доходность
        cumulative = (1 + returns).cumprod()
        
        # Вычисляем просадку
        drawdown = (cumulative - cumulative.expanding().max()) / cumulative.expanding().max()
        
        return abs(drawdown.min())
    
    def _calculate_regime_separation(self, bull_return: float, bear_return: float, sideways_return: float) -> float:
        """
        Вычисление разделения режимов
        
        Args:
            bull_return: Средняя доходность в бычьем режиме
            bear_return: Средняя доходность в медвежьем режиме
            sideways_return: Средняя доходность в боковом режиме
            
        Returns:
            Коэффициент разделения режимов
        """
        # Вычисляем дисперсию между режимами
        returns_array = np.array([bull_return, bear_return, sideways_return])
        mean_return = np.mean(returns_array)
        
        # Коэффициент разделения = стандартное отклонение доходности между режимами
        separation = np.std(returns_array) / abs(mean_return) if mean_return != 0 else 0
        
        return separation
    
    def calculate_risk_adjusted_metrics(self, data: pd.DataFrame, predictions: np.ndarray) -> Dict[str, float]:
        """
        Вычисление риск-скорректированных метрик
        
        Args:
            data: DataFrame с рыночными данными
            predictions: Массив предсказаний
            
        Returns:
            Словарь с риск-скорректированными метриками
        """
        returns = data['close'].pct_change().dropna()
        
        # Разделяем по предсказаниям
        bull_mask = predictions == 1
        bear_mask = predictions == -1
        sideways_mask = predictions == 0
        
        metrics = {}
        
        for regime, mask in [('bull', bull_mask), ('bear', bear_mask), ('sideways', sideways_mask)]:
            if mask.any():
                regime_returns = returns[mask]
                
                # Коэффициент Шарпа
                sharpe = regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0
                
                # Коэффициент Сортино (учитывает только негативную волатильность)
                negative_returns = regime_returns[regime_returns < 0]
                sortino = regime_returns.mean() / negative_returns.std() if len(negative_returns) > 0 else 0
                
                # Коэффициент Кальмара (доходность / максимальная просадка)
                max_drawdown = self._calculate_max_drawdown(regime_returns)
                calmar = regime_returns.mean() / max_drawdown if max_drawdown > 0 else 0
                
                # Коэффициент Стерлинга (доходность / средняя просадка)
                avg_drawdown = abs(regime_returns[regime_returns < 0].mean()) if (regime_returns < 0).any() else 0
                sterling = regime_returns.mean() / avg_drawdown if avg_drawdown > 0 else 0
                
                metrics.update({
                    f'{regime}_sharpe': sharpe,
                    f'{regime}_sortino': sortino,
                    f'{regime}_calmar': calmar,
                    f'{regime}_sterling': sterling
                })
        
        return metrics
    
    def calculate_capacity_metrics(self, data: pd.DataFrame, predictions: np.ndarray, 
                                 position_sizes: List[float]) -> Dict[str, float]:
        """
        Вычисление метрик емкости стратегии
        
        Args:
            data: DataFrame с рыночными данными
            predictions: Массив предсказаний
            position_sizes: Список размеров позиций для тестирования
            
        Returns:
            Словарь с метриками емкости
        """
        capacity_metrics = {}
        
        for size in position_sizes:
            # Симуляция влияния на рынок
            market_impact = self._simulate_market_impact(predictions, size)
            
            # Корректировка доходности с учетом влияния
            adjusted_returns = self._calculate_adjusted_returns(data, market_impact)
            
            # Вычисляем метрики для скорректированной доходности
            adjusted_metrics = self.calculate_economic_metrics(data, adjusted_returns)
            
            capacity_metrics[f'capacity_{size}'] = adjusted_metrics
        
        return capacity_metrics
    
    def _simulate_market_impact(self, predictions: np.ndarray, position_size: float) -> np.ndarray:
        """
        Симуляция влияния на рынок от торговых позиций
        
        Args:
            predictions: Массив предсказаний
            position_size: Размер позиции
            
        Returns:
            Массив влияния на рынок
        """
        # Упрощенная модель влияния на рынок
        impact_factor = position_size * 0.001  # 0.1% на каждый миллион
        return predictions * impact_factor
    
    def _calculate_adjusted_returns(self, data: pd.DataFrame, market_impact: np.ndarray) -> np.ndarray:
        """
        Расчет скорректированной доходности с учетом влияния на рынок
        
        Args:
            data: DataFrame с рыночными данными
            market_impact: Массив влияния на рынок
            
        Returns:
            Массив скорректированной доходности
        """
        base_returns = data['close'].pct_change().dropna()
        adjusted_returns = base_returns - market_impact[:len(base_returns)]
        return adjusted_returns
    
    def get_comprehensive_score(self, metrics: Dict[str, float]) -> float:
        """
        Вычисление комплексного скора классификатора
        
        Args:
            metrics: Словарь с метриками
            
        Returns:
            Комплексный скор
        """
        # Веса для различных метрик
        weights = {
            'economic_value': 0.35,
            'regime_separation': 0.25,
            'trend_efficiency': 0.20,
            'return_spread': 0.20
        }
        
        # Нормализуем метрики (0-1)
        normalized_metrics = {}
        for key, value in metrics.items():
            if key in weights:
                # Простая нормализация (можно улучшить)
                normalized_metrics[key] = min(max(value, 0), 1)
        
        # Вычисляем взвешенный скор
        comprehensive_score = sum(
            normalized_metrics.get(key, 0) * weight 
            for key, weight in weights.items()
        )
        
        return comprehensive_score
    
    def compare_classifiers(self, classifier_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Сравнение нескольких классификаторов
        
        Args:
            classifier_results: Словарь с результатами классификаторов
            
        Returns:
            Словарь с рейтингом классификаторов
        """
        scores = {}
        
        for classifier_name, metrics in classifier_results.items():
            score = self.get_comprehensive_score(metrics)
            scores[classifier_name] = score
        
        # Сортируем по убыванию скора
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_scores
