"""
Дифференцированная система оценки индикаторов с учетом их класса

Использует разные веса метрик для разных классов индикаторов согласно рекомендациям.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from .indicator_classifier import IndicatorType, IndicatorClassifier


@dataclass
class Trade:
    """Представление одной сделки"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_type: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    holding_period: pd.Timedelta


@dataclass
class PerformanceMetrics:
    """Метрики производительности стратегии"""
    # Основные метрики
    total_pnl: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    
    # Дополнительные метрики
    avg_trade_pnl: float
    avg_winning_trade: float
    avg_losing_trade: float
    largest_winning_trade: float
    largest_losing_trade: float
    avg_holding_period: pd.Timedelta
    
    # Специфические метрики по классам
    trend_capture_ratio: float = 0.0
    reversal_accuracy: float = 0.0
    volume_confirmation_rate: float = 0.0
    volatility_adaptation: float = 0.0
    level_accuracy: float = 0.0
    time_based_consistency: float = 0.0


class DifferentiatedEvaluator:
    """
    Система дифференцированной оценки индикаторов с учетом их класса
    
    Использует разные веса метрик для разных классов индикаторов:
    - Trend Following: фокус на trend_capture_ratio, avg_holding_period
    - Oscillators: фокус на reversal_accuracy, win_rate
    - Volume Based: фокус на volume_confirmation_rate
    - Volatility: фокус на volatility_adaptation, max_drawdown
    - Levels Zones: фокус на level_accuracy, win_rate
    - Systemic: фокус на time_based_consistency
    """
    
    # Веса метрик для разных классов индикаторов
    CLASS_WEIGHTS = {
        'trend_following': {
            'total_return': 0.25,
            'sharpe_ratio': 0.20,
            'max_drawdown': 0.15,
            'win_rate': 0.08,
            'profit_factor': 0.10,
            'avg_trade_pnl': 0.05,
            'trend_capture_ratio': 0.12,
            'avg_holding_period': 0.05
        },
        
        'oscillators': {
            'total_return': 0.20,
            'sharpe_ratio': 0.15,
            'max_drawdown': 0.20,
            'win_rate': 0.20,
            'profit_factor': 0.10,
            'avg_trade_pnl': 0.05,
            'reversal_accuracy': 0.10
        },
        
        'volume_based': {
            'total_return': 0.15,
            'sharpe_ratio': 0.15,
            'max_drawdown': 0.15,
            'win_rate': 0.15,
            'profit_factor': 0.15,
            'volume_confirmation_rate': 0.25
        },
        
        'volatility': {
            'total_return': 0.20,
            'sharpe_ratio': 0.20,
            'max_drawdown': 0.25,
            'win_rate': 0.10,
            'profit_factor': 0.10,
            'volatility_adaptation': 0.15
        },
        
        'levels_zones': {
            'total_return': 0.25,
            'sharpe_ratio': 0.15,
            'max_drawdown': 0.15,
            'win_rate': 0.25,
            'profit_factor': 0.10,
            'level_accuracy': 0.10
        },
        
        'systemic': {
            'total_return': 0.25,
            'sharpe_ratio': 0.20,
            'max_drawdown': 0.20,
            'win_rate': 0.15,
            'profit_factor': 0.10,
            'time_based_consistency': 0.10
        }
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.classifier = IndicatorClassifier()
    
    def evaluate_strategy(self, trades: List[Trade], data: pd.DataFrame, 
                         indicator_name: str, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Полная оценка стратегии с учетом типа индикатора
        
        Args:
            trades: Список сделок
            data: Данные для анализа
            indicator_name: Название индикатора
            initial_capital: Начальный капитал
            
        Returns:
            Dict с оценкой и метриками
        """
        if not trades:
            return self._get_empty_metrics(indicator_name)
        
        # Определение типа индикатора
        indicator_type = self.classifier.get_indicator_type(indicator_name)
        indicator_type_str = indicator_type.value
        
        # Базовые метрики
        base_metrics = self._calculate_base_metrics(trades, initial_capital)
        
        # Специфические метрики для класса
        class_specific_metrics = self._calculate_class_specific_metrics(
            trades, data, indicator_type_str
        )
        
        # Комбинированные метрики
        all_metrics = {**base_metrics.__dict__, **class_specific_metrics}
        
        # Расчет итогового score
        final_score = self._calculate_weighted_score(all_metrics, indicator_type_str)
        
        return {
            'score': final_score,
            'metrics': all_metrics,
            'indicator_name': indicator_name,
            'indicator_type': indicator_type_str,
            'weights_used': self.CLASS_WEIGHTS.get(indicator_type_str, {})
        }
    
    def _calculate_base_metrics(self, trades: List[Trade], initial_capital: float) -> PerformanceMetrics:
        """Расчет базовых метрик производительности"""
        if not trades:
            return self._get_empty_metrics_object()
        
        # PnL и возвраты
        total_pnl = sum(trade.pnl for trade in trades)
        total_return = total_pnl / initial_capital
        
        # Win rate
        winning_trades = [t for t in trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Profit factor
        total_profit = sum(t.pnl for t in trades if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Максимальная просадка (упрощенный расчет)
        equity_curve = self._calculate_equity_curve(trades, initial_capital)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Sharpe ratio (упрощенный)
        returns = [t.pnl_pct for t in trades if t.pnl_pct is not None and not np.isnan(t.pnl_pct)]
        if returns and np.std(returns) > 0:
            # Для 15-минутных данных: 252 торговых дня * 24 часа * 4 (15-минутных периода в часе)
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 4)
        else:
            sharpe_ratio = 0.0
        
        # Дополнительные метрики
        avg_trade_pnl = np.mean([t.pnl for t in trades]) if trades else 0
        avg_winning_trade = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in trades if t.pnl < 0]
        avg_losing_trade = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        largest_winning_trade = max([t.pnl for t in trades]) if trades else 0
        largest_losing_trade = min([t.pnl for t in trades]) if trades else 0
        
        avg_holding_period = np.mean([t.holding_period for t in trades]) if trades else pd.Timedelta(0)
        
        return PerformanceMetrics(
            total_pnl=total_pnl,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_pnl=avg_trade_pnl,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            largest_winning_trade=largest_winning_trade,
            largest_losing_trade=largest_losing_trade,
            avg_holding_period=avg_holding_period
        )
    
    def _calculate_class_specific_metrics(self, trades: List[Trade], data: pd.DataFrame, 
                                         indicator_type: str) -> Dict[str, float]:
        """Расчет специфических метрик для класса индикатора"""
        metrics = {}
        
        if indicator_type == 'trend_following':
            metrics['trend_capture_ratio'] = self._calculate_trend_capture_ratio(trades, data)
            
        elif indicator_type == 'oscillators':
            metrics['reversal_accuracy'] = self._calculate_reversal_accuracy(trades, data)
            
        elif indicator_type == 'volume_based':
            metrics['volume_confirmation_rate'] = self._calculate_volume_confirmation(trades, data)
            
        elif indicator_type == 'volatility':
            metrics['volatility_adaptation'] = self._calculate_volatility_adaptation(trades, data)
            
        elif indicator_type == 'levels_zones':
            metrics['level_accuracy'] = self._calculate_level_accuracy(trades, data)
            
        elif indicator_type == 'systemic':
            metrics['time_based_consistency'] = self._calculate_time_consistency(trades, data)
        
        return metrics
    
    def _calculate_trend_capture_ratio(self, trades: List[Trade], data: pd.DataFrame) -> float:
        """Эффективность захвата тренда для трендовых индикаторов"""
        if not trades or len(data) < 2:
            return 0.0
        
        # Идентификация трендовых периодов (упрощенный подход)
        returns = data['close'].pct_change()
        trend_periods = returns.abs() > returns.rolling(window=20).std() * 1.5
        
        # Подсчет сделок в трендовых периодах
        trend_trades = []
        for trade in trades:
            # Проверяем, был ли период входа трендовым
            if trade.entry_time in data.index:
                entry_idx = data.index.get_loc(trade.entry_time)
                if entry_idx < len(trend_periods) and trend_periods.iloc[entry_idx]:
                    trend_trades.append(trade)
        
        if not trend_trades:
            return 0.0
        
        trend_pnl = sum(t.pnl for t in trend_trades)
        total_pnl = sum(t.pnl for t in trades)
        
        return abs(trend_pnl / total_pnl) if total_pnl != 0 else 0.0
    
    def _calculate_reversal_accuracy(self, trades: List[Trade], data: pd.DataFrame) -> float:
        """Точность определения разворотов для осцилляторов"""
        if not trades or len(data) < 2:
            return 0.0
        
        # Идентификация зон разворота (упрощенный подход)
        # Зоны разворота = локальные максимумы/минимумы
        price_changes = data['close'].pct_change()
        reversal_zones = (price_changes.shift(1) > 0) & (price_changes < 0) | \
                        (price_changes.shift(1) < 0) & (price_changes > 0)
        
        reversal_trades = []
        for trade in trades:
            if trade.entry_time in data.index:
                entry_idx = data.index.get_loc(trade.entry_time)
                if entry_idx < len(reversal_zones) and reversal_zones.iloc[entry_idx]:
                    reversal_trades.append(trade)
        
        if not reversal_trades:
            return 0.0
        
        winning_reversal_trades = [t for t in reversal_trades if t.pnl > 0]
        return len(winning_reversal_trades) / len(reversal_trades)
    
    def _calculate_volume_confirmation(self, trades: List[Trade], data: pd.DataFrame) -> float:
        """Эффективность подтверждения объемами"""
        if not trades or 'volume' not in data.columns:
            return 0.0
        
        confirmations = []
        for trade in trades:
            if trade.entry_time in data.index:
                entry_idx = data.index.get_loc(trade.entry_time)
                if entry_idx >= 3 and entry_idx < len(data) - 2:
                    # Анализируем объем вокруг точки входа
                    start_idx = max(0, entry_idx - 3)
                    end_idx = min(len(data), entry_idx + 2)
                    
                    volume_data = data.iloc[start_idx:end_idx]
                    avg_volume = volume_data['volume'].mean()
                    entry_volume = data.iloc[entry_idx]['volume']
                    
                    # Подтверждение: объем выше среднего
                    confirmation = 1 if entry_volume > avg_volume else 0
                    confirmations.append(confirmation)
        
        return sum(confirmations) / len(confirmations) if confirmations else 0.0
    
    def _calculate_volatility_adaptation(self, trades: List[Trade], data: pd.DataFrame) -> float:
        """Адаптация к волатильности для волатильностных индикаторов"""
        if not trades or len(data) < 20:
            return 0.0
        
        # Расчет волатильности
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        high_vol_threshold = volatility.quantile(0.7)
        
        # Производительность в периоды высокой волатильности
        high_vol_trades = []
        for trade in trades:
            if trade.entry_time in data.index:
                entry_idx = data.index.get_loc(trade.entry_time)
                if entry_idx < len(volatility) and volatility.iloc[entry_idx] > high_vol_threshold:
                    high_vol_trades.append(trade)
        
        if not high_vol_trades:
            return 0.0
        
        high_vol_pnl = sum(t.pnl for t in high_vol_trades)
        normal_trades = [t for t in trades if t not in high_vol_trades]
        normal_pnl = sum(t.pnl for t in normal_trades) if normal_trades else 0
        
        if normal_pnl == 0:
            return 1.0 if high_vol_pnl > 0 else 0.0
        
        return high_vol_pnl / abs(normal_pnl) if normal_pnl != 0 else 0.0
    
    def _calculate_level_accuracy(self, trades: List[Trade], data: pd.DataFrame) -> float:
        """Точность работы с уровнями для индикаторов уровней и зон"""
        if not trades:
            return 0.0
        
        # Упрощенный расчет: процент прибыльных сделок при работе с уровнями
        # (предполагаем, что уровни работают лучше при определенных условиях)
        return len([t for t in trades if t.pnl > 0]) / len(trades) if trades else 0.0
    
    def _calculate_time_consistency(self, trades: List[Trade], data: pd.DataFrame) -> float:
        """Согласованность во времени для системных индикаторов"""
        if not trades:
            return 0.0
        
        # Группировка сделок по часам дня
        hourly_performance = {}
        for trade in trades:
            hour = trade.entry_time.hour
            if hour not in hourly_performance:
                hourly_performance[hour] = []
            hourly_performance[hour].append(trade.pnl)
        
        if len(hourly_performance) < 2:
            return 0.0
        
        # Расчет согласованности (низкая вариативность между часами)
        hourly_returns = [np.mean(pnls) for pnls in hourly_performance.values()]
        consistency = 1.0 - (np.std(hourly_returns) / (abs(np.mean(hourly_returns)) + 1e-8))
        
        return max(0.0, min(1.0, consistency))
    
    def _calculate_equity_curve(self, trades: List[Trade], initial_capital: float) -> List[float]:
        """Расчет кривой капитала"""
        equity = [initial_capital]
        current_capital = initial_capital
        
        for trade in trades:
            current_capital += trade.pnl
            equity.append(current_capital)
        
        return equity
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Расчет максимальной просадки"""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        
        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        
        return float(np.max(drawdown))
    
    def _calculate_weighted_score(self, metrics: Dict[str, float], indicator_type: str) -> float:
        """Расчет взвешенного score с учетом типа индикатора"""
        weights = self.CLASS_WEIGHTS.get(indicator_type, {})
        
        if not weights:
            self.logger.warning(f"Не найдены веса для типа {indicator_type}, используем дефолтные")
            weights = self.CLASS_WEIGHTS['trend_following']
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                # Нормализация метрики
                normalized_value = self._normalize_metric(metric_name, metrics[metric_name])
                total_score += normalized_value * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """Нормализация значения метрики к диапазону [0, 1]"""
        normalization_rules = {
            'total_return': lambda x: max(0, min(1, (x + 0.5) / 1.0)),  # [-50%, +50%] -> [0, 1]
            'sharpe_ratio': lambda x: max(0, min(1, (x + 1) / 3)),      # [-1, 2] -> [0, 1]
            'max_drawdown': lambda x: max(0, min(1, 1 - x / 0.5)),      # [0, 50%] -> [1, 0]
            'win_rate': lambda x: max(0, min(1, x)),                    # [0, 1] как есть
            'profit_factor': lambda x: max(0, min(1, x / 3)),           # [0, 3] -> [0, 1]
            'avg_trade_pnl': lambda x: max(0, min(1, (x + 100) / 200)), # Нормализация PnL
            'trend_capture_ratio': lambda x: max(0, min(1, x)),
            'reversal_accuracy': lambda x: max(0, min(1, x)),
            'volume_confirmation_rate': lambda x: max(0, min(1, x)),
            'volatility_adaptation': lambda x: max(0, min(1, x)),
            'level_accuracy': lambda x: max(0, min(1, x)),
            'time_based_consistency': lambda x: max(0, min(1, x)),
            'avg_holding_period': lambda x: max(0, min(1, 1 - (x.total_seconds() / 86400) / 7))  # Нормализация периода
        }
        
        if metric_name in normalization_rules:
            return normalization_rules[metric_name](value)
        
        # Для неизвестных метрик используем сигмоиду
        return 1 / (1 + np.exp(-value))
    
    def _get_empty_metrics(self, indicator_name: str) -> Dict[str, Any]:
        """Возвращает метрики для пустого списка сделок"""
        indicator_type = self.classifier.get_indicator_type(indicator_name)
        
        return {
            'score': 0.0,
            'metrics': self._get_empty_metrics_object().__dict__,
            'indicator_name': indicator_name,
            'indicator_type': indicator_type.value,
            'weights_used': {}
        }
    
    def _get_empty_metrics_object(self) -> PerformanceMetrics:
        """Создает пустой объект метрик"""
        return PerformanceMetrics(
            total_pnl=0, total_return=0, sharpe_ratio=0, max_drawdown=0,
            win_rate=0, profit_factor=0, total_trades=0, avg_trade_pnl=0,
            avg_winning_trade=0, avg_losing_trade=0, largest_winning_trade=0,
            largest_losing_trade=0, avg_holding_period=pd.Timedelta(0)
        )

