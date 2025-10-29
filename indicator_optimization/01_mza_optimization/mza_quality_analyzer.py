"""
🔍 АНАЛИЗАТОР КАЧЕСТВА MZA
Детальный анализ качества определения рыночных зон
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class MZAQualityAnalyzer:
    """
    Анализатор качества определения рыночных зон MZA
    
    Метрики:
    1. Economic Value - разделение доходности по зонам
    2. Zone Stability - стабильность зон
    3. Signal Consistency - точность сигналов
    4. Adaptability Score - разумность параметров
    """
    
    def __init__(self):
        self.results = {}
        
    def analyze_mza_quality(self, data: pd.DataFrame, zones: np.ndarray, 
                           params: Dict, timeframe: str = "15m") -> Dict:
        """
        Полный анализ качества MZA
        
        Args:
            data: Данные OHLCV
            zones: Предсказания зон (-1, 0, 1)
            params: Параметры MZA
            timeframe: Таймфрейм для анализа
            
        Returns:
            Словарь с метриками качества
        """
        print(f"🔍 АНАЛИЗ КАЧЕСТВА MZA ДЛЯ {timeframe}")
        print("=" * 50)
        
        # Вычисляем доходность
        returns = data['close'].pct_change().dropna()
        
        # Выравниваем индексы
        max_lookback = max(params.get('slowMALength', 50), params.get('adxLength', 14))
        aligned_returns = returns.iloc[max_lookback:]
        aligned_zones = zones[max_lookback:len(aligned_returns)+1]
        
        if len(aligned_returns) == 0 or len(aligned_zones) == 0:
            print("❌ Недостаточно данных для анализа")
            return {}
        
        # 1. ECONOMIC VALUE (40%)
        economic_value = self._calculate_economic_value(aligned_returns, aligned_zones)
        
        # 2. ZONE STABILITY (25%)
        zone_stability = self._calculate_zone_stability(aligned_zones)
        
        # 3. SIGNAL CONSISTENCY (20%)
        signal_consistency = self._calculate_signal_consistency(aligned_zones, aligned_returns)
        
        # 4. ADAPTABILITY SCORE (15%)
        adaptability_score = self._calculate_adaptability_score(params)
        
        # Композитный скор
        composite_score = (
            economic_value * 0.4 +
            zone_stability * 0.25 +
            signal_consistency * 0.2 +
            adaptability_score * 0.15
        )
        
        # Дополнительные метрики
        zone_distribution = self._calculate_zone_distribution(aligned_zones)
        performance_by_zone = self._calculate_performance_by_zone(aligned_returns, aligned_zones)
        
        results = {
            'timeframe': timeframe,
            'economic_value': economic_value,
            'zone_stability': zone_stability,
            'signal_consistency': signal_consistency,
            'adaptability_score': adaptability_score,
            'composite_score': composite_score,
            'zone_distribution': zone_distribution,
            'performance_by_zone': performance_by_zone,
            'total_periods': len(aligned_zones),
            'zone_changes': np.sum(np.diff(aligned_zones) != 0)
        }
        
        self._print_analysis_results(results)
        return results
    
    def _calculate_economic_value(self, returns: pd.Series, zones: np.ndarray) -> float:
        """Вычисляет Economic Value"""
        bull_returns = returns[zones == 1]
        bear_returns = returns[zones == -1]
        sideways_returns = returns[zones == 0]
        
        if len(bull_returns) == 0 or len(bear_returns) == 0:
            return 0.0
        
        return_spread = abs(bull_returns.mean() - bear_returns.mean())
        sideways_volatility = sideways_returns.std() if len(sideways_returns) > 0 else 1
        
        economic_value = return_spread / (1 + sideways_volatility)
        return max(economic_value, 0)
    
    def _calculate_zone_stability(self, zones: np.ndarray) -> float:
        """Вычисляет стабильность зон"""
        if len(zones) == 0:
            return 0.0
        
        zone_changes = np.sum(np.diff(zones) != 0)
        stability = 1 - (zone_changes / len(zones))
        return max(stability, 0)
    
    def _calculate_signal_consistency(self, zones: np.ndarray, returns: pd.Series) -> float:
        """Вычисляет согласованность сигналов"""
        if len(zones) == 0 or len(returns) == 0:
            return 0.0
        
        correct_signals = 0
        total_signals = 0
        
        for i in range(1, len(zones)):
            if zones[i] != 0:  # Не нейтральная зона
                price_direction = 1 if returns.iloc[i] > 0 else -1 if returns.iloc[i] < 0 else 0
                if zones[i] == price_direction:
                    correct_signals += 1
                total_signals += 1
        
        return correct_signals / total_signals if total_signals > 0 else 0.0
    
    def _calculate_adaptability_score(self, params: Dict) -> float:
        """Вычисляет адаптивность системы"""
        score = 1.0
        
        # Проверяем соотношения параметров
        if params.get('fastMALength', 20) >= params.get('slowMALength', 50):
            score -= 0.3
        
        if params.get('macdFast', 12) >= params.get('macdSlow', 26):
            score -= 0.3
        
        # Проверяем веса
        total_weight = (params.get('trendWeightBase', 40) + 
                       params.get('momentumWeightBase', 30) + 
                       params.get('priceActionWeightBase', 30))
        
        if abs(total_weight - 100) > 5:  # Допуск 5%
            score -= 0.2
        
        return max(score, 0)
    
    def _calculate_zone_distribution(self, zones: np.ndarray) -> Dict:
        """Вычисляет распределение зон"""
        total = len(zones)
        if total == 0:
            return {'bull': 0, 'bear': 0, 'sideways': 0}
        
        return {
            'bull': np.sum(zones == 1) / total,
            'bear': np.sum(zones == -1) / total,
            'sideways': np.sum(zones == 0) / total
        }
    
    def _calculate_performance_by_zone(self, returns: pd.Series, zones: np.ndarray) -> Dict:
        """Вычисляет производительность по зонам"""
        bull_returns = returns[zones == 1]
        bear_returns = returns[zones == -1]
        sideways_returns = returns[zones == 0]
        
        return {
            'bull_mean': bull_returns.mean() if len(bull_returns) > 0 else 0,
            'bull_std': bull_returns.std() if len(bull_returns) > 0 else 0,
            'bear_mean': bear_returns.mean() if len(bear_returns) > 0 else 0,
            'bear_std': bear_returns.std() if len(bear_returns) > 0 else 0,
            'sideways_mean': sideways_returns.mean() if len(sideways_returns) > 0 else 0,
            'sideways_std': sideways_returns.std() if len(sideways_returns) > 0 else 0
        }
    
    def _print_analysis_results(self, results: Dict):
        """Выводит результаты анализа"""
        print(f"\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print(f"   🎯 Economic Value: {results['economic_value']:.6f}")
        print(f"   🔄 Zone Stability: {results['zone_stability']:.3f}")
        print(f"   ✅ Signal Consistency: {results['signal_consistency']:.3f}")
        print(f"   🧠 Adaptability Score: {results['adaptability_score']:.3f}")
        print(f"   🏆 Composite Score: {results['composite_score']:.6f}")
        
        print(f"\n📈 РАСПРЕДЕЛЕНИЕ ЗОН:")
        dist = results['zone_distribution']
        print(f"   🟢 Bull: {dist['bull']:.1%}")
        print(f"   🔴 Bear: {dist['bear']:.1%}")
        print(f"   ⚪ Sideways: {dist['sideways']:.1%}")
        
        print(f"\n💰 ПРОИЗВОДИТЕЛЬНОСТЬ ПО ЗОНАМ:")
        perf = results['performance_by_zone']
        print(f"   🟢 Bull: {perf['bull_mean']:.4f} ± {perf['bull_std']:.4f}")
        print(f"   🔴 Bear: {perf['bear_mean']:.4f} ± {perf['bear_std']:.4f}")
        print(f"   ⚪ Sideways: {perf['sideways_mean']:.4f} ± {perf['sideways_std']:.4f}")
        
        print(f"\n📊 СТАТИСТИКА:")
        print(f"   📅 Всего периодов: {results['total_periods']}")
        print(f"   🔄 Смен зон: {results['zone_changes']}")
        print(f"   📊 Частота смен: {results['zone_changes']/results['total_periods']:.1%}")
        
        # Оценка качества
        self._evaluate_quality(results)
    
    def _evaluate_quality(self, results: Dict):
        """Оценивает качество MZA"""
        print(f"\n🎯 ОЦЕНКА КАЧЕСТВА:")
        
        # Economic Value
        ev = results['economic_value']
        if ev > 0.01:
            ev_grade = "🟢 ОТЛИЧНО"
        elif ev > 0.005:
            ev_grade = "🟡 ХОРОШО"
        elif ev > 0.001:
            ev_grade = "🟠 УДОВЛЕТВОРИТЕЛЬНО"
        else:
            ev_grade = "🔴 ПЛОХО"
        print(f"   Economic Value: {ev_grade}")
        
        # Zone Stability
        zs = results['zone_stability']
        if zs > 0.8:
            zs_grade = "🟢 ОЧЕНЬ СТАБИЛЬНО"
        elif zs > 0.6:
            zs_grade = "🟡 СТАБИЛЬНО"
        elif zs > 0.4:
            zs_grade = "🟠 НЕСТАБИЛЬНО"
        else:
            zs_grade = "🔴 ОЧЕНЬ НЕСТАБИЛЬНО"
        print(f"   Zone Stability: {zs_grade}")
        
        # Signal Consistency
        sc = results['signal_consistency']
        if sc > 0.6:
            sc_grade = "🟢 ВЫСОКАЯ ТОЧНОСТЬ"
        elif sc > 0.5:
            sc_grade = "🟡 СРЕДНЯЯ ТОЧНОСТЬ"
        else:
            sc_grade = "🔴 НИЗКАЯ ТОЧНОСТЬ"
        print(f"   Signal Consistency: {sc_grade}")
        
        # Общая оценка
        composite = results['composite_score']
        if composite > 0.01:
            overall_grade = "🟢 ОТЛИЧНОЕ КАЧЕСТВО"
        elif composite > 0.005:
            overall_grade = "🟡 ХОРОШЕЕ КАЧЕСТВО"
        elif composite > 0.001:
            overall_grade = "🟠 УДОВЛЕТВОРИТЕЛЬНОЕ КАЧЕСТВО"
        else:
            overall_grade = "🔴 ПЛОХОЕ КАЧЕСТВО"
        print(f"\n🏆 ОБЩАЯ ОЦЕНКА: {overall_grade}")
    
    def compare_timeframes(self, results_list: List[Dict]):
        """Сравнивает результаты по таймфреймам"""
        print(f"\n📊 СРАВНЕНИЕ ПО ТАЙМФРЕЙМАМ:")
        print("=" * 60)
        
        # Создаем DataFrame для сравнения
        df = pd.DataFrame(results_list)
        
        print(f"{'Таймфрейм':<8} {'Economic Value':<15} {'Zone Stability':<15} {'Signal Cons':<12} {'Composite':<12}")
        print("-" * 70)
        
        for _, row in df.iterrows():
            print(f"{row['timeframe']:<8} {row['economic_value']:<15.6f} {row['zone_stability']:<15.3f} "
                  f"{row['signal_consistency']:<12.3f} {row['composite_score']:<12.6f}")
        
        # Находим лучший таймфрейм
        best_idx = df['composite_score'].idxmax()
        best_timeframe = df.loc[best_idx, 'timeframe']
        best_score = df.loc[best_idx, 'composite_score']
        
        print(f"\n🏆 ЛУЧШИЙ ТАЙМФРЕЙМ: {best_timeframe} (Composite Score: {best_score:.6f})")
        
        return df

def analyze_mza_quality_demo():
    """Демонстрация анализа качества MZA"""
    print("🔍 ДЕМОНСТРАЦИЯ АНАЛИЗАТОРА КАЧЕСТВА MZA")
    print("=" * 50)
    
    # Создаем тестовые данные
    np.random.seed(42)
    n_periods = 1000
    
    # Генерируем синтетические данные
    data = pd.DataFrame({
        'open': np.random.randn(n_periods).cumsum() + 50000,
        'high': np.random.randn(n_periods).cumsum() + 50000 + np.abs(np.random.randn(n_periods)),
        'low': np.random.randn(n_periods).cumsum() + 50000 - np.abs(np.random.randn(n_periods)),
        'close': np.random.randn(n_periods).cumsum() + 50000,
        'volume': np.random.randint(1000, 10000, n_periods)
    })
    
    # Генерируем синтетические зоны
    zones = np.random.choice([-1, 0, 1], n_periods, p=[0.3, 0.4, 0.3])
    
    # Параметры MZA
    params = {
        'adxLength': 14,
        'adxThreshold': 20,
        'fastMALength': 20,
        'slowMALength': 50,
        'trendWeightBase': 40,
        'momentumWeightBase': 30,
        'priceActionWeightBase': 30
    }
    
    # Анализируем качество
    analyzer = MZAQualityAnalyzer()
    results = analyzer.analyze_mza_quality(data, zones, params, "15m")
    
    return results

if __name__ == "__main__":
    analyze_mza_quality_demo()
