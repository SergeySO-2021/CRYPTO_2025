"""
Система оптимизации параметров MZA для TradingView
Находит оптимальные параметры в Python, экспортирует для Pine Script
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import itertools
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class MZAParameterOptimizer:
    """
    Оптимизатор параметров MZA для TradingView
    
    Особенности:
    - Оптимизирует только ключевые параметры MZA
    - Использует исторические данные BTC
    - Экспортирует результаты в формат Pine Script
    - Минимальные изменения исходного кода MZA
    """
    
    def __init__(self):
        # Ключевые параметры MZA для оптимизации
        self.optimizable_params = {
            # Trend Indicators
            'adx_length': {'min': 10, 'max': 20, 'step': 1, 'default': 14},
            'adx_threshold': {'min': 15, 'max': 30, 'step': 1, 'default': 20},
            'fast_ma_length': {'min': 15, 'max': 25, 'step': 1, 'default': 20},
            'slow_ma_length': {'min': 40, 'max': 60, 'step': 1, 'default': 50},
            
            # Momentum Indicators
            'rsi_length': {'min': 10, 'max': 20, 'step': 1, 'default': 14},
            'rsi_overbought': {'min': 65, 'max': 80, 'step': 1, 'default': 70},
            'rsi_oversold': {'min': 20, 'max': 35, 'step': 1, 'default': 30},
            'stoch_length': {'min': 10, 'max': 20, 'step': 1, 'default': 14},
            'macd_fast': {'min': 8, 'max': 16, 'step': 1, 'default': 12},
            'macd_slow': {'min': 20, 'max': 30, 'step': 1, 'default': 26},
            
            # Price Action Indicators
            'bb_length': {'min': 15, 'max': 25, 'step': 1, 'default': 20},
            'bb_std': {'min': 1.5, 'max': 3.0, 'step': 0.1, 'default': 2.0},
            'atr_length': {'min': 10, 'max': 20, 'step': 1, 'default': 14},
            
            # Dynamic Weights (для адаптации)
            'trend_weight_high_vol': {'min': 0.4, 'max': 0.6, 'step': 0.05, 'default': 0.5},
            'momentum_weight_high_vol': {'min': 0.3, 'max': 0.4, 'step': 0.05, 'default': 0.35},
            'price_action_weight_low_vol': {'min': 0.4, 'max': 0.6, 'step': 0.05, 'default': 0.45}
        }
        
        # Результаты оптимизации
        self.optimization_results = {}
        self.best_parameters = {}
        
    def generate_parameter_combinations(self, max_combinations: int = 1000) -> List[Dict]:
        """
        Генерирует комбинации параметров для тестирования
        
        Args:
            max_combinations: Максимальное количество комбинаций
            
        Returns:
            Список словарей с параметрами
        """
        # Создаем списки значений для каждого параметра
        param_values = {}
        for param, config in self.optimizable_params.items():
            values = np.arange(config['min'], config['max'] + config['step'], config['step'])
            param_values[param] = values
        
        # Генерируем комбинации
        combinations = []
        
        # Если комбинаций слишком много, используем случайную выборку
        total_combinations = np.prod([len(values) for values in param_values.values()])
        
        if total_combinations <= max_combinations:
            # Генерируем все комбинации
            for combination in itertools.product(*param_values.values()):
                param_dict = dict(zip(param_values.keys(), combination))
                combinations.append(param_dict)
        else:
            # Случайная выборка
            for _ in range(max_combinations):
                param_dict = {}
                for param, values in param_values.items():
                    param_dict[param] = np.random.choice(values)
                combinations.append(param_dict)
        
        print(f"📊 Сгенерировано {len(combinations)} комбинаций параметров")
        return combinations
    
    def test_parameter_combination(self, params: Dict, data: pd.DataFrame) -> Dict:
        """
        Тестирует комбинацию параметров на данных
        
        Args:
            params: Словарь с параметрами
            data: Данные для тестирования
            
        Returns:
            Словарь с результатами тестирования
        """
        try:
            # Создаем MZA классификатор с тестовыми параметрами
            mza_classifier = self.create_test_mza_classifier(params)
            
            # Тестируем на данных
            predictions = mza_classifier.fit_predict(data)
            
            # Вычисляем метрики качества
            metrics = self.calculate_quality_metrics(data, predictions)
            
            return {
                'parameters': params,
                'metrics': metrics,
                'success': True
            }
            
        except Exception as e:
            return {
                'parameters': params,
                'error': str(e),
                'success': False
            }
    
    def create_test_mza_classifier(self, params: Dict):
        """Создает MZA классификатор с тестовыми параметрами"""
        # Импортируем VectorizedMZAClassifier
        import sys
        from pathlib import Path
        
        # Добавляем пути к классификаторам
        project_root = Path(__file__).parent.parent.parent
        classifiers_path = project_root / 'compare_analyze_indicators' / 'classifiers'
        sys.path.insert(0, str(classifiers_path))
        
        try:
            from mza_classifier_vectorized import VectorizedMZAClassifier
            print("✅ VectorizedMZAClassifier загружен успешно")
            return VectorizedMZAClassifier(parameters=params)
        except ImportError as e:
            print(f"❌ Не удалось загрузить VectorizedMZAClassifier: {e}")
            print("🔄 Пробуем загрузить из base_optimization_system...")
            
            # Пробуем импортировать из base_optimization_system
            try:
                sys.path.insert(0, str(project_root / 'indicator_optimization'))
                from base_optimization_system import VectorizedMZAClassifier
                print("✅ VectorizedMZAClassifier загружен из base_optimization_system")
                return VectorizedMZAClassifier(parameters=params)
            except ImportError as e2:
                print(f"❌ Не удалось загрузить VectorizedMZAClassifier: {e2}")
                raise ImportError("Не удалось загрузить VectorizedMZAClassifier ни из одного источника")
    
    def create_simple_mza_classifier(self, params: Dict):
        """Создает упрощенную версию MZA для тестирования"""
        class SimpleMZAClassifier:
            def __init__(self, parameters):
                self.params = parameters
                
            def fit_predict(self, data):
                # Упрощенная логика MZA
                predictions = np.zeros(len(data))
                
                # Простая логика на основе RSI и MA
                rsi_length = self.params.get('rsi_length', 14)
                rsi_overbought = self.params.get('rsi_overbought', 70)
                rsi_oversold = self.params.get('rsi_oversold', 30)
                
                # Вычисляем RSI
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_length).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_length).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                # Простая классификация
                for i in range(len(data)):
                    if i < rsi_length:
                        predictions[i] = 0  # Недостаточно данных
                    elif rsi.iloc[i] > rsi_overbought:
                        predictions[i] = -1  # Bear
                    elif rsi.iloc[i] < rsi_oversold:
                        predictions[i] = 1   # Bull
                    else:
                        predictions[i] = 0   # Sideways
                
                return predictions
        
        return SimpleMZAClassifier(params)
    
    def calculate_quality_metrics(self, data: pd.DataFrame, predictions: np.ndarray) -> Dict:
        """
        Вычисляет метрики качества классификации
        
        Args:
            data: Данные
            predictions: Предсказания классификатора
            
        Returns:
            Словарь с метриками
        """
        # Вычисляем доходность
        returns = data['close'].pct_change().dropna()
        
        # Выравниваем индексы
        aligned_returns = returns.iloc[1:]
        aligned_predictions = predictions[1:len(aligned_returns)+1]
        
        # Разделяем по предсказаниям
        bull_mask = aligned_predictions == 1
        bear_mask = aligned_predictions == -1
        sideways_mask = aligned_predictions == 0
        
        # Метрики доходности
        bull_returns = aligned_returns[bull_mask].mean() if bull_mask.any() else 0
        bear_returns = aligned_returns[bear_mask].mean() if bear_mask.any() else 0
        sideways_returns = aligned_returns[sideways_mask].mean() if sideways_mask.any() else 0
        
        # Economic Value
        return_spread = abs(bull_returns - bear_returns)
        sideways_vol = aligned_returns[sideways_mask].std() if sideways_mask.any() else 0
        economic_value = return_spread / (1 + sideways_vol) if sideways_vol > 0 else return_spread
        
        # Стабильность зон
        zone_changes = np.sum(np.diff(aligned_predictions) != 0)
        zone_stability = 1 - (zone_changes / len(aligned_predictions))
        
        # Распределение зон
        zone_distribution = {
            'bull': np.sum(bull_mask),
            'bear': np.sum(bear_mask),
            'sideways': np.sum(sideways_mask)
        }
        
        return {
            'economic_value': economic_value,
            'return_spread': return_spread,
            'zone_stability': zone_stability,
            'zone_distribution': zone_distribution,
            'bull_returns': bull_returns,
            'bear_returns': bear_returns,
            'sideways_returns': sideways_returns
        }
    
    def optimize_parameters(self, data: pd.DataFrame, timeframes: List[str] = None) -> Dict:
        """
        Оптимизирует параметры MZA на данных
        
        Args:
            data: Данные для оптимизации
            timeframes: Список таймфреймов для тестирования
            
        Returns:
            Словарь с результатами оптимизации
        """
        print("🚀 НАЧИНАЕМ ОПТИМИЗАЦИЮ ПАРАМЕТРОВ MZA")
        print("=" * 50)
        
        # Генерируем комбинации параметров
        combinations = self.generate_parameter_combinations(max_combinations=500)
        
        best_score = -float('inf')
        best_params = None
        results = []
        
        print(f"📊 Тестируем {len(combinations)} комбинаций параметров...")
        
        for i, params in enumerate(combinations):
            if i % 50 == 0:
                print(f"🔄 Прогресс: {i}/{len(combinations)} ({i/len(combinations)*100:.1f}%)")
            
            # Тестируем комбинацию
            result = self.test_parameter_combination(params, data)
            
            if result['success']:
                score = result['metrics']['economic_value']
                results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"🎯 Новый лучший результат: {score:.6f}")
        
        # Сохраняем результаты
        self.optimization_results = {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': results,
            'total_tested': len(combinations),
            'successful_tests': len(results)
        }
        
        print(f"\n✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print(f"📊 Протестировано: {len(combinations)} комбинаций")
        print(f"✅ Успешных тестов: {len(results)}")
        print(f"🏆 Лучший Economic Value: {best_score:.6f}")
        
        return self.optimization_results
    
    def export_to_pine_script(self, output_file: str = "optimized_mza_params.pine") -> str:
        """
        Экспортирует оптимальные параметры в формат Pine Script
        
        Args:
            output_file: Имя файла для экспорта
            
        Returns:
            Строка с кодом Pine Script
        """
        if not self.optimization_results:
            raise ValueError("Сначала необходимо провести оптимизацию")
        
        best_params = self.optimization_results['best_parameters']
        
        # Создаем код Pine Script
        pine_code = f"""// Оптимизированные параметры MZA для TradingView
// Сгенерировано автоматически на основе исторических данных BTC
// Дата оптимизации: {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}
// Лучший Economic Value: {self.optimization_results['best_score']:.6f}

//@version=5
indicator("Optimized MZA [BullByte]", overlay=true)

// ===== ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ =====

// Trend Indicators
adx_length = input.int({best_params.get('adx_length', 14)}, "ADX Length", minval=10, maxval=20)
adx_threshold = input.int({best_params.get('adx_threshold', 20)}, "ADX Threshold", minval=15, maxval=30)
fast_ma_length = input.int({best_params.get('fast_ma_length', 20)}, "Fast MA Length", minval=15, maxval=25)
slow_ma_length = input.int({best_params.get('slow_ma_length', 50)}, "Slow MA Length", minval=40, maxval=60)

// Momentum Indicators
rsi_length = input.int({best_params.get('rsi_length', 14)}, "RSI Length", minval=10, maxval=20)
rsi_overbought = input.int({best_params.get('rsi_overbought', 70)}, "RSI Overbought", minval=65, maxval=80)
rsi_oversold = input.int({best_params.get('rsi_oversold', 30)}, "RSI Oversold", minval=20, maxval=35)
stoch_length = input.int({best_params.get('stoch_length', 14)}, "Stochastic Length", minval=10, maxval=20)
macd_fast = input.int({best_params.get('macd_fast', 12)}, "MACD Fast", minval=8, maxval=16)
macd_slow = input.int({best_params.get('macd_slow', 26)}, "MACD Slow", minval=20, maxval=30)

// Price Action Indicators
bb_length = input.int({best_params.get('bb_length', 20)}, "Bollinger Bands Length", minval=15, maxval=25)
bb_std = input.float({best_params.get('bb_std', 2.0)}, "Bollinger Bands Std Dev", minval=1.5, maxval=3.0)
atr_length = input.int({best_params.get('atr_length', 14)}, "ATR Length", minval=10, maxval=20)

// Dynamic Weights
trend_weight_high_vol = input.float({best_params.get('trend_weight_high_vol', 0.5)}, "Trend Weight (High Vol)", minval=0.4, maxval=0.6)
momentum_weight_high_vol = input.float({best_params.get('momentum_weight_high_vol', 0.35)}, "Momentum Weight (High Vol)", minval=0.3, maxval=0.4)
price_action_weight_low_vol = input.float({best_params.get('price_action_weight_low_vol', 0.45)}, "Price Action Weight (Low Vol)", minval=0.4, maxval=0.6)

// ===== ОСТАЛЬНОЙ КОД MZA ОСТАЕТСЯ БЕЗ ИЗМЕНЕНИЙ =====
// Здесь должен быть вставлен оригинальный код MZA с использованием
// оптимизированных параметров выше

// Пример использования параметров:
// rsi_value = ta.rsi(close, rsi_length)
// adx_value = ta.adx(high, low, close, adx_length)
// bb_upper = ta.sma(close, bb_length) + bb_std * ta.stdev(close, bb_length)
// bb_lower = ta.sma(close, bb_length) - bb_std * ta.stdev(close, bb_length)

// ===== РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ =====
// Economic Value: {self.optimization_results['best_score']:.6f}
// Протестировано комбинаций: {self.optimization_results['total_tested']}
// Успешных тестов: {self.optimization_results['successful_tests']}
"""
        
        # Сохраняем в файл
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(pine_code)
        
        print(f"📄 Оптимизированные параметры экспортированы в {output_file}")
        return pine_code
    
    def create_optimization_report(self) -> str:
        """Создает отчет об оптимизации"""
        if not self.optimization_results:
            return "Оптимизация не проводилась"
        
        report = f"""
# 📊 ОТЧЕТ ОБ ОПТИМИЗАЦИИ MZA ПАРАМЕТРОВ

## 🎯 Результаты оптимизации:
- **Лучший Economic Value:** {self.optimization_results['best_score']:.6f}
- **Протестировано комбинаций:** {self.optimization_results['total_tested']}
- **Успешных тестов:** {self.optimization_results['successful_tests']}
- **Дата оптимизации:** {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}

## 🏆 Оптимальные параметры:
"""
        
        for param, value in self.optimization_results['best_parameters'].items():
            report += f"- **{param}:** {value}\n"
        
        # Топ-5 результатов
        if len(self.optimization_results['all_results']) >= 5:
            report += "\n## 📈 Топ-5 результатов:\n"
            sorted_results = sorted(
                self.optimization_results['all_results'], 
                key=lambda x: x['metrics']['economic_value'], 
                reverse=True
            )[:5]
            
            for i, result in enumerate(sorted_results, 1):
                ev = result['metrics']['economic_value']
                report += f"{i}. Economic Value: {ev:.6f}\n"
        
        return report


# Пример использования
if __name__ == "__main__":
    # Создаем оптимизатор
    optimizer = MZAParameterOptimizer()
    
    # Загружаем данные BTC (пример)
    # data = pd.read_csv('df_btc_1h.csv')
    
    # Оптимизируем параметры
    # results = optimizer.optimize_parameters(data)
    
    # Экспортируем в Pine Script
    # pine_code = optimizer.export_to_pine_script()
    
    print("🎯 MZA Parameter Optimizer готов к работе!")
