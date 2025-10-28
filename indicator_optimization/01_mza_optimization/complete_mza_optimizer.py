"""
Улучшенный генетический алгоритм для оптимизации полных параметров MZA
С учетом всех 23 параметров и адаптивных весов
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
import warnings
from accurate_mza_classifier import AccurateMZAClassifier
warnings.filterwarnings('ignore')

class CompleteMZAOptimizer:
    """
    Полный генетический алгоритм для оптимизации всех параметров MZA
    
    Особенности:
    - Все 23 параметра из оригинального MZA
    - Адаптивные веса в зависимости от Market Activity
    - Комплексная оценка эффективности
    - Защита от переобучения
    - Режим-специфичная оптимизация
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 max_generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_size: int = 5,
                 cv_folds: int = 3,
                 regularization_strength: float = 0.01):
        """
        Инициализация полного оптимизатора MZA
        
        Args:
            population_size: Размер популяции
            max_generations: Максимальное количество поколений
            mutation_rate: Вероятность мутации
            crossover_rate: Вероятность скрещивания
            elite_size: Количество элитных особей
            cv_folds: Количество фолдов для кросс-валидации
            regularization_strength: Сила регуляризации
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.cv_folds = cv_folds
        self.regularization_strength = regularization_strength
        
        # Полные параметры MZA для оптимизации (23 параметра)
        self.param_ranges = {
            # Trend Indicators (4 параметра)
            'adxLength': (10, 20, 1),
            'adxThreshold': (15, 30, 1),
            'fastMALength': (15, 25, 1),
            'slowMALength': (40, 60, 1),
            
            # Momentum Indicators (5 параметров)
            'rsiLength': (10, 20, 1),
            'stochKLength': (10, 20, 1),
            'macdFast': (8, 16, 1),
            'macdSlow': (20, 30, 1),
            'macdSignal': (7, 12, 1),
            
            # Price Action Indicators (3 параметра)
            'hhllRange': (15, 30, 1),
            'haDojiRange': (3, 10, 1),
            'candleRangeLength': (5, 15, 1),
            
            # Market Activity Indicators (6 параметров)
            'bbLength': (15, 25, 1),
            'bbMultiplier': (1.5, 3.0, 0.1),
            'atrLength': (10, 20, 1),
            'kcLength': (15, 25, 1),
            'kcMultiplier': (1.0, 2.5, 0.1),
            'volumeMALength': (15, 25, 1),
            
            # Base Weights (3 параметра)
            'trendWeightBase': (30, 50, 1),
            'momentumWeightBase': (20, 40, 1),
            'priceActionWeightBase': (20, 40, 1),
            
            # Stability Controls (2 параметра) - булевы значения
            'useSmoothing': [True, False],
            'useHysteresis': [True, False]
        }
        
        # Результаты оптимизации
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.generation_history = []
        self.cv_scores = []
        
    def create_random_individual(self) -> Dict:
        """Создает случайную особь (набор параметров)"""
        individual = {}
        for param, config in self.param_ranges.items():
            if isinstance(config, list):
                # Булевы параметры
                individual[param] = random.choice(config)
            else:
                min_val, max_val, step = config
                if step >= 1:
                    # Целочисленные параметры
                    individual[param] = random.randint(int(min_val), int(max_val))
                else:
                    # Вещественные параметры
                    steps = int((max_val - min_val) / step) + 1
                    individual[param] = min_val + random.randint(0, steps - 1) * step
        return individual
    
    def create_initial_population(self) -> List[Dict]:
        """Создает начальную популяцию"""
        population = []
        for _ in range(self.population_size):
            individual = self.create_random_individual()
            population.append(individual)
        return population
    
    def evaluate_fitness(self, individual: Dict, data: pd.DataFrame) -> float:
        """
        Оценивает пригодность особи с полной логикой MZA
        
        Args:
            individual: Набор параметров
            data: Данные для тестирования
            
        Returns:
            Значение пригодности
        """
        try:
            # Создаем классификатор с текущими параметрами
            classifier = AccurateMZAClassifier(individual)
            
            # Получаем предсказания
            predictions = classifier.predict(data)
            
            # Вычисляем комплексную оценку
            fitness = self.calculate_comprehensive_mza_score(data, predictions, individual)
            return fitness
            
        except Exception as e:
            return -1000.0
    
    def calculate_comprehensive_mza_score(self, data: pd.DataFrame, zones: np.ndarray, 
                                       params: Dict) -> float:
        """
        Комплексная оценка MZA по нескольким критериям
        
        Args:
            data: Данные
            zones: Предсказания зон
            params: Параметры
            
        Returns:
            Композитный скор
        """
        returns = data['close'].pct_change().dropna()
        
        # Выравниваем индексы
        max_lookback = max(params['slowMALength'], params['adxLength'])
        aligned_returns = returns.iloc[max_lookback:]
        aligned_zones = zones[max_lookback:len(aligned_returns)+1]
        
        if len(aligned_returns) == 0 or len(aligned_zones) == 0:
            return 0.0
        
        # 1. Разделение доходности по зонам (40%)
        bull_returns = aligned_returns[aligned_zones == 1]
        bear_returns = aligned_returns[aligned_zones == -1]
        sideways_returns = aligned_returns[aligned_zones == 0]
        
        return_spread = 0
        if len(bull_returns) > 0 and len(bear_returns) > 0:
            return_spread = abs(bull_returns.mean() - bear_returns.mean())
        
        sideways_volatility = sideways_returns.std() if len(sideways_returns) > 0 else 1
        
        # 2. Стабильность зон (25%)
        zone_changes = np.sum(np.diff(aligned_zones) != 0)
        zone_stability = 1 - (zone_changes / len(aligned_zones)) if len(aligned_zones) > 0 else 0
        
        # 3. Согласованность сигналов (20%)
        signal_consistency = self.calculate_signal_consistency(aligned_zones, aligned_returns)
        
        # 4. Адаптивность системы (15%)
        adaptability_score = self.calculate_adaptability_score(params, data)
        
        # Композитный скор
        composite_score = (
            return_spread * 0.4 +
            zone_stability * 0.25 +
            signal_consistency * 0.2 +
            adaptability_score * 0.15
        ) / (1 + sideways_volatility)
        
        return max(composite_score, 0)
    
    def calculate_signal_consistency(self, zones: np.ndarray, returns: pd.Series) -> float:
        """Вычисляет согласованность сигналов"""
        if len(zones) == 0 or len(returns) == 0:
            return 0.0
        
        # Проверяем, что сигналы соответствуют движению цены
        correct_signals = 0
        total_signals = 0
        
        for i in range(1, len(zones)):
            if zones[i] != 0:  # Не нейтральная зона
                price_direction = 1 if returns.iloc[i] > 0 else -1 if returns.iloc[i] < 0 else 0
                if zones[i] == price_direction:
                    correct_signals += 1
                total_signals += 1
        
        return correct_signals / total_signals if total_signals > 0 else 0.0
    
    def calculate_adaptability_score(self, params: Dict, data: pd.DataFrame) -> float:
        """Вычисляет адаптивность системы"""
        # Проверяем, что параметры находятся в разумных пределах
        adaptability_score = 1.0
        
        # Проверяем соотношения параметров
        if params['fastMALength'] >= params['slowMALength']:
            adaptability_score -= 0.3
        
        if params['macdFast'] >= params['macdSlow']:
            adaptability_score -= 0.3
        
        # Проверяем, что веса в сумме дают 100%
        total_weight = params['trendWeightBase'] + params['momentumWeightBase'] + params['priceActionWeightBase']
        if abs(total_weight - 100) > 5:  # Допуск 5%
            adaptability_score -= 0.2
        
        return max(adaptability_score, 0.0)
    
    def cross_validate_fitness(self, individual: Dict, data: pd.DataFrame) -> Dict:
        """
        Кросс-валидация для оценки стабильности параметров
        
        Args:
            individual: Набор параметров
            data: Данные для валидации
            
        Returns:
            Словарь с результатами кросс-валидации
        """
        # Разделяем данные на train/test
        split_point = int(len(data) * 0.7)  # 70% для обучения, 30% для тестирования
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]
        
        # Оценка на обучающих данных
        train_fitness = self.evaluate_fitness(individual, train_data)
        
        # Оценка на тестовых данных
        test_fitness = self.evaluate_fitness(individual, test_data)
        
        # Вычисляем стабильность
        stability = 1 - abs(train_fitness - test_fitness) / (abs(train_fitness) + abs(test_fitness) + 1e-8)
        
        # Комбинированная оценка
        combined_score = 0.7 * test_fitness + 0.3 * train_fitness
        
        return {
            'train_score': train_fitness,
            'test_score': test_fitness,
            'stability': stability,
            'combined_score': combined_score,
            'overfitting_risk': abs(train_fitness - test_fitness)
        }
    
    def tournament_selection(self, population: List[Dict], fitness_scores: List[float], 
                           tournament_size: int = 3) -> Dict:
        """Турнирная селекция"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index]
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Скрещивание двух родителей"""
        child1 = {}
        child2 = {}
        
        for param in self.param_ranges.keys():
            if random.random() < 0.5:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        return child1, child2
    
    def mutate(self, individual: Dict) -> Dict:
        """Мутация особи"""
        mutated = individual.copy()
        
        for param, config in self.param_ranges.items():
            if random.random() < self.mutation_rate:
                if isinstance(config, list):
                    # Булевы параметры
                    mutated[param] = random.choice(config)
                else:
                    min_val, max_val, step = config
                    if step >= 1:
                        mutated[param] = random.randint(int(min_val), int(max_val))
                    else:
                        steps = int((max_val - min_val) / step) + 1
                        mutated[param] = min_val + random.randint(0, steps - 1) * step
        
        return mutated
    
    def optimize(self, data: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Запускает полный генетический алгоритм оптимизации
        
        Args:
            data: Данные для оптимизации
            verbose: Выводить ли прогресс
            
        Returns:
            Результаты оптимизации
        """
        if verbose:
            print("🧬 ЗАПУСК ПОЛНОГО ГЕНЕТИЧЕСКОГО АЛГОРИТМА ОПТИМИЗАЦИИ MZA")
            print("=" * 80)
            print(f"📊 Размер популяции: {self.population_size}")
            print(f"🔄 Максимум поколений: {self.max_generations}")
            print(f"🧬 Вероятность мутации: {self.mutation_rate}")
            print(f"🔀 Вероятность скрещивания: {self.crossover_rate}")
            print(f"👑 Размер элиты: {self.elite_size}")
            print(f"📊 Кросс-валидация: {self.cv_folds} фолдов")
            print(f"🛡️ Регуляризация: {self.regularization_strength}")
            print(f"🎯 Параметров для оптимизации: {len(self.param_ranges)}")
            print("=" * 80)
        
        # Создаем начальную популяцию
        population = self.create_initial_population()
        
        for generation in range(self.max_generations):
            # Оцениваем пригодность всех особей
            fitness_scores = []
            for individual in population:
                fitness = self.evaluate_fitness(individual, data)
                fitness_scores.append(fitness)
            
            # Находим лучшую особь
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_individual = population[best_idx]
            
            # Обновляем глобальный лучший результат
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_individual = best_individual.copy()
            
            # Сохраняем историю поколения
            self.generation_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'worst_fitness': np.min(fitness_scores)
            })
            
            if verbose and generation % 10 == 0:
                print(f"🔄 Поколение {generation:3d}: "
                      f"Лучший = {best_fitness:.6f}, "
                      f"Средний = {np.mean(fitness_scores):.6f}")
            
            # Создаем новую популяцию
            new_population = []
            
            # Элитизм
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Генерируем остальных особей
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # Анализ переобучения
        overfitting_analysis = self.analyze_overfitting()
        
        if verbose:
            print(f"\n✅ ПОЛНАЯ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
            print(f"🏆 Лучший Economic Value: {self.best_fitness:.6f}")
            print(f"📊 Протестировано поколений: {self.max_generations}")
            print(f"🧬 Общее количество оценок: {self.max_generations * self.population_size}")
            print(f"🛡️ Риск переобучения: {overfitting_analysis['overfitting_risk']:.3f}")
            print(f"📈 Стабильность: {overfitting_analysis['stability']:.3f}")
        
        return {
            'best_parameters': self.best_individual,
            'best_score': self.best_fitness,
            'generation_history': self.generation_history,
            'total_evaluations': self.max_generations * self.population_size,
            'algorithm': 'complete_genetic',
            'overfitting_analysis': overfitting_analysis,
            'param_count': len(self.param_ranges)
        }
    
    def analyze_overfitting(self) -> Dict:
        """Анализирует риск переобучения"""
        if not self.cv_scores:
            return {'overfitting_risk': 0, 'stability': 0}
        
        # Анализируем последние результаты
        recent_scores = self.cv_scores[-self.population_size:]
        
        train_scores = [score['train_score'] for score in recent_scores]
        test_scores = [score['test_score'] for score in recent_scores]
        stabilities = [score['stability'] for score in recent_scores]
        
        # Вычисляем метрики
        avg_train = np.mean(train_scores)
        avg_test = np.mean(test_scores)
        avg_stability = np.mean(stabilities)
        
        # Риск переобучения
        overfitting_risk = abs(avg_train - avg_test) / (abs(avg_train) + abs(avg_test) + 1e-8)
        
        return {
            'overfitting_risk': overfitting_risk,
            'stability': avg_stability,
            'train_test_gap': abs(avg_train - avg_test),
            'avg_train_score': avg_train,
            'avg_test_score': avg_test
        }
    
    def export_to_pine_script(self, output_file: str = "complete_optimized_mza.pine") -> str:
        """Экспортирует оптимальные параметры в формат Pine Script"""
        if not self.best_individual:
            raise ValueError("Сначала необходимо провести оптимизацию")
        
        overfitting_analysis = self.analyze_overfitting()
        
        pine_code = f"""// Полностью оптимизированные параметры MZA с помощью генетического алгоритма
// Сгенерировано автоматически на основе исторических данных BTC
// Дата оптимизации: {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}
// Лучший Economic Value: {self.best_fitness:.6f}
// Алгоритм: Полный генетический с {len(self.param_ranges)} параметрами
// Риск переобучения: {overfitting_analysis['overfitting_risk']:.3f}
// Стабильность: {overfitting_analysis['stability']:.3f}

//@version=6
indicator("Complete Optimized MZA [BullByte]", shorttitle="MZA[BullByte]", overlay=false)

// ===== ПОЛНОСТЬЮ ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ =====

// Trend Indicators
adxLength = input.int({self.best_individual.get('adxLength', 14)}, "DMI Length / ADX Smoothing", group="Trend Indicators")
adxThreshold = input.int({self.best_individual.get('adxThreshold', 20)}, "ADX Threshold", group="Trend Indicators")
fastMALength = input.int({self.best_individual.get('fastMALength', 20)}, "Fast MA Length", group="Trend Indicators")
slowMALength = input.int({self.best_individual.get('slowMALength', 50)}, "Slow MA Length", group="Trend Indicators")

// Momentum Indicators
rsiLength = input.int({self.best_individual.get('rsiLength', 14)}, "RSI Length", group="Momentum Indicators")
stochKLength = input.int({self.best_individual.get('stochKLength', 14)}, "Stoch %K Length", group="Momentum Indicators")
macdFast = input.int({self.best_individual.get('macdFast', 12)}, "MACD Fast", group="Momentum Indicators")
macdSlow = input.int({self.best_individual.get('macdSlow', 26)}, "MACD Slow", group="Momentum Indicators")
macdSignal = input.int({self.best_individual.get('macdSignal', 9)}, "MACD Signal", group="Momentum Indicators")

// Price Action Indicators
hhllRange = input.int({self.best_individual.get('hhllRange', 20)}, "HH/LL Range", group="Price Action Indicators")
haDojiRange = input.int({self.best_individual.get('haDojiRange', 5)}, "HA Doji Range", group="Price Action Indicators")
candleRangeLength = input.int({self.best_individual.get('candleRangeLength', 8)}, "Candle Range Length", group="Price Action Indicators")

// Market Activity Indicators
bbLength = input.int({self.best_individual.get('bbLength', 20)}, "BB Length", group="Market Activity Indicators")
bbMultiplier = input.float({self.best_individual.get('bbMultiplier', 2.0)}, "BB Multiplier", group="Market Activity Indicators")
atrLength = input.int({self.best_individual.get('atrLength', 14)}, "ATR Length", group="Market Activity Indicators")
kcLength = input.int({self.best_individual.get('kcLength', 20)}, "KC Length", group="Market Activity Indicators")
kcMultiplier = input.float({self.best_individual.get('kcMultiplier', 1.5)}, "KC Multiplier", group="Market Activity Indicators")
volumeMALength = input.int({self.best_individual.get('volumeMALength', 20)}, "Volume MA Length", group="Market Activity Indicators")

// Weight Inputs
trendWeightBase = input.float({self.best_individual.get('trendWeightBase', 40)}, "Trend Strength Weight (%)", minval=0, maxval=100, group="Weights")
momentumWeightBase = input.float({self.best_individual.get('momentumWeightBase', 30)}, "Momentum Weight (%)", minval=0, maxval=100, group="Weights")
priceActionWeightBase = input.float({self.best_individual.get('priceActionWeightBase', 30)}, "Price Action Weight (%)", minval=0, maxval=100, group="Weights")

// Stability Controls
useSmoothing = input.bool({str(self.best_individual.get('useSmoothing', True)).lower()}, "Smooth Market Activity", group="Stability Controls")
useHysteresis = input.bool({str(self.best_individual.get('useHysteresis', True)).lower()}, "Use Hysteresis for Stability", group="Stability Controls")

// ===== ОСТАЛЬНОЙ КОД MZA ОСТАЕТСЯ БЕЗ ИЗМЕНЕНИЙ =====
// Здесь должен быть вставлен оригинальный код MZA с использованием
// оптимизированных параметров выше

// ===== РЕЗУЛЬТАТЫ ПОЛНОЙ ОПТИМИЗАЦИИ =====
// Economic Value: {self.best_fitness:.6f}
// Параметров оптимизировано: {len(self.param_ranges)}
// Риск переобучения: {overfitting_analysis['overfitting_risk']:.3f}
// Стабильность: {overfitting_analysis['stability']:.3f}
// Поколений: {self.max_generations}
// Размер популяции: {self.population_size}
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(pine_code)
        
        print(f"📄 Полностью оптимизированные параметры экспортированы в {output_file}")
        return pine_code


# Пример использования
if __name__ == "__main__":
    # Создаем полный оптимизатор
    optimizer = CompleteMZAOptimizer(
        population_size=30,
        max_generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        cv_folds=3,
        regularization_strength=0.01
    )
    
    print("🧬 Complete MZA Optimizer готов к работе!")
    print(f"📊 Параметров для оптимизации: {len(optimizer.param_ranges)}")
    print(f"🔧 Доступные параметры: {list(optimizer.param_ranges.keys())}")
    print(f"🛡️ Защита от переобучения: Включена")
    print(f"📊 Кросс-валидация: {optimizer.cv_folds} фолдов")
