# 🚀 МОДУЛЬ ОПТИМИЗАЦИИ MZA С ГЕНЕТИЧЕСКИМ АЛГОРИТМОМ
# ==================================================

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from data_loader import DataManager
import warnings
warnings.filterwarnings('ignore')

class MZAOptimizer:
    """
    Оптимизатор MZA с генетическим алгоритмом
    """
    
    def __init__(self, 
                 population_size: int = 30,
                 max_generations: int = 50,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.8,
                 elite_size: int = 5,
                 cv_folds: int = 3,
                 regularization: float = 0.01):
        """
        Инициализация оптимизатора
        
        Args:
            population_size: Размер популяции
            max_generations: Максимальное количество поколений
            mutation_rate: Вероятность мутации
            crossover_rate: Вероятность кроссовера
            elite_size: Размер элиты
            cv_folds: Количество фолдов для кросс-валидации
            regularization: Коэффициент регуляризации
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.cv_folds = cv_folds
        self.regularization = regularization
        
        # Параметры MZA для оптимизации
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
            
            # Stability Controls (2 параметра)
            'useSmoothing': [True, False],
            'useHysteresis': [True, False]
        }
        
        self.best_score = -float('inf')
        self.best_parameters = None
        self.generation_history = []
        
    def create_random_individual(self) -> Dict:
        """
        Создание случайной особи (набора параметров)
        
        Returns:
            Словарь с параметрами
        """
        individual = {}
        
        for param, param_range in self.param_ranges.items():
            # Проверяем тип параметра
            if isinstance(param_range, list):
                # Булевые параметры представлены как список
                individual[param] = random.choice(param_range)
            else:
                # Числовые параметры представлены как кортеж (min, max, step)
                min_val, max_val, step = param_range
                if isinstance(min_val, int):
                    individual[param] = random.randint(min_val, max_val)
                else:
                    individual[param] = round(random.uniform(min_val, max_val), 1)
        
        return individual
    
    def calculate_fitness(self, individual: Dict, data: pd.DataFrame) -> float:
        """
        Расчет фитнеса особи
        
        Args:
            individual: Параметры особи
            data: Данные для тестирования
            
        Returns:
            Значение фитнеса
        """
        try:
            # Импортируем классификатор
            from accurate_mza_classifier import AccurateMZAClassifier
            
            # Создаем классификатор
            classifier = AccurateMZAClassifier(individual)
            
            # Получаем предсказания
            predictions = classifier.predict(data)
            
            # Рассчитываем Economic Value
            returns = data['close'].pct_change().dropna()
            
            # Выравниваем индексы
            min_length = min(len(returns), len(predictions))
            returns = returns.iloc[:min_length]
            predictions = predictions[:min_length]
            
            # Разделяем доходность по зонам
            bull_returns = returns[predictions == 1]
            bear_returns = returns[predictions == -1]
            sideways_returns = returns[predictions == 0]
            
            # Рассчитываем метрики
            if len(bull_returns) > 0 and len(bear_returns) > 0:
                return_spread = abs(bull_returns.mean() - bear_returns.mean())
            else:
                return_spread = 0
            
            if len(sideways_returns) > 0:
                sideways_volatility = sideways_returns.std()
            else:
                sideways_volatility = 1
            
            # Стабильность зон
            zone_changes = np.sum(np.diff(predictions) != 0)
            zone_stability = 1 - (zone_changes / len(predictions)) if len(predictions) > 1 else 1
            
            # Композитный скор
            fitness = (return_spread * 0.4 + zone_stability * 0.6) / (1 + sideways_volatility)
            
            # Применяем регуляризацию
            fitness = fitness - self.regularization * len([p for p in individual.values() if p > 50])
            
            return max(fitness, 0)
            
        except Exception as e:
            return -1000.0
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        Кроссовер двух родителей
        
        Args:
            parent1: Первый родитель
            parent2: Второй родитель
            
        Returns:
            Два потомка
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for param in self.param_ranges.keys():
            if random.random() < 0.5:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        return child1, child2
    
    def mutate(self, individual: Dict) -> Dict:
        """
        Мутация особи
        
        Args:
            individual: Особая для мутации
            
        Returns:
            Мутированная особь
        """
        mutated = individual.copy()
        
        for param, param_range in self.param_ranges.items():
            if random.random() < self.mutation_rate:
                if isinstance(param_range, list):
                    # Булевые параметры
                    mutated[param] = random.choice(param_range)
                else:
                    # Числовые параметры
                    min_val, max_val, step = param_range
                    if isinstance(min_val, int):
                        mutated[param] = random.randint(min_val, max_val)
                    else:
                        mutated[param] = round(random.uniform(min_val, max_val), 1)
        
        return mutated
    
    def select_parents(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """
        Селекция родителей для следующего поколения
        
        Args:
            population: Популяция
            fitness_scores: Оценки фитнеса
            
        Returns:
            Список родителей
        """
        # Турнирная селекция
        parents = []
        
        for _ in range(self.population_size):
            # Выбираем случайных особей для турнира
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Выбираем лучшего из турнира
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_index])
        
        return parents
    
    def optimize(self, data: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Запуск оптимизации
        
        Args:
            data: Данные для оптимизации
            verbose: Выводить ли прогресс
            
        Returns:
            Результаты оптимизации
        """
        if verbose:
            print("🚀 ЗАПУСК ОПТИМИЗАЦИИ MZA")
            print("=" * 50)
            print(f"📊 Размер популяции: {self.population_size}")
            print(f"🔄 Максимум поколений: {self.max_generations}")
            print(f"🧬 Вероятность мутации: {self.mutation_rate}")
            print(f"🔀 Вероятность кроссовера: {self.crossover_rate}")
            print(f"👑 Размер элиты: {self.elite_size}")
            print(f"📊 Кросс-валидация: {self.cv_folds} фолдов")
            print(f"🛡️ Регуляризация: {self.regularization}")
            print(f"🎯 Параметров для оптимизации: {len(self.param_ranges)}")
            print("=" * 50)
        
        # Инициализация популяции
        population = [self.create_random_individual() for _ in range(self.population_size)]
        
        for generation in range(self.max_generations):
            # Оценка фитнеса
            fitness_scores = [self.calculate_fitness(ind, data) for ind in population]
            
            # Обновление лучшего результата
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_score:
                self.best_score = fitness_scores[best_idx]
                self.best_parameters = population[best_idx].copy()
            
            # Сохранение истории
            self.generation_history.append({
                'generation': generation,
                'best_fitness': max(fitness_scores),
                'avg_fitness': np.mean(fitness_scores),
                'best_individual': population[best_idx].copy()
            })
            
            if verbose and generation % 10 == 0:
                print(f"🔄 Поколение {generation:3d}: Лучший = {max(fitness_scores):.6f}, "
                      f"Средний = {np.mean(fitness_scores):.6f}")
            
            # Селекция родителей
            parents = self.select_parents(population, fitness_scores)
            
            # Создание нового поколения
            new_population = []
            
            # Элитизм - сохраняем лучших
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Генерация потомков
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Обрезаем до нужного размера
            population = new_population[:self.population_size]
        
        # Финальная оценка
        final_fitness_scores = [self.calculate_fitness(ind, data) for ind in population]
        best_idx = np.argmax(final_fitness_scores)
        
        if final_fitness_scores[best_idx] > self.best_score:
            self.best_score = final_fitness_scores[best_idx]
            self.best_parameters = population[best_idx].copy()
        
        if verbose:
            print(f"\n🎉 ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
            print(f"🏆 Лучший Economic Value: {self.best_score:.6f}")
            print(f"📊 Протестировано поколений: {self.max_generations}")
            print(f"🧬 Общее количество оценок: {self.population_size * self.max_generations}")
        
        return {
            'best_score': self.best_score,
            'best_parameters': self.best_parameters,
            'generations': self.max_generations,
            'total_evaluations': self.population_size * self.max_generations,
            'generation_history': self.generation_history
        }
    
    def get_parameter_summary(self, parameters: Dict) -> str:
        """
        Получение сводки по параметрам
        
        Args:
            parameters: Словарь с параметрами
            
        Returns:
            Строка со сводкой
        """
        summary = "🔧 ЛУЧШИЕ ПАРАМЕТРЫ MZA:\n"
        summary += "=" * 30 + "\n"
        
        # Группируем параметры по категориям
        categories = {
            'Trend Indicators': ['adxLength', 'adxThreshold', 'fastMALength', 'slowMALength'],
            'Momentum Indicators': ['rsiLength', 'stochKLength', 'macdFast', 'macdSlow', 'macdSignal'],
            'Price Action Indicators': ['hhllRange', 'haDojiRange', 'candleRangeLength'],
            'Market Activity Indicators': ['bbLength', 'bbMultiplier', 'atrLength', 'kcLength', 'kcMultiplier', 'volumeMALength'],
            'Base Weights': ['trendWeightBase', 'momentumWeightBase', 'priceActionWeightBase'],
            'Stability Controls': ['useSmoothing', 'useHysteresis']
        }
        
        for category, params in categories.items():
            summary += f"\n📊 {category}:\n"
            for param in params:
                if param in parameters:
                    summary += f"   {param}: {parameters[param]}\n"
        
        return summary

def optimize_mza_for_timeframe(data: pd.DataFrame, 
                              timeframe: str,
                              population_size: int = 30,
                              max_generations: int = 50) -> Dict:
    """
    Оптимизация MZA для конкретного таймфрейма
    
    Args:
        data: Данные для оптимизации
        timeframe: Название таймфрейма
        population_size: Размер популяции
        max_generations: Количество поколений
        
    Returns:
        Результаты оптимизации
    """
    print(f"🚀 ОПТИМИЗАЦИЯ MZA ДЛЯ ТАЙМФРЕЙМА {timeframe}")
    print("=" * 60)
    
    optimizer = MZAOptimizer(
        population_size=population_size,
        max_generations=max_generations
    )
    
    results = optimizer.optimize(data, verbose=True)
    
    print(f"\n✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА ДЛЯ {timeframe}")
    print(f"🏆 Лучший Economic Value: {results['best_score']:.6f}")
    
    return results

def optimize_mza_all_timeframes(data_dict: Dict[str, pd.DataFrame],
                               population_size: int = 25,
                               max_generations: int = 40) -> Dict[str, Dict]:
    """
    Оптимизация MZA для всех таймфреймов
    
    Args:
        data_dict: Словарь с данными по таймфреймам
        population_size: Размер популяции
        max_generations: Количество поколений
        
    Returns:
        Словарь с результатами по таймфреймам
    """
    print("🚀 ОПТИМИЗАЦИЯ MZA ДЛЯ ВСЕХ ТАЙМФРЕЙМОВ")
    print("=" * 60)
    
    results = {}
    
    for tf, data in data_dict.items():
        print(f"\n📊 Оптимизация для {tf}...")
        results[tf] = optimize_mza_for_timeframe(
            data, tf, population_size, max_generations
        )
    
    # Находим лучший таймфрейм
    best_tf = max(results.keys(), key=lambda x: results[x]['best_score'])
    
    print(f"\n🎯 ЛУЧШИЙ ТАЙМФРЕЙМ: {best_tf}")
    print(f"🏆 Лучший Economic Value: {results[best_tf]['best_score']:.6f}")
    
    return results

if __name__ == "__main__":
    # Пример использования
    print("🚀 ТЕСТИРОВАНИЕ МОДУЛЯ ОПТИМИЗАЦИИ MZA")
    print("=" * 50)
    
    # Загружаем данные
    manager = DataManager()
    data = manager.load_data(['15m'])
    
    if data and '15m' in data:
        print(f"✅ Данные загружены: {len(data['15m'])} записей")
        
        # Тестируем оптимизацию
        optimizer = MZAOptimizer(population_size=10, max_generations=5)
        results = optimizer.optimize(data['15m'], verbose=True)
        
        print(f"\n🎉 Тест завершен!")
        print(f"🏆 Лучший скор: {results['best_score']:.6f}")
    else:
        print("❌ Не удалось загрузить данные для тестирования")
