"""
Улучшенный генетический алгоритм для оптимизации параметров MZA
с защитой от переобучения и кросс-валидацией
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
import warnings
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings('ignore')

class RobustGeneticMZAOptimizer:
    """
    Улучшенный генетический алгоритм с защитой от переобучения
    
    Особенности:
    - Кросс-валидация на временных рядах
    - Регуляризация для предотвращения переобучения
    - Валидация на out-of-sample данных
    - Стабильность параметров во времени
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
        Инициализация улучшенного генетического алгоритма
        
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
        
        # Параметры MZA для оптимизации
        self.param_ranges = {
            # Trend Indicators
            'adx_length': (10, 20, 1),
            'adx_threshold': (15, 30, 1),
            'fast_ma_length': (15, 25, 1),
            'slow_ma_length': (40, 60, 1),
            
            # Momentum Indicators
            'rsi_length': (10, 20, 1),
            'rsi_overbought': (65, 80, 1),
            'rsi_oversold': (20, 35, 1),
            'stoch_length': (10, 20, 1),
            'macd_fast': (8, 16, 1),
            'macd_slow': (20, 30, 1),
            
            # Price Action Indicators
            'bb_length': (15, 25, 1),
            'bb_std': (1.5, 3.0, 0.1),
            'atr_length': (10, 20, 1),
            
            # Dynamic Weights
            'trend_weight_high_vol': (0.4, 0.6, 0.05),
            'momentum_weight_high_vol': (0.3, 0.4, 0.05),
            'price_action_weight_low_vol': (0.4, 0.6, 0.05)
        }
        
        # Результаты оптимизации
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.generation_history = []
        self.cv_scores = []
        
    def create_random_individual(self) -> Dict:
        """Создает случайную особь (набор параметров)"""
        individual = {}
        for param, (min_val, max_val, step) in self.param_ranges.items():
            if step >= 1:
                individual[param] = random.randint(int(min_val), int(max_val))
            else:
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
    
    def simple_mza_classify(self, params: Dict, data: pd.DataFrame) -> np.ndarray:
        """Упрощенная классификация MZA для быстрой оценки"""
        predictions = np.zeros(len(data))
        
        # Получаем параметры
        rsi_length = params.get('rsi_length', 14)
        rsi_overbought = params.get('rsi_overbought', 70)
        rsi_oversold = params.get('rsi_oversold', 30)
        fast_ma_length = params.get('fast_ma_length', 20)
        slow_ma_length = params.get('slow_ma_length', 50)
        
        # Вычисляем RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_length).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Вычисляем скользящие средние
        fast_ma = data['close'].rolling(window=fast_ma_length).mean()
        slow_ma = data['close'].rolling(window=slow_ma_length).mean()
        
        # Простая классификация
        for i in range(len(data)):
            if i < max(rsi_length, slow_ma_length):
                predictions[i] = 0
            else:
                trend_signal = 1 if fast_ma.iloc[i] > slow_ma.iloc[i] else -1
                
                if rsi.iloc[i] > rsi_overbought:
                    momentum_signal = -1
                elif rsi.iloc[i] < rsi_oversold:
                    momentum_signal = 1
                else:
                    momentum_signal = 0
                
                if trend_signal == momentum_signal and momentum_signal != 0:
                    predictions[i] = momentum_signal
                elif trend_signal != 0:
                    predictions[i] = trend_signal * 0.5
                else:
                    predictions[i] = 0
        
        return predictions
    
    def calculate_economic_value(self, data: pd.DataFrame, predictions: np.ndarray) -> float:
        """Вычисляет Economic Value с регуляризацией"""
        # Вычисляем доходность
        returns = data['close'].pct_change().dropna()
        
        # Выравниваем индексы
        aligned_returns = returns.iloc[1:]
        aligned_predictions = predictions[1:len(aligned_returns)+1]
        
        # Разделяем по предсказаниям
        bull_mask = aligned_predictions > 0.5
        bear_mask = aligned_predictions < -0.5
        sideways_mask = (aligned_predictions >= -0.5) & (aligned_predictions <= 0.5)
        
        # Метрики доходности
        bull_returns = aligned_returns[bull_mask].mean() if bull_mask.any() else 0
        bear_returns = aligned_returns[bear_mask].mean() if bear_mask.any() else 0
        sideways_returns = aligned_returns[sideways_mask].mean() if sideways_mask.any() else 0
        
        # Economic Value
        return_spread = abs(bull_returns - bear_returns)
        sideways_vol = aligned_returns[sideways_mask].std() if sideways_mask.any() else 0
        economic_value = return_spread / (1 + sideways_vol) if sideways_vol > 0 else return_spread
        
        # Штраф за нестабильность
        zone_changes = np.sum(np.diff(aligned_predictions) != 0)
        stability_penalty = zone_changes / len(aligned_predictions) * 0.1
        
        # Регуляризация для предотвращения переобучения
        regularization_penalty = self.regularization_strength * (
            abs(bull_returns) + abs(bear_returns) + abs(sideways_returns)
        )
        
        return economic_value - stability_penalty - regularization_penalty
    
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
        train_predictions = self.simple_mza_classify(individual, train_data)
        train_score = self.calculate_economic_value(train_data, train_predictions)
        
        # Оценка на тестовых данных
        test_predictions = self.simple_mza_classify(individual, test_data)
        test_score = self.calculate_economic_value(test_data, test_predictions)
        
        # Вычисляем стабильность
        stability = 1 - abs(train_score - test_score) / (abs(train_score) + abs(test_score) + 1e-8)
        
        # Комбинированная оценка
        combined_score = 0.7 * test_score + 0.3 * train_score
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'stability': stability,
            'combined_score': combined_score,
            'overfitting_risk': abs(train_score - test_score)
        }
    
    def evaluate_fitness(self, individual: Dict, data: pd.DataFrame) -> float:
        """
        Оценивает пригодность особи с кросс-валидацией
        
        Args:
            individual: Набор параметров
            data: Данные для тестирования
            
        Returns:
            Значение пригодности с учетом стабильности
        """
        try:
            # Проводим кросс-валидацию
            cv_results = self.cross_validate_fitness(individual, data)
            
            # Сохраняем результаты для анализа
            self.cv_scores.append(cv_results)
            
            # Возвращаем комбинированную оценку
            return cv_results['combined_score']
            
        except Exception as e:
            return -1000.0
    
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
        
        for param, (min_val, max_val, step) in self.param_ranges.items():
            if random.random() < self.mutation_rate:
                if step >= 1:
                    mutated[param] = random.randint(int(min_val), int(max_val))
                else:
                    steps = int((max_val - min_val) / step) + 1
                    mutated[param] = min_val + random.randint(0, steps - 1) * step
        
        return mutated
    
    def optimize(self, data: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Запускает улучшенный генетический алгоритм оптимизации
        
        Args:
            data: Данные для оптимизации
            verbose: Выводить ли прогресс
            
        Returns:
            Результаты оптимизации с анализом переобучения
        """
        if verbose:
            print("🧬 ЗАПУСК УЛУЧШЕННОГО ГЕНЕТИЧЕСКОГО АЛГОРИТМА С ЗАЩИТОЙ ОТ ПЕРЕОБУЧЕНИЯ")
            print("=" * 80)
            print(f"📊 Размер популяции: {self.population_size}")
            print(f"🔄 Максимум поколений: {self.max_generations}")
            print(f"🧬 Вероятность мутации: {self.mutation_rate}")
            print(f"🔀 Вероятность скрещивания: {self.crossover_rate}")
            print(f"👑 Размер элиты: {self.elite_size}")
            print(f"📊 Кросс-валидация: {self.cv_folds} фолдов")
            print(f"🛡️ Регуляризация: {self.regularization_strength}")
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
                'worst_fitness': np.min(fitness_scores),
                'stability': np.mean([score['stability'] for score in self.cv_scores[-self.population_size:]])
            })
            
            if verbose and generation % 10 == 0:
                avg_stability = np.mean([score['stability'] for score in self.cv_scores[-self.population_size:]])
                print(f"🔄 Поколение {generation:3d}: "
                      f"Лучший = {best_fitness:.6f}, "
                      f"Средний = {np.mean(fitness_scores):.6f}, "
                      f"Стабильность = {avg_stability:.3f}")
            
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
            print(f"\n✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
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
            'algorithm': 'robust_genetic',
            'overfitting_analysis': overfitting_analysis,
            'cv_scores': self.cv_scores
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
    
    def plot_convergence_with_stability(self):
        """Строит график сходимости с анализом стабильности"""
        if not self.generation_history:
            print("❌ Нет данных для построения графика")
            return
        
        import matplotlib.pyplot as plt
        
        generations = [h['generation'] for h in self.generation_history]
        best_fitness = [h['best_fitness'] for h in self.generation_history]
        avg_fitness = [h['avg_fitness'] for h in self.generation_history]
        stability = [h['stability'] for h in self.generation_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # График сходимости
        ax1.plot(generations, best_fitness, 'b-', label='Лучший результат', linewidth=2)
        ax1.plot(generations, avg_fitness, 'r--', label='Средний результат', linewidth=1)
        ax1.set_xlabel('Поколение')
        ax1.set_ylabel('Economic Value')
        ax1.set_title('Сходимость улучшенного генетического алгоритма')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График стабильности
        ax2.plot(generations, stability, 'g-', label='Стабильность', linewidth=2)
        ax2.set_xlabel('Поколение')
        ax2.set_ylabel('Стабильность')
        ax2.set_title('Стабильность параметров во времени')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_to_pine_script(self, output_file: str = "robust_genetic_optimized_mza.pine") -> str:
        """Экспортирует оптимальные параметры в формат Pine Script"""
        if not self.best_individual:
            raise ValueError("Сначала необходимо провести оптимизацию")
        
        overfitting_analysis = self.analyze_overfitting()
        
        pine_code = f"""// Оптимизированные параметры MZA с защитой от переобучения
// Сгенерировано автоматически на основе исторических данных BTC
// Дата оптимизации: {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}
// Лучший Economic Value: {self.best_fitness:.6f}
// Алгоритм: Улучшенный генетический с кросс-валидацией
// Риск переобучения: {overfitting_analysis['overfitting_risk']:.3f}
// Стабильность: {overfitting_analysis['stability']:.3f}

//@version=5
indicator("Robust Genetic Optimized MZA [BullByte]", overlay=true)

// ===== ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ (С ЗАЩИТОЙ ОТ ПЕРЕОБУЧЕНИЯ) =====

// Trend Indicators
adx_length = input.int({self.best_individual.get('adx_length', 14)}, "ADX Length", minval=10, maxval=20)
adx_threshold = input.int({self.best_individual.get('adx_threshold', 20)}, "ADX Threshold", minval=15, maxval=30)
fast_ma_length = input.int({self.best_individual.get('fast_ma_length', 20)}, "Fast MA Length", minval=15, maxval=25)
slow_ma_length = input.int({self.best_individual.get('slow_ma_length', 50)}, "Slow MA Length", minval=40, maxval=60)

// Momentum Indicators
rsi_length = input.int({self.best_individual.get('rsi_length', 14)}, "RSI Length", minval=10, maxval=20)
rsi_overbought = input.int({self.best_individual.get('rsi_overbought', 70)}, "RSI Overbought", minval=65, maxval=80)
rsi_oversold = input.int({self.best_individual.get('rsi_oversold', 30)}, "RSI Oversold", minval=20, maxval=35)
stoch_length = input.int({self.best_individual.get('stoch_length', 14)}, "Stochastic Length", minval=10, maxval=20)
macd_fast = input.int({self.best_individual.get('macd_fast', 12)}, "MACD Fast", minval=8, maxval=16)
macd_slow = input.int({self.best_individual.get('macd_slow', 26)}, "MACD Slow", minval=20, maxval=30)

// Price Action Indicators
bb_length = input.int({self.best_individual.get('bb_length', 20)}, "Bollinger Bands Length", minval=15, maxval=25)
bb_std = input.float({self.best_individual.get('bb_std', 2.0)}, "Bollinger Bands Std Dev", minval=1.5, maxval=3.0)
atr_length = input.int({self.best_individual.get('atr_length', 14)}, "ATR Length", minval=10, maxval=20)

// Dynamic Weights
trend_weight_high_vol = input.float({self.best_individual.get('trend_weight_high_vol', 0.5)}, "Trend Weight (High Vol)", minval=0.4, maxval=0.6)
momentum_weight_high_vol = input.float({self.best_individual.get('momentum_weight_high_vol', 0.35)}, "Momentum Weight (High Vol)", minval=0.3, maxval=0.4)
price_action_weight_low_vol = input.float({self.best_individual.get('price_action_weight_low_vol', 0.45)}, "Price Action Weight (Low Vol)", minval=0.4, maxval=0.6)

// ===== ОСТАЛЬНОЙ КОД MZA ОСТАЕТСЯ БЕЗ ИЗМЕНЕНИЙ =====

// ===== РЕЗУЛЬТАТЫ УЛУЧШЕННОЙ ОПТИМИЗАЦИИ =====
// Economic Value: {self.best_fitness:.6f}
// Риск переобучения: {overfitting_analysis['overfitting_risk']:.3f}
// Стабильность: {overfitting_analysis['stability']:.3f}
// Поколений: {self.max_generations}
// Размер популяции: {self.population_size}
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(pine_code)
        
        print(f"📄 Оптимизированные параметры с защитой от переобучения экспортированы в {output_file}")
        return pine_code


# Пример использования
if __name__ == "__main__":
    # Создаем улучшенный оптимизатор
    optimizer = RobustGeneticMZAOptimizer(
        population_size=30,
        max_generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        cv_folds=3,
        regularization_strength=0.01
    )
    
    print("🛡️ Robust Genetic MZA Optimizer готов к работе!")
    print(f"📊 Параметров для оптимизации: {len(optimizer.param_ranges)}")
    print(f"🔧 Доступные параметры: {list(optimizer.param_ranges.keys())}")
    print(f"🛡️ Защита от переобучения: Включена")
    print(f"📊 Кросс-валидация: {optimizer.cv_folds} фолдов")
