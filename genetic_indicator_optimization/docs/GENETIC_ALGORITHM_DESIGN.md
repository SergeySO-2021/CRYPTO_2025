# 🧬 ДИЗАЙН ГЕНЕТИЧЕСКОГО АЛГОРИТМА

**Дата создания:** 2025-11-10  
**Дата обновления:** 2025-11-16  
**Версия:** 2.0.0

> **⚠️ ВАЖНО:** Этот документ был обновлен после критического анализа рекомендаций DeepSeek.  
> См. `docs/UPDATED_PLAN_CRITICAL_ANALYSIS.md` для детального анализа и обоснования изменений.  
> Краткая сводка: `docs/PLAN_SUMMARY.md`

---

## 🎯 1. ОБЩАЯ АРХИТЕКТУРА

### 1.1. Компоненты системы

```
┌─────────────────────────────────────────────────────────┐
│                    ГЕНЕТИЧЕСКИЙ АЛГОРИТМ                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │ Инициализация│ -> │  Селекция    │ -> │ Кроссовер│ │
│  │  Популяции   │    │              │    │          │ │
│  └──────────────┘    └──────────────┘    └──────────┘ │
│         ↑                    │                  │       │
│         │                    │                  │       │
│         │                    ↓                  ↓       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │   Оценка     │    │   Мутация    │    │  Элитизм │ │
│  │ Приспособлен.│    │              │    │          │ │
│  └──────────────┘    └──────────────┘    └──────────┘ │
│         │                    │                  │         │
│         └───────────────────┴──────────────────┘         │
│                           │                              │
│                           ↓                              │
│              ┌──────────────────────┐                   │
│              │  Новая Популяция     │                   │
│              └──────────────────────┘                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 1.2. Поток данных

```
Данные (OHLCV + Order Book)
    ↓
[Генерация начальной популяции]
    ↓
[Для каждой особи:]
    ├─> Расчет индикаторов
    ├─> Генерация сигналов
    ├─> Бэктестинг
    └─> Расчет PnL (fitness)
    ↓
[Селекция лучших особей]
    ↓
[Кроссовер + Мутация]
    ↓
[Создание нового поколения]
    ↓
[Проверка критериев остановки]
    ├─> Если не выполнены -> повтор
    └─> Если выполнены -> возврат лучшей особи
```

---

## 🧬 2. ПРЕДСТАВЛЕНИЕ ОСОБИ

### 2.1. Структура хромосомы

```python
class Individual:
    """
    Представление особи в генетическом алгоритме
    """
    def __init__(self):
        self.indicators = {
            'rsi': {
                'period': 14,
                'overbought': 70,
                'oversold': 30,
                'stop_loss': 0.02,      # 2% от цены входа
                'take_profit': 0.04,    # 4% от цены входа
                'enabled': True         # Использовать ли индикатор
            },
            'macd': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'stop_loss': 0.015,
                'take_profit': 0.035,
                'enabled': True
            },
            # ... остальные индикаторы
        }
        
        self.strategy_params = {
            'max_positions': 1,         # Максимум одновременных позиций
            'position_size': 1.0,         # Размер позиции (1.0 = 100%)
            'use_order_book': False,      # Использовать ли order book
            'order_book_weight': 0.3       # Вес order book в решении
        }
        
        self.fitness = None              # Приспособленность
        self.backtest_results = None     # Результаты бэктеста
```

### 2.2. Кодирование параметров

#### 2.2.1. Параметры индикаторов

| Индикатор | Параметры | Диапазон значений |
|-----------|-----------|-------------------|
| RSI | period | 8-30 (целое) |
| | overbought | 60-90 (целое) |
| | oversold | 10-40 (целое) |
| MACD | fast_period | 5-30 (целое) |
| | slow_period | 15-70 (целое) |
| | signal_period | 5-30 (целое) |
| Bollinger Bands | period | 15-40 (целое) |
| | std_dev | 1.5-4.0 (float, шаг 0.1) |
| SuperTrend | atr_period | 5-30 (целое) |
| | atr_multiplier | 1.5-6.0 (float, шаг 0.1) |

#### 2.2.2. Стоп-лоссы и тейк-профиты

| Параметр | Диапазон | Шаг |
|----------|----------|-----|
| stop_loss | 0.5% - 5% | 0.1% |
| take_profit | 1% - 10% | 0.1% |

**Ограничения:**
- `take_profit > stop_loss * 1.5` (минимум соотношение 1.5:1)
- `stop_loss < 5%` (максимальный риск)

---

## 🔄 3. ОПЕРАТОРЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА

### 3.1. Инициализация популяции

```python
def initialize_population(size, indicators_config):
    """
    Создание начальной популяции
    
    Args:
        size: Размер популяции
        indicators_config: Конфигурация индикаторов с диапазонами
    
    Returns:
        list: Список особей
    """
    population = []
    
    for _ in range(size):
        individual = Individual()
        
        # Случайная инициализация параметров для каждого индикатора
        for indicator_name, params in indicators_config.items():
            individual.indicators[indicator_name] = {
                param: random_value_in_range(param_range)
                for param, param_range in params.items()
            }
            individual.indicators[indicator_name]['enabled'] = random.choice([True, False])
        
        # Инициализация стоп-лоссов и тейк-профитов
        for indicator_name in individual.indicators:
            stop_loss = random.uniform(0.005, 0.05)
            take_profit = random.uniform(stop_loss * 1.5, 0.1)
            individual.indicators[indicator_name]['stop_loss'] = stop_loss
            individual.indicators[indicator_name]['take_profit'] = take_profit
        
        population.append(individual)
    
    return population
```

### 3.2. Функция приспособленности

```python
def calculate_fitness(individual, data, initial_capital=10000):
    """
    Расчет приспособленности особи
    
    Args:
        individual: Особая для оценки
        data: Исторические данные
        initial_capital: Начальный капитал
    
    Returns:
        float: Оценка приспособленности
    """
    # 1. Бэктестинг стратегии
    backtest_results = backtest_strategy(individual, data, initial_capital)
    
    # 2. Основные метрики
    total_pnl = backtest_results['total_pnl']  # Общая прибыльность
    sharpe_ratio = backtest_results['sharpe_ratio']
    max_drawdown = backtest_results['max_drawdown']
    win_rate = backtest_results['win_rate']
    profit_factor = backtest_results['profit_factor']
    total_trades = backtest_results['total_trades']
    
    # 3. Нормализация метрик (гибридный подход)
    # Первые 30% поколений: rank-based, остальные: absolute values
    use_rank_based = (generation / max_generations) < 0.3
    
    if use_rank_based:
        # Rank-based нормализация для разнообразия в начале
        normalized_pnl = rank_normalize([ind.total_pnl for ind in population], total_pnl)
        normalized_sharpe = rank_normalize([ind.sharpe_ratio for ind in population], sharpe_ratio)
        normalized_drawdown = rank_normalize([ind.max_drawdown for ind in population], max_drawdown, reverse=True)
    else:
        # Absolute values для точности в конце
        normalized_pnl = total_pnl * 100
        normalized_sharpe = sharpe_ratio * 10
        normalized_drawdown = (1 - max_drawdown) * 50
    
    # 4. Взвешенная функция приспособленности (эволюция по этапам)
    # MVP: PnL 60%, Sharpe 30%, Drawdown 10%
    # Этап 1: PnL 40%, Sharpe 25%, Drawdown 20%, Win Rate 10%, Profit Factor 5%
    # Этап 2: PnL 35%, Sharpe 25%, Drawdown 20%, Win Rate 10%, Profit Factor 10%
    
    stage = determine_stage(generation, max_generations)
    
    if stage == 'mvp':
        weights = {'pnl': 0.60, 'sharpe': 0.30, 'drawdown': 0.10}
        base_score = (
            normalized_pnl * weights['pnl'] +
            normalized_sharpe * weights['sharpe'] +
            normalized_drawdown * weights['drawdown']
        )
    elif stage == 'stage1':
        weights = {'pnl': 0.40, 'sharpe': 0.25, 'drawdown': 0.20, 'win_rate': 0.10, 'profit_factor': 0.05}
        base_score = (
            normalized_pnl * weights['pnl'] +
            normalized_sharpe * weights['sharpe'] +
            normalized_drawdown * weights['drawdown'] +
            win_rate * 100 * weights['win_rate'] +
            profit_factor * 10 * weights['profit_factor']
        )
    else:  # stage2 или final
        weights = {'pnl': 0.35, 'sharpe': 0.25, 'drawdown': 0.20, 'win_rate': 0.10, 'profit_factor': 0.10}
        base_score = (
            normalized_pnl * weights['pnl'] +
            normalized_sharpe * weights['sharpe'] +
            normalized_drawdown * weights['drawdown'] +
            win_rate * 100 * weights['win_rate'] +
            profit_factor * 10 * weights['profit_factor']
        )
    
    # 5. Штрафы за плохие характеристики (мультипликативные)
    if total_trades < 20:  # Мало сделок
        base_score *= 0.5
    
    if max_drawdown > 0.5:  # Просадка > 50%
        base_score *= 0.2  # Сильный штраф
    
    if win_rate < 0.3:  # Win rate < 30%
        base_score *= 0.7
    
    if profit_factor < 1.0:  # Убыточная стратегия
        base_score *= 0.3
    
    # 6. Финальный score
    fitness = base_score
    
    # Сохраняем результаты в особи
    individual.fitness = fitness
    individual.backtest_results = backtest_results
    
    return fitness
```

### 3.3. Селекция

#### 3.3.1. Турнирная селекция

```python
def tournament_selection(population, tournament_size=3):
    """
    Турнирная селекция
    
    Args:
        population: Текущая популяция
        tournament_size: Размер турнира
    
    Returns:
        Individual: Выбранная особь
    """
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness)
```

#### 3.3.2. Рулеточная селекция

```python
def roulette_selection(population):
    """
    Рулеточная селекция (пропорциональная приспособленности)
    
    Args:
        population: Текущая популяция
    
    Returns:
        Individual: Выбранная особь
    """
    # Нормализация fitness (все значения должны быть положительными)
    min_fitness = min(ind.fitness for ind in population)
    normalized_fitness = [ind.fitness - min_fitness + 1 for ind in population]
    
    # Вероятности выбора
    total_fitness = sum(normalized_fitness)
    probabilities = [f / total_fitness for f in normalized_fitness]
    
    # Выбор на основе вероятностей
    return np.random.choice(population, p=probabilities)
```

### 3.4. Кроссовер

```python
def crossover(parent1, parent2, crossover_rate=0.8):
    """
    Создание потомка из двух родителей
    
    Args:
        parent1, parent2: Родительские особи
        crossover_rate: Вероятность кроссовера
    
    Returns:
        Individual: Потомок
    """
    if random.random() > crossover_rate:
        # Если кроссовер не произошел, возвращаем копию лучшего родителя
        return copy.deepcopy(parent1 if parent1.fitness > parent2.fitness else parent2)
    
    child = Individual()
    
    # Для каждого индикатора
    for indicator_name in parent1.indicators:
        if random.random() < 0.5:
            # Берем параметры от первого родителя
            child.indicators[indicator_name] = copy.deepcopy(parent1.indicators[indicator_name])
        else:
            # Берем параметры от второго родителя
            child.indicators[indicator_name] = copy.deepcopy(parent2.indicators[indicator_name])
        
        # Смешивание параметров внутри индикатора
        if random.random() < 0.3:
            for param in child.indicators[indicator_name]:
                if param == 'enabled':
                    continue
                if random.random() < 0.5:
                    child.indicators[indicator_name][param] = parent1.indicators[indicator_name][param]
                else:
                    child.indicators[indicator_name][param] = parent2.indicators[indicator_name][param]
    
    # Кроссовер стратегических параметров
    child.strategy_params = copy.deepcopy(
        parent1.strategy_params if random.random() < 0.5 else parent2.strategy_params
    )
    
    return child
```

### 3.5. Мутация

```python
def mutate(individual, mutation_rate=0.1, indicators_config=None):
    """
    Мутация параметров особи
    
    Args:
        individual: Особая для мутации
        mutation_rate: Вероятность мутации каждого параметра
        indicators_config: Конфигурация с диапазонами параметров
    """
    for indicator_name, params in individual.indicators.items():
        # Мутация параметров индикатора
        for param, value in params.items():
            if param == 'enabled':
                # Мутация включения/выключения индикатора
                if random.random() < mutation_rate * 0.5:
                    params[param] = not params[param]
                continue
            
            if random.random() < mutation_rate:
                if param in ['stop_loss', 'take_profit']:
                    # Мутация стоп-лосса/тейк-профита
                    if param == 'stop_loss':
                        new_value = value * random.uniform(0.7, 1.3)
                        new_value = max(0.005, min(0.05, new_value))
                    else:  # take_profit
                        new_value = value * random.uniform(0.7, 1.3)
                        new_value = max(params['stop_loss'] * 1.5, min(0.1, new_value))
                    
                    params[param] = round(new_value, 4)
                
                else:
                    # Мутация параметров индикатора
                    if indicators_config and indicator_name in indicators_config:
                        param_range = indicators_config[indicator_name].get(param)
                        if param_range:
                            if isinstance(value, int):
                                new_value = random.randint(param_range['min'], param_range['max'])
                            else:
                                new_value = random.uniform(param_range['min'], param_range['max'])
                                new_value = round(new_value, 2)
                            params[param] = new_value
                        else:
                            # Случайная мутация в пределах ±20%
                            if isinstance(value, int):
                                new_value = int(value * random.uniform(0.8, 1.2))
                            else:
                                new_value = value * random.uniform(0.8, 1.2)
                            params[param] = round(new_value, 2) if isinstance(new_value, float) else new_value
    
    return individual
```

### 3.6. Элитизм

```python
def elitism(population, elite_size=5):
    """
    Сохранение лучших особей
    
    Args:
        population: Текущая популяция
        elite_size: Количество элитных особей
    
    Returns:
        list: Список элитных особей
    """
    sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
    return [copy.deepcopy(ind) for ind in sorted_population[:elite_size]]
```

---

## 🔄 4. ОСНОВНОЙ ЦИКЛ АЛГОРИТМА

```python
def genetic_algorithm(data, indicators_config, ga_params):
    """
    Основной цикл генетического алгоритма
    
    Args:
        data: Исторические данные
        indicators_config: Конфигурация индикаторов
        ga_params: Параметры ГА
    
    Returns:
        Individual: Лучшая найденная особь
    """
    # 1. Инициализация
    population = initialize_population(
        ga_params['population_size'],
        indicators_config
    )
    
    # 2. Оценка начальной популяции
    for individual in population:
        calculate_fitness(individual, data)
    
    best_individual = max(population, key=lambda x: x.fitness)
    best_fitness_history = [best_individual.fitness]
    stagnation_counter = 0
    
    # 3. Основной цикл
    for generation in range(ga_params['max_generations']):
        # 3.1. Элитизм
        elite = elitism(population, ga_params['elite_size'])
        
        # 3.2. Создание нового поколения
        new_population = elite.copy()
        
        while len(new_population) < ga_params['population_size']:
            # Селекция
            parent1 = tournament_selection(population, ga_params['tournament_size'])
            parent2 = tournament_selection(population, ga_params['tournament_size'])
            
            # Кроссовер
            child = crossover(parent1, parent2, ga_params['crossover_rate'])
            
            # Мутация
            child = mutate(child, ga_params['mutation_rate'], indicators_config)
            
            # Оценка потомка
            calculate_fitness(child, data)
            
            new_population.append(child)
        
        # 3.3. Обновление популяции
        population = new_population
        
        # 3.4. Обновление лучшей особи
        current_best = max(population, key=lambda x: x.fitness)
        if current_best.fitness > best_individual.fitness:
            best_individual = copy.deepcopy(current_best)
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        best_fitness_history.append(best_individual.fitness)
        
        # 3.5. Логирование
        if generation % 10 == 0:
            print(f"Generation {generation}: Best fitness = {best_individual.fitness:.4f}")
        
        # 3.6. Проверка критериев остановки
        if stagnation_counter >= ga_params['stagnation_limit']:
            print(f"Stopped due to stagnation after {generation} generations")
            break
        
        if best_individual.fitness >= ga_params['target_fitness']:
            print(f"Target fitness reached after {generation} generations")
            break
    
    return best_individual, best_fitness_history
```

---

## 📊 5. ПАРАМЕТРЫ АЛГОРИТМА

### 5.1. Рекомендуемые значения (поэтапный подход)

**ВАЖНО:** Параметры адаптированы после критического анализа рекомендаций DeepSeek.
См. `docs/UPDATED_PLAN_CRITICAL_ANALYSIS.md` для деталей.

```python
# MVP (Этап 0) - Быстрая итерация
GA_PARAMS_MVP = {
    'population_size': 40,        # Уменьшено с 50 для быстрой итерации
    'max_generations': 50,        # Уменьшено для MVP
    'crossover_rate': 0.8,
    'mutation_rate': 0.15,
    'elite_size': 3,              # Меньше для большего разнообразия
    'tournament_size': 3,
    'stagnation_limit': 15,       # Раннее прекращение для MVP
    'selection_method': 'tournament'
}

# Этап 1: Индивидуальная оптимизация
GA_PARAMS_STAGE1 = {
    'population_size': 60,        # Постепенное увеличение
    'max_generations': 80,
    'crossover_rate': 0.8,
    'mutation_rate': 0.15,
    'elite_size': 5,
    'tournament_size': 3,
    'stagnation_limit': 20,
    'selection_method': 'tournament'
}

# Этап 2: Интеграция
GA_PARAMS_STAGE2 = {
    'population_size': 90,        # Увеличено для сложного пространства поиска
    'max_generations': 100,
    'crossover_rate': 0.8,
    'mutation_rate': 0.15,
    'elite_size': 5,
    'tournament_size': 3,
    'stagnation_limit': 25,
    'selection_method': 'tournament'
}

# Финальная оптимизация
GA_PARAMS_FINAL = {
    'population_size': 110,        # Максимальный размер для production
    'max_generations': 120,
    'crossover_rate': 0.8,
    'mutation_rate': 0.15,
    'elite_size': 5,
    'tournament_size': 3,
    'stagnation_limit': 30,
    'selection_method': 'tournament'
}
```

### 5.2. Адаптивные параметры

Для улучшения сходимости можно использовать:

- **Адаптивная мутация:** Увеличение mutation_rate при застое
- **Адаптивный кроссовер:** Изменение crossover_rate в зависимости от разнообразия популяции
- **Динамический размер популяции:** Увеличение при низком разнообразии

---

## 🎯 6. ОПТИМИЗАЦИЯ ПРОИЗВОДИТЕЛЬНОСТИ

### 6.1. Параллелизация

```python
from multiprocessing import Pool

def evaluate_population_parallel(population, data):
    """
    Параллельная оценка популяции
    """
    with Pool(processes=4) as pool:
        results = pool.starmap(
            calculate_fitness,
            [(ind, data) for ind in population]
        )
    
    for individual, fitness in zip(population, results):
        individual.fitness = fitness
```

### 6.2. Кэширование

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def calculate_indicator_cached(indicator_name, params_hash, data_hash):
    """
    Кэширование результатов расчета индикаторов
    """
    # Расчет индикатора
    pass
```

---

## 📚 СВЯЗАННЫЕ ДОКУМЕНТЫ

### Дополнительные ресурсы:
- [Анализ подходов к стоп-лоссам и тейк-профитам (OsEngine)](OSENGINE_STOP_TAKE_ANALYSIS.md)
  - Детальный анализ различных подходов к риск-менеджменту
  - Рекомендации по реализации для нашего проекта
  - **Статус:** Готово к использованию, интеграция запланирована на будущее

---

**Дата создания:** 2025-11-10  
**Дата обновления:** 2025-11-16  
**Версия:** 2.0.0

