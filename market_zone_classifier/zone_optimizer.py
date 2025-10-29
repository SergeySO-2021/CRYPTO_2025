"""
Market Zone Classifier - Parameter Optimizer
Оптимизация параметров классификатора зон
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Try to import optimization libraries
try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("⚠️ scikit-optimize не установлен. Используем Genetic Algorithm")
    print("💡 Установите: pip install scikit-optimize")

try:
    from deap import base, creator, tools, algorithms
    GA_AVAILABLE = True
except ImportError:
    GA_AVAILABLE = False
    print("⚠️ DEAP не установлен. Используем простой Random Search")
    print("💡 Установите: pip install deap")


class ZoneClassifierOptimizer:
    """Оптимизатор параметров Market Zone Classifier"""
    
    def __init__(self, data, optimization_method='bayesian', 
                 metric_weights=None):
        """
        Args:
            data: DataFrame с OHLCV данными
            optimization_method: 'bayesian', 'genetic', или 'random'
            metric_weights: Словарь с весами метрик. По умолчанию:
                {
                    'stability': 0.3,    # Стабильность зон
                    'separation': 0.3,   # Разделение зон
                    'economic': 0.4      # Экономическая ценность
                }
        """
        self.data = data.copy()
        self.method = optimization_method
        
        # Настройка весов метрик
        if metric_weights is None:
            self.metric_weights = {
                'stability': 0.3,
                'separation': 0.3,
                'economic': 0.4
            }
        else:
            self.metric_weights = metric_weights
            # Нормализуем веса
            total = sum(self.metric_weights.values())
            self.metric_weights = {k: v/total for k, v in self.metric_weights.items()}
        
        # Проверяем доступность методов
        if optimization_method == 'bayesian' and not BAYESIAN_AVAILABLE:
            print("⚠️ Bayesian Optimization недоступен, используем Genetic Algorithm")
            self.method = 'genetic'
        
        if optimization_method == 'genetic' and not GA_AVAILABLE:
            print("⚠️ Genetic Algorithm недоступен, используем Random Search")
            self.method = 'random'
    
    def classify_zones(self, lookback_period, trend_threshold, 
                      volatility_period, volatility_threshold):
        """Классификация зон с заданными параметрами"""
        df = self.data.copy()
        
        # Price extremes (без look ahead)
        df['highest_high'] = df['high'].shift(1).rolling(window=lookback_period-1).max()
        df['lowest_low'] = df['low'].shift(1).rolling(window=lookback_period-1).min()
        df['price_range'] = df['highest_high'] - df['lowest_low']
        
        # Moving averages
        df['fast_ma'] = df['close'].shift(1).rolling(window=10).mean()
        df['slow_ma'] = df['close'].shift(1).rolling(window=20).mean()
        df['ma_slope'] = df['fast_ma'] - df['slow_ma']
        
        # Volatility
        df['atr'] = df.apply(
            lambda row: max(
                row['high'] - row['low'],
                abs(row['high'] - df.loc[row.name - 1, 'close']) if row.name > 0 else 0,
                abs(row['low'] - df.loc[row.name - 1, 'close']) if row.name > 0 else 0
            ), axis=1
        ) if len(df) > 0 else pd.Series()
        df['atr'] = df['atr'].rolling(window=volatility_period).mean()
        df['atr_ma'] = df['atr'].rolling(window=volatility_period).mean()
        df['volatility_ratio'] = df['atr'] / df['atr_ma']
        
        # Bollinger Bands
        df['bb_basis'] = df['close'].rolling(window=volatility_period).mean()
        df['bb_dev'] = df['close'].rolling(window=volatility_period).std()
        df['bb_width'] = (df['bb_basis'] + 2 * df['bb_dev'] - (df['bb_basis'] - 2 * df['bb_dev'])) / df['bb_basis']
        
        # Trend classification
        df['trend_up'] = (df['ma_slope'] > trend_threshold) & (df['close'].shift(1) > df['fast_ma'])
        df['trend_down'] = (df['ma_slope'] < -trend_threshold) & (df['close'].shift(1) < df['fast_ma'])
        
        # Volatility classification
        df['high_volatility'] = (df['volatility_ratio'] > volatility_threshold) | (df['bb_width'] > 0.1)
        df['low_volatility'] = (df['volatility_ratio'] < (1 / volatility_threshold)) & (df['bb_width'] < 0.05)
        
        # Zone determination
        df['primary_zone'] = np.where(df['trend_up'], 1,
                                    np.where(df['trend_down'], -1, 0))
        df['secondary_zone'] = np.where(df['high_volatility'], 2,
                                       np.where(df['low_volatility'], -2, 0))
        df['zone'] = df['primary_zone'] + df['secondary_zone']
        
        return df['zone'].dropna()
    
    def calculate_zone_stability(self, zones):
        """Вычисление стабильности зон"""
        if len(zones) < 2:
            return 0.0
        
        changes = (zones != zones.shift(1)).sum()
        total = len(zones) - 1
        stability = 1 - (changes / total) if total > 0 else 0
        return max(0, min(1, stability))
    
    def calculate_zone_separation(self, zones, returns):
        """Вычисление разделения зон по доходности"""
        if len(zones) == 0 or len(returns) == 0:
            return 0.0
        
        try:
            by_zone = returns.groupby(zones).mean()
            if len(by_zone) < 2:
                return 0.0
            
            separation = by_zone.max() - by_zone.min()
            return abs(separation) if not pd.isna(separation) else 0.0
        except:
            return 0.0
    
    def calculate_economic_value(self, zones, returns):
        """Вычисление экономической ценности разделения зон"""
        if len(zones) == 0 or len(returns) == 0:
            return 0.0
        
        try:
            by_zone = returns.groupby(zones).mean()
            
            # Средние доходы для бычьих и медвежьих зон
            bull_zones = [1, 2, 3]
            bear_zones = [-1, -2, -3]
            
            bull_returns = by_zone[by_zone.index.isin(bull_zones)].mean()
            bear_returns = by_zone[by_zone.index.isin(bear_zones)].mean()
            
            if pd.isna(bull_returns):
                bull_returns = 0
            if pd.isna(bear_returns):
                bear_returns = 0
            
            economic_value = abs(bull_returns - bear_returns)
            return economic_value if not pd.isna(economic_value) else 0.0
        except:
            return 0.0
    
    def calculate_combined_score(self, zones, returns):
        """Вычисление комбинированной оценки качества
        
        Использует веса из self.metric_weights:
        - stability: стабильность зон
        - separation: разделение зон
        - economic: экономическая ценность
        """
        stability = self.calculate_zone_stability(zones) * self.metric_weights.get('stability', 0.3)
        separation = self.calculate_zone_separation(zones, returns) * self.metric_weights.get('separation', 0.3)
        economic = self.calculate_economic_value(zones, returns) * self.metric_weights.get('economic', 0.4)
        
        return stability + separation + economic
    
    def get_metric_breakdown(self, zones, returns):
        """Получить детальную разбивку метрик"""
        stability = self.calculate_zone_stability(zones)
        separation = self.calculate_zone_separation(zones, returns)
        economic = self.calculate_economic_value(zones, returns)
        
        return {
            'stability': stability,
            'separation': separation,
            'economic': economic,
            'stability_weighted': stability * self.metric_weights.get('stability', 0.3),
            'separation_weighted': separation * self.metric_weights.get('separation', 0.3),
            'economic_weighted': economic * self.metric_weights.get('economic', 0.4),
            'total_score': self.calculate_combined_score(zones, returns)
        }
    
    def objective_function(self, params):
        """Целевая функция для оптимизации"""
        lookback, trend_thresh, vol_period, vol_thresh = params
        
        # Классифицируем зоны
        try:
            zones = self.classify_zones(
                int(lookback), 
                float(trend_thresh),
                int(vol_period),
                float(vol_thresh)
            )
            
            # Вычисляем доходность
            returns = self.data['close'].pct_change().dropna()
            
            # Выравниваем индексы
            common_idx = zones.index.intersection(returns.index)
            if len(common_idx) < 100:  # Минимум данных
                return -1000  # Очень плохая оценка
            
            zones_aligned = zones.loc[common_idx]
            returns_aligned = returns.loc[common_idx]
            
            # Вычисляем метрики
            score = self.calculate_combined_score(zones_aligned, returns_aligned)
            
            # Минимизируем отрицательный score
            return -score
            
        except Exception as e:
            return -1000  # Ошибка - очень плохая оценка
    
    def optimize_bayesian(self, n_calls=100, n_initial_points=20):
        """Bayesian Optimization"""
        if not BAYESIAN_AVAILABLE:
            raise ImportError("scikit-optimize не установлен")
        
        # Пространство параметров
        dimensions = [
            Integer(5, 100, name='lookbackPeriod'),
            Real(0.1, 2.0, name='trendThreshold'),
            Integer(5, 50, name='volatilityPeriod'),
            Real(0.5, 3.0, name='volatilityThreshold')
        ]
        
        # Оптимизация
        result = gp_minimize(
            func=self.objective_function,
            dimensions=dimensions,
            n_calls=n_calls,
            acq_func='EI',
            n_initial_points=n_initial_points,
            random_state=42,
            verbose=True
        )
        
        # Получаем количество итераций (используем доступные атрибуты)
        try:
            # Пробуем разные способы получить количество итераций
            if hasattr(result, 'x_iters') and len(result.x_iters) > 0:
                iterations = len(result.x_iters)
            elif hasattr(result, 'func_vals') and len(result.func_vals) > 0:
                iterations = len(result.func_vals)
            else:
                iterations = n_calls  # Используем переданное значение
        except:
            iterations = n_calls
        
        best_params = {
            'lookbackPeriod': int(result.x[0]),
            'trendThreshold': float(result.x[1]),
            'volatilityPeriod': int(result.x[2]),
            'volatilityThreshold': float(result.x[3]),
            'score': -result.fun,
            'iterations': iterations,
            'method': 'Bayesian Optimization'
        }
        
        return best_params
    
    def optimize_genetic(self, population_size=30, generations=50):
        """Genetic Algorithm Optimization"""
        if not GA_AVAILABLE:
            raise ImportError("DEAP не установлен")
        
        # Настройка DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Гены (параметры)
        toolbox.register("attr_lookback", np.random.randint, 5, 101)
        toolbox.register("attr_trend", lambda: np.round(np.random.uniform(0.1, 2.0), 1))
        toolbox.register("attr_vol_period", np.random.randint, 5, 51)
        toolbox.register("attr_vol_thresh", lambda: np.round(np.random.uniform(0.5, 3.0), 1))
        
        toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.attr_lookback, toolbox.attr_trend,
                         toolbox.attr_vol_period, toolbox.attr_vol_thresh), n=1)
        
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", lambda ind: (self.objective_function(ind),))
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate_individual, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Мутация
        def mutate_individual(individual):
            if np.random.random() < 0.25:
                individual[0] = np.random.randint(5, 101)  # lookback
            if np.random.random() < 0.25:
                individual[1] = np.round(np.random.uniform(0.1, 2.0), 1)  # trend
            if np.random.random() < 0.25:
                individual[2] = np.random.randint(5, 51)  # vol_period
            if np.random.random() < 0.25:
                individual[3] = np.round(np.random.uniform(0.5, 3.0), 1)  # vol_thresh
            return individual
        
        toolbox.register("mutate", mutate_individual)
        
        # Создаем популяцию
        population = toolbox.population(n=population_size)
        
        # Оцениваем начальную популяцию
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Эволюция
        best_individual = None
        best_fitness = float('-inf')
        
        for gen in range(generations):
            # Отбор
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            # Кроссовер
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < 0.8:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Мутация
            for mutant in offspring:
                if np.random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Оценка новых особей
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Заменяем популяцию
            population[:] = offspring
            
            # Трек лучшей особи
            best_gen = tools.selBest(population, 1)[0]
            if best_gen.fitness.values[0] > best_fitness:
                best_fitness = best_gen.fitness.values[0]
                best_individual = best_gen
            
            if (gen + 1) % 10 == 0:
                print(f"Поколение {gen + 1}/{generations}: Лучший score = {-best_fitness:.6f}")
        
        best_params = {
            'lookbackPeriod': int(best_individual[0]),
            'trendThreshold': float(best_individual[1]),
            'volatilityPeriod': int(best_individual[2]),
            'volatilityThreshold': float(best_individual[3]),
            'score': -best_fitness,
            'iterations': generations * population_size,
            'method': 'Genetic Algorithm'
        }
        
        return best_params
    
    def optimize_random(self, n_trials=500):
        """Random Search Optimization"""
        best_score = float('-inf')
        best_params = None
        
        for i in range(n_trials):
            params = [
                np.random.randint(5, 101),  # lookback
                np.round(np.random.uniform(0.1, 2.0), 1),  # trend_thresh
                np.random.randint(5, 51),  # vol_period
                np.round(np.random.uniform(0.5, 3.0), 1)  # vol_thresh
            ]
            
            score = -self.objective_function(params)
            
            if score > best_score:
                best_score = score
                best_params = params
                print(f"Попытка {i+1}/{n_trials}: Новый лучший score = {score:.6f}")
        
        return {
            'lookbackPeriod': int(best_params[0]),
            'trendThreshold': float(best_params[1]),
            'volatilityPeriod': int(best_params[2]),
            'volatilityThreshold': float(best_params[3]),
            'score': best_score,
            'iterations': n_trials,
            'method': 'Random Search'
        }
    
    def optimize(self, **kwargs):
        """Главный метод оптимизации"""
        print(f"🚀 Запуск оптимизации методом: {self.method}")
        print("=" * 50)
        
        if self.method == 'bayesian':
            return self.optimize_bayesian(
                n_calls=kwargs.get('n_calls', 100),
                n_initial_points=kwargs.get('n_initial_points', 20)
            )
        elif self.method == 'genetic':
            return self.optimize_genetic(
                population_size=kwargs.get('population_size', 30),
                generations=kwargs.get('generations', 50)
            )
        else:  # random
            return self.optimize_random(
                n_trials=kwargs.get('n_trials', 500)
            )


def load_btc_data(base_path=None):
    """
    Загрузка данных BTC из CSV файлов
    
    Args:
        base_path: Базовый путь к корню проекта (опционально)
                   По умолчанию определяется автоматически
    
    Returns:
        Словарь с данными по таймфреймам
    """
    import sys
    import os
    
    # Определяем базовый путь
    if base_path is None:
        # Пробуем найти корень проекта
        current_file = os.path.abspath(__file__)
        if 'market_zone_classifier' in current_file:
            # Находим корень проекта (на уровень выше market_zone_classifier)
            base_path = os.path.dirname(os.path.dirname(current_file))
        else:
            # Используем текущую рабочую директорию
            base_path = os.getcwd()
    
    print(f"📁 Базовый путь: {base_path}")
    
    # Список таймфреймов
    timeframes = ['15m', '30m', '1h', '4h', '1d']
    
    # Приоритет файлов для каждого таймфрейма
    file_priorities = [
        'df_btc_{tf}_complete.csv',
        'df_btc_{tf}_matching.csv',
        'df_btc_{tf}_large.csv',
        'df_btc_{tf}_real.csv',
        'df_btc_{tf}.csv'
    ]
    
    dataframes = {}
    
    print("\n📊 ЗАГРУЗКА ДАННЫХ BTC ИЗ ФАЙЛОВ")
    print("=" * 40)
    
    for tf in timeframes:
        df = None
        
        # Пробуем загрузить из разных вариантов файлов
        for file_template in file_priorities:
            filename = file_template.format(tf=tf)
            filepath = os.path.join(base_path, filename)
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    print(f"✅ {tf}: Загружено из {filename} ({len(df)} записей)")
                    
                    # Проверяем наличие необходимых колонок
                    required_columns = ['timestamps', 'open', 'high', 'low', 'close']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        print(f"⚠️ {tf}: Отсутствуют колонки: {missing_columns}")
                        df = None
                        continue
                    
                    # Конвертируем timestamps
                    if 'timestamps' in df.columns:
                        df['timestamps'] = pd.to_datetime(df['timestamps'])
                        df.set_index('timestamps', inplace=True)
                    
                    # Добавляем volume если отсутствует
                    if 'volume' not in df.columns:
                        print(f"⚠️ {tf}: Volume отсутствует, добавляем синтетический")
                        price_range = df['high'] - df['low']
                        avg_price = df['close'].mean()
                        np.random.seed(42)
                        random_factor = np.random.uniform(0.5, 2.0, len(df))
                        df['volume'] = (price_range * avg_price * random_factor).astype(int)
                    
                    # Сортируем по индексу
                    df.sort_index(inplace=True)
                    
                    dataframes[tf] = df
                    break
                    
                except Exception as e:
                    print(f"❌ {tf}: Ошибка загрузки {filename} - {e}")
                    continue
        
        if df is None:
            print(f"❌ {tf}: Не удалось загрузить данные")
    
    if not dataframes:
        print("\n⚠️ ВНИМАНИЕ: Не удалось загрузить данные из файлов!")
        print("💡 Убедитесь, что файлы находятся в корне проекта:")
        print(f"   {base_path}")
        print("\n📋 Ожидаемые имена файлов:")
        for tf in timeframes:
            print(f"   - df_btc_{tf}.csv (или другие варианты)")
        print("\n💡 Создаем синтетические данные для демонстрации...")
        
        # Создаем синтетические данные для демонстрации
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='1H')
        np.random.seed(42)
        
        data = []
        price = 30000
        
        for date in dates:
            price += np.random.normal(0, 100)
            data.append({
                'open': price + np.random.normal(0, 50),
                'high': price + abs(np.random.normal(0, 100)),
                'low': price - abs(np.random.normal(0, 100)),
                'close': price,
                'volume': np.random.uniform(1000, 10000)
            })
        
        df = pd.DataFrame(data, index=dates)
        dataframes['1h'] = df
        print(f"✅ Создано {len(df)} синтетических записей для таймфрейма 1h")
    
    return dataframes


if __name__ == "__main__":
    print("🔧 Market Zone Classifier - Parameter Optimizer")
    print("=" * 50)
    
    # Загружаем данные
    print("📊 Загрузка данных...")
    btc_data = load_btc_data()
    
    # Выбираем таймфрейм (например, 1h)
    if '1h' in btc_data:
        df = btc_data['1h']
        print(f"✅ Загружено {len(df)} записей для таймфрейма 1h")
    else:
        # Берем первый доступный таймфрейм
        tf = list(btc_data.keys())[0]
        df = btc_data[tf]
        print(f"✅ Загружено {len(df)} записей для таймфрейма {tf}")
    
    # Создаем оптимизатор
    print("\n🎯 Выбор метода оптимизации...")
    
    # Проверяем доступность методов
    if BAYESIAN_AVAILABLE:
        print("✅ Bayesian Optimization доступен - РЕКОМЕНДУЕТСЯ")
        method = 'bayesian'
    elif GA_AVAILABLE:
        print("✅ Genetic Algorithm доступен")
        method = 'genetic'
    else:
        print("⚠️ Только Random Search доступен")
        method = 'random'
    
    optimizer = ZoneClassifierOptimizer(df, optimization_method=method)
    
    # Запускаем оптимизацию
    print(f"\n🚀 Запуск оптимизации ({method})...")
    results = optimizer.optimize()
    
    # Выводим результаты
    print("\n" + "=" * 50)
    print("🏆 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("=" * 50)
    print(f"Метод: {results['method']}")
    print(f"Итераций: {results['iterations']}")
    print(f"Лучший Score: {results['score']:.6f}")
    print(f"\n📊 ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:")
    print(f"   Lookback Period: {results['lookbackPeriod']}")
    print(f"   Trend Threshold: {results['trendThreshold']}")
    print(f"   Volatility Period: {results['volatilityPeriod']}")
    print(f"   Volatility Threshold: {results['volatilityThreshold']}")
    print("\n✅ Оптимизация завершена!")
