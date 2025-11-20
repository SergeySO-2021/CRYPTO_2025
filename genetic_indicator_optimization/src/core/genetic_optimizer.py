"""
Simple genetic algorithm skeleton that optimizes MVP strategy parameters.

This optimizer loads GA + strategy configs, generates candidate genomes
according to the defined search space, evaluates them via the MVP pipeline
(indicator calculation → signals → backtest), and keeps track of the best
configuration based on validation metrics.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from functools import partial

import yaml
import numpy as np

try:
    from multiprocessing import cpu_count
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False
    cpu_count = lambda: 1

try:
    from pathos.multiprocessing import ProcessingPool
    PATHOS_AVAILABLE = True
except ImportError:
    PATHOS_AVAILABLE = False

from analysis import IndicatorPipeline
from data_loader import DataLoader
from strategies.mvp_strategy import MVPStrategy, MVPSignalConfig
from .simple_backtester import SimpleBacktester, BacktestConfig


@dataclass
class Individual:
    genes: Dict[str, Any]
    fitness: Optional[float] = None
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class GeneticOptimizer:
    """
    Genetic algorithm driver that evaluates configurations via MVP backtests.
    """

    def __init__(
        self,
        ga_config_path: str,
        strategy_config_path: str,
        data_path: Optional[str] = None,
        population_size: Optional[int] = None,
        max_generations: Optional[int] = None,
    ):
        self.ga_config = self._load_yaml(ga_config_path)
        self.strategy_config = self._load_yaml(strategy_config_path)
        self.search_space = self.ga_config.get("search_space", {})
        if not self.search_space:
            raise ValueError("Search space is empty. Define parameters in ga_config.yaml::search_space.")

        self.population_size = population_size or self.ga_config["genetic_algorithm"]["population_size"]
        self.max_generations = max_generations or self.ga_config["genetic_algorithm"]["max_generations"]
        self.crossover_rate = self.ga_config["genetic_algorithm"]["crossover_rate"]
        self.mutation_rate = self.ga_config["genetic_algorithm"]["mutation_rate"]
        self.elite_size = self.ga_config["genetic_algorithm"]["elite_size"]
        self.tournament_size = self.ga_config["genetic_algorithm"]["tournament_size"]
        
        # Параллелизация
        ga_params = self.ga_config["genetic_algorithm"]
        self.use_parallel = ga_params.get("use_parallel", False) and PATHOS_AVAILABLE
        if not PATHOS_AVAILABLE and ga_params.get("use_parallel", False):
            print("[WARN] pathos not available, parallel evaluation disabled. Install with: pip install pathos")
        self.n_jobs = ga_params.get("n_jobs", min(4, cpu_count() if MULTIPROCESSING_AVAILABLE else 1))
        
        # Early stopping
        self.stagnation_limit = ga_params.get("stagnation_limit", None)
        self.target_fitness = ga_params.get("target_fitness", None)
        
        # Адаптивная мутация
        self.adaptive_mutation = ga_params.get("adaptive_mutation", False)
        self.adaptive_mutation_max = ga_params.get("adaptive_mutation_max", 0.25)
        self.fixed_generations = ga_params.get("fixed_generations", 0)

        loader = DataLoader(data_path=data_path)
        self.raw_data = loader.load_data()
        train_df, val_df, test_df = loader.split_data()
        self.split_indices = {
            "train": train_df.index,
            "val": val_df.index,
            "test": test_df.index,
        }
        self.full_index = self.raw_data.index

        self.fitness_weights = self.ga_config["fitness"]["weights"]
        self.fitness_penalties = self.ga_config["fitness"].get("penalties", {})
        self.fitness_bonuses = self.ga_config["fitness"].get("bonuses", {})
        self.evaluation_cache: Dict[Tuple[Tuple[str, Any], ...], Individual] = {}

    def run(self) -> Individual:
        population = [Individual(genes=self._random_genes()) for _ in range(self.population_size)]
        best_individual: Optional[Individual] = None
        best_fitness = -math.inf
        stagnation_count = 0

        if self.use_parallel:
            print(f"[GA] Using parallel evaluation with {self.n_jobs} processes (pathos)")

        for generation in range(self.max_generations):
            # Параллельная или последовательная оценка
            if self.use_parallel:
                evaluated_population = self._evaluate_population_parallel(population)
            else:
                evaluated_population = [self._evaluate(ind) for ind in population]
            
            evaluated_population.sort(key=lambda ind: ind.fitness or -math.inf, reverse=True)

            current_best = evaluated_population[0]
            current_fitness = current_best.fitness or -math.inf
            
            # Обновляем лучшую особь
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_individual = copy.deepcopy(current_best)
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Логирование прогресса
            avg_fitness = sum((ind.fitness or -math.inf) for ind in evaluated_population) / len(evaluated_population)
            print(
                f"[GA] Generation {generation+1}/{self.max_generations} "
                f"best={current_fitness:.2f} avg={avg_fitness:.2f} "
                f"stagnation={stagnation_count}/{self.stagnation_limit or 'N/A'}"
            )

            # Early stopping по target fitness
            if self.target_fitness and current_fitness >= self.target_fitness:
                print(f"[GA] Early stopping: target fitness {self.target_fitness} reached!")
                break

            # Early stopping по stagnation
            if self.stagnation_limit and stagnation_count >= self.stagnation_limit:
                print(f"[GA] Early stopping: no improvement for {stagnation_count} generations")
                break

            if generation + 1 == self.max_generations:
                break

            # Адаптивная мутация
            current_mutation_rate = self._get_adaptive_mutation_rate(generation, stagnation_count)
            
            next_population = evaluated_population[: self.elite_size]
            while len(next_population) < self.population_size:
                parent_a = self._tournament_select(evaluated_population)
                parent_b = self._tournament_select(evaluated_population)

                if random.random() < self.crossover_rate:
                    child_genes = self._crossover(parent_a.genes, parent_b.genes)
                else:
                    child_genes = copy.deepcopy(parent_a.genes)

                child_genes = self._mutate(child_genes, mutation_rate=current_mutation_rate)
                next_population.append(Individual(genes=child_genes))

            population = next_population

        if best_individual is None:
            raise RuntimeError("GA finished without evaluating any individuals.")
        return best_individual

    # ---------------- GA helpers ---------------- #

    def _random_genes(self) -> Dict[str, Any]:
        genes = {}
        for name, definition in self.search_space.items():
            genes[name] = self._random_value(definition)
        # Применяем constraints после генерации случайных генов
        genes = self._apply_constraints(genes)
        return genes

    def _apply_constraints(self, genes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Применяет constraints к генам для Long/Short параметров и WOBI весов.
        
        Constraints:
        - long_signal_multiplier + short_signal_multiplier >= 1.6
        - entry_threshold_long >= entry_threshold_short
        - WOBI веса: сумма должна быть > 0 (нормализуется в _build_feature_dataset)
        """
        constrained = copy.deepcopy(genes)
        
        # Constraint 1: long_signal_multiplier + short_signal_multiplier >= 1.6
        if "long_signal_multiplier" in constrained and "short_signal_multiplier" in constrained:
            long_mult = constrained["long_signal_multiplier"]
            short_mult = constrained["short_signal_multiplier"]
            sum_mult = long_mult + short_mult
            
            if sum_mult < 1.6:
                # Корректируем оба множителя пропорционально
                scale = 1.6 / sum_mult
                constrained["long_signal_multiplier"] = min(1.2, max(0.8, long_mult * scale))
                constrained["short_signal_multiplier"] = min(1.2, max(0.8, short_mult * scale))
        
        # Constraint 2: entry_threshold_long >= entry_threshold_short
        if "entry_threshold_long" in constrained and "entry_threshold_short" in constrained:
            long_thresh = constrained["entry_threshold_long"]
            short_thresh = constrained["entry_threshold_short"]
            
            if long_thresh < short_thresh:
                # Устанавливаем long_thresh = short_thresh (минимальная корректировка)
                constrained["entry_threshold_long"] = short_thresh
        
        # Constraint 3: WOBI веса - гарантируем, что хотя бы один вес > 0
        wobi_weights = ["wobi_weight_ratio3", "wobi_weight_ratio5", "wobi_weight_ratio8", "wobi_weight_ratio60"]
        if all(w in constrained for w in wobi_weights):
            total = sum(constrained[w] for w in wobi_weights)
            if total <= 0:
                # Если сумма <= 0, устанавливаем равномерные веса
                for w in wobi_weights:
                    constrained[w] = 0.25

        # Constraint 4: MACD fast < slow
        if "macd_fast_period" in constrained and "macd_slow_period" in constrained:
            fast_def = self.search_space.get("macd_fast_period", {})
            slow_def = self.search_space.get("macd_slow_period", {})
            fast_min = fast_def.get("min", 1)
            fast_max = fast_def.get("max", 100)
            slow_min = slow_def.get("min", fast_min + 1)
            slow_max = slow_def.get("max", fast_max + 10)

            fast = max(fast_min, min(constrained["macd_fast_period"], fast_max))
            slow = max(slow_min, min(constrained["macd_slow_period"], slow_max))

            if fast >= slow:
                slow = min(slow_max, max(fast + 1, slow_min))
                if slow <= fast:
                    fast = max(fast_min, min(slow - 1, fast_max))
                    slow = min(slow_max, max(fast + 1, slow_min))

            constrained["macd_fast_period"] = fast
            constrained["macd_slow_period"] = slow
        
        return constrained

    def _mutate(self, genes: Dict[str, Any], mutation_rate: Optional[float] = None) -> Dict[str, Any]:
        mutated = copy.deepcopy(genes)
        rate = mutation_rate if mutation_rate is not None else self.mutation_rate
        for name, definition in self.search_space.items():
            if random.random() <= rate:
                mutated[name] = self._mutated_value(mutated[name], definition)
        # Применяем constraints после мутации
        mutated = self._apply_constraints(mutated)
        return mutated

    def _get_adaptive_mutation_rate(self, generation: int, stagnation_count: int) -> float:
        """Вычисляет адаптивную мутацию на основе поколения и застоя."""
        if not self.adaptive_mutation:
            return self.mutation_rate
        
        base_rate = self.mutation_rate
        
        # В фиксированных поколениях используем базовую мутацию
        if generation < self.fixed_generations:
            return base_rate
        
        # Увеличиваем мутацию при застое
        if stagnation_count > 5:
            adaptive_rate = base_rate * (1 + stagnation_count * 0.05)
            return min(self.adaptive_mutation_max, adaptive_rate)
        
        # Увеличиваем мутацию в конце эволюции
        progress = generation / self.max_generations
        if progress > 0.7:
            return min(self.adaptive_mutation_max, base_rate * 1.5)
        
        return base_rate

    def _crossover(self, genes_a: Dict[str, Any], genes_b: Dict[str, Any]) -> Dict[str, Any]:
        child = {}
        for name in genes_a.keys():
            child[name] = random.choice([genes_a[name], genes_b[name]])
        # Применяем constraints после кроссовера
        child = self._apply_constraints(child)
        return child

    def _tournament_select(self, population: List[Individual]) -> Individual:
        contenders = random.sample(population, k=min(self.tournament_size, len(population)))
        contenders.sort(key=lambda ind: ind.fitness or -math.inf, reverse=True)
        return contenders[0]

    def _evaluate_population_parallel(self, population: List[Individual]) -> List[Individual]:
        """
        Параллельная оценка популяции с использованием pathos.
        Pathos использует dill для сериализации, что позволяет работать с методами классов.
        """
        if not PATHOS_AVAILABLE:
            print("[WARN] pathos not available, using sequential evaluation")
            return [self._evaluate(ind) for ind in population]
        
        try:
            with ProcessingPool(nodes=self.n_jobs) as pool:
                # pathos может pickle методы классов благодаря dill
                evaluated = pool.map(self._evaluate, population)
                return evaluated
        except Exception as e:
            print(f"[WARN] Parallel evaluation failed: {e}, using sequential")
            return [self._evaluate(ind) for ind in population]

    # ---------------- Evaluation ---------------- #

    def _evaluate(self, individual: Individual) -> Individual:
        cache_key = tuple(sorted(individual.genes.items()))
        if cache_key in self.evaluation_cache:
            cached = self.evaluation_cache[cache_key]
            return copy.deepcopy(cached)

        enriched = self._build_feature_dataset(individual.genes)
        segments = self._prepare_segments(enriched)
        strategy_config = self._build_strategy_config(individual.genes)
        signal_config = MVPSignalConfig.from_dict(strategy_config)
        backtest_config = BacktestConfig.from_dict(strategy_config.get("risk"))

        strategy = MVPStrategy(signal_config)
        metrics: Dict[str, Dict[str, Any]] = {}

        for split_name, df in segments.items():
            if df.empty:
                metrics[split_name] = self._empty_metrics()
                continue
            signals = strategy.generate_signals(df)
            backtester = SimpleBacktester(backtest_config)
            result = backtester.run(df, signals)
            metrics[split_name] = result.metrics

        fitness = self._calculate_fitness(metrics, individual.genes)

        evaluated = Individual(
            genes=copy.deepcopy(individual.genes),
            fitness=fitness,
            metrics=metrics,
        )
        self.evaluation_cache[cache_key] = evaluated
        return evaluated

    def _build_feature_dataset(self, genes: Dict[str, Any]):

        indicator_params = {
            "rsi": {"period": genes["rsi_period"]},
            "atr": {"period": genes["atr_period"]},
        }
        
        # Добавляем параметры MACD, если они есть в генах
        if "macd_fast_period" in genes and "macd_slow_period" in genes and "macd_signal_period" in genes:
            indicator_params["macd"] = {
                "fast_period": genes["macd_fast_period"],
                "slow_period": genes["macd_slow_period"],
                "signal_period": genes["macd_signal_period"],
            }
        
        # Добавляем параметры Bollinger, если они есть в генах
        if "bollinger_period" in genes and "bollinger_std_dev" in genes:
            indicator_params["bollinger"] = {
                "period": genes["bollinger_period"],
                "std_dev": genes["bollinger_std_dev"],
            }
        
        # Добавляем параметры WOBI (веса глубин), если они есть в генах
        if all(f"wobi_weight_ratio{k}" in genes for k in [3, 5, 8, 60]):
            # Нормализуем веса, чтобы сумма была равна 1.0
            w3 = max(0.0, genes["wobi_weight_ratio3"])
            w5 = max(0.0, genes["wobi_weight_ratio5"])
            w8 = max(0.0, genes["wobi_weight_ratio8"])
            w60 = max(0.0, genes["wobi_weight_ratio60"])
            total = w3 + w5 + w8 + w60
            if total > 0:
                indicator_params["wobi"] = {
                    "weights": {
                        "ratio3": w3 / total,
                        "ratio5": w5 / total,
                        "ratio8": w8 / total,
                        "ratio60": w60 / total,
                    }
                }
        
        pipeline = IndicatorPipeline(self.raw_data, params=indicator_params)
        artifacts = pipeline.run()
        return artifacts.data.dropna()

    def _prepare_segments(self, enriched):
        def subset(index):
            subset_df = enriched.reindex(index)
            return subset_df.dropna()

        return {
            "train": subset(self.split_indices["train"]),
            "val": subset(self.split_indices["val"]),
            "test": subset(self.split_indices["test"]),
            "full": enriched.dropna(),
        }

    def _build_strategy_config(self, genes: Dict[str, Any]) -> Dict[str, Any]:
        cfg = copy.deepcopy(self.strategy_config)

        signals_cfg = cfg.setdefault("signals", {})
        
        # RSI параметры
        rsi_cfg = signals_cfg.setdefault("rsi", {})
        rsi_cfg["column"] = f"rsi_{genes['rsi_period']}"
        if "rsi_weight" in genes:
            rsi_cfg["weight"] = float(genes["rsi_weight"])
        if "rsi_buy_below_long" in genes:
            rsi_cfg["buy_below_long"] = float(genes["rsi_buy_below_long"])
        if "rsi_sell_above_short" in genes:
            rsi_cfg["sell_above_short"] = float(genes["rsi_sell_above_short"])
        
        # MACD параметры
        macd_cfg = signals_cfg.setdefault("macd", {})
        if "macd_weight" in genes:
            macd_cfg["weight"] = float(genes["macd_weight"])
        # Параметры MACD (fast/slow/signal) обрабатываются в _build_feature_dataset
        
        # Bollinger параметры
        boll_cfg = signals_cfg.setdefault("bollinger", {})
        if "bollinger_weight" in genes:
            boll_cfg["weight"] = float(genes["bollinger_weight"])
        # Параметры Bollinger (period/std_dev) обрабатываются в _build_feature_dataset
        
        # WOBI параметры
        wobi_cfg = signals_cfg.setdefault("wobi", {})
        if "wobi_weight" in genes:
            wobi_cfg["weight"] = float(genes["wobi_weight"])
        # Веса глубин WOBI обрабатываются в _build_feature_dataset

        risk_cfg = cfg.setdefault("risk", {})
        risk_cfg["stop_loss_pct"] = genes["stop_loss_pct"]
        risk_cfg["take_profit_pct"] = genes["take_profit_pct"]

        atr_cfg = risk_cfg.setdefault("atr", {})
        atr_cfg["enabled"] = True
        atr_cfg["period"] = genes["atr_period"]
        atr_cfg["stop_multiplier"] = genes["atr_stop_multiplier"]
        atr_cfg["trailing_multiplier"] = genes["atr_trailing_multiplier"]

        # Time filter parameters (оптимизируются ГА)
        # ГА может решить: включить фильтр (1) или торговать весь день (0)
        # Если включен, ГА также выбирает начало окна (0-23) и длину (1-12 часов)
        time_cfg = cfg.setdefault("time_filter", {})
        enabled_gene = genes.get("time_filter_enabled")
        if enabled_gene is not None:
            time_cfg["enabled"] = bool(enabled_gene)
        enabled = time_cfg.get("enabled", False)

        if enabled:
            # Фильтр включен: используем параметры из генов
            default_start = time_cfg.get("allowed_hours_start", 0)
            default_end = time_cfg.get("allowed_hours_end", default_start + 4)
            start = int(genes.get("time_window_start", default_start))
            base_length = max(1, default_end - default_start)
            length = int(genes.get("time_window_length", base_length))
            length = max(1, length)
            # Убеждаемся, что окно не выходит за границы суток
            end = min(start + length, 24)
            if end <= start:
                end = min(start + 4, 24)
            time_cfg["allowed_hours_start"] = start
            time_cfg["allowed_hours_end"] = end
        else:
            # Фильтр выключен: торгуем весь день (параметры start/end игнорируются)
            time_cfg["enabled"] = False

        # Long/Short балансировка параметры (оптимизируются ГА)
        # ГА может оптимизировать множители сигналов и пороги входа для балансировки Long/Short
        combination_cfg = cfg.setdefault("combination", {})
        if "long_signal_multiplier" in genes:
            combination_cfg["long_signal_multiplier"] = float(genes["long_signal_multiplier"])
        if "short_signal_multiplier" in genes:
            combination_cfg["short_signal_multiplier"] = float(genes["short_signal_multiplier"])
        if "entry_threshold_long" in genes:
            combination_cfg["entry_threshold_long"] = float(genes["entry_threshold_long"])
        if "entry_threshold_short" in genes:
            combination_cfg["entry_threshold_short"] = float(genes["entry_threshold_short"])

        return cfg

    def _safe_metric_value(self, value: float, metric_name: str, default: float = 0.0) -> float:
        """
        Безопасное вычисление метрик с обработкой edge cases.
        
        Обрабатывает Infinity и NaN значения, ограничивая их разумными пределами.
        """
        if np.isinf(value) or np.isnan(value):
            # Для profit_factor: Infinity -> ограничиваем сверху
            if metric_name == "profit_factor":
                return 10.0  # Максимальное разумное значение
            # Для sharpe: Infinity -> высокое, но конечное значение
            elif metric_name == "sharpe_ratio":
                return 50.0 if value > 0 else -50.0
            else:
                return default
        return float(value)

    def _validate_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Валидация и очистка метрик.
        
        Обрабатывает бесконечности, NaN, нулевую просадку и гарантирует корректные диапазоны.
        """
        if not metrics:
            return self._empty_metrics()
        
        validated = copy.deepcopy(metrics)
        
        # Замена бесконечностей
        validated["profit_factor"] = self._safe_metric_value(
            validated.get("profit_factor", 0.0), "profit_factor", 0.0
        )
        
        validated["sharpe_ratio"] = self._safe_metric_value(
            validated.get("sharpe_ratio", 0.0), "sharpe_ratio", 0.0
        )
        
        # Обработка нулевой просадки
        max_drawdown = validated.get("max_drawdown", 0.0)
        if max_drawdown == 0.0 or np.isnan(max_drawdown):
            validated["max_drawdown"] = 1e-5  # Избегаем деления на ноль
        
        # Гарантия, что win_rate в [0, 1]
        win_rate = validated.get("win_rate", 0.0)
        if np.isnan(win_rate) or np.isinf(win_rate):
            validated["win_rate"] = 0.0
        else:
            validated["win_rate"] = max(0.0, min(1.0, float(win_rate)))
        
        # Гарантия, что total_return конечное значение
        total_return = validated.get("total_return", 0.0)
        if np.isinf(total_return) or np.isnan(total_return):
            validated["total_return"] = 0.0
        
        return validated

    def _passes_hard_constraints(self, metrics: Dict[str, Any]) -> bool:
        """
        Жёсткие ограничения - отсекаем плохие решения до вычисления fitness.
        
        Returns:
            True если решение проходит все constraints, False иначе
        """
        trades = metrics.get("total_trades", 0)
        
        # Минимум 10 сделок на валидации
        if trades < 10:
            return False
        
        # Максимальная просадка не более 20%
        max_drawdown = metrics.get("max_drawdown", 1.0)
        if not (np.isnan(max_drawdown) or np.isinf(max_drawdown)):
            if max_drawdown > 0.20:
                return False
        
        # Win rate не менее 25%
        win_rate = metrics.get("win_rate", 0.0)
        if not (np.isnan(win_rate) or np.isinf(win_rate)):
            if win_rate < 0.25:
                return False
        
        return True

    def _apply_trade_count_penalties(self, metrics: Dict[str, Any]) -> Tuple[float, float]:
        """
        ЖЁСТКИЕ штрафы за малое количество сделок.
        
        Returns:
            Tuple[аддитивный_штраф, мультипликативный_множитель]
        """
        trades = metrics.get("total_trades", 0)
        
        # Аддитивные штрафы (работают даже с Infinity)
        penalty = 0.0
        
        if trades < 5:
            penalty += 1000.0  # Очень жёсткий штраф
        elif trades < 10:
            penalty += 500.0
        elif trades < 20:
            penalty += 200.0
        elif trades < 30:
            penalty += 100.0
        
        # Мультипликативные штрафы (дополнительно, усиленные)
        multiplier = 1.0
        if trades < 10:
            multiplier *= 0.1  # Сильное уменьшение (было 0.5)
        elif trades < 20:
            multiplier *= 0.3  # Было 0.7
        elif trades < 30:
            multiplier *= 0.6
        
        return penalty, multiplier

    def _calculate_fitness(self, metrics: Dict[str, Dict[str, Any]], genes: Optional[Dict[str, Any]] = None) -> float:
        # 1. Предварительная валидация метрик
        val_metrics_raw = metrics.get("val") or self._empty_metrics()
        train_metrics_raw = metrics.get("train") or self._empty_metrics()
        test_metrics_raw = metrics.get("test") or self._empty_metrics()
        
        val_metrics = self._validate_metrics(val_metrics_raw)
        train_metrics = self._validate_metrics(train_metrics_raw)
        test_metrics = self._validate_metrics(test_metrics_raw)
        
        # 2. Жёсткие проверки (hard constraints) - отсекаем плохие решения
        if not self._passes_hard_constraints(val_metrics):
            return -float('inf')
        # Жёсткий запрет на отрицательный Test return
        if test_metrics.get("total_return", 0.0) < 0:
            return -float("inf")
        
        weights = self.fitness_weights

        # 3. Основной fitness на validation (борьба с переобучением)
        # Используем безопасные метрики
        pnl_score = val_metrics["total_return"] * 100
        sharpe_score = val_metrics["sharpe_ratio"] * 10
        drawdown_score = (1 - val_metrics["max_drawdown"]) * 100
        win_rate_score = val_metrics["win_rate"] * 100
        profit_factor_score = val_metrics["profit_factor"] * 10

        base_score = (
            pnl_score * weights.get("pnl", 0)
            + sharpe_score * weights.get("sharpe_ratio", 0)
            + drawdown_score * weights.get("max_drawdown", 0)
            + win_rate_score * weights.get("win_rate", 0)
            + profit_factor_score * weights.get("profit_factor", 0)
        )

        score = base_score
        
        # Штраф за переобучение: если train намного лучше val
        overfitting_penalty = self._calculate_overfitting_penalty(train_metrics, val_metrics)
        score -= overfitting_penalty
        
        # Штраф за нестабильность: большая разница между train и val
        stability_penalty = self._calculate_stability_penalty(train_metrics, val_metrics)
        score -= stability_penalty
        
        # Бонус за стабильность: если train и val близки
        stability_bonus = self._calculate_stability_bonus(train_metrics, val_metrics)
        score += stability_bonus
        
        # Штраф за разницу между validation и test
        val_test_penalty = self._calculate_val_test_penalty(val_metrics, test_metrics)
        score -= val_test_penalty
        
        # Штраф за слишком широкое временное окно
        time_window_penalty = self._calculate_time_window_penalty(genes or {})
        score -= time_window_penalty
        
        # ЖЁСТКИЕ штрафы за малое количество сделок (аддитивные + мультипликативные)
        trade_penalty, trade_multiplier = self._apply_trade_count_penalties(val_metrics)
        score -= trade_penalty
        
        # Логирование для отладки (только если есть значительные штрафы/бонусы)
        if (
            overfitting_penalty > 10
            or stability_penalty > 10
            or stability_bonus > 5
            or trade_penalty > 100
            or val_test_penalty > 10
            or time_window_penalty > 10
        ):
            print(
                f"  [Fitness] base={base_score:.2f} "
                f"overfit_penalty={overfitting_penalty:.2f} "
                f"stability_penalty={stability_penalty:.2f} "
                f"stability_bonus={stability_bonus:.2f} "
                f"trade_penalty={trade_penalty:.2f} "
                f"val_test_penalty={val_test_penalty:.2f} "
                f"time_window_penalty={time_window_penalty:.2f} "
                f"final={score:.2f}"
            )
        
        # Мультипликативные штрафы (существующие + новые)
        penalties_multiplier = self._apply_penalties(val_metrics) * trade_multiplier
        score *= penalties_multiplier
        
        # Аддитивные бонусы
        bonuses = self._apply_bonuses(val_metrics)
        score += bonuses
        
        # 4. Гарантия конечного значения
        if np.isinf(score) or np.isnan(score):
            return -1000.0
        
        return score

    def _apply_penalties(self, metrics: Dict[str, Any]) -> float:
        multiplier = 1.0
        rules = {
            "trades_below_10": metrics["total_trades"] < 10,
            "trades_below_20": metrics["total_trades"] < 20,
            "drawdown_above_40": metrics["max_drawdown"] > 0.40,
            "drawdown_above_50": metrics["max_drawdown"] > 0.50,
            "win_rate_below_30": metrics["win_rate"] < 0.30,
            "profit_factor_below_1": metrics["profit_factor"] < 1.0,
            "sharpe_below_0": metrics["sharpe_ratio"] < 0.0,
        }
        for key, condition in rules.items():
            if condition and key in self.fitness_penalties:
                multiplier *= self.fitness_penalties[key]
        return multiplier

    def _apply_bonuses(self, metrics: Dict[str, Any]) -> float:
        bonus = 0.0
        conditions = {
            "sharpe_above_1": metrics["sharpe_ratio"] > 1.0,
            "sharpe_above_2": metrics["sharpe_ratio"] > 2.0,
            "drawdown_below_20": metrics["max_drawdown"] < 0.20,
            "drawdown_below_10": metrics["max_drawdown"] < 0.10,
            "win_rate_above_50": metrics["win_rate"] > 0.50,
            "profit_factor_above_2": metrics["profit_factor"] > 2.0,
        }
        for key, condition in conditions.items():
            if condition and key in self.fitness_bonuses:
                bonus += self.fitness_bonuses[key]
        return bonus

    def _calculate_val_test_penalty(self, val_metrics: Dict[str, Any], test_metrics: Dict[str, Any]) -> float:
        """
        Штраф за разницу между validation и test метриками.
        Цель — удерживать gap < 50% для доходности и Sharpe.
        """
        penalty = 0.0

        val_return = val_metrics.get("total_return", 0.0)
        test_return = test_metrics.get("total_return", 0.0)
        return_base = max(abs(val_return), 1e-4)
        return_gap = abs(test_return - val_return) / return_base
        if return_gap > 0.5:
            penalty += (return_gap - 0.5) * 400.0

        val_sharpe = val_metrics.get("sharpe_ratio", 0.0)
        test_sharpe = test_metrics.get("sharpe_ratio", 0.0)
        sharpe_base = max(abs(val_sharpe), 0.1)
        sharpe_gap = abs(test_sharpe - val_sharpe) / sharpe_base
        if sharpe_gap > 0.5:
            penalty += (sharpe_gap - 0.5) * 40.0

        return penalty

    def _calculate_time_window_penalty(self, genes: Dict[str, Any]) -> float:
        """
        Штраф за слишком широкое временное окно (> 8 часов).
        Применяется только если time_filter включён.
        """
        if not genes:
            return 0.0
        if genes.get("time_filter_enabled", 0) != 1:
            return 0.0
        window = genes.get("time_window_length", 24)
        if window <= 8:
            return 0.0
        excess = window - 8
        return excess * 50.0

    def _calculate_overfitting_penalty(self, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]) -> float:
        """
        Штраф за переобучение: если train метрики намного лучше val.
        Это указывает на то, что стратегия запомнила паттерны train, которые не работают на новых данных.
        """
        penalty = 0.0
        
        # Штраф за разницу в return (если train намного лучше val)
        train_return = train_metrics.get("total_return", 0.0)
        val_return = val_metrics.get("total_return", 0.0)
        if train_return > val_return:
            # Если train положительный, а val отрицательный - сильный штраф
            if train_return > 0 and val_return < 0:
                penalty += abs(train_return - val_return) * 500  # Сильный штраф
            # Если оба положительные, но train намного лучше
            elif train_return > val_return * 1.5:
                penalty += (train_return - val_return) * 200
        
        # Штраф за разницу в Sharpe (если train намного лучше val)
        train_sharpe = train_metrics.get("sharpe_ratio", 0.0)
        val_sharpe = val_metrics.get("sharpe_ratio", 0.0)
        if train_sharpe > val_sharpe:
            # Если train положительный, а val отрицательный - сильный штраф
            if train_sharpe > 0 and val_sharpe < 0:
                penalty += abs(train_sharpe - val_sharpe) * 20
            # Если оба положительные, но train намного лучше
            elif train_sharpe > val_sharpe * 1.5:
                penalty += (train_sharpe - val_sharpe) * 10
        
        return penalty

    def _calculate_stability_penalty(self, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]) -> float:
        """
        Штраф за нестабильность: большая разница между train и val метриками.
        Даже если обе метрики хорошие, большая разница указывает на нестабильность стратегии.
        """
        penalty = 0.0
        
        # Разница в return (абсолютная)
        return_diff = abs(train_metrics.get("total_return", 0.0) - val_metrics.get("total_return", 0.0))
        if return_diff > 0.10:  # Разница больше 10%
            penalty += return_diff * 100
        
        # Разница в Sharpe (абсолютная)
        sharpe_diff = abs(train_metrics.get("sharpe_ratio", 0.0) - val_metrics.get("sharpe_ratio", 0.0))
        if sharpe_diff > 5.0:  # Разница больше 5
            penalty += sharpe_diff * 2
        
        # Разница в max drawdown (абсолютная)
        dd_diff = abs(train_metrics.get("max_drawdown", 0.0) - val_metrics.get("max_drawdown", 0.0))
        if dd_diff > 0.15:  # Разница больше 15%
            penalty += dd_diff * 50
        
        return penalty

    def _calculate_stability_bonus(self, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]) -> float:
        """
        Бонус за стабильность: если train и val метрики близки.
        Это указывает на то, что стратегия работает стабильно на разных данных.
        """
        bonus = 0.0
        
        # Близость return (если разница меньше 5%)
        return_diff = abs(train_metrics.get("total_return", 0.0) - val_metrics.get("total_return", 0.0))
        if return_diff < 0.05:
            bonus += 10.0  # Бонус за стабильность return
        
        # Близость Sharpe (если разница меньше 2)
        sharpe_diff = abs(train_metrics.get("sharpe_ratio", 0.0) - val_metrics.get("sharpe_ratio", 0.0))
        if sharpe_diff < 2.0:
            bonus += 5.0  # Бонус за стабильность Sharpe
        
        # Близость max drawdown (если разница меньше 5%)
        dd_diff = abs(train_metrics.get("max_drawdown", 0.0) - val_metrics.get("max_drawdown", 0.0))
        if dd_diff < 0.05:
            bonus += 5.0  # Бонус за стабильность drawdown
        
        # Дополнительный бонус, если обе метрики положительные и близкие
        train_return = train_metrics.get("total_return", 0.0)
        val_return = val_metrics.get("total_return", 0.0)
        if train_return > 0 and val_return > 0 and return_diff < 0.03:
            bonus += 15.0  # Большой бонус за стабильную прибыльность
        
        return bonus

    # ---------------- Value helpers ---------------- #

    def _random_value(self, definition: Dict[str, Any]):
        if definition["type"] == "int":
            return random.randint(definition["min"], definition["max"])
        if definition["type"] == "float":
            step = definition.get("step")
            if step:
                steps = int((definition["max"] - definition["min"]) / step)
                return round(definition["min"] + random.randint(0, steps) * step, 6)
            return round(random.uniform(definition["min"], definition["max"]), 6)
        raise ValueError(f"Unsupported parameter type {definition['type']}")

    def _mutated_value(self, value, definition: Dict[str, Any]):
        if definition["type"] == "int":
            delta = random.choice([-2, -1, 1, 2])
            mutated = value + delta
            return max(definition["min"], min(definition["max"], mutated))
        elif definition["type"] == "float":
            amplitude = (definition["max"] - definition["min"]) * 0.1
            mutated = value + random.uniform(-amplitude, amplitude)
            mutated = max(definition["min"], min(definition["max"], mutated))
            return round(mutated, 6)
        return value

    # ---------------- Misc helpers ---------------- #

    @staticmethod
    def _empty_metrics() -> Dict[str, Any]:
        return {
            "total_pnl": 0.0,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
        }

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

