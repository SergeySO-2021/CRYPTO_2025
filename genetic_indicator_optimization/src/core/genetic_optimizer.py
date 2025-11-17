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

import yaml

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

        for generation in range(self.max_generations):
            evaluated_population = [self._evaluate(ind) for ind in population]
            evaluated_population.sort(key=lambda ind: ind.fitness or -math.inf, reverse=True)

            current_best = evaluated_population[0]
            if best_individual is None or (current_best.fitness or -math.inf) > (best_individual.fitness or -math.inf):
                best_individual = copy.deepcopy(current_best)

            print(
                f"[GA] Generation {generation+1}/{self.max_generations} "
                f"best fitness={current_best.fitness:.2f} genes={current_best.genes}"
            )

            if generation + 1 == self.max_generations:
                break

            next_population = evaluated_population[: self.elite_size]
            while len(next_population) < self.population_size:
                parent_a = self._tournament_select(evaluated_population)
                parent_b = self._tournament_select(evaluated_population)

                if random.random() < self.crossover_rate:
                    child_genes = self._crossover(parent_a.genes, parent_b.genes)
                else:
                    child_genes = copy.deepcopy(parent_a.genes)

                child_genes = self._mutate(child_genes)
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
        return genes

    def _mutate(self, genes: Dict[str, Any]) -> Dict[str, Any]:
        mutated = copy.deepcopy(genes)
        for name, definition in self.search_space.items():
            if random.random() <= self.mutation_rate:
                mutated[name] = self._mutated_value(mutated[name], definition)
        return mutated

    def _crossover(self, genes_a: Dict[str, Any], genes_b: Dict[str, Any]) -> Dict[str, Any]:
        child = {}
        for name in genes_a.keys():
            child[name] = random.choice([genes_a[name], genes_b[name]])
        return child

    def _tournament_select(self, population: List[Individual]) -> Individual:
        contenders = random.sample(population, k=min(self.tournament_size, len(population)))
        contenders.sort(key=lambda ind: ind.fitness or -math.inf, reverse=True)
        return contenders[0]

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

        fitness = self._calculate_fitness(metrics)

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
        rsi_cfg = signals_cfg.setdefault("rsi", {})
        rsi_cfg["column"] = f"rsi_{genes['rsi_period']}"

        wobi_cfg = signals_cfg.setdefault("wobi", {})
        wobi_cfg["weight"] = genes["wobi_weight"]

        risk_cfg = cfg.setdefault("risk", {})
        risk_cfg["stop_loss_pct"] = genes["stop_loss_pct"]
        risk_cfg["take_profit_pct"] = genes["take_profit_pct"]

        atr_cfg = risk_cfg.setdefault("atr", {})
        atr_cfg["enabled"] = True
        atr_cfg["period"] = genes["atr_period"]
        atr_cfg["stop_multiplier"] = genes["atr_stop_multiplier"]
        atr_cfg["trailing_multiplier"] = genes["atr_trailing_multiplier"]

        return cfg

    def _calculate_fitness(self, metrics: Dict[str, Dict[str, Any]]) -> float:
        val_metrics = metrics.get("val") or self._empty_metrics()
        weights = self.fitness_weights

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
        score *= self._apply_penalties(val_metrics)
        score += self._apply_bonuses(val_metrics)
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

