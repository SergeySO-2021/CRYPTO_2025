"""
Entry-point for running the MVP genetic algorithm search.
"""

import argparse
import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from core import GeneticOptimizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run genetic search for MVP strategy parameters")
    parser.add_argument(
        "--ga-config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "ga_config.yaml"),
        help="Path to GA configuration file",
    )
    parser.add_argument(
        "--strategy-config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "mvp_strategy_config.yaml"),
        help="Path to MVP strategy configuration file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional override for data CSV path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ga_best.json",
        help="Where to store best genome & metrics summary",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Optional override for population size",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Optional override for GA generations",
    )
    args = parser.parse_args()

    optimizer = GeneticOptimizer(
        ga_config_path=args.ga_config,
        strategy_config_path=args.strategy_config,
        data_path=args.data_path,
        population_size=args.population_size,
        max_generations=args.max_generations,
    )
    best = optimizer.run()

    summary = {
        "genes": best.genes,
        "fitness": best.fitness,
        "metrics": best.metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[GA] Best individual saved to {output_path}")


if __name__ == "__main__":
    main()

