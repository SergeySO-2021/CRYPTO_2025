"""
CLI utility to run the MVP pipeline end-to-end:
data loading -> indicator calculation -> signal generation -> backtest metrics.
Outputs aggregated metrics and optional trade logs per dataset split.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml

# Ensure src/ is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from data_loader import DataLoader
from analysis import IndicatorPipeline
from strategies.mvp_strategy import MVPStrategy, MVPSignalConfig
from core import SimpleBacktester, BacktestConfig


def trades_to_dataframe(trades) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "position_type",
                "pnl",
                "pnl_pct",
                "holding_minutes",
            ]
        )

    rows = []
    for trade in trades:
        rows.append(
            {
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "position_type": trade.position_type,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "holding_minutes": trade.holding_period.total_seconds() / 60.0,
            }
        )
    return pd.DataFrame(rows)


def run_segment(name: str, data: pd.DataFrame, strategy: MVPStrategy, backtester: SimpleBacktester) -> Dict[str, Any]:
    if data.empty:
        return {
            "split": name,
            "metrics": {
                "total_pnl": 0.0,
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
            },
            "trades": pd.DataFrame(),
        }

    signals = strategy.generate_signals(data)
    result = backtester.run(data, signals)
    trades_df = trades_to_dataframe(result.trades)

    return {
        "split": name,
        "metrics": result.metrics,
        "trades": trades_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Run MVP indicator + backtest pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV with BTCUSDT 15m data (defaults to project dataframe)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/stage0",
        help="Directory to store metrics/trades outputs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "mvp_strategy_config.yaml"),
        help="YAML config with signal/risk parameters",
    )
    parser.add_argument(
        "--skip-trades",
        action="store_true",
        help="Do not export trade logs (only summary metrics)",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Override stop loss percentage (e.g., 0.01 = 1%)",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Override take profit percentage (e.g., 0.02 = 2%)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(data_path=args.data_path)
    raw_data = loader.load_data()
    train_data, val_data, test_data = loader.split_data()

    pipeline = IndicatorPipeline(raw_data)
    artifacts = pipeline.run()
    enriched = artifacts.data.dropna()

    config_path = Path(args.config)
    config_data: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
    else:
        print(f"[WARN] Config file {config_path} not found. Using defaults.")

    signal_config = MVPSignalConfig.from_dict(config_data)
    risk_config = config_data.get("risk")
    backtest_config = BacktestConfig.from_dict(risk_config)

    if args.stop_loss is not None:
        backtest_config.stop_loss_pct = args.stop_loss
    if args.take_profit is not None:
        backtest_config.take_profit_pct = args.take_profit

    # Keep thresholds in sync with signal config
    backtest_config.entry_threshold = signal_config.entry_threshold
    backtest_config.exit_threshold = signal_config.exit_threshold

    def subset(df_idx):
        subset_df = enriched.reindex(df_idx)
        return subset_df.dropna()

    splits = {
        "train": subset(train_data.index),
        "val": subset(val_data.index),
        "test": subset(test_data.index),
        "full": enriched.dropna(),
    }

    backtester = SimpleBacktester(backtest_config)
    strategy = MVPStrategy(signal_config)

    metrics_summary = {}

    for split_name, df in splits.items():
        segment_result = run_segment(split_name, df, strategy, backtester)
        metrics_summary[split_name] = segment_result["metrics"]

        if not args.skip_trades:
            trades_df = segment_result["trades"]
            trades_path = output_dir / f"trades_{split_name}.csv"
            trades_df.to_csv(trades_path, index=False)
            print(f"[INFO] Saved {len(trades_df)} trades for {split_name} -> {trades_path}")

    metrics_path = output_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[INFO] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

