"""
MVP strategy logic: convert baseline indicators into trading votes/signals.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np
import pandas as pd


@dataclass
class MVPSignalConfig:
    """
    Configuration for mapping indicators to discrete votes.
    """

    rsi_column: str = "rsi_14"
    rsi_buy_below: float = 35.0
    rsi_sell_above: float = 65.0
    macd_line: str = "macd_line"
    macd_signal: str = "macd_signal"
    macd_deadband: float = 0.0
    boll_upper: str = "boll_upper"
    boll_lower: str = "boll_lower"
    wobi_column: str = "wobi_zscore"
    wobi_buy_below: float = -0.5
    wobi_sell_above: float = 0.5
    indicator_weights: Dict[str, float] = field(
        default_factory=lambda: {"rsi": 1.0, "macd": 1.0, "bollinger": 1.0, "wobi": 1.0}
    )
    entry_threshold: float = 0.5
    exit_threshold: float = 0.1

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any] | None) -> "MVPSignalConfig":
        instance = cls()
        if not cfg:
            return instance

        signals_cfg = cfg.get("signals", {})

        rsi_cfg = signals_cfg.get("rsi", {})
        instance.rsi_column = rsi_cfg.get("column", instance.rsi_column)
        instance.rsi_buy_below = rsi_cfg.get("buy_below", instance.rsi_buy_below)
        instance.rsi_sell_above = rsi_cfg.get("sell_above", instance.rsi_sell_above)

        macd_cfg = signals_cfg.get("macd", {})
        instance.macd_line = macd_cfg.get("line", instance.macd_line)
        instance.macd_signal = macd_cfg.get("signal", instance.macd_signal)
        instance.macd_deadband = macd_cfg.get("deadband", instance.macd_deadband)

        boll_cfg = signals_cfg.get("bollinger", {})
        instance.boll_upper = boll_cfg.get("upper", instance.boll_upper)
        instance.boll_lower = boll_cfg.get("lower", instance.boll_lower)

        wobi_cfg = signals_cfg.get("wobi", {})
        instance.wobi_column = wobi_cfg.get("column", instance.wobi_column)
        instance.wobi_buy_below = wobi_cfg.get("buy_below", instance.wobi_buy_below)
        instance.wobi_sell_above = wobi_cfg.get("sell_above", instance.wobi_sell_above)

        weights = instance.indicator_weights.copy()
        weights["rsi"] = rsi_cfg.get("weight", weights.get("rsi", 1.0))
        weights["macd"] = macd_cfg.get("weight", weights.get("macd", 1.0))
        weights["bollinger"] = boll_cfg.get("weight", weights.get("bollinger", 1.0))
        weights["wobi"] = wobi_cfg.get("weight", weights.get("wobi", 1.0))
        instance.indicator_weights = weights

        combo_cfg = cfg.get("combination", {})
        instance.entry_threshold = combo_cfg.get("entry_threshold", instance.entry_threshold)
        instance.exit_threshold = combo_cfg.get("exit_threshold", instance.exit_threshold)

        return instance


class MVPStrategy:
    """
    Generates combined signals using a simple majority vote from baseline indicators.
    """

    def __init__(self, config: MVPSignalConfig | None = None):
        self.config = config or MVPSignalConfig()

    def build_votes(self, data: pd.DataFrame) -> pd.DataFrame:
        votes = pd.DataFrame(index=data.index)

        if self.config.rsi_column in data.columns:
            rsi = data[self.config.rsi_column]
            votes["rsi_vote"] = np.where(
                rsi <= self.config.rsi_buy_below,
                1,
                np.where(rsi >= self.config.rsi_sell_above, -1, 0),
            )

        if self.config.macd_line in data.columns and self.config.macd_signal in data.columns:
            macd_diff = data[self.config.macd_line] - data[self.config.macd_signal]
            votes["macd_vote"] = np.where(
                macd_diff >= self.config.macd_deadband,
                1,
                np.where(macd_diff <= -self.config.macd_deadband, -1, 0),
            )

        if {"close", self.config.boll_upper, self.config.boll_lower}.issubset(data.columns):
            close = data["close"]
            votes["bollinger_vote"] = np.where(
                close <= data[self.config.boll_lower],
                1,
                np.where(close >= data[self.config.boll_upper], -1, 0),
            )

        if self.config.wobi_column in data.columns:
            wobi = data[self.config.wobi_column]
            votes["wobi_vote"] = np.where(
                wobi <= self.config.wobi_buy_below,
                1,
                np.where(wobi >= self.config.wobi_sell_above, -1, 0),
            )

        votes.fillna(0, inplace=True)
        return votes

    def aggregate_signal(self, votes: pd.DataFrame) -> pd.Series:
        weights = self.config.indicator_weights
        active_columns: List[str] = [col for col in votes.columns if col.split("_vote")[0] in weights]

        if not active_columns:
            raise ValueError("No indicator votes available for aggregation")

        weighted_sum = None
        total_weight = 0.0

        for column in active_columns:
            indicator_name = column.replace("_vote", "")
            weight = weights.get(indicator_name, 0.0)
            if weight <= 0:
                continue

            vote_series = votes[column] * weight
            weighted_sum = vote_series if weighted_sum is None else weighted_sum.add(vote_series, fill_value=0.0)
            total_weight += weight

        if weighted_sum is None or total_weight == 0:
            raise ValueError("Indicator weights sum to zero; adjust MVPSignalConfig.indicator_weights")

        combined_signal = weighted_sum / total_weight
        combined_signal = combined_signal.clip(-1.0, 1.0)
        return combined_signal

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        votes = self.build_votes(data)
        combined = self.aggregate_signal(votes)

        signals = votes.copy()
        signals["combined_signal"] = combined
        signals["entry_long"] = combined >= self.config.entry_threshold
        signals["entry_short"] = combined <= -self.config.entry_threshold
        signals["exit_long"] = combined <= self.config.exit_threshold
        signals["exit_short"] = combined >= -self.config.exit_threshold

        return signals

