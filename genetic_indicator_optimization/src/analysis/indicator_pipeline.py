"""
Indicator calculation pipeline for MVP stage.
Computes baseline technical + order book features (RSI, MACD, Bollinger Bands, WOBI).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np


DEFAULT_PARAMS = {
    "rsi": {"period": 14},
    "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    "bollinger": {"period": 20, "std_dev": 2.0},
    "atr": {"period": 14},
    "wobi": {
        "weights": {
            "ratio3": 0.4,
            "ratio5": 0.3,
            "ratio8": 0.2,
            "ratio60": 0.1,
        }
    },
}


@dataclass
class IndicatorArtifacts:
    """
    Container for indicator outputs and diagnostics.
    """

    data: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


class IndicatorPipeline:
    """
    Compute MVP indicator stack on top of price + order-book features.
    """

    def __init__(self, data: pd.DataFrame, params: Optional[Dict[str, Dict[str, Any]]] = None):
        if data is None or data.empty:
            raise ValueError("IndicatorPipeline expects a non-empty DataFrame")

        required_cols = {"close"}
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for indicator pipeline: {missing_cols}")

        self.base_df = data.copy()
        self.params = DEFAULT_PARAMS.copy()
        if params:
            # Shallow merge for provided parameters
            for key, value in params.items():
                if isinstance(value, dict) and isinstance(self.params.get(key), dict):
                    merged = self.params[key].copy()
                    merged.update(value)
                    self.params[key] = merged
                else:
                    self.params[key] = value

    def run(self, indicators: Tuple[str, ...] = ("rsi", "macd", "bollinger", "atr", "wobi")) -> IndicatorArtifacts:
        df = self.base_df.copy()
        metadata = {}

        if "rsi" in indicators:
            rsi_period = self.params["rsi"]["period"]
            df[f"rsi_{rsi_period}"] = self._compute_rsi(df["close"], rsi_period)
            metadata["rsi"] = {"period": rsi_period}

        if "macd" in indicators:
            macd_params = self.params["macd"]
            macd_line, signal_line, histogram = self._compute_macd(
                df["close"],
                macd_params["fast_period"],
                macd_params["slow_period"],
                macd_params["signal_period"],
            )
            df["macd_line"] = macd_line
            df["macd_signal"] = signal_line
            df["macd_hist"] = histogram
            metadata["macd"] = macd_params

        if "bollinger" in indicators:
            bb_params = self.params["bollinger"]
            mid, upper, lower = self._compute_bollinger(df["close"], bb_params["period"], bb_params["std_dev"])
            df["boll_mid"] = mid
            df["boll_upper"] = upper
            df["boll_lower"] = lower
            df["boll_width"] = (upper - lower) / mid.replace(0, np.nan)
            metadata["bollinger"] = bb_params

        if "atr" in indicators:
            atr_params = self.params["atr"]
            atr_series = self._compute_atr(df, atr_params["period"])
            df[f"atr_{atr_params['period']}"] = atr_series
            metadata["atr"] = atr_params

        if "wobi" in indicators:
            wobi_params = self.params["wobi"]
            wobi_series = self._compute_wobi(df, wobi_params["weights"])
            df["wobi"] = wobi_series
            df["wobi_zscore"] = (wobi_series - wobi_series.rolling(96, min_periods=30).mean()) / (
                wobi_series.rolling(96, min_periods=30).std(ddof=0)
            )
            metadata["wobi"] = {"weights": wobi_params["weights"], "zscore_window": 96}

        metadata["rows"] = len(df)
        metadata["null_stats"] = df.isna().sum().to_dict()

        return IndicatorArtifacts(data=df, metadata=metadata)

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _compute_macd(close: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _compute_bollinger(close: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        mid = close.rolling(window=period, min_periods=period).mean()
        rolling_std = close.rolling(window=period, min_periods=period).std(ddof=0)
        upper = mid + std_dev * rolling_std
        lower = mid - std_dev * rolling_std
        return mid, upper, lower

    @staticmethod
    def _compute_wobi(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        numerator = None
        total_weight = 0.0

        for column, weight in weights.items():
            if column not in df.columns:
                continue
            total_weight += weight
            series = df[column] * weight
            numerator = series if numerator is None else numerator.add(series, fill_value=0.0)

        if numerator is None or total_weight == 0:
            raise ValueError("Cannot compute WOBI: no ratio columns available with provided weights")

        wobi = numerator / total_weight
        return wobi

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        return atr

