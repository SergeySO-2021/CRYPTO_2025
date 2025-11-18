"""
Minimal backtester used for MVP GA experiments.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from .differentiated_evaluator import Trade


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    position_size_pct: float = 1.0  # fraction of capital to allocate per trade
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02
    entry_threshold: float = 0.5
    exit_threshold: float = 0.1
    max_holding_bars: int = 96  # one trading day on 15m timeframe
    min_holding_bars: int = 0  # минимальное время удержания (в барах, 2 часа = 8 баров на 15m)
    maker_fee_pct: float = 0.0002   # 2 bps
    taker_fee_pct: float = 0.0007   # 7 bps
    emergency_exit_reasons: Tuple[str, ...] = ("stop_loss", "time_stop")
    use_atr_stop: bool = False
    atr_period: int = 14
    atr_stop_multiplier: float = 1.0
    atr_trailing_multiplier: float = 0.0
    turnover_cooldown_bars: int = 0
    max_trades_per_day: Optional[int] = None

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "BacktestConfig":
        if not cfg:
            return cls()
        default = cls()
        config = cls(
            initial_capital=cfg.get("initial_capital", default.initial_capital),
            position_size_pct=cfg.get("position_size_pct", default.position_size_pct),
            stop_loss_pct=cfg.get("stop_loss_pct", default.stop_loss_pct),
            take_profit_pct=cfg.get("take_profit_pct", default.take_profit_pct),
            entry_threshold=cfg.get("entry_threshold", default.entry_threshold),
            exit_threshold=cfg.get("exit_threshold", default.exit_threshold),
            max_holding_bars=cfg.get("max_holding_bars", default.max_holding_bars),
            min_holding_bars=cfg.get("min_holding_bars", default.min_holding_bars),
            maker_fee_pct=cfg.get("maker_fee_pct", default.maker_fee_pct),
            taker_fee_pct=cfg.get("taker_fee_pct", default.taker_fee_pct),
            emergency_exit_reasons=tuple(
                cfg.get("emergency_exit_reasons", list(default.emergency_exit_reasons))
            ),
            turnover_cooldown_bars=cfg.get("turnover", {}).get(
                "cooldown_bars", default.turnover_cooldown_bars
            ),
            max_trades_per_day=cfg.get("turnover", {}).get(
                "max_trades_per_day", default.max_trades_per_day
            ),
        )
        atr_cfg = cfg.get("atr") or {}
        config.use_atr_stop = atr_cfg.get("enabled", default.use_atr_stop)
        config.atr_period = atr_cfg.get("period", default.atr_period)
        config.atr_stop_multiplier = atr_cfg.get("stop_multiplier", default.atr_stop_multiplier)
        config.atr_trailing_multiplier = atr_cfg.get(
            "trailing_multiplier", default.atr_trailing_multiplier
        )
        return config


@dataclass
class BacktestResult:
    trades: List[Trade]
    metrics: Dict[str, Any]
    equity_curve: List[float]


class SimpleBacktester:
    """
    Stateful backtester that opens a single position at a time based on combined signals.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(self, data: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        if "close" not in data.columns:
            raise ValueError("Price data must contain 'close'")
        if "combined_signal" not in signals.columns:
            raise ValueError("Signals must contain 'combined_signal'")

        trades: List[Trade] = []
        capital = self.config.initial_capital
        position = None  # 'long' or 'short'
        entry_price = None
        entry_time = None
        position_notional = None
        entry_fee = 0.0
        holding_bars = 0
        equity_curve = [capital]
        stop_price = None
        take_price = None
        best_price = None

        stop_loss_pct = self.config.stop_loss_pct
        take_profit_pct = self.config.take_profit_pct
        atr_column = f"atr_{self.config.atr_period}"

        cooldown_counter = 0
        daily_trade_counts: Dict[pd.Timestamp.date, int] = {}

        for timestamp, row in data.iterrows():
            current_day = timestamp.date()
            if current_day not in daily_trade_counts:
                daily_trade_counts[current_day] = 0

            if cooldown_counter > 0:
                cooldown_counter -= 1

            price = row["close"]
            signal_row = signals.loc[timestamp]
            combined = signal_row["combined_signal"]

            if position is None:
                if self.config.max_trades_per_day is not None and \
                        daily_trade_counts[current_day] >= self.config.max_trades_per_day:
                    continue
                if cooldown_counter > 0:
                    continue

                if signal_row.get("entry_long", False):
                    position = "long"
                    entry_price = price
                    entry_time = timestamp
                    position_notional = capital * self.config.position_size_pct
                    entry_fee = position_notional * self.config.maker_fee_pct
                    stop_price, take_price = self._initialize_targets(
                        position,
                        entry_price,
                        stop_loss_pct,
                        take_profit_pct,
                        row.get(atr_column),
                    )
                    best_price = entry_price
                    daily_trade_counts[current_day] += 1
                    holding_bars = 0
                elif signal_row.get("entry_short", False):
                    position = "short"
                    entry_price = price
                    entry_time = timestamp
                    position_notional = capital * self.config.position_size_pct
                    entry_fee = position_notional * self.config.maker_fee_pct
                    stop_price, take_price = self._initialize_targets(
                        position,
                        entry_price,
                        stop_loss_pct,
                        take_profit_pct,
                        row.get(atr_column),
                    )
                    best_price = entry_price
                    daily_trade_counts[current_day] += 1
                    holding_bars = 0
                continue

            holding_bars += 1
            exit_reason = None
            atr_value = row.get(atr_column)

            if position == "long":
                if best_price is None or price > best_price:
                    best_price = price
                stop_price = self._maybe_trail_stop(
                    position, stop_price, best_price, atr_value
                )
                if stop_price is not None and price <= stop_price:
                    exit_reason = "stop_loss"
                elif take_price is not None and price >= take_price:
                    exit_reason = "take_profit"
                elif (signal_row.get("exit_long", False) or combined <= -self.config.exit_threshold) and \
                     holding_bars >= self.config.min_holding_bars:
                    exit_reason = "signal_flip"
            else:  # short
                if best_price is None or price < best_price:
                    best_price = price
                stop_price = self._maybe_trail_stop(
                    position, stop_price, best_price, atr_value
                )
                if stop_price is not None and price >= stop_price:
                    exit_reason = "stop_loss"
                elif take_price is not None and price <= take_price:
                    exit_reason = "take_profit"
                elif (signal_row.get("exit_short", False) or combined >= self.config.exit_threshold) and \
                     holding_bars >= self.config.min_holding_bars:
                    exit_reason = "signal_flip"

            if exit_reason is None and holding_bars >= self.config.max_holding_bars:
                exit_reason = "time_stop"

            if exit_reason is not None:
                pnl, pnl_pct = self._close_trade(
                    position,
                    entry_price,
                    price,
                    position_notional,
                    entry_fee,
                    exit_reason,
                )
                trade = Trade(
                    entry_time=entry_time,
                    exit_time=timestamp,
                    entry_price=entry_price,
                    exit_price=price,
                    position_type=position,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_period=pd.Timedelta(minutes=15 * holding_bars),
                )
                trades.append(trade)
                capital += pnl
                equity_curve.append(capital)

                position = None
                entry_price = None
                entry_time = None
                position_notional = None
                entry_fee = 0.0
                stop_price = None
                take_price = None
                best_price = None
                holding_bars = 0
                cooldown_counter = self.config.turnover_cooldown_bars

        # Force close on final bar
        if position is not None and entry_price is not None:
            final_price = data.iloc[-1]["close"]
            final_time = data.index[-1]
            pnl, pnl_pct = self._close_trade(
                position,
                entry_price,
                final_price,
                position_notional,
                entry_fee,
                "time_stop",
            )
            trade = Trade(
                entry_time=entry_time,
                exit_time=final_time,
                entry_price=entry_price,
                exit_price=final_price,
                position_type=position,
                pnl=pnl,
                pnl_pct=pnl_pct,
                holding_period=pd.Timedelta(minutes=15 * holding_bars),
            )
            trades.append(trade)
            capital += pnl
            equity_curve.append(capital)

        metrics = self._compute_metrics(trades, equity_curve)
        return BacktestResult(trades=trades, metrics=metrics, equity_curve=equity_curve)

    def _close_trade(
        self,
        position: str,
        entry_price: float,
        exit_price: float,
        position_notional: Optional[float],
        entry_fee: float,
        exit_reason: str,
    ) -> tuple[float, float]:
        if position_notional is None:
            raise ValueError("Position notional is not defined for closing trade")
        direction = 1 if position == "long" else -1
        gross_pct = direction * ((exit_price / entry_price) - 1)

        exit_fee_rate = (
            self.config.taker_fee_pct
            if exit_reason in self.config.emergency_exit_reasons
            else self.config.maker_fee_pct
        )
        exit_fee = position_notional * exit_fee_rate
        total_fee = entry_fee + exit_fee

        net_pct = gross_pct - (total_fee / position_notional)
        pnl = net_pct * position_notional
        return pnl, net_pct

    def _initialize_targets(
        self,
        position: str,
        entry_price: float,
        base_stop_pct: float,
        base_take_pct: float,
        atr_value: Optional[float],
    ) -> Tuple[float, float]:
        stop_pct = base_stop_pct
        if self.config.use_atr_stop and atr_value is not None and not np.isnan(atr_value):
            atr_pct = atr_value / entry_price if entry_price != 0 else 0
            stop_pct = max(stop_pct, atr_pct * self.config.atr_stop_multiplier)

        if position == "long":
            stop_price = entry_price * (1 - stop_pct)
            take_price = entry_price * (1 + base_take_pct)
        else:
            stop_price = entry_price * (1 + stop_pct)
            take_price = entry_price * (1 - base_take_pct)
        return stop_price, take_price

    def _maybe_trail_stop(
        self,
        position: str,
        current_stop: Optional[float],
        best_price: Optional[float],
        atr_value: Optional[float],
    ) -> Optional[float]:
        if current_stop is None:
            return None
        if not self.config.use_atr_stop or self.config.atr_trailing_multiplier <= 0:
            return current_stop
        if atr_value is None or np.isnan(atr_value) or best_price is None:
            return current_stop

        offset = atr_value * self.config.atr_trailing_multiplier
        if position == "long":
            trailing_stop = best_price - offset
            return max(current_stop, trailing_stop)
        else:
            trailing_stop = best_price + offset
            return min(current_stop, trailing_stop)

    def _compute_metrics(self, trades: List[Trade], equity_curve: List[float]) -> Dict[str, Any]:
        if not trades:
            return {
                "total_pnl": 0.0,
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
            }

        total_pnl = sum(t.pnl for t in trades)
        total_return = total_pnl / self.config.initial_capital

        winning = [t.pnl for t in trades if t.pnl > 0]
        losing = [abs(t.pnl) for t in trades if t.pnl < 0]
        win_rate = len(winning) / len(trades) if trades else 0.0
        profit_factor = sum(winning) / sum(losing) if losing else float("inf")

        pnl_pct = [t.pnl_pct for t in trades if t.pnl_pct is not None]
        if pnl_pct and np.std(pnl_pct) > 0:
            sharpe_ratio = np.mean(pnl_pct) / np.std(pnl_pct) * np.sqrt(252 * 24 * 4)
        else:
            sharpe_ratio = 0.0

        max_drawdown = self._max_drawdown(equity_curve)

        return {
            "total_pnl": float(total_pnl),
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "total_trades": len(trades),
        }

    @staticmethod
    def _max_drawdown(equity_curve: List[float]) -> float:
        if not equity_curve:
            return 0.0
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        return float(np.max(dd))

