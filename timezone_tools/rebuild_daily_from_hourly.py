"""
Rebuild daily OHLC candles from hourly data aligned to UTC-3.
"""

from pathlib import Path

import pandas as pd


SOURCE = Path(__file__).resolve().parent.parent / "df_btc_1h_complete.csv"
TARGET = Path(__file__).resolve().parent.parent / "df_btc_1d.csv"


def rebuild_daily() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Hourly source file not found: {SOURCE}")

    df = pd.read_csv(SOURCE)
    if "timestamps" not in df.columns:
        raise ValueError("Source file missing 'timestamps' column")

    df["timestamps"] = pd.to_datetime(df["timestamps"], errors="raise")
    df = df.sort_values("timestamps")
    df = df.set_index("timestamps")

    resampled = pd.DataFrame(
        {
            "open": df["open"].resample("1D").first(),
            "high": df["high"].resample("1D").max(),
            "low": df["low"].resample("1D").min(),
            "close": df["close"].resample("1D").last(),
            "volume": df["volume"].resample("1D").sum(),
        }
    ).dropna()

    resampled = resampled.reset_index()
    resampled["timestamps"] = resampled["timestamps"].dt.strftime("%Y-%m-%d %H:%M:%S")

    resampled.to_csv(TARGET, index=False)
    print(
        f"[OK] Rebuilt daily data -> {TARGET.name}: {len(resampled)} rows, "
        f"range {resampled['timestamps'].iloc[0]} -> {resampled['timestamps'].iloc[-1]}"
    )


if __name__ == "__main__":
    rebuild_daily()

