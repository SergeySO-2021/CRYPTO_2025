"""
Fetch sample klines from Binance and compare with local CSV data.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import datetime as dt

import pandas as pd
import requests


TESTS: List[Tuple[str, str, str]] = [
    ("df_btc_1h_complete.csv", "1h", "2024-02-10 19:00:00"),
    ("df_btc_1h_complete.csv", "1h", "2024-10-07 02:00:00"),
    ("df_btc_30m_complete.csv", "30m", "2024-02-10 19:30:00"),
    ("df_btc_15m_complete.csv", "15m", "2024-02-11 04:45:00"),
    ("df_btc_1d.csv", "1d", "2024-02-11 21:00:00"),
]

INTERVAL_MS = {
    "1h": 3600_000,
    "30m": 1800_000,
    "15m": 900_000,
    "1d": 86_400_000,
}


def fetch_binance_kline(symbol: str, interval: str, ts_local: str) -> Dict:
    ts = pd.to_datetime(ts_local)
    ts_utc = ts + pd.Timedelta(hours=3)  # convert from UTC-3 to UTC

    start_ms = int(ts_utc.value // 10**6)
    end_ms = start_ms + INTERVAL_MS[interval] - 1

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1,
    }
    resp = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError("No kline data returned")

    kline = data[0]
    return {
        "open_time_utc": dt.datetime.utcfromtimestamp(kline[0] / 1000),
        "open": float(kline[1]),
        "high": float(kline[2]),
        "low": float(kline[3]),
        "close": float(kline[4]),
        "volume": float(kline[5]),
    }


def main() -> None:
    base = Path(__file__).resolve().parent.parent

    for filename, interval, ts in TESTS:
        path = base / filename
        df = pd.read_csv(path)
        row = df.loc[df["timestamps"] == ts]
        if row.empty:
            print(f"[WARN] row not found in {filename} for {ts}")
            continue

        row_dict = row.iloc[0].to_dict()
        kline = fetch_binance_kline("BTCUSDT", interval, ts)

        print(f"--- {filename} interval={interval} ts={ts} ---")
        print("local   :", row_dict)
        print("binance :", kline)

        for key in ("open", "high", "low", "close"):
            diff = row_dict[key] - kline[key]
            print(f"  diff {key:<5}: {diff:>12.6f}")

        if "volume" in row_dict:
            diff_vol = row_dict["volume"] - kline["volume"]
            print(f"  diff volume: {diff_vol:>12.6f}")

        print()


if __name__ == "__main__":
    main()

