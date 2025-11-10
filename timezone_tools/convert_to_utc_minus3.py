"""
Utility script to convert CSV candle data timestamps to UTC-3.

The script adjusts the `timestamps` column by -3 hours for the
following files located in the project root:
    - df_btc_15m_complete.csv
    - df_btc_30m_complete.csv
    - df_btc_1h_complete.csv
    - df_btc_1d.csv

For daily data the timestamps are converted to full datetime strings
with the time component preserved after the shift.
"""

from pathlib import Path

import pandas as pd


FILES = [
    ("df_btc_15m_complete.csv", False),
    ("df_btc_30m_complete.csv", False),
    ("df_btc_1h_complete.csv", False),
    ("df_btc_1d.csv", True),
]

OFFSET_HOURS = -3  # convert from UTC to UTC-3
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def convert_file(path: Path, treat_as_daily: bool) -> None:
    df = pd.read_csv(path)
    if "timestamps" not in df.columns:
        raise ValueError(f"{path.name}: missing 'timestamps' column")

    dt = pd.to_datetime(df["timestamps"], errors="raise")
    dt_shifted = dt + pd.Timedelta(hours=OFFSET_HOURS)

    df["timestamps"] = dt_shifted.dt.strftime(TIME_FORMAT)
    df.to_csv(path, index=False)

    print(
        f"[OK] {path.name}: {len(df)} rows, "
        f"range {df['timestamps'].iloc[0]} -> {df['timestamps'].iloc[-1]}"
    )


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    for filename, is_daily in FILES:
        file_path = base / filename
        if not file_path.exists():
            print(f"[WARN] {file_path} not found; skipping")
            continue
        convert_file(file_path, is_daily)


if __name__ == "__main__":
    main()

