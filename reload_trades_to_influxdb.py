"""
Перезагрузка данных с trades полями в InfluxDB
"""

import sys
import io
from pathlib import Path
import pandas as pd

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append(str(Path(__file__).parent))

from binance_data_collector.utils.influxdb_client import InfluxDBWriter
from binance_data_collector.utils.logger import setup_logger

logger = setup_logger("reload_trades_to_influxdb")

csv_file = r"C:\Users\XE\Desktop\CRYPTO_2025\binance_data_collector\BTCUSDT_15m_COMBINED.csv"

print("=" * 80)
print("ПЕРЕЗАГРУЗКА ДАННЫХ С TRADES В INFLUXDB")
print("=" * 80)

# Загрузка CSV
print(f"\n[ЗАГРУЗКА] CSV файл...")
df = pd.read_csv(csv_file)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

print(f"[OK] Загружено {len(df):,} строк")
print(f"[INFO] Trades данных: {df['trades_buy_volume'].notna().sum():,} строк")

# Подключение
print(f"\n[ПОДКЛЮЧЕНИЕ] InfluxDB...")
writer = InfluxDBWriter(
    url="http://localhost:8086",
    token="my-super-secret-admin-token",
    org="crypto",
    bucket="binance_data"
)

if writer.client is None:
    print("[ERROR] Не удалось подключиться!")
    sys.exit(1)

# Запись данных (записываем ВСЕ поля, включая NaN)
print(f"\n[ЗАПИСЬ] Загрузка данных в InfluxDB...")
from influxdb_client import Point

chunk_size = 10000
total_chunks = (len(df) + chunk_size - 1) // chunk_size

for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunk_num = (i // chunk_size) + 1
    
    print(f"   [ЧАНК {chunk_num}/{total_chunks}] {len(chunk)} записей...")
    
    points = []
    for timestamp, row in chunk.iterrows():
        point = Point("btc_combined") \
            .tag("symbol", "BTCUSDT") \
            .tag("timeframe", "15m") \
            .time(pd.Timestamp(timestamp))
        
        # OHLCV поля
        if 'open' in row:
            point = point.field("open", float(row['open']) if pd.notna(row['open']) else 0.0)
        if 'high' in row:
            point = point.field("high", float(row['high']) if pd.notna(row['high']) else 0.0)
        if 'low' in row:
            point = point.field("low", float(row['low']) if pd.notna(row['low']) else 0.0)
        if 'close' in row:
            point = point.field("close", float(row['close']) if pd.notna(row['close']) else 0.0)
        if 'volume' in row:
            point = point.field("volume", float(row['volume']) if pd.notna(row['volume']) else 0.0)
        if 'quote_volume' in row:
            point = point.field("quote_volume", float(row['quote_volume']) if pd.notna(row['quote_volume']) else 0.0)
        if 'taker_buy_base' in row:
            point = point.field("taker_buy_base", float(row['taker_buy_base']) if pd.notna(row['taker_buy_base']) else 0.0)
        if 'taker_buy_quote' in row:
            point = point.field("taker_buy_quote", float(row['taker_buy_quote']) if pd.notna(row['taker_buy_quote']) else 0.0)
        
        # Trades поля (записываем только если не NaN)
        if 'trades_buy_volume' in row and pd.notna(row['trades_buy_volume']):
            point = point.field("trades_buy_volume", float(row['trades_buy_volume']))
        if 'trades_sell_volume' in row and pd.notna(row['trades_sell_volume']):
            point = point.field("trades_sell_volume", float(row['trades_sell_volume']))
        if 'trades_count' in row and pd.notna(row['trades_count']):
            point = point.field("trades_count", int(row['trades_count']))
        
        points.append(point)
    
    writer.write_api.write(bucket="binance_data", record=points)
    print(f"   [OK] Чанк {chunk_num} записан")

print(f"\n[OK] Все данные перезагружены!")
print(f"[INFO] Загружено: {len(df):,} записей")
print(f"[INFO] Период: {df.index[0]} - {df.index[-1]}")

writer.close()

print("\n" + "=" * 80)
print("[OK] ГОТОВО! Теперь проверьте данные: py check_influxdb_data.py")
print("=" * 80)

