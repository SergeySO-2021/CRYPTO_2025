import pandas as pd
import sys
import io
from influxdb_client import InfluxDBClient

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("СРАВНЕНИЕ: CSV vs InfluxDB")
print("=" * 60)

# CSV
csv_file = r"binance_data_collector\BTCUSDT_15m_COMBINED.csv"
df = pd.read_csv(csv_file)
print(f"\n[CSV] Файл:")
print(f"   Строк: {len(df):,}")
print(f"   Колонки: {list(df.columns)}")
print(f"   Период: {df['time'].min()} - {df['time'].max()}")

# InfluxDB
client = InfluxDBClient(url='http://localhost:8086', token='my-super-secret-admin-token', org='crypto')
query_api = client.query_api()

# Подсчет записей
count_query = 'from(bucket: "binance_data") |> range(start: -365d) |> filter(fn: (r) => r["_measurement"] == "btc_combined") |> filter(fn: (r) => r["_field"] == "close") |> count()'
result = query_api.query(count_query)
count = 0
if result:
    for table in result:
        for record in table.records:
            count = record.get_value()
            break

print(f"\n[INFLUXDB] База данных:")
print(f"   Записей с close: {count:,}")

# Проверка полей
fields_query = 'from(bucket: "binance_data") |> range(start: -365d) |> filter(fn: (r) => r["_measurement"] == "btc_combined") |> group() |> distinct(column: "_field")'
result = query_api.query(fields_query)
fields = []
if result:
    for table in result:
        for record in table.records:
            fields.append(record.get_value())

print(f"   Доступные поля: {fields}")

print("\n" + "=" * 60)
print("✅ CSV файл - это исходные данные")
print("✅ InfluxDB - это база данных, куда мы загрузили CSV")
print("✅ Grafana читает данные ИЗ InfluxDB, а не из CSV")
print("=" * 60)

client.close()

