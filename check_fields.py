from influxdb_client import InfluxDBClient

client = InfluxDBClient(url='http://localhost:8086', token='my-super-secret-admin-token', org='crypto')
query_api = client.query_api()

# Проверяем taker_buy_quote
result = query_api.query('from(bucket: "binance_data") |> range(start: -365d) |> filter(fn: (r) => r["_measurement"] == "btc_combined") |> filter(fn: (r) => r["_field"] == "taker_buy_quote") |> limit(n: 1)')
print("taker_buy_quote:", "ЕСТЬ" if list(result) else "НЕТ")

client.close()

