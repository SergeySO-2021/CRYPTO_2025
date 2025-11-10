"""
Скрипт для проверки данных в InfluxDB
"""

import sys
import io
from pathlib import Path

# Устанавливаем кодировку для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append(str(Path(__file__).parent))

try:
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.query_api import QueryApi
except ImportError:
    print("❌ Не установлен influxdb-client!")
    print("   Установите: pip install influxdb-client")
    sys.exit(1)

# Настройки
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "my-super-secret-admin-token"
ORG = "crypto"
BUCKET = "binance_data"

print("=" * 80)
print("ПРОВЕРКА ДАННЫХ В INFLUXDB")
print("=" * 80)

# Подключение
print(f"\n[ПОДКЛЮЧЕНИЕ] InfluxDB: {INFLUXDB_URL}")
try:
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=ORG)
    query_api = client.query_api()
    print("[OK] Подключение успешно!")
except Exception as e:
    print(f"[ERROR] Ошибка подключения: {e}")
    sys.exit(1)

# Проверка bucket
print(f"\n[BUCKET] Проверка: {BUCKET}")
try:
    buckets_api = client.buckets_api()
    buckets = buckets_api.find_buckets()
    bucket_names = [b.name for b in buckets]
    
    if BUCKET in bucket_names:
        print(f"[OK] Bucket '{BUCKET}' найден!")
    else:
        print(f"[ERROR] Bucket '{BUCKET}' НЕ найден!")
        print(f"   Доступные buckets: {bucket_names}")
        sys.exit(1)
except Exception as e:
    print(f"[WARNING] Не удалось проверить bucket: {e}")

# Проверка measurement
print(f"\n[MEASUREMENT] Проверка: btc_combined")
flux_query = f'''
from(bucket: "{BUCKET}")
  |> range(start: -365d)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> group()
  |> distinct(column: "_field")
  |> limit(n: 20)
'''

try:
    result = query_api.query(flux_query)
    
    if result:
        print("[OK] Measurement 'btc_combined' найден!")
        print("\n[ПОЛЯ] Доступные поля (_field):")
        fields = []
        for table in result:
            for record in table.records:
                field = record.get_value()
                fields.append(field)
                print(f"   - {field}")
        
        # Проверка необходимых полей
        required_fields = ["close", "high", "low", "trades_buy_volume", "trades_sell_volume"]
        missing_fields = [f for f in required_fields if f not in fields]
        
        if missing_fields:
            print(f"\n[WARNING] Отсутствуют поля: {missing_fields}")
        else:
            print(f"\n[OK] Все необходимые поля присутствуют!")
    else:
        print("[ERROR] Measurement 'btc_combined' НЕ найден или пуст!")
        print("   Проверьте, что данные были загружены")
        sys.exit(1)
        
except Exception as e:
    print(f"[ERROR] Ошибка запроса: {e}")
    import traceback
    traceback.print_exc()

# Проверка количества записей
print(f"\n[КОЛИЧЕСТВО] Проверка записей:")
flux_query = f'''
from(bucket: "{BUCKET}")
  |> range(start: -365d)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "close")
  |> count()
'''

try:
    result = query_api.query(flux_query)
    if result:
        for table in result:
            for record in table.records:
                count = record.get_value()
                print(f"   Записей с полем 'close': {count:,}")
                break
except Exception as e:
    print(f"⚠️ Не удалось подсчитать записи: {e}")

# Проверка временного диапазона
print(f"\n[ВРЕМЯ] Проверка временного диапазона:")
flux_query = f'''
from(bucket: "{BUCKET}")
  |> range(start: -365d)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "close")
  |> first()
  |> map(fn: (r) => ({{ r with _value: r._time }}))
  |> yield(name: "first_record")
'''

try:
    result = query_api.query(flux_query)
    first_time = None
    for table in result:
        for record in table.records:
            first_time = record.get_time()
            break
    
    if first_time:
        print(f"   Первая запись: {first_time}")
except Exception as e:
    print(f"⚠️ Не удалось получить первую запись: {e}")

flux_query = f'''
from(bucket: "{BUCKET}")
  |> range(start: -365d)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "close")
  |> last()
  |> map(fn: (r) => ({{ r with _value: r._time }}))
  |> yield(name: "last_record")
'''

try:
    result = query_api.query(flux_query)
    last_time = None
    for table in result:
        for record in table.records:
            last_time = record.get_time()
            break
    
    if last_time:
        print(f"   Последняя запись: {last_time}")
except Exception as e:
    print(f"⚠️ Не удалось получить последнюю запись: {e}")

# Тестовый запрос для цены
print(f"\n[ТЕСТ] Запрос для цены (последние 5 записей):")
flux_query = f'''
from(bucket: "{BUCKET}")
  |> range(start: -365d)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "close")
  |> limit(n: 5)
'''

try:
    result = query_api.query(flux_query)
    if result:
        print("[OK] Запрос выполнен успешно!")
        print("\n   Последние 5 записей:")
        for table in result:
            for record in table.records:
                print(f"   {record.get_time()}: {record.get_value()}")
    else:
        print("[ERROR] Запрос вернул пустой результат!")
except Exception as e:
    print(f"[ERROR] Ошибка тестового запроса: {e}")
    import traceback
    traceback.print_exc()

# Тестовый запрос для объемов
print(f"\n[ТЕСТ] Запрос для объемов покупок (последние 5 записей):")
flux_query = f'''
from(bucket: "{BUCKET}")
  |> range(start: -365d)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "trades_buy_volume")
  |> limit(n: 5)
'''

try:
    result = query_api.query(flux_query)
    if result:
        print("[OK] Запрос выполнен успешно!")
        print("\n   Последние 5 записей:")
        for table in result:
            for record in table.records:
                print(f"   {record.get_time()}: {record.get_value()}")
    else:
        print("[WARNING] Запрос вернул пустой результат (возможно, нет данных для этого периода)")
except Exception as e:
    print(f"[WARNING] Ошибка тестового запроса: {e}")

print("\n" + "=" * 80)
print("[OK] ПРОВЕРКА ЗАВЕРШЕНА")
print("=" * 80)
print("\n💡 Если данные не найдены:")
print("   1. Проверьте, что скрипт load_combined_to_influxdb.py выполнился успешно")
print("   2. Проверьте временной диапазон в Grafana (данные с 2024-01-01)")
print("   3. Проверьте, что measurement называется 'btc_combined'")

client.close()

