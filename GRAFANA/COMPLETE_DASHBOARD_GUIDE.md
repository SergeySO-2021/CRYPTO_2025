# 📊 ПОЛНАЯ ИНСТРУКЦИЯ: Создание дашборда BTC Price & Volume

## 🎯 Цель
Создать дашборд в Grafana с:
1. Графиком цены BTC (цена закрытия 15-минутных свечей)
2. При наведении показывать Max и Min значения
3. Объемы покупок (зеленым цветом, эффект "пламени", в долларах, шкала справа)
4. Объемы продаж (красным цветом, эффект "пламени", в долларах, шкала справа)

---

## ⚡ БЫСТРЫЙ СТАРТ (если данные уже загружены)

**Если данные уже в InfluxDB, но нет trades полей:**

1. **Проверьте доступные поля:**
   ```bash
   py check_influxdb_data.py
   ```

2. **Используйте упрощенные запросы:**
   - **Объемы покупок**: используйте поле `taker_buy_quote` (Вариант B в ШАГ 6.3)
   - **Объемы продаж**: используйте расчет `quote_volume - taker_buy_quote` (Вариант B в ШАГ 7.3)

3. **Создайте дашборд** по инструкции ниже, используя **Вариант B** для объемов

**Эти запросы работают с существующими данными и не требуют перезагрузки!**

---

## ⚠️ ВАЖНО: ПРОВЕРКА ДОСТУПНЫХ ПОЛЕЙ

**Сначала проверьте, какие поля есть в InfluxDB:**

1. Запустите скрипт проверки:
   ```bash
   py check_influxdb_data.py
   ```

2. Или проверьте через Grafana:
   - Откройте Grafana → Add visualization
   - Data source: InfluxDB → Flux
   - Запрос:
   ```flux
   from(bucket: "binance_data")
     |> range(start: -365d)
     |> filter(fn: (r) => r["_measurement"] == "btc_combined")
     |> group()
     |> distinct(column: "_field")
   ```

**Если в InfluxDB НЕТ полей `trades_buy_volume` и `trades_sell_volume`:**
- Используйте **Вариант B** в запросах ниже (использует `taker_buy_quote` и `quote_volume`)
- Или перезагрузите данные (см. ШАГ 2)

---

## 📋 ШАГ 1: ПРОВЕРКА ДАННЫХ

### 1.1. Проверьте наличие CSV файла
Путь: `C:\Users\XE\Desktop\CRYPTO_2025\binance_data_collector\BTCUSDT_15m_COMBINED.csv`

Если файла нет, создайте его:
```bash
cd C:\Users\XE\Desktop\CRYPTO_2025
py aggregate_trades_to_15m.py
```

Затем сохраните в CSV:
```python
# Создайте файл save_to_csv.py
import pickle
import pandas as pd

pkl_path = r"C:\Users\XE\Desktop\CRYPTO_2025\binance_data_collector\BTCUSDT_15m_20251104_011229.pkl"
output_dir = r"C:\Users\XE\Desktop\CRYPTO_2025\binance_data_collector"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

df_ohlcv = data['ohlcv_df']
df_trades = data['trades_df']

# Агрегируем trades
trades_15m = df_trades.resample('15min').agg({
    'buy_volume': 'sum',
    'sell_volume': 'sum',
    'quantity': 'sum',
    'price': 'mean'
})
trades_15m['total_volume'] = trades_15m['buy_volume'] + trades_15m['sell_volume']
trades_15m['buy_sell_ratio'] = trades_15m['buy_volume'] / trades_15m['sell_volume'].replace(0, 1)
trades_15m['trade_count'] = df_trades.resample('15min').size()
trades_15m.columns = ['trades_buy_volume', 'trades_sell_volume', 'trades_total_quantity', 'trades_avg_price', 'trades_total_volume', 'trades_buy_sell_ratio', 'trades_count']

# Объединяем
df_combined = df_ohlcv.join(trades_15m, how='left')
df_combined_csv = df_combined.reset_index()
df_combined_csv.rename(columns={'timestamp': 'time'}, inplace=True)

# Сохраняем
csv_file = f"{output_dir}\\BTCUSDT_15m_COMBINED.csv"
df_combined_csv.to_csv(csv_file, index=False, encoding='utf-8')
print(f"✅ Сохранено: {csv_file}")
```

---

## 📋 ШАГ 2: ЗАГРУЗКА ДАННЫХ В INFLUXDB

**⚠️ ВАЖНО:** 
- Если trades данные (`trades_buy_volume`, `trades_sell_volume`) не загрузились в InfluxDB
- **ИСПОЛЬЗУЙТЕ Вариант B в запросах** - он использует `taker_buy_quote` и `quote_volume`, которые уже есть в InfluxDB
- Или перезагрузите данные с trades полями (см. ниже)

### 2.1. Проверьте, что InfluxDB запущен
```bash
# Проверьте, что InfluxDB работает
# Обычно это через docker-compose или напрямую
```

### 2.2. Загрузите данные в InfluxDB

**Обычная загрузка:**
```bash
cd C:\Users\XE\Desktop\CRYPTO_2025
py load_combined_to_influxdb.py
```

**Если trades данные не загрузились, перезагрузите:**
```bash
py reload_trades_to_influxdb.py
```

**Ожидаемый результат:**
```
✅ Загружено 64,498 строк
✅ Подключение к InfluxDB установлено
✅ Все данные успешно записаны в InfluxDB!
```

**После загрузки проверьте:**
```bash
py check_influxdb_data.py
```

Должны быть видны поля: `trades_buy_volume`, `trades_sell_volume`

### 2.3. Проверьте данные в InfluxDB (опционально)
Откройте InfluxDB UI: http://localhost:8086
- Логин: admin
- Пароль: admin123 (или ваш токен)
- Проверьте bucket: `binance_data`
- Measurement: `btc_combined`

---

## 📋 ШАГ 3: ОТКРЫТЬ GRAFANA

### 3.1. Откройте Grafana
```
http://localhost:3001
```

### 3.2. Войдите в систему
- Логин: `admin`
- Пароль: `admin123`

### 3.3. Проверьте Data Source
1. Перейдите: **Configuration (⚙️) → Data Sources**
2. Нажмите на **InfluxDB**
3. Проверьте настройки:
   - **URL**: `http://influxdb:8086` или `http://localhost:8086`
   - **Database/Bucket**: `binance_data`
   - **Organization**: `crypto`
   - **Token**: ваш токен
4. Нажмите **Save & Test**
5. Должно появиться: ✅ **Data source is working**

---

## 📋 ШАГ 4: СОЗДАНИЕ НОВОГО ДАШБОРДА

### 4.1. Создайте новый дашборд
1. В левом меню: **Dashboards (📊)**
2. Нажмите **New** → **New Dashboard**
3. Нажмите **Add visualization** (синяя кнопка)

---

## 📋 ШАГ 5: ПАНЕЛЬ 1 - ГРАФИК ЦЕНЫ BTC

### 5.1. Настройте Data Source
1. В панели редактирования найдите **Query**
2. Выберите **Data source**: `InfluxDB`
3. Переключитесь на **Flux** (вкладка или переключатель)

### 5.2. Запрос для цены закрытия
**Вставьте этот запрос:**
```flux
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "close")
  |> aggregateWindow(every: 15m, fn: last, createEmpty: false)
```

### 5.3. Добавьте запросы для Max и Min
**Нажмите "+ Query" два раза, чтобы добавить еще 2 запроса:**

**Запрос B (Max):**
```flux
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "high")
  |> aggregateWindow(every: 15m, fn: max, createEmpty: false)
```

**Запрос C (Min):**
```flux
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "low")
  |> aggregateWindow(every: 15m, fn: min, createEmpty: false)
```

### 5.4. Настройте панель
1. **Вкладка "Panel options"** (справа):
   - **Title**: `BTC Price (Close) - 15m`
   - **Description**: `График цены закрытия с Max/Min при наведении`

2. **Вкладка "Field"** (или "Standard options"):
   - **Unit**: `currencyUSD`
   - **Decimals**: `2`

3. **Вкладка "Overrides"** (Field overrides):
   - Нажмите **+ Add field override**
   - **Field with name**: `high`
   - Настройки:
     - **Hide in graph**: `true`
     - **Hide in legend**: `true`
     - **Display name**: `Max`
   - Нажмите **+ Add field override** еще раз
   - **Field with name**: `low`
   - Настройки:
     - **Hide in graph**: `true`
     - **Hide in legend**: `true`
     - **Display name**: `Min`

4. **Вкладка "Tooltip"**:
   - **Mode**: `Multi`

5. **Вкладка "Legend"**:
   - **Show legend**: `On`
   - **Display mode**: `Table`
   - **Placement**: `Bottom`
   - **Show calculations**: `max`, `min`, `last`

6. Нажмите **Apply** (вверху справа)

---

## 📋 ШАГ 6: ПАНЕЛЬ 2 - ОБЪЕМЫ ПОКУПОК (ЗЕЛЕНОЕ ПЛАМЯ)

### 6.1. Добавьте новую панель
1. Нажмите **Add** (вверху дашборда)
2. Выберите **Visualization**

### 6.2. Настройте Data Source
- **Data source**: `InfluxDB`
- **Flux** режим

### 6.3. Запрос для объемов покупок в USD

**ВАЖНО:** Если поля `trades_buy_volume` нет в InfluxDB, используйте альтернативный вариант ниже!

#### Вариант A: Если trades_buy_volume есть (используйте pivot):
```flux
data = from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> map(fn: (r) => ({ r with _value: r.trades_buy_volume * r.close, _field: "buy_volume_usd" }))
  |> drop(columns: ["trades_buy_volume", "close", "open", "high", "low", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "trades_sell_volume", "trades_total_quantity", "trades_avg_price", "trades_total_volume", "trades_buy_sell_ratio", "trades_count"])
```

#### Вариант B: Если trades_buy_volume нет (используйте taker_buy_quote) - РЕКОМЕНДУЕТСЯ:
```flux
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "taker_buy_quote")
  |> aggregateWindow(every: 15m, fn: sum, createEmpty: false)
```
**Примечание:** `taker_buy_quote` - это уже объем покупок в USDT (долларах)! Это поле есть в ваших данных и работает для всех записей.

### 6.4. Настройте визуализацию
1. **Panel options**:
   - **Title**: `Market Buy Volume (Green Flame) - USD`
   - **Description**: `Объемы покупок в долларах`

2. **Visualization** (внизу):
   - Выберите тип: **Time series**

3. **Field**:
   - **Unit**: `currencyUSD`
   - **Decimals**: `1`

4. **Field overrides**:
   - **+ Add field override**
   - **Field with name**: `buy_volume_usd` (или любое поле)
   - **Color**: `green`
   - **Axis placement**: `Right`

5. **Visualization options** (вкладка "Graph styles" или "Options"):
   - **Draw style**: `Bars`
   - **Fill opacity**: `80`
   - **Gradient mode**: `Opacity`
   - **Line width**: `0`
   - **Bar width**: `0.97`
   - **Bar radius**: `0`

6. **Y-axis** (Axis):
   - **Left Y**: `Off` (или скрыть)
   - **Right Y**: `On`
   - **Right Y Unit**: `currencyUSD`
   - **Right Y Decimals**: `1`
   - **Right Y Label**: `Volume (USD)`

7. **Tooltip**:
   - **Mode**: `Multi`

8. Нажмите **Apply**

---

## 📋 ШАГ 7: ПАНЕЛЬ 3 - ОБЪЕМЫ ПРОДАЖ (КРАСНОЕ ПЛАМЯ)

### 7.1. Добавьте новую панель
1. Нажмите **Add** → **Visualization**

### 7.2. Настройте Data Source
- **Data source**: `InfluxDB`
- **Flux** режим

### 7.3. Запрос для объемов продаж в USD

#### Вариант A: Если trades_sell_volume есть:
```flux
data = from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> map(fn: (r) => ({ r with _value: r.trades_sell_volume * r.close, _field: "sell_volume_usd" }))
  |> drop(columns: ["trades_sell_volume", "close", "open", "high", "low", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "trades_buy_volume", "trades_total_quantity", "trades_avg_price", "trades_total_volume", "trades_buy_sell_ratio", "trades_count"])
```

#### Вариант B: Если trades_sell_volume нет (используйте расчет) - РЕКОМЕНДУЕТСЯ:
```flux
buy_vol = from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "taker_buy_quote")
  |> aggregateWindow(every: 15m, fn: sum, createEmpty: false)

total_vol = from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "quote_volume")
  |> aggregateWindow(every: 15m, fn: sum, createEmpty: false)

join(tables: {buy: buy_vol, total: total_vol}, on: ["_time"])
  |> map(fn: (r) => ({ r with _value: r.total__value - r.buy__value }))
  |> drop(columns: ["buy__value", "total__value", "buy__field", "total__field"])
```
**Примечание:** `quote_volume - taker_buy_quote` = объем продаж в USDT. Это поле есть в ваших данных и работает для всех записей.

### 7.4. Настройте визуализацию
1. **Panel options**:
   - **Title**: `Market Sell Volume (Red Flame) - USD`
   - **Description**: `Объемы продаж в долларах`

2. **Visualization**:
   - Тип: **Time series**

3. **Field**:
   - **Unit**: `currencyUSD`
   - **Decimals**: `1`

4. **Field overrides**:
   - **+ Add field override**
   - **Field with name**: `sell_volume_usd` (или любое поле)
   - **Color**: `red`
   - **Axis placement**: `Right`

5. **Visualization options**:
   - **Draw style**: `Bars`
   - **Fill opacity**: `80`
   - **Gradient mode**: `Opacity`
   - **Line width**: `0`
   - **Bar width**: `0.97`
   - **Bar radius**: `0`

6. **Y-axis**:
   - **Left Y**: `Off`
   - **Right Y**: `On`
   - **Right Y Unit**: `currencyUSD`
   - **Right Y Decimals**: `1`
   - **Right Y Label**: `Volume (USD)`

7. **Tooltip**:
   - **Mode**: `Multi`

8. Нажмите **Apply**

---

## 📋 ШАГ 8: РАСПОЛОЖЕНИЕ ПАНЕЛЕЙ

### 8.1. Расположите панели
1. **Панель 1 (Цена)**: Вверху, на всю ширину
2. **Панель 2 (Покупки)**: Слева, под ценой
3. **Панель 3 (Продажи)**: Справа, под ценой

Для изменения размера:
- Наведите на панель → нажмите иконку настроек (⚙️)
- Внизу найдите **Panel size**
- Или перетащите углы панели

---

## 📋 ШАГ 9: НАСТРОЙКА ВРЕМЕННОГО ДИАПАЗОНА

### 9.1. Выберите период
В правом верхнем углу дашборда:
1. Нажмите на временной диапазон (например, "Last 6 hours")
2. Выберите период, где есть данные:
   - **Last 7 days**
   - Или конкретные даты: **2024-01-01** до **2025-11-02**

---

## ⚠️ УСТРАНЕНИЕ ПРОБЛЕМ

### Проблема: "No data"
**Решения:**
1. Проверьте временной диапазон (данные с 2024-01-01)
2. Проверьте, что данные загружены:
   ```bash
   # Проверьте InfluxDB
   # Или перезагрузите данные:
   py load_combined_to_influxdb.py
   ```
3. Проверьте measurement: должно быть `btc_combined`
4. Проверьте bucket: должно быть `binance_data`

### Проблема: Ошибка в Flux запросе
**Решения:**
1. Проверьте синтаксис запроса
2. Убедитесь, что все кавычки правильные
3. Попробуйте упрощенный запрос:
   ```flux
   from(bucket: "binance_data")
     |> range(start: -365d)
     |> filter(fn: (r) => r["_measurement"] == "btc_combined")
     |> filter(fn: (r) => r["_field"] == "close")
   ```

### Проблема: Объемы не показываются
**Решения:**
1. Проверьте, что запрос использует `pivot` и `map`
2. Убедитесь, что `trades_buy_volume` и `close` существуют в данных
3. Проверьте временной диапазон (trades данные только до 2024-03-30)

### Проблема: Шкала не справа
**Решения:**
1. В Field overrides: **Axis placement** → `Right`
2. Или в Y-axis настройках: **Right Y** → `On`

---

## ✅ ПРОВЕРКА

### Что должно работать:
1. ✅ График цены отображается
2. ✅ При наведении видны Max и Min
3. ✅ Объемы покупок - зеленые столбцы справа
4. ✅ Объемы продаж - красные столбцы справа
5. ✅ Значения в долларах (тысячах/миллионах)
6. ✅ Tooltip показывает значения в USD

---

## 📝 БЫСТРЫЕ ЗАПРОСЫ ДЛЯ КОПИРОВАНИЯ

### Цена закрытия:
```flux
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "close")
  |> aggregateWindow(every: 15m, fn: last, createEmpty: false)
```

### Max (скрыт, только в tooltip):
```flux
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "high")
  |> aggregateWindow(every: 15m, fn: max, createEmpty: false)
```

### Min (скрыт, только в tooltip):
```flux
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "low")
  |> aggregateWindow(every: 15m, fn: min, createEmpty: false)
```

### Объемы покупок в USD:

**Вариант A (если trades_buy_volume есть):**
```flux
data = from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> map(fn: (r) => ({ r with _value: r.trades_buy_volume * r.close, _field: "buy_volume_usd" }))
  |> drop(columns: ["trades_buy_volume", "close", "open", "high", "low", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "trades_sell_volume", "trades_total_quantity", "trades_avg_price", "trades_total_volume", "trades_buy_sell_ratio", "trades_count"])
```

**Вариант B (РЕКОМЕНДУЕТСЯ - использует taker_buy_quote):**
```flux
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "taker_buy_quote")
  |> aggregateWindow(every: 15m, fn: sum, createEmpty: false)
```

### Объемы продаж в USD:

**Вариант A (если trades_sell_volume есть):**
```flux
data = from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> map(fn: (r) => ({ r with _value: r.trades_sell_volume * r.close, _field: "sell_volume_usd" }))
  |> drop(columns: ["trades_sell_volume", "close", "open", "high", "low", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "trades_buy_volume", "trades_total_quantity", "trades_avg_price", "trades_total_volume", "trades_buy_sell_ratio", "trades_count"])
```

**Вариант B (РЕКОМЕНДУЕТСЯ - использует расчет):**
```flux
buy_vol = from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "taker_buy_quote")
  |> aggregateWindow(every: 15m, fn: sum, createEmpty: false)

total_vol = from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "quote_volume")
  |> aggregateWindow(every: 15m, fn: sum, createEmpty: false)

join(tables: {buy: buy_vol, total: total_vol}, on: ["_time"])
  |> map(fn: (r) => ({ r with _value: r.total__value - r.buy__value }))
  |> drop(columns: ["buy__value", "total__value", "buy__field", "total__field"])
```

---

## 🎉 ГОТОВО!

После выполнения всех шагов у вас будет рабочий дашборд с графиком цены и объемов в виде "пламени"!

