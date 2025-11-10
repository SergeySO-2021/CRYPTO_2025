# 🚀 БЫСТРЫЙ СТАРТ: Визуализация df_btc_15m_complete.csv в Grafana

## 📋 ШАГИ ДЛЯ ЗАГРУЗКИ ДАННЫХ И ВИЗУАЛИЗАЦИИ

### ШАГ 1: Запуск инфраструктуры Grafana + InfluxDB

```bash
cd GRAFANA/grafana
docker-compose up -d
```

**Проверка:**
- InfluxDB: http://localhost:8086
- Grafana: http://localhost:3001 (или http://localhost:3000, если порт 3000 свободен)

### ШАГ 2: Установка зависимостей (если еще не установлены)

```bash
pip install influxdb-client pandas
```

### ШАГ 3: Загрузка CSV данных в InfluxDB

```bash
cd C:\Users\XE\Desktop\CRYPTO_2025
python GRAFANA/scripts/load_csv_to_influxdb.py \
    --csv-file df_btc_15m_complete.csv \
    --symbol BTCUSDT \
    --timeframe 15m \
    --influxdb-url http://localhost:8086 \
    --influxdb-token my-super-secret-admin-token \
    --org crypto \
    --bucket binance_data
```

**Или более короткий вариант (используются значения по умолчанию):**

```bash
python GRAFANA/scripts/load_csv_to_influxdb.py --csv-file df_btc_15m_complete.csv
```

### ШАГ 4: Настройка Grafana Data Source

1. Откройте Grafana: http://localhost:3001 (или http://localhost:3000, если используется другая установка)
   - Логин: `admin`
   - Пароль: `admin123`

2. Перейдите в **Configuration → Data Sources → Add data source**

3. Выберите **InfluxDB**

4. Настройте подключение:
   - **Query Language**: Flux
   - **URL**: `http://influxdb:8086` (или `http://localhost:8086` если вне Docker)
   - **Organization**: `crypto`
   - **Token**: `my-super-secret-admin-token`
   - **Default Bucket**: `binance_data`

5. Нажмите **Save & Test**

### ШАГ 5: Создание дашборда

#### Вариант A: Импорт готового дашборда

Если есть готовый дашборд в `GRAFANA/grafana/dashboards/`, импортируйте его:
1. **Dashboards → Import**
2. Выберите файл `dashboard.json`
3. Выберите Data Source: `InfluxDB`

#### Вариант B: Создание нового дашборда

1. **Dashboards → New → New Dashboard**
2. **Add → Visualization**

3. **Создайте панель "BTC Price (15m)":**
   - В Query Builder используйте Flux:
   ```flux
   from(bucket: "binance_data")
     |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
     |> filter(fn: (r) => r["_measurement"] == "ohlcv")
     |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
     |> filter(fn: (r) => r["timeframe"] == "15m")
     |> filter(fn: (r) => r["_field"] == "close")
     |> aggregateWindow(every: 15m, fn: last, createEmpty: false)
   ```
   - Panel type: **Time series**
   - Title: "BTC Price (Close)"

4. **Добавьте панель "Volume":**
   ```flux
   from(bucket: "binance_data")
     |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
     |> filter(fn: (r) => r["_measurement"] == "ohlcv")
     |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
     |> filter(fn: (r) => r["timeframe"] == "15m")
     |> filter(fn: (r) => r["_field"] == "volume")
     |> aggregateWindow(every: 15m, fn: sum, createEmpty: false)
   ```
   - Panel type: **Time series**
   - Title: "Volume"

5. **Добавьте панель "OHLC Candles":**
   
   Для создания свечного графика создайте 4 запроса:
   
   **Open:**
   ```flux
   from(bucket: "binance_data")
     |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
     |> filter(fn: (r) => r["_measurement"] == "ohlcv")
     |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
     |> filter(fn: (r) => r["timeframe"] == "15m")
     |> filter(fn: (r) => r["_field"] == "open")
     |> aggregateWindow(every: 15m, fn: first, createEmpty: false)
   ```
   
   **High:**
   ```flux
   from(bucket: "binance_data")
     |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
     |> filter(fn: (r) => r["_measurement"] == "ohlcv")
     |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
     |> filter(fn: (r) => r["timeframe"] == "15m")
     |> filter(fn: (r) => r["_field"] == "high")
     |> aggregateWindow(every: 15m, fn: max, createEmpty: false)
   ```
   
   **Low:**
   ```flux
   from(bucket: "binance_data")
     |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
     |> filter(fn: (r) => r["_measurement"] == "ohlcv")
     |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
     |> filter(fn: (r) => r["timeframe"] == "15m")
     |> filter(fn: (r) => r["_field"] == "low")
     |> aggregateWindow(every: 15m, fn: min, createEmpty: false)
   ```
   
   **Close:**
   ```flux
   from(bucket: "binance_data")
     |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
     |> filter(fn: (r) => r["_measurement"] == "ohlcv")
     |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
     |> filter(fn: (r) => r["timeframe"] == "15m")
     |> filter(fn: (r) => r["_field"] == "close")
     |> aggregateWindow(every: 15m, fn: last, createEmpty: false)
   ```
   
   - Panel type: **Time series**
   - Title: "OHLC Candles"

### ШАГ 6: Настройка переменных для интерактивности

Для удобного переключения периодов:

1. **Dashboard Settings → Variables → New**

2. **Создайте переменную `$date_range`:**
   - Name: `date_range`
   - Type: `Interval`
   - Label: `Date Range`
   - Values: `1h, 6h, 12h, 24h, 7d, 30d, 90d, 1y`

3. Используйте в запросах:
   ```flux
   |> range(start: $date_range)
   ```

### ШАГ 7: Сохранение дашборда

1. Нажмите **Save dashboard**
2. Укажите название: "BTC 15m Analysis"
3. Нажмите **Save**

---

## ✅ ПРОВЕРКА РАБОТЫ

После выполнения всех шагов вы должны увидеть:

1. ✅ Данные загружены в InfluxDB (проверьте в InfluxDB UI: http://localhost:8086)
2. ✅ Data Source подключен в Grafana (зеленая галочка)
3. ✅ Дашборд отображает графики с данными BTC
4. ✅ Можно zoom и панорамировать графики
5. ✅ Можно переключать временные периоды

---

## 🐛 РЕШЕНИЕ ПРОБЛЕМ

### Данные не отображаются

1. **Проверьте, что данные записаны в InfluxDB:**
   - Откройте http://localhost:8086
   - Data Explorer → выберите bucket `binance_data`
   - Проверьте наличие measurement `ohlcv`

2. **Проверьте временной диапазон:**
   - В Grafana выберите правильный временной диапазон (правый верхний угол)

3. **Проверьте запросы:**
   - Edit панели → Query Inspector → проверьте данные

### Ошибка подключения к InfluxDB

1. Проверьте, что InfluxDB запущен:
   ```bash
   docker ps
   ```

2. Проверьте токен и настройки в Data Source

3. Проверьте логи:
   ```bash
   docker logs influxdb
   ```

---

## 📚 ДОПОЛНИТЕЛЬНЫЕ РЕСУРСЫ

- [GRAFANA_SETUP.md](./GRAFANA_SETUP.md) - Детальная настройка Grafana
- [VISUALIZATION_ARCHITECTURE.md](./VISUALIZATION_ARCHITECTURE.md) - Архитектура системы визуализации

---

**Готово! Теперь вы можете визуализировать данные BTC 15m в Grafana!** 🎉

