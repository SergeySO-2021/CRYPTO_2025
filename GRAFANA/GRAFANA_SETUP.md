# 📊 Настройка Grafana для визуализации данных Binance

Это руководство поможет настроить Grafana для просмотра данных Binance в реальном времени.

## 🎯 Что будет настроено

- **InfluxDB** - база данных временных рядов для хранения данных
- **Grafana** - платформа для визуализации
- **Автоматическая запись данных** из скриптов сбора в InfluxDB
- **Дашборд с графиками** для всех метрик BTC

## 🚀 Быстрый старт (Docker)

Самый простой способ - использовать Docker Compose:

### 1. Установка Docker

Убедитесь, что у вас установлен Docker и Docker Compose:
- [Docker Desktop для Windows](https://www.docker.com/products/docker-desktop/)

### 2. Запуск сервисов

```bash
cd GRAFANA/grafana
docker-compose up -d
```

Это запустит:
- **InfluxDB** на `http://localhost:8086`
- **Grafana** на `http://localhost:3000`

### 3. Первоначальная настройка InfluxDB

Откройте `http://localhost:8086` и создайте:
- **Username**: `admin`
- **Password**: `admin123`
- **Organization**: `crypto`
- **Bucket**: `binance_data`

Или используйте предустановленные значения из `docker-compose.yml`.

### 4. Вход в Grafana

Откройте `http://localhost:3000`:
- **Username**: `admin`
- **Password**: `admin123`

Дашборд уже настроен автоматически!

## 📝 Ручная установка (без Docker)

### 1. Установка InfluxDB

#### Windows:
1. Скачайте InfluxDB: https://portal.influxdata.com/downloads/
2. Распакуйте архив
3. Запустите `influxd.exe`

#### Linux:
```bash
wget https://dl.influxdata.com/influxdb/releases/influxdb2-2.7.0-linux-amd64.tar.gz
tar xvzf influxdb2-2.7.0-linux-amd64.tar.gz
cd influxdb2-2.7.0-linux-amd64
./influxd
```

### 2. Установка Grafana

#### Windows:
1. Скачайте Grafana: https://grafana.com/grafana/download
2. Установите и запустите сервис

#### Linux:
```bash
wget https://dl.grafana.com/oss/release/grafana-10.2.0.linux-amd64.tar.gz
tar -zxvf grafana-10.2.0.linux-amd64.tar.gz
cd grafana-10.2.0
./bin/grafana-server
```

### 3. Настройка InfluxDB

1. Откройте `http://localhost:8086`
2. Создайте аккаунт
3. Создайте Organization: `crypto`
4. Создайте Bucket: `binance_data`
5. Создайте API Token (Settings → Tokens → Generate)

### 4. Настройка Grafana

1. Откройте `http://localhost:3000` (логин: admin/admin)
2. Settings → Data Sources → Add data source
3. Выберите InfluxDB
4. Настройте:
   - **URL**: `http://localhost:8086`
   - **Organization**: `crypto`
   - **Bucket**: `binance_data`
   - **Token**: ваш токен из InfluxDB
   - **Query Language**: Flux

5. Импортируйте дашборд из `GRAFANA/grafana/dashboards/dashboard.json`

## 🔧 Настройка записи данных

### Установка зависимостей

```bash
pip install influxdb-client
```

### Запись исторических данных

```bash
python scripts/collect_to_influxdb.py \
    --mode historical \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --influxdb-url http://localhost:8086 \
    --influxdb-token YOUR_TOKEN_HERE
```

### Запись данных в реальном времени

```bash
python scripts/collect_to_influxdb.py \
    --mode realtime \
    --influxdb-url http://localhost:8086 \
    --influxdb-token YOUR_TOKEN_HERE
```

## 📊 Структура дашборда

Дашборд включает следующие графики:

1. **BTC Price (Close)** - цена закрытия
2. **Volume** - объем торгов
3. **Market Buy/Sell Volume** - объемы покупок и продаж
4. **Liquidations** - ликвидации (long/short/total)
5. **Open Interest** - открытый интерес
6. **Order Book Depth** - глубина order book (3%, 8%, 15%, 60%)
7. **Order Book Imbalance** - дисбаланс на разных глубинах

## ⚙️ Настройка автоматического обновления

В Grafana дашборд обновляется каждые 10 секунд. Для изменения:

1. Откройте дашборд
2. Settings → Time options
3. Измените "Auto refresh"

## 🔒 Безопасность

**Важно для продакшена:**

1. Измените пароли по умолчанию:
   - В `docker-compose.yml` измените пароли
   - В Grafana: Configuration → Users → Change password

2. Используйте переменные окружения для токенов:
```bash
export INFLUXDB_TOKEN="your-secure-token"
export GRAFANA_ADMIN_PASSWORD="your-secure-password"
```

3. Настройте firewall для ограничения доступа

## 🐛 Решение проблем

### InfluxDB не запускается

```bash
# Проверьте логи
docker logs influxdb

# Проверьте порт
netstat -an | findstr 8086  # Windows
lsof -i :8086              # Linux/Mac
```

### Grafana не подключается к InfluxDB

1. Проверьте, что InfluxDB запущен
2. Проверьте URL и токен в настройках Grafana
3. Проверьте логи: `docker logs grafana`

### Данные не отображаются

1. Проверьте, что данные записываются в InfluxDB:
```python
# Примечание: модуль influxdb_client находится в binance_data_collector/utils/
from binance_data_collector.utils.influxdb_client import InfluxDBWriter
writer = InfluxDBWriter()
# Проверьте соединение
```

2. Проверьте временной диапазон в Grafana (верхний правый угол)
3. Проверьте запросы в панелях (Edit → Query Inspector)

## 📚 Полезные ссылки

- [Документация InfluxDB](https://docs.influxdata.com/)
- [Документация Grafana](https://grafana.com/docs/)
- [Flux язык запросов](https://docs.influxdata.com/flux/)

## 🎨 Кастомизация дашборда

Вы можете создать свои панели:

1. В Grafana: Add → Visualization
2. Выберите тип графика
3. В Query используйте Flux:
```flux
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_advanced")
  |> filter(fn: (r) => r["_field"] == "close")
  |> aggregateWindow(every: 15m, fn: last, createEmpty: false)
```

## ✅ Готово!

Теперь у вас есть полная система мониторинга данных Binance в реальном времени!


