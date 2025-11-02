# 📊 Grafana Dashboard для Binance BTC Data

Быстрый старт для визуализации данных BTC в реальном времени.

## 🚀 Быстрый запуск (3 шага)

### 1. Запуск Grafana и InfluxDB

```bash
docker-compose up -d
```

Это запустит:
- InfluxDB на `http://localhost:8086`
- Grafana на `http://localhost:3000`

### 2. Настройка InfluxDB (первый запуск)

Откройте `http://localhost:8086` и используйте:
- Username: `admin`
- Password: `admin123`
- Organization: `crypto`
- Bucket: `binance_data`

### 3. Запуск сбора данных

```bash
cd ..
pip install influxdb-client
python scripts/collect_to_influxdb.py --mode realtime
```

### 4. Открыть Grafana

Откройте `http://localhost:3000`:
- Username: `admin`
- Password: `admin123`

Дашборд "Binance BTC Advanced Data" должен быть уже настроен!

## 📊 Что вы увидите

- **BTC Price** - цена в реальном времени
- **Volume** - объемы торгов
- **Market Buy/Sell** - покупки и продажи
- **Liquidations** - ликвидации (long/short)
- **Open Interest** - открытый интерес
- **Order Book Depth** - глубина стакана на разных уровнях (3%, 8%, 15%, 60%)

## ⚙️ Остановка

```bash
docker-compose down
```

## 🔧 Изменение паролей

Отредактируйте `docker-compose.yml` перед первым запуском для изменения паролей по умолчанию.


