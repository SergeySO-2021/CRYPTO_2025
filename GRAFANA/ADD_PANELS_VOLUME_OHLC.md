# 📊 ДОБАВЛЕНИЕ ПАНЕЛЕЙ: Объем и OHLC Свечи

## 🎯 Цель: Добавить панели для визуализации объема и OHLC свечей BTC

---

## 📋 ПАНЕЛЬ 1: Объем (Volume)

### Шаг 1: Добавьте новую панель

1. В вашем дашборде нажмите **Add** (кнопка сверху)
2. Выберите **Visualization**

### Шаг 2: Настройте Data Source

1. В верхней части панели найдите выпадающий список **Data source**
2. Выберите **InfluxDB**

### Шаг 3: Переключитесь на Flux

1. Найдите переключатель **Query Builder / Flux**
2. Переключитесь на **Flux**

### Шаг 4: Вставьте запрос для Volume

Вставьте следующий запрос (без символов ```):

```
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "ohlcv")
  |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
  |> filter(fn: (r) => r["timeframe"] == "15m")
  |> filter(fn: (r) => r["_field"] == "volume")
  |> aggregateWindow(every: 15m, fn: sum, createEmpty: false)
```

### Шаг 5: Настройте панель

1. В правой части экрана найдите настройки панели
2. Установите:
   - **Title:** `Volume (BTC)`
   - **Panel type:** `Time series`
   - **Unit:** `short` (или `decbytes` для байтов)
3. Нажмите **Apply** (или кнопку сохранения справа вверху)

### Шаг 6: Сохраните панель

1. Нажмите **Save** в правом верхнем углу дашборда
2. Введите название дашборда (если нужно)
3. Нажмите **Save**

---

## 📊 ПАНЕЛЬ 2: OHLC Свечи

Для создания свечного графика нужно добавить 4 серии данных (Open, High, Low, Close) в одну панель.

### Шаг 1: Добавьте новую панель

1. Нажмите **Add → Visualization**

### Шаг 2: Настройте Data Source

1. Выберите **InfluxDB** в Data source

### Шаг 3: Переключитесь на Flux

1. Переключитесь на **Flux**

### Шаг 4: Добавьте первый запрос (Open)

В разделе запросов найдите кнопку **+ Add query** или **Query A**

Вставьте запрос для Open:

```
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "ohlcv")
  |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
  |> filter(fn: (r) => r["timeframe"] == "15m")
  |> filter(fn: (r) => r["_field"] == "open")
  |> aggregateWindow(every: 15m, fn: first, createEmpty: false)
```

### Шаг 5: Добавьте второй запрос (High)

1. Нажмите **+ Add query** (или **Query B**)

Вставьте запрос для High:

```
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "ohlcv")
  |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
  |> filter(fn: (r) => r["timeframe"] == "15m")
  |> filter(fn: (r) => r["_field"] == "high")
  |> aggregateWindow(every: 15m, fn: max, createEmpty: false)
```

### Шаг 6: Добавьте третий запрос (Low)

1. Нажмите **+ Add query** (или **Query C**)

Вставьте запрос для Low:

```
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "ohlcv")
  |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
  |> filter(fn: (r) => r["timeframe"] == "15m")
  |> filter(fn: (r) => r["_field"] == "low")
  |> aggregateWindow(every: 15m, fn: min, createEmpty: false)
```

### Шаг 7: Добавьте четвертый запрос (Close)

1. Нажмите **+ Add query** (или **Query D**)

Вставьте запрос для Close:

```
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "ohlcv")
  |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
  |> filter(fn: (r) => r["timeframe"] == "15m")
  |> filter(fn: (r) => r["_field"] == "close")
  |> aggregateWindow(every: 15m, fn: last, createEmpty: false)
```

### Шаг 8: Настройте панель OHLC

1. В правой части экрана найдите настройки панели
2. Установите:
   - **Title:** `BTC OHLC Candles (15m)`
   - **Panel type:** `Time series`
   - **Unit:** `currencyUSD` (для цены в долларах)
   - **Decimals:** `2`

3. **Настройте легенду:**
   - Включите **Show legend**
   - Настройте отображение: **As table** или **As list**

4. **Настройте цвета** (опционально):
   - Open: синий
   - High: зеленый
   - Low: красный
   - Close: оранжевый

5. Нажмите **Apply**

### Шаг 9: Сохраните панель

1. Нажмите **Save** в правом верхнем углу дашборда

---

## 📊 АЛЬТЕРНАТИВА: Упрощенная панель OHLC (только Close и High/Low)

Если хотите более простой график, можно создать панель только с Close и добавить High/Low как области:

### Запрос 1 (Close - основная линия):

```
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "ohlcv")
  |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
  |> filter(fn: (r) => r["timeframe"] == "15m")
  |> filter(fn: (r) => r["_field"] == "close")
  |> aggregateWindow(every: 15m, fn: last, createEmpty: false)
```

### Запрос 2 (High - верхняя граница):

```
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "ohlcv")
  |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
  |> filter(fn: (r) => r["timeframe"] == "15m")
  |> filter(fn: (r) => r["_field"] == "high")
  |> aggregateWindow(every: 15m, fn: max, createEmpty: false)
```

### Запрос 3 (Low - нижняя граница):

```
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "ohlcv")
  |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
  |> filter(fn: (r) => r["timeframe"] == "15m")
  |> filter(fn: (r) => r["_field"] == "low")
  |> aggregateWindow(every: 15m, fn: min, createEmpty: false)
```

**Настройте панель:**
- **Visualization type:** `Time series`
- **Fill opacity:** 0.1 (для High/Low как области)
- **Line width:** 2 (для Close)

---

## 🎨 НАСТРОЙКА ВНЕШНЕГО ВИДА

### Для панели Volume:

1. **Цвет:** Выберите яркий цвет (например, синий или оранжевый)
2. **Fill:** Включите заливку снизу (Fill below)
3. **Fill opacity:** 0.3-0.5

### Для панели OHLC:

1. **Line width:** 1-2 для всех линий
2. **Point size:** 0 (чтобы не показывать точки)
3. **Legend:** Показывать как таблицу для удобства

---

## 📐 РАСПОЛОЖЕНИЕ ПАНЕЛЕЙ

После добавления панелей, вы можете:

1. **Перемещать панели:** Перетаскивайте за заголовок
2. **Изменять размер:** Перетаскивайте углы панели
3. **Расположить рядом:** 
   - Price (Close) - сверху, на всю ширину
   - Volume - снизу слева (половина ширины)
   - OHLC - снизу справа (половина ширины)

---

## ✅ ПРОВЕРКА РАБОТЫ

После добавления панелей:

1. **Убедитесь, что данные отображаются:**
   - Volume должен показывать столбцы
   - OHLC должен показывать 4 линии (Open, High, Low, Close)

2. **Проверьте временной диапазон:**
   - В правом верхнем углу выберите период с данными (например, Last 12 months)

3. **Проверьте Query Inspector:**
   - Если график пустой, откройте Query Inspector
   - Проверьте, есть ли данные в таблице

---

## 🔧 ЧАСТЫЕ ПРОБЛЕМЫ

### Проблема: Volume показывает нули

**Решение:**
- Проверьте, что поле `volume` существует в данных
- Убедитесь, что временной диапазон включает данные

### Проблема: OHLC линии не видны или перекрываются

**Решение:**
- Убедитесь, что все 4 запроса добавлены
- Проверьте цвета линий (они должны отличаться)
- Увеличьте ширину линий (Line width)

### Проблема: График пустой

**Решение:**
- Проверьте временной диапазон на дашборде
- Используйте конкретные даты вместо `v.timeRangeStart`:
  ```
  |> range(start: 2024-01-31T00:00:00Z, stop: 2025-10-27T23:59:59Z)
  ```

---

## 📋 ГОТОВЫЕ ЗАПРОСЫ (Скопируйте и вставьте)

### Volume (один запрос):

```
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "ohlcv")
  |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
  |> filter(fn: (r) => r["timeframe"] == "15m")
  |> filter(fn: (r) => r["_field"] == "volume")
  |> aggregateWindow(every: 15m, fn: sum, createEmpty: false)
```

### OHLC - Open (Query A):

```
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "ohlcv")
  |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
  |> filter(fn: (r) => r["timeframe"] == "15m")
  |> filter(fn: (r) => r["_field"] == "open")
  |> aggregateWindow(every: 15m, fn: first, createEmpty: false)
```

### OHLC - High (Query B):

```
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "ohlcv")
  |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
  |> filter(fn: (r) => r["timeframe"] == "15m")
  |> filter(fn: (r) => r["_field"] == "high")
  |> aggregateWindow(every: 15m, fn: max, createEmpty: false)
```

### OHLC - Low (Query C):

```
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "ohlcv")
  |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
  |> filter(fn: (r) => r["timeframe"] == "15m")
  |> filter(fn: (r) => r["_field"] == "low")
  |> aggregateWindow(every: 15m, fn: min, createEmpty: false)
```

### OHLC - Close (Query D):

```
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "ohlcv")
  |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
  |> filter(fn: (r) => r["timeframe"] == "15m")
  |> filter(fn: (r) => r["_field"] == "close")
  |> aggregateWindow(every: 15m, fn: last, createEmpty: false)
```

---

**Готово! Теперь у вас есть панели для объема и OHLC свечей!** 🎉

