# 🔧 УПРОЩЕННЫЕ FLUX ЗАПРОСЫ (если pivot не работает)

## 📊 Альтернативные запросы для объемов в USD

### Вариант 1: Использование join (если pivot не работает)

#### Объемы покупок в USD:
```flux
buy_vol = from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "trades_buy_volume")
  |> aggregateWindow(every: 15m, fn: sum, createEmpty: false)

price = from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "close")
  |> aggregateWindow(every: 15m, fn: last, createEmpty: false)

join(tables: {buy: buy_vol, price: price}, on: ["_time"])
  |> map(fn: (r) => ({ r with _value: r.buy__value * r.price__value }))
  |> drop(columns: ["buy__value", "price__value", "buy__field", "price__field"])
```

#### Объемы продаж в USD:
```flux
sell_vol = from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "trades_sell_volume")
  |> aggregateWindow(every: 15m, fn: sum, createEmpty: false)

price = from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "close")
  |> aggregateWindow(every: 15m, fn: last, createEmpty: false)

join(tables: {sell: sell_vol, price: price}, on: ["_time"])
  |> map(fn: (r) => ({ r with _value: r.sell__value * r.price__value }))
  |> drop(columns: ["sell__value", "price__value", "sell__field", "price__field"])
```

---

## 📊 Вариант 2: Простые запросы (если join не работает)

### Просто объемы покупок в BTC (потом умножим на цену в Grafana):

#### Объемы покупок (BTC):
```flux
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "trades_buy_volume")
  |> aggregateWindow(every: 15m, fn: sum, createEmpty: false)
```

**Затем в Grafana:**
- Создайте еще один запрос для цены
- Используйте Transformations → Multiply для умножения

---

## 📊 Вариант 3: Использование quote_volume (если есть)

Если в данных есть поле `quote_volume` (объем в USDT), можно использовать его напрямую:

```flux
from(bucket: "binance_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "quote_volume")
  |> aggregateWindow(every: 15m, fn: sum, createEmpty: false)
```

Но это общий объем, не разделенный на покупки/продажи.

---

## 🔍 ДИАГНОСТИКА ПРОБЛЕМ

### Проверка доступных полей:
```flux
from(bucket: "binance_data")
  |> range(start: -365d)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> group()
  |> distinct(column: "_field")
```

Этот запрос покажет все доступные поля в measurement `btc_combined`.

### Проверка данных:
```flux
from(bucket: "binance_data")
  |> range(start: -365d)
  |> filter(fn: (r) => r["_measurement"] == "btc_combined")
  |> filter(fn: (r) => r["_field"] == "close")
  |> limit(n: 10)
```

Этот запрос покажет последние 10 записей цены.

---

## 💡 РЕКОМЕНДАЦИИ

1. **Начните с простых запросов** - сначала проверьте, что данные вообще есть
2. **Используйте Query Inspector** в Grafana для отладки запросов
3. **Проверяйте временной диапазон** - ваши данные с 2024-01-01
4. **Используйте `-365d` вместо `v.timeRangeStart`** для тестирования

