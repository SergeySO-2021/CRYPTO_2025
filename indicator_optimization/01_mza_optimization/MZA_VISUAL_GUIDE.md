# 📊 **MZA OPTIMIZED - ВИЗУАЛЬНАЯ СХЕМА РАБОТЫ**

## 🎯 **АЛГОРИТМ РАБОТЫ MZA**

```
📈 ВХОДНЫЕ ДАННЫЕ
├── Цена (OHLC)
├── Объем (Volume)
└── Время (Timestamps)

    ↓

🔍 АНАЛИЗ ПО ТРЕМ КАТЕГОРИЯМ

📊 TREND STRENGTH (49% веса)
├── ADX (12 периодов, порог 28)
├── MA Slope (15/40 периоды)
└── Ichimoku Diff

⚡ MOMENTUM (21% веса)
├── RSI (20 периодов)
├── Stochastic (11 периодов)
└── MACD (14/22/9)

🎯 PRICE ACTION (25% веса)
├── HH/LL Range (27 периодов)
├── HA Doji (9 периодов)
└── Candle Range (11 периодов)

    ↓

📊 MARKET ACTIVITY ANALYSIS
├── Bollinger Bands (22 периода, 3.0 множитель)
├── ATR (13 периодов)
├── Keltner Channels (23 периода, 1.8 множитель)
└── Volume MA (25 периодов)

    ↓

⚖️ АДАПТИВНЫЕ ВЕСА
├── High Activity: Trend 50%, Momentum 35%, Price Action 15%
├── Medium Activity: Trend 49%, Momentum 21%, Price Action 25%
└── Low Activity: Trend 25%, Momentum 20%, Price Action 55%

    ↓

🧮 ФИНАЛЬНЫЙ РАСЧЕТ
Net Score = (Trend × Trend Weight) + (Momentum × Momentum Weight) + (Price Action × Price Action Weight)

    ↓

🎯 ОПРЕДЕЛЕНИЕ ЗОНЫ
├── Net Score ≥ +2 → BULLISH (🟢)
├── Net Score ≤ -2 → BEARISH (🔴)
└── -2 < Net Score < +2 → SIDEWAYS (⚪)

    ↓

🛡️ ГИСТЕРЕЗИС (защита от ложных сигналов)
├── Проверка стабильности сигнала
├── Подтверждение в течение 2+ баров
└── Финальная зона

    ↓

📊 ВЫВОД РЕЗУЛЬТАТА
├── Осциллятор с цветовой индикацией
├── Dashboard с детальной информацией
└── Торговые рекомендации
```

---

## 🎨 **ВИЗУАЛЬНЫЕ СИГНАЛЫ**

### 📈 **Осциллятор (нижняя панель)**
```
    +5 ┤
       │  🟢 BULLISH ZONE
    +2 ├─────────────────────────────
       │
     0 ├─────────────────────────────
       │
   -2 ├─────────────────────────────
       │  🔴 BEARISH ZONE
   -5 ┤
```

### 🎯 **Цветовая схема**
- **🟢 Зеленый** = BULLISH (покупки)
- **🔴 Красный** = BEARISH (продажи)
- **⚪ Серый** = SIDEWAYS (ожидание)

---

## 📊 **DASHBOARD СТРУКТУРА**

```
┌─────────────────────────────────────────────────────────┐
│ INDICATOR    │ VALUE │ REGIME │ SCORE │ COMMENTS         │
├─────────────────────────────────────────────────────────┤
│ TREND STRENGTH│ BULLISH│ BULLISH│  +3  │ Highly Bullish  │
│ ADX          │ 32.5  │ BULL   │  +1  │ Strong trend     │
│ MA Slope     │ 125.3 │ BULL   │  +1  │ Fast above slow │
│ Ichimoku Diff│ 45.2  │ BULL   │  +1  │ Senkou A > B    │
├─────────────────────────────────────────────────────────┤
│ MOMENTUM     │ BULLISH│ BULLISH│  +2  │ Strong Momentum │
│ RSI          │ 68.4  │ BULL   │  +1  │ Above threshold  │
│ Stoch %K     │ 75.2  │ BULL   │  +1  │ Above threshold  │
│ MACD Hist    │ 12.5  │ BULL   │  +1  │ Positive hist    │
├─────────────────────────────────────────────────────────┤
│ PRICE ACTION │ BULLISH│ BULLISH│  +2  │ Bullish Action  │
│ HH/LL Flat   │ 1250  │ BULL   │  +1  │ Near highs       │
│ HA Doji      │ 45.3  │ BULL   │  +1  │ Bullish HA       │
│ Candle Range │ 89.2  │ BULL   │  +1  │ Large range      │
├─────────────────────────────────────────────────────────┤
│ MARKET ACTIVITY STATE │ High │ Volatile Market          │
├─────────────────────────────────────────────────────────┤
│ CATEGORY WEIGHTS                                      │
│ Trend: 50% │ Momentum: 35% │ Price Action: 15%       │
├─────────────────────────────────────────────────────────┤
│ OVERALL SCORE: +4.2 │ Zone: BULLISH │ Trend Dominant   │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 **ТОРГОВЫЕ СИГНАЛЫ**

### 🟢 **BULLISH SIGNALS**
```
Net Score: +2 to +5
Zone: BULLISH
Background: Green
Action: BUY/LONG
```

### 🔴 **BEARISH SIGNALS**
```
Net Score: -2 to -5
Zone: BEARISH
Background: Red
Action: SELL/SHORT
```

### ⚪ **SIDEWAYS SIGNALS**
```
Net Score: -2 to +2
Zone: SIDEWAYS
Background: Gray
Action: WAIT/HOLD
```

---

## ⚙️ **НАСТРОЙКИ ПО ТАЙМФРЕЙМАМ**

### 🏃 **1-5 минут (Скальпинг)**
```
ADX Threshold: 25 (более чувствительный)
useSmoothing: True (сглаживание)
Dashboard Size: Tiny
```

### 📊 **15 минут (Оптимизировано)**
```
Все параметры по умолчанию
Economic Value: 0.592989
```

### 📅 **1-4 часа (Свинг)**
```
ADX Threshold: 30 (менее чувствительный)
useSmoothing: True (сглаживание)
Dashboard Size: Normal
```

### 📆 **1 день (Позиционная)**
```
ADX Threshold: 35 (очень строгий)
useSmoothing: True (максимальное сглаживание)
Dashboard Size: Normal
```

---

## 🚨 **ВАЖНЫЕ ПРЕДУПРЕЖДЕНИЯ**

### ⚠️ **НЕ ТОРГУЙТЕ:**
- Только по MZA без подтверждения
- Против сильного тренда
- В новостное время без анализа
- Без стоп-лоссов

### ✅ **ВСЕГДА ИСПОЛЬЗУЙТЕ:**
- Стоп-лоссы
- Управление рисками
- Подтверждение другими индикаторами
- Анализ объема

---

## 📈 **ПРИМЕРЫ ТОРГОВЛИ**

### 📊 **Пример 1: Сильный бычий сигнал**
```
Net Score: +4.2
Zone: BULLISH
Trend Dominant
Action: BUY
Stop Loss: За предыдущий минимум
Take Profit: При смене зоны на SIDEWAYS
```

### 📊 **Пример 2: Слабый медвежий сигнал**
```
Net Score: -2.1
Zone: BEARISH
Momentum Dominant
Action: SELL (осторожно)
Stop Loss: За предыдущий максимум
Take Profit: При Net Score > -1
```

### 📊 **Пример 3: Боковая зона**
```
Net Score: +0.8
Zone: SIDEWAYS
Price Action Dominant
Action: WAIT
Strategy: Готовиться к пробою
```

---

**🎯 Помните: MZA - это инструмент анализа, а не гарантия прибыли!**
