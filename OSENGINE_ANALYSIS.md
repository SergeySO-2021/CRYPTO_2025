# Подробный анализ OsEngine для проекта оптимизации индикаторов

## 1. Использование Trading Classifier в OsEngine

**Ответ: НЕТ, Trading Classifier НЕ используется в OsEngine.**

Поиск по кодовой базе OsEngine не выявил упоминаний `trading_classifier`, `trend_classifier` или `classifier`. Это отдельный Python-пакет, который у вас добавлен как submodule в `indicators/trading_classifier_iziceros/`.

---

## 2. Подробно про метод Trading Classifier

### Как работает метод:

**Trading Classifier (Segmenter)** - это алгоритм сегментации временных рядов на зоны с похожим трендом.

#### Алгоритм работы:

1. **Скользящее окно (Window):**
   - Берется окно размером `N` свечей
   - Вычисляется линейная регрессия (полином 1-й степени) для цены в этом окне
   - Получается `slope` (наклон) и `offset` (смещение)

2. **Перекрывающиеся окна:**
   - Окна сдвигаются с шагом `offset = N * overlap_ratio`
   - Например: N=20, overlap_ratio=0.33 → шаг = 6.6 ≈ 7 свечей

3. **Определение нового сегмента:**
   - Сравниваются `slope` и `offset` текущего окна с предыдущим
   - Если разница превышает пороги `alpha` (для slope) или `beta` (для offset) → создается новый сегмент
   - Метрики сравнения: `ABSOLUTE_ERROR` или `RELATIVE_ABSOLUTE_ERROR`

4. **Результат:**
   - Список сегментов с характеристиками:
     - `start`, `stop` - границы сегмента
     - `slope` - наклон тренда
     - `offset` - смещение
     - `std` - стандартное отклонение (волатильность)
     - `span` - размах значений (нормализованный)
     - `slopes_std`, `offsets_std` - стабильность тренда

### Работа со свечными данными:

**Да, работает, но только с одним столбцом!**

```python
# Trading Classifier работает ТОЛЬКО с одним столбцом:
seg = Segmenter(df=df, column="close")  # или "Adj Close"
# Он НЕ использует open, high, low, volume напрямую!
```

**Ограничения:**
- Использует только одну цену (обычно `close`)
- Не учитывает OHLC напрямую
- Не учитывает объемы

**Решение:** Можно расширить, используя результаты сегментации для анализа других метрик.

---

## 3. Подбор параметров для разных таймфреймов

### Основные параметры:

```python
Config(
    N=60,                    # Размер окна (количество свечей)
    overlap_ratio=0.33,      # Перекрытие окон (0.33 = 33%)
    alpha=2,                 # Порог для slope (относительная ошибка)
    beta=2,                  # Порог для offset (относительная ошибка)
    metrics_for_alpha=Metrics.RELATIVE_ABSOLUTE_ERROR,
    metrics_for_beta=Metrics.RELATIVE_ABSOLUTE_ERROR
)
```

### Рекомендации по таймфреймам:

#### 1 минута:
```python
Config(N=20-30, overlap_ratio=0.33, alpha=2, beta=2)
# Малые окна, быстрая реакция на изменения
```

#### 5 минут:
```python
Config(N=40-60, overlap_ratio=0.33, alpha=2, beta=2)
# Средние окна
```

#### 15 минут:
```python
Config(N=60-80, overlap_ratio=0.33, alpha=2, beta=2)
# Ваш текущий таймфрейм - оптимально N=60-80
```

#### 1 час:
```python
Config(N=80-120, overlap_ratio=0.33, alpha=2, beta=2)
# Большие окна для долгосрочных трендов
```

#### 1 день:
```python
Config(N=120-200, overlap_ratio=0.33, alpha=2, beta=2)
# Очень большие окна
```

### Метод подбора:

1. **Начните с дефолтных значений:**
   ```python
   seg = Segmenter(df=df, column="close", n=60)
   ```

2. **Проверьте количество сегментов:**
   ```python
   seg.calculate_segments()
   print(f"Количество сегментов: {len(seg.segments)}")
   # Хорошо: 10-30 сегментов на год данных
   # Слишком много (>50): увеличить N или alpha/beta
   # Слишком мало (<5): уменьшить N или alpha/beta
   ```

3. **Визуализируйте:**
   ```python
   seg.plot_segments()
   # Проверьте, что сегменты логичны
   ```

4. **Оптимизируйте:**
   - **N слишком большое** → слишком мало сегментов → уменьшить N
   - **N слишком маленькое** → слишком много сегментов → увеличить N
   - **alpha/beta слишком маленькие** → слишком много сегментов → увеличить
   - **alpha/beta слишком большие** → слишком мало сегментов → уменьшить

### Формула для расчета N:

```
N ≈ (количество свечей в торговом дне) / 4
```

Примеры:
- 1m: 1440 свечей/день → N ≈ 360 (но лучше 20-30 для быстрой реакции)
- 15m: 96 свечей/день → N ≈ 24 (но лучше 60-80 для стабильности)
- 1h: 24 свечи/день → N ≈ 6 (но лучше 80-120 для трендов)

---

## 4. Вариант 1: Гибридный подход - ДЕТАЛЬНО

### Этап 1: Базовая сегментация (Trading Classifier)

```python
from trend_classifier import Segmenter, Config

# 1. Сегментация по тренду
config = Config(N=60, overlap_ratio=0.33, alpha=2, beta=2)
seg = Segmenter(df=df, column="close", config=config)
seg.calculate_segments()

# Получаем базовые сегменты с трендом
for segment in seg.segments:
    print(f"Сегмент {segment.start}-{segment.stop}: slope={segment.slope:.4f}")
```

### Этап 2: Расширение классификации зон

#### 2.1. Добавление волатильности:

```python
import pandas as pd
import numpy as np

def classify_volatility(df, segment):
    """Классифицирует волатильность сегмента"""
    segment_data = df.iloc[segment.start:segment.stop]
    
    # Метод 1: ATR (Average True Range)
    high = segment_data['high']
    low = segment_data['low']
    close = segment_data['close']
    close_prev = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close_prev)
    tr3 = abs(low - close_prev)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.mean()
    
    # Метод 2: Стандартное отклонение (уже есть в segment.std)
    std = segment.std
    
    # Метод 3: Размах (high - low)
    price_range = (high.max() - low.min()) / close.mean()
    
    # Классификация
    atr_median = df['high'].rolling(100).std().median()  # Базовый уровень
    
    if atr > atr_median * 1.5:
        return "high_vol"
    elif atr < atr_median * 0.5:
        return "low_vol"
    else:
        return "medium_vol"
```

#### 2.2. Добавление объемов:

```python
def classify_volume(df, segment):
    """Классифицирует объемы сегмента"""
    segment_data = df.iloc[segment.start:segment.stop]
    
    # Средний объем в сегменте
    avg_volume = segment_data['volume'].mean()
    
    # Медианный объем за последние N периодов
    volume_median = df['volume'].rolling(100).median().iloc[-1]
    
    # Отношение buy/sell объемов (если есть)
    if 'trades_buy_volume' in segment_data.columns:
        buy_vol = segment_data['trades_buy_volume'].sum()
        sell_vol = segment_data['trades_sell_volume'].sum()
        buy_sell_ratio = buy_vol / (sell_vol + 0.001)
        
        if buy_sell_ratio > 1.2:
            volume_pressure = "buy_pressure"
        elif buy_sell_ratio < 0.8:
            volume_pressure = "sell_pressure"
        else:
            volume_pressure = "balanced"
    else:
        volume_pressure = "unknown"
    
    # Классификация
    if avg_volume > volume_median * 1.5:
        return f"high_volume_{volume_pressure}"
    elif avg_volume < volume_median * 0.5:
        return f"low_volume_{volume_pressure}"
    else:
        return f"medium_volume_{volume_pressure}"
```

#### 2.3. Определение рыночного режима:

```python
def classify_market_regime(df, segment):
    """Определяет рыночный режим: trending/ranging/breakout"""
    segment_data = df.iloc[segment.start:segment.stop]
    
    # 1. Тренд (из Trading Classifier)
    slope = segment.slope
    slope_std = segment.slopes_std
    
    # 2. Волатильность
    price_std = segment_data['close'].std()
    price_mean = segment_data['close'].mean()
    cv = price_std / price_mean  # Коэффициент вариации
    
    # 3. ADX (Average Directional Index) - если есть
    # Или упрощенный расчет:
    high = segment_data['high']
    low = segment_data['low']
    close = segment_data['close']
    
    # Простой ADX-подобный индикатор
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    
    plus_di = 100 * (plus_dm.rolling(14).mean() / tr.rolling(14).mean())
    minus_di = 100 * (minus_dm.rolling(14).mean() / tr.rolling(14).mean())
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
    adx = dx.rolling(14).mean().iloc[-1]
    
    # 4. Breakout detection
    price_range = high.max() - low.min()
    price_position = (close.iloc[-1] - low.min()) / (price_range + 0.001)
    
    # Классификация
    if abs(slope) > 0.001 and slope_std < abs(slope) * 0.5:  # Сильный стабильный тренд
        if slope > 0:
            return "trending_up"
        else:
            return "trending_down"
    elif adx > 25:  # Сильный тренд по ADX
        if slope > 0:
            return "trending_up"
        else:
            return "trending_down"
    elif cv < 0.01 and abs(slope) < 0.0001:  # Низкая волатильность, нет тренда
        return "ranging"
    elif price_position > 0.9 or price_position < 0.1:  # Цена у границ
        return "breakout"
    else:
        return "ranging"
```

### Этап 3: Создание матрицы зон

```python
def create_market_zones(df, seg):
    """Создает полную матрицу рыночных зон"""
    zones = []
    
    for segment in seg.segments:
        # Базовые характеристики
        trend_slope = segment.slope
        trend_direction = "up" if trend_slope > 0 else "down" if trend_slope < 0 else "flat"
        
        # Дополнительные классификации
        volatility = classify_volatility(df, segment)
        volume = classify_volume(df, segment)
        regime = classify_market_regime(df, segment)
        
        # Создание уникальной зоны
        zone = {
            "start": segment.start,
            "stop": segment.stop,
            "trend_slope": trend_slope,
            "trend_direction": trend_direction,
            "volatility": volatility,
            "volume": volume,
            "regime": regime,
            "zone_id": f"{trend_direction}_{volatility}_{regime}",
            "std": segment.std,
            "span": segment.span
        }
        
        zones.append(zone)
    
    return pd.DataFrame(zones)
```

### Этап 4: Интеграция с OsOptimizer

```python
# Для каждой зоны создаем отдельную оптимизацию
zone_results = {}

for zone_id in zones['zone_id'].unique():
    zone_data = zones[zones['zone_id'] == zone_id]
    
    # Фильтруем историю для этой зоны
    zone_periods = []
    for _, zone in zone_data.iterrows():
        zone_periods.append((zone['start'], zone['stop']))
    
    # Запускаем оптимизацию только для периодов этой зоны
    optimal_params = optimize_for_zone(zone_periods, indicator_params)
    zone_results[zone_id] = optimal_params

# Сохраняем результаты
import json
with open('zone_optimal_params.json', 'w') as f:
    json.dump(zone_results, f)
```

### Этап 5: Применение в реальной торговле

```python
def get_current_zone_and_params(df_current, zone_classifier, zone_results):
    """Определяет текущую зону и возвращает оптимальные параметры"""
    
    # 1. Определяем текущую зону
    current_segment = zone_classifier.get_current_segment(df_current)
    current_zone = create_zone_from_segment(current_segment, df_current)
    
    # 2. Находим похожую зону в истории
    best_match_zone = find_best_match_zone(current_zone, zone_results.keys())
    
    # 3. Возвращаем оптимальные параметры
    return zone_results[best_match_zone]
```

---

## 5. Фильтрация по метрикам в OsOptimizer

### Доступные метрики:

Из кода `OptimizerReport.cs` видно следующие метрики:

1. **TotalProfit** - общая прибыль
2. **PositionsCount** - количество позиций
3. **MaxDrawDawn** - максимальная просадка
4. **AverageProfit** - средняя прибыль
5. **AverageProfitPercent** - средняя прибыль в процентах
6. **ProfitFactor** - профит-фактор
7. **PayOffRatio** - соотношение прибыли к убыткам
8. **Recovery** - восстановление после просадки
9. **SharpRatio** - коэффициент Шарпа

### Как работает фильтрация:

```csharp
// Из OptimizerMaster.cs
_filterProfitValue = 10;              // Минимальная прибыль
_filterProfitIsOn = false;            // Включить фильтр?
_filterMaxDrawDownValue = -10;        // Максимальная просадка
_filterMaxDrawDownIsOn = false;       // Включить фильтр?
_filterMiddleProfitValue = 0.001m;    // Средняя прибыль
_filterMiddleProfitIsOn = false;       // Включить фильтр?
_filterProfitFactorValue = 1;         // Минимальный профит-фактор
_filterProfitFactorIsOn = false;      // Включить фильтр?
```

### Пример настройки фильтров:

```python
# В OsOptimizer можно настроить:
filters = {
    "profit_min": 1000,           # Минимальная прибыль $1000
    "max_drawdown_max": -500,     # Максимальная просадка не более $500
    "profit_factor_min": 1.5,     # Профит-фактор не менее 1.5
    "average_profit_min": 50,     # Средняя прибыль не менее $50
    "positions_count_min": 10     # Минимум 10 сделок
}

# Только результаты, прошедшие все фильтры, попадут в финальный отчет
```

### Логика фильтрации:

```python
def filter_optimization_results(reports, filters):
    """Фильтрует результаты оптимизации"""
    filtered = []
    
    for report in reports:
        # Проверка всех фильтров
        if filters.get('profit_min') and report.TotalProfit < filters['profit_min']:
            continue
        if filters.get('max_drawdown_max') and report.MaxDrawDawn < filters['max_drawdown_max']:
            continue
        if filters.get('profit_factor_min') and report.ProfitFactor < filters['profit_factor_min']:
            continue
        if filters.get('average_profit_min') and report.AverageProfit < filters['average_profit_min']:
            continue
        if filters.get('positions_count_min') and report.PositionsCount < filters['positions_count_min']:
            continue
        
        # Все фильтры пройдены
        filtered.append(report)
    
    return filtered
```

---

## 6. VolatilityStageRotation - ДЕТАЛЬНО

### Как определяются стадии волатильности:

Из кода `BollingerTrendVolatilityStagesFilter.cs` видно использование индикатора `VolatilityStagesAW`:

```csharp
// Создание индикатора VolatilityStages
_volatilityStages = IndicatorsFactory.CreateIndicatorByName("VolatilityStagesAW", ...);
_volatilityStages.ParametersDigit[0].Value = _volatilitySlowSmaLength.ValueInt;  // 25
_volatilityStages.ParametersDigit[1].Value = _volatilityFastSmaLength.ValueInt;  // 7
_volatilityStages.ParametersDigit[2].Value = _volatilityChannelDeviation.ValueDecimal; // 0.5
```

**Логика определения стадий (предположительно):**

1. **Вычисляется волатильность:**
   - Быстрая SMA волатильности (7 периодов)
   - Медленная SMA волатильности (25 периодов)

2. **Определение стадий:**
   - **Стадия 1:** Низкая волатильность (быстрая < медленной - deviation)
   - **Стадия 2:** Средняя волатильность (быстрая ≈ медленной)
   - **Стадия 3:** Высокая волатильность (быстрая > медленной + deviation)
   - **Стадия 4:** Экстремальная волатильность

3. **Использование в фильтрации:**
   ```csharp
   decimal stage = _volatilityStages.DataSeries[0].Values[...];
   if (stage != _volatilityStageToTrade.ValueString.ToDecimal()) {
       return; // Не торгуем в этой стадии
   }
   ```

### Как фильтруются сигналы:

```csharp
// Из BollingerTrendVolatilityStagesFilter.cs

// 1. Основной сигнал (Bollinger)
if (lastPrice > lastPcUp && _regime.ValueString != "OnlyShort") {
    
    // 2. Фильтр по волатильности (ОПЦИОНАЛЬНО)
    if (_volatilityFilterIsOn.ValueBool == true) {
        decimal stage = _volatilityStages.DataSeries[0].Values[...];
        
        // Торгуем ТОЛЬКО если стадия совпадает
        if (stage != _volatilityStageToTrade.ValueString.ToDecimal()) {
            return; // Пропускаем сигнал
        }
    }
    
    // 3. Сигнал прошел фильтры - открываем позицию
    _tab.BuyAtMarket(GetVolume(_tab));
}
```

**Логика фильтрации:**
1. Получаем основной сигнал (например, пробой Bollinger)
2. Проверяем текущую стадию волатильности
3. Если стадия не подходит → пропускаем сигнал
4. Если стадия подходит → открываем позицию

---

## 7. Анализ лимитных ордеров (стаканов) в OsEngine

**Ответ: ДА, есть обширный анализ стаканов!**

### Где находится:

1. **Класс MarketDepth** - `OsEngine/project/OsEngine/Entity/MarketDepth.cs`
   - Структура стакана
   - Методы работы со стаканом

2. **Роботы для анализа стакана:**
   - `HighFrequencyTrader.cs` - анализ стакана для HFT
   - `MarketDepthScreener.cs` - скринер на основе стакана
   - `Fisher.cs` - торговля на основе стакана

3. **События обновления стакана:**
   ```csharp
   _tab.MarketDepthUpdateEvent += _tab_MarketDepthUpdateEvent;
   ```

### Примеры использования:

#### Пример 1: HighFrequencyTrader
```csharp
// Анализ уровней стакана
for (int i = 0; i < marketDepth.Bids.Count && i < _maxLevelsInMarketDepth.ValueInt; i++)
{
    if (marketDepth.Bids[i].Bid > lastVolume)
    {
        buyPrice = marketDepth.Bids[i].Price.ToDecimal() + _tab.Security.PriceStep;
        lastVolume = Convert.ToInt32(marketDepth.Bids[i].Bid);
    }
}
```

#### Пример 2: MarketDepthScreener
```csharp
private void LogicEntry(MarketDepth marketDepth, BotTabSimple tab)
{
    // Анализ стакана
    // Поиск аномалий, дисбалансов
}
```

### Структура MarketDepth:

```csharp
public class MarketDepth
{
    public DateTime Time;                    // Время обновления
    public List<MarketDepthLevel> Asks;     // Продажи (лучшая цена с индексом 0)
    public List<MarketDepthLevel> Bids;     // Покупки (лучшая цена с индексом 0)
    public string SecurityNameCode;         // Инструмент
    public decimal AskSummVolume;           // Суммарный объем продаж
    public decimal BidSummVolume;           // Суммарный объем покупок
}

public class MarketDepthLevel
{
    public double Price;    // Цена уровня
    public double Ask;      // Объем продаж на уровне
    public double Bid;      // Объем покупок на уровне
}
```

### Что можно анализировать:

1. **Дисбаланс стакана:**
   - Соотношение объемов покупок/продаж
   - Концентрация объемов на уровнях

2. **Глубина стакана:**
   - Количество уровней
   - Объемы на каждом уровне

3. **Динамика стакана:**
   - Изменение объемов во времени
   - Скорость обновления

4. **Spread анализ:**
   - Разница между лучшими ценами покупки/продажи
   - Ширина стакана

---

## Итоговые рекомендации:

1. **Trading Classifier** - использовать как основу для сегментации
2. **Расширить классификацию** - добавить волатильность, объемы, режимы
3. **OsOptimizer** - оптимизировать параметры для каждой зоны отдельно
4. **VolatilityStageRotation** - использовать логику фильтрации по стадиям
5. **MarketDepth** - анализировать стакан для дополнительных сигналов

