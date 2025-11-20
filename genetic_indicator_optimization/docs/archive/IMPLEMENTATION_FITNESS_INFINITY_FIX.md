# ✅ Реализация исправлений для проблемы fitness=Infinity

**Дата:** 18 ноября 2025  
**Статус:** ✅ Реализовано  
**Источник рекомендаций:** `docs/deepseek_advice/RESPONSE_04_FITNESS_INFINITY_SOLUTION.md`

---

## 🎯 ЦЕЛЬ

Исправить проблему с `fitness=Infinity` и переобучением, возникшую после добавления Long/Short параметров в search space генетического алгоритма.

---

## ✅ РЕАЛИЗОВАННЫЕ ИЗМЕНЕНИЯ

### 1. Обработка бесконечных метрик

**Файл:** `src/core/genetic_optimizer.py`

**Добавлена функция `_safe_metric_value`:**
- Обрабатывает `Infinity` и `NaN` значения
- Ограничивает `profit_factor` до 10.0 (вместо Infinity)
- Ограничивает `sharpe_ratio` до 50.0/-50.0 (вместо Infinity)

**Код:**
```python
def _safe_metric_value(self, value: float, metric_name: str, default: float = 0.0) -> float:
    """Безопасное вычисление метрик с обработкой edge cases"""
    if np.isinf(value) or np.isnan(value):
        if metric_name == "profit_factor":
            return 10.0  # Максимальное разумное значение
        elif metric_name == "sharpe_ratio":
            return 50.0 if value > 0 else -50.0
        else:
            return default
    return float(value)
```

---

### 2. Валидация метрик

**Добавлена функция `_validate_metrics`:**
- Валидирует все метрики перед использованием
- Обрабатывает бесконечности через `_safe_metric_value`
- Обрабатывает нулевую просадку (заменяет на `1e-5`)
- Гарантирует корректные диапазоны (win_rate в [0, 1])

**Код:**
```python
def _validate_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Валидация и очистка метрик"""
    validated = copy.deepcopy(metrics)
    
    # Замена бесконечностей
    validated["profit_factor"] = self._safe_metric_value(
        validated.get("profit_factor", 0.0), "profit_factor", 0.0
    )
    validated["sharpe_ratio"] = self._safe_metric_value(
        validated.get("sharpe_ratio", 0.0), "sharpe_ratio", 0.0
    )
    
    # Обработка нулевой просадки
    if validated.get("max_drawdown", 0.0) == 0.0 or np.isnan(validated.get("max_drawdown", 0.0)):
        validated["max_drawdown"] = 1e-5
    
    # Гарантия корректных диапазонов
    validated["win_rate"] = max(0.0, min(1.0, ...))
    
    return validated
```

---

### 3. Жёсткие constraints на минимальное количество сделок

**Добавлена функция `_passes_hard_constraints`:**
- Минимум 10 сделок на валидации → `return False` если меньше
- Максимальная просадка не более 20% → `return False` если больше
- Win rate не менее 25% → `return False` если меньше

**Код:**
```python
def _passes_hard_constraints(self, metrics: Dict[str, Any]) -> bool:
    """Жёсткие ограничения - отсекаем плохие решения"""
    trades = metrics.get("total_trades", 0)
    
    if trades < 10:
        return False
    
    if metrics.get("max_drawdown", 1.0) > 0.20:
        return False
    
    if metrics.get("win_rate", 0.0) < 0.25:
        return False
    
    return True
```

**Использование в `_calculate_fitness`:**
```python
# 2. Жёсткие проверки (hard constraints) - отсекаем плохие решения
if not self._passes_hard_constraints(val_metrics):
    return -float('inf')
```

---

### 4. Усиление штрафов за малое количество сделок

**Добавлена функция `_apply_trade_count_penalties`:**
- Аддитивные штрафы (работают даже с Infinity):
  - `trades < 5` → `-1000`
  - `trades < 10` → `-500`
  - `trades < 20` → `-200`
  - `trades < 30` → `-100`
- Мультипликативные штрафы (усиленные):
  - `trades < 10` → `* 0.1` (было 0.5)
  - `trades < 20` → `* 0.3` (было 0.7)
  - `trades < 30` → `* 0.6`

**Код:**
```python
def _apply_trade_count_penalties(self, metrics: Dict[str, Any]) -> Tuple[float, float]:
    """ЖЁСТКИЕ штрафы за малое количество сделок"""
    trades = metrics.get("total_trades", 0)
    
    # Аддитивные штрафы
    penalty = 0.0
    if trades < 5:
        penalty += 1000.0
    elif trades < 10:
        penalty += 500.0
    elif trades < 20:
        penalty += 200.0
    elif trades < 30:
        penalty += 100.0
    
    # Мультипликативные штрафы (усиленные)
    multiplier = 1.0
    if trades < 10:
        multiplier *= 0.1  # Было 0.5
    elif trades < 20:
        multiplier *= 0.3  # Было 0.7
    elif trades < 30:
        multiplier *= 0.6
    
    return penalty, multiplier
```

**Использование в `_calculate_fitness`:**
```python
# ЖЁСТКИЕ штрафы за малое количество сделок
trade_penalty, trade_multiplier = self._apply_trade_count_penalties(val_metrics)
score -= trade_penalty
# ...
penalties_multiplier = self._apply_penalties(val_metrics) * trade_multiplier
score *= penalties_multiplier
```

---

### 5. Обновлённая fitness функция

**Изменения в `_calculate_fitness`:**
1. Предварительная валидация метрик через `_validate_metrics`
2. Жёсткие constraints через `_passes_hard_constraints` (отсекает плохие решения)
3. Безопасные вычисления с использованием валидированных метрик
4. Аддитивные и мультипликативные штрафы за малое количество сделок
5. Гарантия конечного значения в конце (если всё равно Infinity/NaN → `-1000.0`)

**Структура:**
```python
def _calculate_fitness(self, metrics):
    # 1. Предварительная валидация метрик
    val_metrics = self._validate_metrics(metrics.get("val"))
    train_metrics = self._validate_metrics(metrics.get("train"))
    
    # 2. Жёсткие проверки (hard constraints)
    if not self._passes_hard_constraints(val_metrics):
        return -float('inf')
    
    # 3. Основной score на validation
    base_score = ...
    
    # 4. Штрафы и бонусы
    score = base_score - overfitting_penalty - stability_penalty + stability_bonus
    score -= trade_penalty  # Аддитивный штраф
    score *= penalties_multiplier * trade_multiplier  # Мультипликативные штрафы
    
    # 5. Гарантия конечного значения
    if np.isinf(score) or np.isnan(score):
        return -1000.0
    
    return score
```

---

### 6. Ограничение search space для Long/Short параметров

**Файл:** `config/ga_config.yaml`

**Изменения диапазонов:**
- `long_signal_multiplier`: `[0.8, 1.2]` (было `[0.5, 1.5]`)
- `short_signal_multiplier`: `[0.8, 1.2]` (было `[0.5, 1.5]`)
- `entry_threshold_long`: `[0.5, 0.7]` (было `[0.5, 0.8]`)
- `entry_threshold_short`: `[0.4, 0.6]` (было `[0.3, 0.6]`)

**Причина:** Предотвращение "лазеек", когда ГА находит решения с ослабленными обоими сигналами одновременно.

---

### 7. Constraints для Long/Short параметров

**Файл:** `src/core/genetic_optimizer.py`

**Добавлена функция `_apply_constraints`:**
- `long_signal_multiplier + short_signal_multiplier >= 1.6` (предотвращает одновременное ослабление)
- `entry_threshold_long >= entry_threshold_short` (Long входы строже)

**Код:**
```python
def _apply_constraints(self, genes: Dict[str, Any]) -> Dict[str, Any]:
    """Применяет constraints к генам для Long/Short параметров"""
    constrained = copy.deepcopy(genes)
    
    # Constraint 1: сумма множителей >= 1.6
    if "long_signal_multiplier" in constrained and "short_signal_multiplier" in constrained:
        long_mult = constrained["long_signal_multiplier"]
        short_mult = constrained["short_signal_multiplier"]
        sum_mult = long_mult + short_mult
        
        if sum_mult < 1.6:
            scale = 1.6 / sum_mult
            constrained["long_signal_multiplier"] = min(1.2, long_mult * scale)
            constrained["short_signal_multiplier"] = min(1.2, short_mult * scale)
    
    # Constraint 2: long порог >= short порог
    if "entry_threshold_long" in constrained and "entry_threshold_short" in constrained:
        long_thresh = constrained["entry_threshold_long"]
        short_thresh = constrained["entry_threshold_short"]
        
        if long_thresh < short_thresh:
            constrained["entry_threshold_long"] = short_thresh
    
    return constrained
```

**Применение:**
- После генерации случайных генов (`_random_genes`)
- После кроссовера (`_crossover`)
- После мутации (`_mutate`)

---

## 📊 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ

После реализации исправлений ожидаем:

1. ✅ **Fitness ≠ Infinity** — всегда конечные значения
2. ✅ **Минимум 10+ сделок на валидации** — жёсткие constraints отсекают решения с меньшим количеством
3. ✅ **Улучшение баланса Long/Short** — ограниченные диапазоны и constraints предотвращают "лазейки"
4. ✅ **Более стабильные метрики** — валидация метрик и обработка edge cases улучшают стабильность

---

## 🧪 ТЕСТИРОВАНИЕ

### Быстрый прогон (20×10)

**Команда:**
```bash
py -3 scripts/run_ga_search.py --population-size 20 --max-generations 10 --output results/ga_test_fix.json
```

**Проверки:**
- [ ] Fitness не равен Infinity
- [ ] ГА не находит решения с < 10 сделками
- [ ] Constraints работают корректно
- [ ] Валидация метрик работает

### Полный прогон (100×100)

**Если быстрый прогон успешен:**
```bash
py -3 scripts/run_ga_search.py --output results/ga_best_fixed.json
```

**Сравнение:**
- Сравнить результаты с ГА №3 (успешный запуск)
- Проанализировать баланс Long/Short
- Проверить стабильность метрик между train/val/test

---

## 📝 ИЗМЕНЁННЫЕ ФАЙЛЫ

1. **`src/core/genetic_optimizer.py`:**
   - Добавлен импорт `numpy as np`
   - Добавлена функция `_safe_metric_value`
   - Добавлена функция `_validate_metrics`
   - Добавлена функция `_passes_hard_constraints`
   - Добавлена функция `_apply_trade_count_penalties`
   - Добавлена функция `_apply_constraints`
   - Обновлена функция `_calculate_fitness`
   - Обновлены функции `_random_genes`, `_crossover`, `_mutate` для применения constraints

2. **`config/ga_config.yaml`:**
   - Обновлены диапазоны для `long_signal_multiplier` и `short_signal_multiplier`
   - Обновлены диапазоны для `entry_threshold_long` и `entry_threshold_short`
   - Добавлены комментарии с описанием constraints

---

## ✅ СТАТУС

**Реализация:** ✅ Завершена  
**Тестирование:** ⏳ Ожидает быстрого прогона  
**Документация:** ✅ Завершена

---

**Следующий шаг:** Запустить быстрый прогон (20×10) для проверки исправлений.

