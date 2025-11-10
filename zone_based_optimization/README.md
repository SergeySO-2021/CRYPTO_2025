# Zone-Based Optimization Project

> **Примечание:** Метод оказался не очень удачным. См. `PROJECT_FINAL_REPORT.md` для деталей.

## Структура проекта

```
zone_based_optimization/
├── PROJECT_FINAL_REPORT.md       # Единый отчет о проекте
├── README.md                      # Этот файл
├── classification_functions.py   # Функции классификации зон
├── scripts/                      # Скрипты проекта
│   ├── 01_check_environment_simple.py
│   ├── 02_check_data_simple.py
│   ├── 03_prepare_data_simple.py
│   ├── 03b_split_data.py
│   ├── 04_optimize_15m_adaptive.py
│   ├── 17_validate_trend_classifier_quality.py
│   ├── 18_create_simple_classes.py
│   ├── 19_create_indicators_matrix_simple.py
│   └── 20_validate_indicators_accuracy.py
├── data/                          # Данные
│   ├── prepared/
│   ├── train/
│   └── test/
└── results/                       # Результаты
    ├── optimization_15m_adaptive.json
    ├── trend_classifier_validation.json
    ├── zones_with_simple_classes.csv
    ├── indicators_matrix_simple_classes.json
    └── indicators_accuracy_validation.json
```

## Основные результаты

- ✅ Trend Classifier: точность 85.98% по направлениям
- ✅ Оптимизация параметров: score 0.900 (excellent)
- ❌ Классификация индикаторами: точность 4.22% (exact match)

Подробности в `PROJECT_FINAL_REPORT.md`.
