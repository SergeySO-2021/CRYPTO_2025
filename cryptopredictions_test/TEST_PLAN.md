# План тестирования CryptoPredictions на наших данных

## Цель

Оценить возможности библиотеки CryptoPredictions для:
1. Прогнозирования цены BTCUSDT на 15-минутных данных
2. Сравнения различных моделей ML/DL
3. Оценки качества через backtesting

## Этапы

### 1. Подготовка данных ✅
- [x] Адаптировать наши данные BTCUSDT 15m под формат CryptoPredictions
- [x] Сохранить в `data/BTCUSDT-15m-data.csv`
- [x] Создать конфигурационный файл для тестирования

### 2. Установка зависимостей
- [ ] Установить requirements.txt
- [ ] Проверить совместимость версий

### 3. Тестирование моделей
- [ ] Random Forest
- [ ] XGBoost
- [ ] LSTM (опционально)
- [ ] Prophet (опционально)

### 4. Оценка результатов
- [ ] Сравнить метрики (MAE, RMSE, MAPE)
- [ ] Провести backtesting
- [ ] Сравнить с нашим подходом (если применимо)

## Запуск

```bash
# 1. Подготовка данных
python test_on_our_data.py

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Тестирование моделей
python train.py --config-path configs/hydra --config-name test_our_data model=random_forest
python train.py --config-path configs/hydra --config-name test_our_data model=xgboost

# 4. Backtesting
python backtester.py --config-path configs/hydra --config-name backtest
```

## Ожидаемые результаты

- Метрики качества прогнозирования
- Сравнение различных моделей
- Оценка применимости для нашей задачи (классификация режимов)

