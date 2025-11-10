"""
Скрипт для тестирования CryptoPredictions на наших данных BTCUSDT 15m
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(__file__))

def adapt_our_data_to_cryptopredictions_format():
    """
    Адаптирует наши данные BTCUSDT 15m под формат CryptoPredictions
    """
    print("=" * 80)
    print("АДАПТАЦИЯ НАШИХ ДАННЫХ ПОД ФОРМАТ CRYPTOPREDICTIONS")
    print("=" * 80)
    
    # Загружаем наши данные
    data_path = Path(r'..\df_btc_15m_complete.csv')
    if not data_path.exists():
        print(f"[ERROR] Файл не найден: {data_path}")
        return None
    
    print(f"\n[LOAD] Загрузка данных из {data_path.name}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"   [OK] Загружено {len(df)} записей")
    print(f"   Период: {df.index[0]} - {df.index[-1]}")
    
    # Проверяем наличие необходимых колонок
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"   [ERROR] Отсутствуют колонки: {missing}")
        return None
    
    # Адаптируем под формат CryptoPredictions
    # Формат: Date, High, Low, open, close, volume (обязательно)
    df_adapted = pd.DataFrame()
    
    # Дата (обязательно)
    df_adapted['Date'] = pd.to_datetime(df.index)
    
    # OHLC (High и Low с большой буквы, open и close с маленькой)
    df_adapted['High'] = df['high'].values
    df_adapted['Low'] = df['low'].values
    df_adapted['open'] = df['open'].values
    df_adapted['close'] = df['close'].values
    
    # Объем
    df_adapted['volume'] = df['volume'].values
    
    # Сортируем по дате
    df_adapted = df_adapted.sort_values('Date').reset_index(drop=True)
    
    # Убеждаемся, что Date в правильном формате
    df_adapted['Date'] = pd.to_datetime(df_adapted['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Сохраняем в формате CryptoPredictions
    output_path = Path('data') / 'BTCUSDT-15m-data.csv'
    output_path.parent.mkdir(exist_ok=True)
    
    df_adapted.to_csv(output_path, index=False)
    print(f"\n[SAVE] Данные сохранены: {output_path}")
    print(f"   Колонки: {list(df_adapted.columns)}")
    print(f"   Размер: {len(df_adapted)} записей")
    print(f"   Период: {df_adapted['Date'].iloc[0]} - {df_adapted['Date'].iloc[-1]}")
    
    return df_adapted, output_path

def test_models_on_our_data():
    """
    Тестирует несколько моделей CryptoPredictions на наших данных
    """
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ МОДЕЛЕЙ CRYPTOPREDICTIONS")
    print("=" * 80)
    
    # Адаптируем данные
    df_adapted, data_path = adapt_our_data_to_cryptopredictions_format()
    if df_adapted is None:
        return
    
    print(f"\n[INFO] Данные готовы для тестирования")
    print(f"   Путь: {data_path}")
    print(f"   Период: {df_adapted['Date'].min()} - {df_adapted['Date'].max()}")
    print(f"   Записей: {len(df_adapted)}")
    
    # Модели для тестирования (простые, без сложных зависимостей)
    models_to_test = [
        'random_forest',
        'xgboost',
        # 'lstm',  # Требует больше настроек
        # 'prophet',  # Может быть медленным
    ]
    
    print(f"\n[INFO] Будем тестировать модели: {models_to_test}")
    print(f"\n[NOTE] Для полного тестирования используйте:")
    print(f"   python train.py model=random_forest dataset_loader=CoinMarket load_path={data_path}")
    
    # Определяем даты для train/validation
    date_min = pd.to_datetime(df_adapted['Date'].iloc[0])
    date_max = pd.to_datetime(df_adapted['Date'].iloc[-1])
    
    # Разделяем 70% train, 15% validation, 15% test
    train_end = date_min + (date_max - date_min) * 0.70
    valid_end = date_min + (date_max - date_min) * 0.85
    
    train_start = date_min.strftime('%Y-%m-%d %H:%M:%S')
    train_end_str = train_end.strftime('%Y-%m-%d %H:%M:%S')
    valid_start = train_end_str
    valid_end_str = valid_end.strftime('%Y-%m-%d %H:%M:%S')
    
    # Создаем конфиг для тестирования
    config_content = f"""# Конфигурация для тестирования на наших данных
defaults:
  - _self_
  - model: random_forest
  - dataset_loader: CoinMarket
  - metrics

validation_method: simple
symbol: BTCUSDT
load_path: {data_path}
save_dir: results/
dataset_path: {data_path}

dataset_loader:
  window_size: 10
  train_start_date: {train_start}
  train_end_date: {train_end_str}
  valid_start_date: {valid_start}
  valid_end_date: {valid_end_str}
  features: Date, open, High, Low, close, volume
  indicators_names: rsi macd
"""
    
    config_path = Path('configs') / 'hydra' / 'test_our_data.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"\n[SAVE] Конфигурация сохранена: {config_path}")
    print(f"\n[INFO] Для запуска тестирования выполните:")
    print(f"   cd {Path.cwd()}")
    print(f"   python train.py --config-path configs/hydra --config-name test_our_data model=random_forest")
    print(f"   python train.py --config-path configs/hydra --config-name test_our_data model=xgboost")
    
    return data_path, config_path

def analyze_results():
    """
    Анализирует результаты тестирования
    """
    print("\n" + "=" * 80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    results_dir = Path('results')
    if not results_dir.exists():
        print(f"[WARNING] Папка результатов не найдена: {results_dir}")
        print(f"   Запустите сначала train.py для получения результатов")
        return
    
    print(f"[INFO] Результаты будут сохранены в: {results_dir}")
    print(f"   После запуска train.py проверьте файлы в этой папке")

def main():
    """Основная функция"""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ CRYPTOPREDICTIONS НА НАШИХ ДАННЫХ")
    print("=" * 80)
    
    try:
        # 1. Адаптация данных
        data_path, config_path = test_models_on_our_data()
        
        if data_path is None:
            return
        
        # 2. Инструкции по запуску
        print("\n" + "=" * 80)
        print("ИНСТРУКЦИИ ПО ЗАПУСКУ")
        print("=" * 80)
        print(f"""
1. Установите зависимости:
   pip install -r requirements.txt

2. Запустите тестирование моделей:
   python train.py --config-path configs/hydra --config-name test_our_data model=random_forest
   python train.py --config-path configs/hydra --config-name test_our_data model=xgboost

3. Проверьте результаты в папке results/

4. Для backtesting:
   python backtester.py --config-path configs/hydra --config-name backtest
        """)
        
        # 3. Анализ
        analyze_results()
        
        print("\n" + "=" * 80)
        print("[SUCCESS] ПОДГОТОВКА ЗАВЕРШЕНА!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

