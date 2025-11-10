"""
Скрипт 03: Подготовка данных (версия без эмодзи)
Подготавливает данные для работы с Trading Classifier
"""

import sys
import io
import pandas as pd
import os
from pathlib import Path

# Исправление кодировки для Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

def prepare_data(input_file, output_file):
    """
    Подготавливает данные для Trading Classifier
    
    Args:
        input_file: Путь к исходному файлу
        output_file: Путь для сохранения подготовленных данных
    """
    print(f"\n[PREPARE] Подготовка данных из {input_file}")
    
    # Загрузка
    df = pd.read_csv(input_file)
    print(f"   [OK] Загружено {len(df)} записей")
    
    # Определение колонки времени
    time_col = None
    for col in ['timestamps', 'time', 'timestamp', 'datetime']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        print("   [ERROR] Не найдена колонка времени!")
        return None
    
    # Преобразование времени
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)
    print(f"   [OK] Индекс установлен: {df.index.name}")
    
    # Проверка необходимых колонок
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"   [ERROR] Отсутствуют колонки: {missing_cols}")
        return None
    
    # Проверка на пропуски в основных колонках (OHLC)
    missing = df[required_cols].isnull().sum()
    if missing.sum() > 0:
        print(f"   [WARNING] Найдены пропуски в основных колонках:")
        for col, count in missing[missing > 0].items():
            print(f"      {col}: {count} пропусков")
        # Удаляем строки с пропусками в основных колонках
        df = df.dropna(subset=required_cols)
        print(f"   [OK] Удалены строки с пропусками, осталось {len(df)} записей")
    
    # Проверка пропусков в дополнительных колонках (trades_*)
    trades_cols = [col for col in df.columns if col.startswith('trades_')]
    if trades_cols:
        missing_trades = df[trades_cols].isnull().sum()
        if missing_trades.sum() > 0:
            print(f"   [INFO] Пропуски в колонках trades_* (не критично):")
            for col, count in missing_trades[missing_trades > 0].items():
                pct = (count / len(df)) * 100
                print(f"      {col}: {count} пропусков ({pct:.1f}%)")
            print(f"   [INFO] Это нормально - данные о трейдах доступны не для всех периодов")
            print(f"   [INFO] Trading Classifier будет работать только с OHLC данными")
    
    # Сортировка по времени
    df = df.sort_index()
    
    # Проверка дубликатов
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        print(f"   [WARNING] Найдено {duplicates} дубликатов, удаляем...")
        df = df[~df.index.duplicated(keep='first')]
        print(f"   [OK] Осталось {len(df)} уникальных записей")
    
    # Сохранение
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_file)
    print(f"   [OK] Данные сохранены: {output_file}")
    print(f"   Период: {df.index[0]} - {df.index[-1]}")
    print(f"   Размер: {len(df)} записей")
    
    return df

def main():
    try:
        print("=" * 80)
        print("ПОДГОТОВКА ДАННЫХ")
        print("=" * 80)
        
        # Ищем файл данных (относительно корня проекта)
        data_files = [
            '../binance_data_collector/BTCUSDT_15m_COMBINED.csv',
            '../df_btc_15m_complete.csv',
            '../df_btc_15m.csv',
        ]
        
        input_file = None
        for file_path in data_files:
            if os.path.exists(file_path):
                input_file = file_path
                break
        
        if input_file is None:
            print("[ERROR] Файл данных не найден!")
            print("   Проверьте наличие одного из файлов:")
            for file_path in data_files:
                print(f"      - {file_path}")
            return
        
        # Подготовка
        output_file = 'data/prepared/df_btc_15m_prepared.csv'
        df = prepare_data(input_file, output_file)
        
        if df is not None:
            print("\n" + "=" * 80)
            print("[SUCCESS] ПОДГОТОВКА ЗАВЕРШЕНА!")
            print(f"   Файл готов: {output_file}")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("[ERROR] ОШИБКА ПОДГОТОВКИ!")
            print("=" * 80)
    except Exception as e:
        print(f"[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

