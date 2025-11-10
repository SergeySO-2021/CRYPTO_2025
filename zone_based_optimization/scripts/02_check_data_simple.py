"""
Скрипт 02: Проверка данных (версия без эмодзи)
Проверяет наличие и структуру данных для работы
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

def check_data_file(file_path, description):
    """Проверяет файл данных"""
    print(f"\n[CHECK] {description}")
    print(f"   Путь: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"   [FAIL] Файл не найден!")
        return None
    
    try:
        df = pd.read_csv(file_path, nrows=5)  # Читаем только первые 5 строк для проверки
        print(f"   [OK] Файл существует")
        print(f"   Колонки: {list(df.columns)}")
        
        # Проверяем полный файл
        df_full = pd.read_csv(file_path)
        print(f"   Размер: {df_full.shape[0]} строк, {df_full.shape[1]} колонок")
        
        # Проверяем наличие необходимых колонок
        required_cols = ['open', 'high', 'low', 'close']
        if 'timestamps' in df.columns:
            required_cols.append('timestamps')
        elif 'time' in df.columns:
            required_cols.append('time')
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"   [WARNING] Отсутствуют колонки: {missing_cols}")
        else:
            print(f"   [OK] Все необходимые колонки присутствуют")
        
        # Проверяем период данных
        if 'timestamps' in df_full.columns:
            df_full['timestamps'] = pd.to_datetime(df_full['timestamps'])
            print(f"   Период: {df_full['timestamps'].min()} - {df_full['timestamps'].max()}")
        elif 'time' in df_full.columns:
            df_full['time'] = pd.to_datetime(df_full['time'])
            print(f"   Период: {df_full['time'].min()} - {df_full['time'].max()}")
        
        # Проверяем пропуски
        missing = df_full.isnull().sum()
        if missing.sum() > 0:
            print(f"   [WARNING] Пропуски в данных:")
            for col, count in missing[missing > 0].items():
                print(f"      {col}: {count} пропусков")
        else:
            print(f"   [OK] Пропусков нет")
        
        return df_full
        
    except Exception as e:
        print(f"   [ERROR] Ошибка при чтении файла: {e}")
        return None

def main():
    try:
        print("=" * 80)
        print("ПРОВЕРКА ДАННЫХ")
        print("=" * 80)
        
        # Проверяем основные файлы данных (относительно корня проекта)
        data_files = {
            '../df_btc_15m.csv': 'BTC 15m (основной)',
            '../df_btc_15m_complete.csv': 'BTC 15m (расширенный)',
            '../binance_data_collector/BTCUSDT_15m_COMBINED.csv': 'BTC 15m (с объемами)',
        }
        
        found_files = {}
        for file_path, description in data_files.items():
            df = check_data_file(file_path, description)
            if df is not None:
                found_files[file_path] = df
        
        print("\n" + "=" * 80)
        if found_files:
            print(f"[SUCCESS] НАЙДЕНО {len(found_files)} ФАЙЛОВ ДАННЫХ")
            print("\nРекомендуемый файл для работы:")
            
            # Рекомендуем файл с объемами, если есть
            if '../binance_data_collector/BTCUSDT_15m_COMBINED.csv' in found_files:
                print("   [RECOMMEND] binance_data_collector/BTCUSDT_15m_COMBINED.csv (с объемами и trades)")
            elif '../df_btc_15m_complete.csv' in found_files:
                print("   [RECOMMEND] df_btc_15m_complete.csv (расширенный)")
            else:
                print(f"   [RECOMMEND] {list(found_files.keys())[0]}")
        else:
            print("[ERROR] ФАЙЛЫ ДАННЫХ НЕ НАЙДЕНЫ!")
            print("   Убедитесь, что файлы данных находятся в корне проекта")
        print("=" * 80)
        
        return found_files
    except Exception as e:
        print(f"[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    try:
        found_files = main()
        sys.exit(0 if found_files else 1)
    except Exception as e:
        print(f"[ERROR] Ошибка выполнения: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

