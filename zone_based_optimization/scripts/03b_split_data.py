"""
Скрипт 03b: Разделение данных на train и test set (Walk-Forward Analysis)
Критически важно для валидации системы!
"""

import sys
import io
import os
import json
import pandas as pd
from pathlib import Path

# Исправление кодировки для Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

def split_data_walk_forward(df, train_ratio=0.7, verbose=True):
    """
    Разделяет данные на train и test set (walk-forward)
    
    Args:
        df: DataFrame с данными (индекс - datetime)
        train_ratio: Доля данных для train (0.7 = 70%)
        verbose: Выводить информацию
    
    Returns:
        train_df, test_df, split_info
    """
    if verbose:
        print(f"\n[SPLIT] Разделение данных на train/test")
        print(f"   Всего записей: {len(df)}")
        print(f"   Период: {df.index[0]} - {df.index[-1]}")
        print(f"   Дней: {(df.index[-1] - df.index[0]).days}")
    
    # Разделение по времени (не случайное!)
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Информация о разделении
    split_info = {
        'train_ratio': train_ratio,
        'test_ratio': 1 - train_ratio,
        'total_records': len(df),
        'train_records': len(train_df),
        'test_records': len(test_df),
        'split_index': split_idx,
        'train_start': str(train_df.index[0]),
        'train_end': str(train_df.index[-1]),
        'test_start': str(test_df.index[0]),
        'test_end': str(test_df.index[-1]),
        'train_days': (train_df.index[-1] - train_df.index[0]).days,
        'test_days': (test_df.index[-1] - test_df.index[0]).days,
    }
    
    if verbose:
        print(f"\n   [TRAIN SET]")
        print(f"      Записей: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"      Период: {train_df.index[0]} - {train_df.index[-1]}")
        print(f"      Дней: {split_info['train_days']} (~{split_info['train_days']/365:.1f} лет)")
        
        print(f"\n   [TEST SET]")
        print(f"      Записей: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        print(f"      Период: {test_df.index[0]} - {test_df.index[-1]}")
        print(f"      Дней: {split_info['test_days']} (~{split_info['test_days']/365:.1f} лет)")
        
        # Проверка на пересечения
        if train_df.index[-1] < test_df.index[0]:
            print(f"\n   [OK] Нет пересечений между train и test")
        else:
            print(f"\n   [WARNING] Есть пересечения!")
    
    return train_df, test_df, split_info

def main():
    try:
        print("=" * 80)
        print("РАЗДЕЛЕНИЕ ДАННЫХ НА TRAIN И TEST SET")
        print("Walk-Forward Analysis (временное разделение)")
        print("=" * 80)
        
        # Загрузка данных
        data_file = 'data/prepared/df_btc_15m_prepared.csv'
        if not os.path.exists(data_file):
            print(f"\n[ERROR] Файл данных не найден: {data_file}")
            print("   Сначала запустите: py scripts/03_prepare_data_simple.py")
            return
        
        print(f"\n[LOAD] Загрузка данных...")
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        print(f"   [OK] Загружено {len(df)} записей")
        
        # Разделение данных
        train_ratio = 0.7  # 70% для обучения, 30% для тестирования
        train_df, test_df, split_info = split_data_walk_forward(df, train_ratio=train_ratio)
        
        # Создание директорий
        train_dir = Path('data/train')
        test_dir = Path('data/test')
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохранение train set
        train_file = train_dir / 'df_btc_15m_train.csv'
        train_df.to_csv(train_file)
        print(f"\n[SAVE] Train set сохранен: {train_file}")
        
        # Сохранение test set
        test_file = test_dir / 'df_btc_15m_test.csv'
        test_df.to_csv(test_file)
        print(f"[SAVE] Test set сохранен: {test_file}")
        
        # Сохранение информации о разделении
        split_info_file = Path('data/split_info.json')
        with open(split_info_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] Информация о разделении: {split_info_file}")
        
        print("\n" + "=" * 80)
        print("[SUCCESS] РАЗДЕЛЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 80)
        print("\nВАЖНО:")
        print("   - Используйте train set для обучения (оптимизация, создание зон)")
        print("   - Используйте test set ТОЛЬКО для финальной валидации")
        print("   - НИКОГДА не используйте test set для обучения!")
        print("\nСледующий шаг:")
        print("   py scripts/04_optimize_15m_adaptive.py (обновить для использования train set)")
        
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

