"""
Скрипт для диагностики проблем с оптимизацией MZA
"""

import sys
from pathlib import Path
import pandas as pd

print("🔍 ДИАГНОСТИКА ПРОБЛЕМ С ОПТИМИЗАЦИЕЙ MZA")
print("=" * 60)

# 1. Проверка импорта классификатора
print("\n1️⃣ Проверка импорта VectorizedMZAClassifier...")
try:
    project_root = Path(__file__).parent.parent.parent
    classifiers_path = project_root / 'compare_analyze_indicators' / 'classifiers'
    
    print(f"   Корневой каталог: {project_root}")
    print(f"   Путь к классификаторам: {classifiers_path}")
    
    if not classifiers_path.exists():
        print(f"   ❌ Путь не существует!")
        sys.exit(1)
    
    sys.path.insert(0, str(classifiers_path))
    
    from mza_classifier_vectorized import VectorizedMZAClassifier
    print("   ✅ VectorizedMZAClassifier импортирован успешно")
    
    # Проверка создания экземпляра
    print("\n2️⃣ Проверка создания экземпляра классификатора...")
    test_classifier = VectorizedMZAClassifier()
    print("   ✅ Экземпляр создан успешно")
    
    # Проверка с параметрами
    print("\n3️⃣ Проверка создания с параметрами...")
    test_params = {
        'rsi_length': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30
    }
    
    test_classifier = VectorizedMZAClassifier(parameters=test_params)
    print("   ✅ Классификатор создан с параметрами")
    
    # 4. Проверка данных
    print("\n4️⃣ Проверка данных...")
    csv_file = project_root / 'df_btc_1h.csv'
    
    if csv_file.exists():
        print(f"   ✅ Файл найден: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"   📊 Записей: {len(df)}")
        print(f"   📊 Колонки: {list(df.columns)}")
        
        # Подготовка данных
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df.set_index('timestamps', inplace=True)
        
        # Тест на небольшом количестве данных
        print("\n5️⃣ Тестирование классификатора на 100 записях...")
        data_sample = df.tail(100)[['open', 'high', 'low', 'close']].copy()
        
        try:
            predictions = test_classifier.fit_predict(data_sample)
            print(f"   ✅ Классификатор работает!")
            print(f"   📊 Получено предсказаний: {len(predictions)}")
            print(f"   📊 Уникальные значения: {set(predictions)}")
            
        except Exception as e:
            print(f"   ❌ Ошибка при классификации: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ❌ Файл не найден: {csv_file}")
    
    print("\n✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
    
except ImportError as e:
    print(f"   ❌ Ошибка импорта: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\n💡 Если все проверки прошли успешно, проблема может быть в:")
print("   1. Параметрах оптимизации")
print("   2. Формате данных")
print("   3. Логике расчета метрик")

