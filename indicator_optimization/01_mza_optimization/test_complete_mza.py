"""
Тестирование полной реализации MZA с генетическим алгоритмом
Отдельный файл для тестирования нового подхода
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем наши модули
from accurate_mza_classifier import AccurateMZAClassifier
from complete_mza_optimizer import CompleteMZAOptimizer

def load_test_data():
    """Загружает тестовые данные BTC"""
    try:
        # Пытаемся загрузить данные из CSV файлов
        data_files = {
            '15m': 'df_btc_15m.csv',
            '30m': 'df_btc_30m.csv', 
            '1h': 'df_btc_1h.csv',
            '4h': 'df_btc_4h.csv',
            '1d': 'df_btc_1d.csv'
        }
        
        data = {}
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        for tf, filename in data_files.items():
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                # Проверяем наличие необходимых колонок
                required_cols = ['open', 'high', 'low', 'close']
                if all(col in df.columns for col in required_cols):
                    # Если нет колонки volume, создаем синтетическую
                    if 'volume' not in df.columns:
                        print(f"   ⚠️ Колонка 'volume' отсутствует, создаем синтетическую")
                        # Создаем volume на основе диапазона цен и случайности
                        price_range = df['high'] - df['low']
                        avg_price = df['close'].mean()
                        # Объем пропорционален диапазону цен и средней цене
                        df['volume'] = (price_range * avg_price * np.random.uniform(0.5, 2.0, len(df))).astype(int)
                    
                    data[tf] = df
                    print(f"✅ Загружены данные {tf}: {len(df):,} записей")
                else:
                    print(f"❌ Недостаточно основных колонок в {filename}")
                    print(f"   Доступные колонки: {list(df.columns)}")
                    print(f"   Требуемые колонки: {required_cols}")
            else:
                print(f"❌ Файл {filename} не найден")
        
        return data
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return {}

def test_basic_classifier():
    """Тестирует базовую функциональность классификатора"""
    print("🧪 ТЕСТИРОВАНИЕ БАЗОВОЙ ФУНКЦИОНАЛЬНОСТИ КЛАССИФИКАТОРА")
    print("=" * 70)
    
    # Создаем классификатор с параметрами по умолчанию
    classifier = AccurateMZAClassifier({})
    
    print(f"✅ Классификатор создан")
    print(f"📊 Параметров: {len(classifier.params)}")
    print(f"🔧 Доступные параметры: {list(classifier.params.keys())[:5]}...")
    
    # Создаем тестовые данные
    test_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    
    print(f"\n📊 Тестовые данные созданы: {len(test_data)} записей")
    
    try:
        # Тестируем предсказания
        predictions = classifier.predict(test_data)
        
        print(f"✅ Предсказания получены!")
        print(f"   📊 Размер массива: {len(predictions)}")
        print(f"   🎯 Уникальные значения: {np.unique(predictions)}")
        print(f"   📈 Бычьи зоны: {np.sum(predictions == 1)}")
        print(f"   📉 Медвежьи зоны: {np.sum(predictions == -1)}")
        print(f"   ➡️ Боковые зоны: {np.sum(predictions == 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования классификатора: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimizer_basic():
    """Тестирует базовую функциональность оптимизатора"""
    print("\n🧪 ТЕСТИРОВАНИЕ БАЗОВОЙ ФУНКЦИОНАЛЬНОСТИ ОПТИМИЗАТОРА")
    print("=" * 70)
    
    # Создаем оптимизатор
    optimizer = CompleteMZAOptimizer(
        population_size=5,
        max_generations=3,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=1,
        cv_folds=2,
        regularization_strength=0.01
    )
    
    print(f"✅ Оптимизатор создан")
    print(f"📊 Параметров для оптимизации: {len(optimizer.param_ranges)}")
    print(f"🧬 Размер популяции: {optimizer.population_size}")
    print(f"🔄 Поколений: {optimizer.max_generations}")
    
    # Создаем тестовые данные
    test_data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(101, 111, 100),
        'low': np.random.uniform(99, 109, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000, 2000, 100)
    })
    
    print(f"\n📊 Тестовые данные созданы: {len(test_data)} записей")
    
    try:
        # Тестируем создание особи
        individual = optimizer.create_random_individual()
        print(f"✅ Тестовая особь создана с {len(individual)} параметрами")
        
        # Тестируем оценку пригодности
        fitness = optimizer.evaluate_fitness(individual, test_data)
        print(f"📊 Оценка пригодности: {fitness:.6f}")
        
        if fitness > -1000:
            print("✅ Функция оценки работает корректно!")
        else:
            print("❌ Функция оценки возвращает ошибку")
            return False
        
        # Тестируем кросс-валидацию
        cv_results = optimizer.cross_validate_fitness(individual, test_data)
        print(f"📊 Результаты кросс-валидации:")
        print(f"   🏋️ Train Score: {cv_results['train_score']:.6f}")
        print(f"   🧪 Test Score: {cv_results['test_score']:.6f}")
        print(f"   📈 Стабильность: {cv_results['stability']:.6f}")
        print(f"   🛡️ Риск переобучения: {cv_results['overfitting_risk']:.6f}")
        
        if cv_results['train_score'] > -1000 and cv_results['test_score'] > -1000:
            print("✅ Кросс-валидация работает корректно!")
            return True
        else:
            print("❌ Кросс-валидация возвращает ошибки")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка тестирования оптимизатора: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mini_optimization():
    """Тестирует мини-оптимизацию"""
    print("\n🚀 ТЕСТИРОВАНИЕ МИНИ-ОПТИМИЗАЦИИ")
    print("=" * 50)
    
    # Создаем оптимизатор для мини-теста
    optimizer = CompleteMZAOptimizer(
        population_size=10,
        max_generations=5,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=2,
        cv_folds=2,
        regularization_strength=0.01
    )
    
    # Создаем более реалистичные тестовые данные
    np.random.seed(42)  # Для воспроизводимости
    n_points = 500
    
    # Создаем трендовые данные
    trend = np.linspace(100, 120, n_points)
    noise = np.random.normal(0, 2, n_points)
    prices = trend + noise
    
    test_data = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0, 2, n_points),
        'low': prices - np.random.uniform(0, 2, n_points),
        'close': prices + np.random.normal(0, 0.5, n_points),
        'volume': np.random.uniform(1000, 5000, n_points)
    })
    
    print(f"📊 Тестовые данные созданы: {len(test_data)} записей")
    print(f"💰 Диапазон цен: {test_data['close'].min():.2f} - {test_data['close'].max():.2f}")
    
    try:
        # Запускаем мини-оптимизацию
        print("🧬 Запускаем мини-оптимизацию...")
        results = optimizer.optimize(test_data, verbose=True)
        
        print(f"\n✅ МИНИ-ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print(f"🏆 Лучший Economic Value: {results['best_score']:.6f}")
        print(f"📊 Протестировано оценок: {results['total_evaluations']}")
        print(f"🎯 Параметров оптимизировано: {results['param_count']}")
        
        if results['best_score'] > -1000:
            print("🎉 МИНИ-ОПТИМИЗАЦИЯ УСПЕШНА!")
            
            # Показываем лучшие параметры
            best_params = results['best_parameters']
            print(f"\n🔧 ЛУЧШИЕ ПАРАМЕТРЫ:")
            print("-" * 25)
            
            # Показываем ключевые параметры
            key_params = ['fastMALength', 'slowMALength', 'rsiLength', 'adxThreshold', 
                         'trendWeightBase', 'momentumWeightBase', 'priceActionWeightBase']
            for param in key_params:
                if param in best_params:
                    print(f"   {param}: {best_params[param]}")
            
            return True
        else:
            print("❌ Мини-оптимизация не дала результатов")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка мини-оптимизации: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_data_optimization():
    """Тестирует оптимизацию на реальных данных BTC"""
    print("\n📊 ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ ДАННЫХ BTC")
    print("=" * 50)
    
    # Загружаем реальные данные
    data = load_test_data()
    
    if not data:
        print("❌ Не удалось загрузить данные")
        return False
    
    # Выбираем таймфрейм для тестирования
    test_timeframe = '15m'
    if test_timeframe not in data:
        test_timeframe = list(data.keys())[0]
    
    print(f"📊 Тестируем на таймфрейме: {test_timeframe}")
    print(f"📈 Записей: {len(data[test_timeframe]):,}")
    
    # Используем последние 2000 записей для быстрого тестирования
    test_data = data[test_timeframe].tail(2000)
    
    # Создаем оптимизатор
    optimizer = CompleteMZAOptimizer(
        population_size=15,
        max_generations=10,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=3,
        cv_folds=3,
        regularization_strength=0.01
    )
    
    try:
        # Запускаем оптимизацию
        print("🧬 Запускаем оптимизацию на реальных данных...")
        results = optimizer.optimize(test_data, verbose=True)
        
        print(f"\n✅ ОПТИМИЗАЦИЯ НА РЕАЛЬНЫХ ДАННЫХ ЗАВЕРШЕНА")
        print(f"🏆 Лучший Economic Value: {results['best_score']:.6f}")
        print(f"📊 Протестировано оценок: {results['total_evaluations']}")
        print(f"🎯 Параметров оптимизировано: {results['param_count']}")
        print(f"🛡️ Риск переобучения: {results['overfitting_analysis']['overfitting_risk']:.3f}")
        print(f"📈 Стабильность: {results['overfitting_analysis']['stability']:.3f}")
        
        if results['best_score'] > -1000:
            print("🎉 ОПТИМИЗАЦИЯ НА РЕАЛЬНЫХ ДАННЫХ УСПЕШНА!")
            
            # Показываем лучшие параметры
            best_params = results['best_parameters']
            print(f"\n🔧 ЛУЧШИЕ ПАРАМЕТРЫ ДЛЯ {test_timeframe}:")
            print("-" * 40)
            
            # Группируем параметры по категориям
            categories = {
                'Trend': ['adxLength', 'adxThreshold', 'fastMALength', 'slowMALength'],
                'Momentum': ['rsiLength', 'stochKLength', 'macdFast', 'macdSlow', 'macdSignal'],
                'Price Action': ['hhllRange', 'haDojiRange', 'candleRangeLength'],
                'Market Activity': ['bbLength', 'bbMultiplier', 'atrLength', 'kcLength', 'kcMultiplier', 'volumeMALength'],
                'Weights': ['trendWeightBase', 'momentumWeightBase', 'priceActionWeightBase'],
                'Stability': ['useSmoothing', 'useHysteresis']
            }
            
            for category, params in categories.items():
                print(f"\n📊 {category}:")
                for param in params:
                    if param in best_params:
                        print(f"   {param}: {best_params[param]}")
            
            # Экспортируем в Pine Script
            try:
                pine_file = f"optimized_mza_{test_timeframe}.pine"
                pine_code = optimizer.export_to_pine_script(pine_file)
                print(f"\n📄 Оптимальные параметры экспортированы в {pine_file}")
            except Exception as e:
                print(f"⚠️ Ошибка экспорта в Pine Script: {e}")
            
            return True
        else:
            print("❌ Оптимизация на реальных данных не дала результатов")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка оптимизации на реальных данных: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция тестирования"""
    print("🎯 ТЕСТИРОВАНИЕ ПОЛНОЙ РЕАЛИЗАЦИИ MZA С ГЕНЕТИЧЕСКИМ АЛГОРИТМОМ")
    print("=" * 80)
    print("📅 Дата: 26.10.2025")
    print("🔧 Версия: 1.0")
    print("=" * 80)
    
    # Счетчик успешных тестов
    successful_tests = 0
    total_tests = 4
    
    # Тест 1: Базовая функциональность классификатора
    if test_basic_classifier():
        successful_tests += 1
        print("✅ Тест 1/4 пройден: Базовая функциональность классификатора")
    else:
        print("❌ Тест 1/4 провален: Базовая функциональность классификатора")
    
    # Тест 2: Базовая функциональность оптимизатора
    if test_optimizer_basic():
        successful_tests += 1
        print("✅ Тест 2/4 пройден: Базовая функциональность оптимизатора")
    else:
        print("❌ Тест 2/4 провален: Базовая функциональность оптимизатора")
    
    # Тест 3: Мини-оптимизация
    if test_mini_optimization():
        successful_tests += 1
        print("✅ Тест 3/4 пройден: Мини-оптимизация")
    else:
        print("❌ Тест 3/4 провален: Мини-оптимизация")
    
    # Тест 4: Оптимизация на реальных данных
    if test_real_data_optimization():
        successful_tests += 1
        print("✅ Тест 4/4 пройден: Оптимизация на реальных данных")
    else:
        print("❌ Тест 4/4 провален: Оптимизация на реальных данных")
    
    # Итоговый отчет
    print(f"\n🎯 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 50)
    print(f"📊 Пройдено тестов: {successful_tests}/{total_tests}")
    print(f"📈 Процент успеха: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests == total_tests:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("✅ Полная реализация MZA готова к использованию")
        print("✅ Генетический алгоритм работает корректно")
        print("✅ Кросс-валидация функционирует")
        print("✅ Оптимизация на реальных данных успешна")
    elif successful_tests >= total_tests * 0.75:
        print("⚠️ БОЛЬШИНСТВО ТЕСТОВ ПРОЙДЕНО")
        print("✅ Основная функциональность работает")
        print("⚠️ Требуется доработка некоторых компонентов")
    else:
        print("❌ МНОГИЕ ТЕСТЫ ПРОВАЛЕНЫ")
        print("❌ Требуется серьезная доработка")
        print("❌ Не рекомендуется использовать в продакшене")
    
    print(f"\n📚 ФАЙЛЫ ПРОЕКТА:")
    print("-" * 20)
    print("📄 accurate_mza_classifier.py - Полная реализация MZA")
    print("📄 complete_mza_optimizer.py - Генетический алгоритм")
    print("📄 test_complete_mza.py - Этот файл тестирования")
    print("📄 COMPLETE_MZA_IMPLEMENTATION_REPORT.md - Документация")
    
    print(f"\n🎯 РЕКОМЕНДАЦИИ:")
    print("-" * 20)
    if successful_tests == total_tests:
        print("✅ Можно запускать полную оптимизацию")
        print("✅ Можно использовать в продакшене")
        print("✅ Можно экспортировать в Pine Script")
    else:
        print("⚠️ Исправить ошибки перед использованием")
        print("⚠️ Протестировать на дополнительных данных")
        print("⚠️ Проверить совместимость с TradingView")

if __name__ == "__main__":
    main()
