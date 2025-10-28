# 🧪 ТЕСТИРОВАНИЕ ИСПРАВЛЕННОГО ОПТИМИЗАТОРА MZA
# ==================================================

import sys
import os

# Добавляем путь к модулям
current_dir = os.getcwd()
if 'indicator_optimization' not in current_dir:
    sys.path.append(os.path.join(current_dir, 'indicator_optimization', '01_mza_optimization'))
else:
    sys.path.append('.')

print("🧪 ТЕСТИРОВАНИЕ ИСПРАВЛЕННОГО ОПТИМИЗАТОРА MZA")
print("=" * 55)

try:
    # Импортируем модули
    from data_loader import load_btc_data
    from mza_optimizer import MZAOptimizer
    from accurate_mza_classifier import AccurateMZAClassifier
    
    print("✅ Все модули импортированы успешно")
    
    # Загружаем данные
    print("\n📊 Загружаем тестовые данные...")
    data = load_btc_data(['15m'])
    
    if data and '15m' in data:
        print(f"✅ Данные загружены: {len(data['15m'])} записей")
        
        # Тестируем создание оптимизатора
        print("\n🔧 Тестируем создание оптимизатора...")
        optimizer = MZAOptimizer(
            population_size=5,      # Маленькая популяция для теста
            max_generations=2,      # Мало поколений для теста
            mutation_rate=0.15,
            crossover_rate=0.8,
            elite_size=2
        )
        
        print("✅ Оптимизатор создан")
        print(f"📊 Размер популяции: {optimizer.population_size}")
        print(f"🔄 Максимум поколений: {optimizer.max_generations}")
        
        # Тестируем создание случайной особи
        print("\n🧬 Тестируем создание случайной особи...")
        individual = optimizer.create_random_individual()
        print(f"✅ Случайная особь создана: {len(individual)} параметров")
        
        # Показываем несколько параметров
        sample_params = dict(list(individual.items())[:5])
        print("📋 Пример параметров:")
        for param, value in sample_params.items():
            print(f"   {param}: {value}")
        
        # Тестируем мутацию
        print("\n🔄 Тестируем мутацию...")
        mutated = optimizer.mutate(individual)
        print("✅ Мутация выполнена")
        
        # Тестируем расчет фитнеса
        print("\n📊 Тестируем расчет фитнеса...")
        test_data = data['15m'].head(100)  # Маленький набор для теста
        fitness = optimizer.calculate_fitness(individual, test_data)
        print(f"✅ Фитнес рассчитан: {fitness:.6f}")
        
        # Тестируем кроссовер
        print("\n🔀 Тестируем кроссовер...")
        parent1 = optimizer.create_random_individual()
        parent2 = optimizer.create_random_individual()
        child1, child2 = optimizer.crossover(parent1, parent2)
        print("✅ Кроссовер выполнен")
        
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("✅ Оптимизатор готов к работе")
        
    else:
        print("❌ Не удалось загрузить данные")
        
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
