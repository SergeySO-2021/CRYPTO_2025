"""
Безопасный запуск stage2 с управлением памятью
Снижает нагрузку для работы на машинах с ограниченной RAM
"""

import gc
from staged_optimization import StagedOptimization

def run_stage2_optimized(mza_system, combinations=100, data_samples=2500):
    """
    Запуск stage2 с оптимизированными параметрами для экономии памяти
    
    Args:
        mza_system: система с данными
        combinations: количество комбинаций (по умолчанию 100 вместо 150)
        data_samples: количество записей (по умолчанию 2500 вместо 3000)
    """
    # Создаем оптимизатор
    staged_optimizer = StagedOptimization(mza_system)
    
    # Модифицируем параметры stage2 для экономии памяти
    print("🔧 Настройка параметров для экономии памяти:")
    print(f"   - Комбинаций: {combinations} (вместо 150)")
    print(f"   - Записей: {data_samples} (вместо 3000)")
    print(f"   - Параметров: 16 (все)")
    print()
    
    # Изменяем параметры stage2
    staged_optimizer.stages['stage2_balanced_1h']['combinations'] = combinations
    staged_optimizer.stages['stage2_balanced_1h']['data_samples'] = data_samples
    
    # Очищаем память перед запуском
    gc.collect()
    
    print("🚀 Запуск оптимизированного stage2...")
    print("="*70)
    
    # Запускаем stage2
    results = staged_optimizer.run_stage('stage2_balanced_1h')
    
    # Очищаем память после завершения
    gc.collect()
    
    return results

# РЕЖИМЫ ИСПОЛЬЗОВАНИЯ:
# ======================

# 1. ЭКОНОМНЫЙ РЕЖИМ (для машин с < 4GB RAM):
# results = run_stage2_optimized(mza_system, combinations=75, data_samples=2000)

# 2. СБАЛАНСИРОВАННЫЙ РЕЖИМ (рекомендуется):
# results = run_stage2_optimized(mza_system, combinations=100, data_samples=2500)

# 3. ПРЕМИУМ РЕЖИМ (для машин с > 8GB RAM):
# results = run_stage2_optimized(mza_system, combinations=150, data_samples=3000)

if __name__ == "__main__":
    print("💡 Для использования импортируйте функцию и запустите в notebook:")
    print("   from run_stage2_memory_safe import run_stage2_optimized")
    print("   results = run_stage2_optimized(mza_system, combinations=100, data_samples=2500)")


