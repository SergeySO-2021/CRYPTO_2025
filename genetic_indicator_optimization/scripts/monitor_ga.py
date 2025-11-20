"""
Скрипт для мониторинга выполнения генетического алгоритма.

Показывает:
- Статус процессов Python
- Информацию о файле результатов (если создан)
- Прогресс выполнения (поколение, fitness)
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
import sys

def get_python_processes():
    """Получает информацию о процессах Python."""
    try:
        import psutil
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'create_time']):
            try:
                if 'python' in proc.info['name'].lower():
                    create_time = datetime.fromtimestamp(proc.info['create_time'])
                    if (datetime.now() - create_time).total_seconds() < 7200:  # Последние 2 часа
                        processes.append({
                            'pid': proc.info['pid'],
                            'cpu': proc.info['cpu_percent'],
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                            'start_time': create_time,
                            'uptime_min': (datetime.now() - create_time).total_seconds() / 60
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return processes
    except ImportError:
        print("⚠️  psutil не установлен. Установите: pip install psutil")
        return []

def check_results_file(results_path):
    """Проверяет файл результатов и извлекает информацию."""
    if not os.path.exists(results_path):
        return None
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        file_time = datetime.fromtimestamp(os.path.getmtime(results_path))
        file_age_min = (datetime.now() - file_time).total_seconds() / 60
        
        return {
            'exists': True,
            'last_update': file_time,
            'age_min': file_age_min,
            'fitness': data.get('fitness'),
            'genes': data.get('genes', {}),
            'metrics': data.get('metrics', {})
        }
    except Exception as e:
        return {'exists': True, 'error': str(e)}

def format_fitness(fitness):
    """Форматирует fitness для отображения."""
    if fitness is None:
        return "N/A"
    if fitness == float('inf'):
        return "∞ (target reached!)"
    return f"{fitness:.2f}"

def monitor_ga(results_file="results/ga_best_longshort.json", interval=10):
    """Основной цикл мониторинга."""
    results_path = Path(__file__).parent.parent / results_file
    
    print("=" * 70)
    print("🔍 МОНИТОРИНГ ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
    print("=" * 70)
    print(f"📁 Файл результатов: {results_path}")
    print(f"⏱️  Интервал обновления: {interval} секунд")
    print("=" * 70)
    print()
    
    iteration = 0
    try:
        while True:
            iteration += 1
            now = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n[{now}] Проверка #{iteration}")
            print("-" * 70)
            
            # Проверка процессов
            processes = get_python_processes()
            if processes:
                print(f"✅ Найдено процессов Python: {len(processes)}")
                total_cpu = sum(p['cpu'] for p in processes)
                total_memory = sum(p['memory_mb'] for p in processes)
                avg_uptime = sum(p['uptime_min'] for p in processes) / len(processes)
                
                print(f"   📊 CPU: {total_cpu:.1f}% | Память: {total_memory:.1f} MB | Время работы: {avg_uptime:.1f} мин")
                print(f"   🔢 Процессы: {', '.join(str(p['pid']) for p in processes)}")
            else:
                print("⚠️  Процессы Python не найдены (возможно, завершились)")
            
            # Проверка файла результатов
            results = check_results_file(results_path)
            if results is None:
                print("⏳ Файл результатов ещё не создан")
                print("   💡 ГА может быть на этапе инициализации или первого поколения")
            elif 'error' in results:
                print(f"❌ Ошибка чтения файла: {results['error']}")
            else:
                print("✅ Файл результатов найден!")
                print(f"   📅 Последнее обновление: {results['last_update'].strftime('%H:%M:%S')}")
                print(f"   ⏱️  Возраст файла: {results['age_min']:.1f} минут")
                
                fitness = results.get('fitness')
                print(f"   🎯 Fitness: {format_fitness(fitness)}")
                
                # Показываем метрики
                metrics = results.get('metrics', {})
                if 'val' in metrics:
                    val = metrics['val']
                    print(f"   📊 Val: Return={val.get('total_return', 0)*100:.2f}% | "
                          f"Sharpe={val.get('sharpe_ratio', 0):.2f} | "
                          f"Trades={val.get('total_trades', 0)}")
                
                if 'test' in metrics:
                    test = metrics['test']
                    print(f"   📊 Test: Return={test.get('total_return', 0)*100:.2f}% | "
                          f"Sharpe={test.get('sharpe_ratio', 0):.2f} | "
                          f"Trades={test.get('total_trades', 0)}")
                
                # Показываем ключевые параметры
                genes = results.get('genes', {})
                if genes:
                    print("   🧬 Ключевые параметры:")
                    if 'long_signal_multiplier' in genes:
                        print(f"      Long multiplier: {genes['long_signal_multiplier']:.3f}")
                    if 'short_signal_multiplier' in genes:
                        print(f"      Short multiplier: {genes['short_signal_multiplier']:.3f}")
                    if 'entry_threshold_long' in genes:
                        print(f"      Entry threshold Long: {genes['entry_threshold_long']:.3f}")
                    if 'entry_threshold_short' in genes:
                        print(f"      Entry threshold Short: {genes['entry_threshold_short']:.3f}")
            
            print("-" * 70)
            print(f"⏳ Следующая проверка через {interval} секунд... (Ctrl+C для выхода)")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Мониторинг остановлен пользователем")
        sys.exit(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Мониторинг выполнения ГА")
    parser.add_argument("--file", default="results/ga_best_longshort.json", 
                       help="Путь к файлу результатов")
    parser.add_argument("--interval", type=int, default=10,
                       help="Интервал проверки в секундах (по умолчанию 10)")
    args = parser.parse_args()
    
    monitor_ga(args.file, args.interval)

