"""
Простой скрипт мониторинга ГА без внешних зависимостей.
Использует только стандартные библиотеки Python.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
import sys

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
    try:
        return f"{float(fitness):.2f}"
    except:
        return str(fitness)

def monitor_ga(results_file="results/ga_best_longshort.json", interval=10):
    """Основной цикл мониторинга."""
    results_path = Path(__file__).parent.parent / results_file
    
    print("=" * 70)
    print("МОНИТОРИНГ ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
    print("=" * 70)
    print(f"Файл результатов: {results_path}")
    print(f"Интервал обновления: {interval} секунд")
    print("=" * 70)
    print("Нажмите Ctrl+C для остановки")
    print()
    
    iteration = 0
    last_file_size = 0
    
    try:
        while True:
            iteration += 1
            now = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n[{now}] Проверка #{iteration}")
            print("-" * 70)
            
            # Проверка файла результатов
            results = check_results_file(results_path)
            if results is None:
                print("[!] Файл результатов еще не создан")
                print("    ГА может быть на этапе инициализации или первого поколения")
                print("    Обычно файл появляется через 5-15 минут после запуска")
            elif 'error' in results:
                print(f"[ERROR] Ошибка чтения файла: {results['error']}")
            else:
                # Проверяем, изменился ли файл
                current_size = os.path.getsize(results_path)
                file_changed = current_size != last_file_size
                last_file_size = current_size
                
                if file_changed:
                    print("[UPDATE] Файл обновлен!")
                
                print("[OK] Файл результатов найден!")
                print(f"    Последнее обновление: {results['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    Возраст файла: {results['age_min']:.1f} минут")
                print(f"    Размер файла: {current_size / 1024:.1f} KB")
                
                fitness = results.get('fitness')
                print(f"    Fitness: {format_fitness(fitness)}")
                
                # Показываем метрики
                metrics = results.get('metrics', {})
                if 'val' in metrics:
                    val = metrics['val']
                    return_pct = val.get('total_return', 0) * 100
                    sharpe = val.get('sharpe_ratio', 0)
                    trades = val.get('total_trades', 0)
                    win_rate = val.get('win_rate', 0) * 100
                    print(f"    Val: Return={return_pct:+.2f}% | "
                          f"Sharpe={sharpe:.2f} | "
                          f"Win Rate={win_rate:.1f}% | "
                          f"Trades={trades}")
                
                if 'test' in metrics:
                    test = metrics['test']
                    return_pct = test.get('total_return', 0) * 100
                    sharpe = test.get('sharpe_ratio', 0)
                    trades = test.get('total_trades', 0)
                    win_rate = test.get('win_rate', 0) * 100
                    print(f"    Test: Return={return_pct:+.2f}% | "
                          f"Sharpe={sharpe:.2f} | "
                          f"Win Rate={win_rate:.1f}% | "
                          f"Trades={trades}")
                
                if 'train' in metrics:
                    train = metrics['train']
                    return_pct = train.get('total_return', 0) * 100
                    trades = train.get('total_trades', 0)
                    print(f"    Train: Return={return_pct:+.2f}% | Trades={trades}")
                
                # Показываем ключевые параметры Long/Short
                genes = results.get('genes', {})
                if genes:
                    print("    Параметры Long/Short балансировки:")
                    if 'long_signal_multiplier' in genes:
                        print(f"      Long multiplier: {genes['long_signal_multiplier']:.3f}")
                    if 'short_signal_multiplier' in genes:
                        print(f"      Short multiplier: {genes['short_signal_multiplier']:.3f}")
                    if 'entry_threshold_long' in genes:
                        print(f"      Entry threshold Long: {genes['entry_threshold_long']:.3f}")
                    if 'entry_threshold_short' in genes:
                        print(f"      Entry threshold Short: {genes['entry_threshold_short']:.3f}")
                    
                    # Показываем временное окно
                    if 'time_filter_enabled' in genes:
                        enabled = genes.get('time_filter_enabled', 0)
                        if enabled:
                            start = genes.get('time_window_start', 0)
                            length = genes.get('time_window_length', 0)
                            end = min(start + length, 24)
                            print(f"    Временное окно: {start:02d}:00 - {end:02d}:00 UTC ({length} часов)")
                        else:
                            print(f"    Временное окно: отключено (торгуем весь день)")
            
            print("-" * 70)
            print(f"Следующая проверка через {interval} секунд... (Ctrl+C для выхода)")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nМониторинг остановлен пользователем")
        sys.exit(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Простой мониторинг выполнения ГА")
    parser.add_argument("--file", default="results/ga_best_longshort.json", 
                       help="Путь к файлу результатов")
    parser.add_argument("--interval", type=int, default=15,
                       help="Интервал проверки в секундах (по умолчанию 15)")
    args = parser.parse_args()
    
    monitor_ga(args.file, args.interval)

