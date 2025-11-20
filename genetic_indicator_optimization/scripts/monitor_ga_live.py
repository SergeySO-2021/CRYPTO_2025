"""
Живой мониторинг прогона ГА с обновлением каждые 30 секунд.
Показывает текущий статус, fitness, метрики и прогресс.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "ga_best_fixed.json"

def clear_screen():
    """Очищает экран (кроссплатформенный)"""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_time(seconds):
    """Форматирует время в читаемый вид"""
    if seconds < 60:
        return f"{int(seconds)} сек"
    elif seconds < 3600:
        return f"{int(seconds / 60)} мин {int(seconds % 60)} сек"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours} ч {minutes} мин"

def check_status():
    """Проверяет статус прогона"""
    
    if not RESULTS_FILE.exists():
        return {
            'status': 'not_started',
            'message': 'Файл результатов не найден. Прогон ещё не начался или находится на этапе инициализации.'
        }
    
    try:
        # Время последнего изменения
        mtime = RESULTS_FILE.stat().st_mtime
        last_modified = datetime.fromtimestamp(mtime)
        now = datetime.now()
        age_seconds = (now - last_modified).total_seconds()
        
        # Чтение данных
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fitness = data.get('fitness')
        metrics = data.get('metrics', {})
        genes = data.get('genes', {})
        
        # Определение статуса
        if age_seconds < 60:
            status = 'running'
        elif age_seconds < 300:
            status = 'maybe_finished'
        else:
            status = 'finished'
        
        return {
            'status': status,
            'last_modified': last_modified,
            'age_seconds': age_seconds,
            'fitness': fitness,
            'metrics': metrics,
            'genes': genes,
            'file_size': RESULTS_FILE.stat().st_size
        }
        
    except json.JSONDecodeError:
        return {
            'status': 'writing',
            'message': 'Файл существует, но JSON ещё не записан полностью. Прогон активен.'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Ошибка чтения файла: {e}'
        }

def print_status(info):
    """Выводит статус на экран"""
    
    clear_screen()
    
    print("=" * 80)
    print("МОНИТОРИНГ ГЕНЕТИЧЕСКОГО АЛГОРИТМА - ЖИВОЙ РЕЖИМ")
    print("=" * 80)
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Файл: {RESULTS_FILE}")
    print("=" * 80)
    print()
    
    if info['status'] == 'not_started':
        print("[INFO] Прогон ещё не начался")
        print(f"      {info['message']}")
        print()
        print("Ожидание создания файла результатов...")
        return
    
    if info['status'] == 'writing':
        print("[INFO] Прогон активен, файл создаётся")
        print(f"      {info['message']}")
        return
    
    if info['status'] == 'error':
        print(f"[ERROR] {info['message']}")
        return
    
    # Статус выполнения
    age = info['age_seconds']
    if info['status'] == 'running':
        print(f"[ACTIVE] Прогон активен (файл обновлялся {format_time(age)} назад)")
    elif info['status'] == 'maybe_finished':
        print(f"[CHECK] Возможно завершается (файл не обновлялся {format_time(age)})")
    else:
        print(f"[FINISHED] Прогон завершён (файл не обновлялся {format_time(age)})")
    
    print(f"Последнее обновление: {info['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Размер файла: {info['file_size'] / 1024:.1f} KB")
    print()
    
    # Fitness
    fitness = info['fitness']
    if fitness is not None:
        if fitness == float('inf'):
            print(f"[WARNING] Fitness = Infinity (проблема!)")
        elif fitness == -float('inf'):
            print(f"[INFO] Fitness = -Infinity (решение не прошло constraints)")
        else:
            print(f"[OK] Fitness: {fitness:.2f}")
    print()
    
    # Метрики
    metrics = info['metrics']
    print("-" * 80)
    print("МЕТРИКИ ПО СРЕЗАМ")
    print("-" * 80)
    
    for split_name in ['train', 'val', 'test']:
        if split_name in metrics:
            m = metrics[split_name]
            trades = m.get('total_trades', 0)
            return_pct = m.get('total_return', 0.0) * 100
            sharpe = m.get('sharpe_ratio', 0.0)
            max_dd = m.get('max_drawdown', 0.0) * 100
            win_rate = m.get('win_rate', 0.0) * 100
            pf = m.get('profit_factor', 0.0)
            
            print(f"{split_name.upper():6s} | "
                  f"Return: {return_pct:+.2f}% | "
                  f"Sharpe: {sharpe:6.2f} | "
                  f"Max DD: {max_dd:5.2f}% | "
                  f"WR: {win_rate:5.1f}% | "
                  f"PF: {pf:4.2f} | "
                  f"Trades: {trades:3d}")
    
    print()
    
    # Параметры
    genes = info['genes']
    if genes:
        print("-" * 80)
        print("КЛЮЧЕВЫЕ ПАРАМЕТРЫ")
        print("-" * 80)
        
        # Long/Short балансировка
        if 'long_signal_multiplier' in genes and 'short_signal_multiplier' in genes:
            long_mult = genes['long_signal_multiplier']
            short_mult = genes['short_signal_multiplier']
            sum_mult = long_mult + short_mult
            print(f"Long multiplier:  {long_mult:.3f}")
            print(f"Short multiplier: {short_mult:.3f}")
            print(f"Сумма: {sum_mult:.3f} {'[OK]' if sum_mult >= 1.6 else '[WARNING]'}")
            print()
        
        if 'entry_threshold_long' in genes and 'entry_threshold_short' in genes:
            long_thresh = genes['entry_threshold_long']
            short_thresh = genes['entry_threshold_short']
            print(f"Entry threshold Long:  {long_thresh:.3f}")
            print(f"Entry threshold Short: {short_thresh:.3f}")
            print(f"Long >= Short: {'[OK]' if long_thresh >= short_thresh else '[WARNING]'}")
            print()
        
        # Временное окно
        if 'time_filter_enabled' in genes:
            enabled = genes.get('time_filter_enabled', 0)
            if enabled:
                start = genes.get('time_window_start', 0)
                length = genes.get('time_window_length', 0)
                end = min(start + length, 24)
                print(f"Временное окно: {start:02d}:00 - {end:02d}:00 UTC ({length} часов)")
            else:
                print("Временное окно: отключено")
            print()
        
        # Другие параметры
        key_params = ['rsi_period', 'stop_loss_pct', 'take_profit_pct', 'atr_period']
        for param in key_params:
            if param in genes:
                value = genes[param]
                if isinstance(value, float):
                    print(f"{param:20s}: {value:.4f}")
                else:
                    print(f"{param:20s}: {value}")
    
    print()
    print("=" * 80)
    print(f"Следующее обновление через 30 секунд... (Ctrl+C для выхода)")
    print("=" * 80)

def main():
    """Основной цикл мониторинга"""
    print("Запуск мониторинга...")
    print("Обновление каждые 30 секунд")
    print("Нажмите Ctrl+C для остановки")
    print()
    time.sleep(2)
    
    try:
        while True:
            info = check_status()
            print_status(info)
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\nМониторинг остановлен пользователем")
        print("Прогон ГА продолжает работать в фоне")

if __name__ == "__main__":
    main()

