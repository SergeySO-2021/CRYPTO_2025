"""
Простой скрипт для проверки статуса прогона ГА.

Проверяет:
1. Существует ли файл результатов
2. Когда он был последний раз изменён
3. Текущий fitness и метрики
4. На каком поколении остановился (если есть логи)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"

def check_ga_status(result_file="ga_best_fixed.json"):
    """Проверяет статус прогона ГА"""
    
    result_path = RESULTS_DIR / result_file
    
    print("=" * 60)
    print("ПРОВЕРКА СТАТУСА ПРОГОНА ГА")
    print("=" * 60)
    print()
    
    # Проверка существования файла
    if not result_path.exists():
        print(f"[INFO] Файл результатов не найден: {result_path}")
        print("[INFO] Прогон ещё не начался или файл не создан")
        return False
    
    # Проверка времени последнего изменения
    mtime = result_path.stat().st_mtime
    last_modified = datetime.fromtimestamp(mtime)
    now = datetime.now()
    time_diff = now - last_modified
    
    print(f"[INFO] Файл результатов: {result_path}")
    print(f"[INFO] Последнее изменение: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if time_diff.total_seconds() < 60:
        print(f"[STATUS] Файл обновлялся {int(time_diff.total_seconds())} секунд назад - ПРОГОН АКТИВЕН")
    elif time_diff.total_seconds() < 300:
        print(f"[STATUS] Файл обновлялся {int(time_diff.total_seconds() / 60)} минут назад - возможно завершается")
    else:
        print(f"[STATUS] Файл не обновлялся {int(time_diff.total_seconds() / 60)} минут - ПРОГОН ЗАВЕРШЁН")
    
    print()
    
    # Чтение результатов
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("=" * 60)
        print("ТЕКУЩИЕ РЕЗУЛЬТАТЫ")
        print("=" * 60)
        print()
        
        # Fitness
        fitness = data.get("fitness")
        if fitness is not None:
            if fitness == float('inf'):
                print(f"[WARNING] Fitness = Infinity (проблема!)")
            elif fitness == -float('inf'):
                print(f"[INFO] Fitness = -Infinity (решение не прошло constraints)")
            else:
                print(f"[OK] Fitness: {fitness:.2f}")
        else:
            print("[WARNING] Fitness не найден")
        
        print()
        
        # Метрики по срезам
        metrics = data.get("metrics", {})
        
        for split_name in ["train", "val", "test"]:
            if split_name in metrics:
                split_metrics = metrics[split_name]
                trades = split_metrics.get("total_trades", 0)
                return_pct = split_metrics.get("total_return", 0.0) * 100
                sharpe = split_metrics.get("sharpe_ratio", 0.0)
                max_dd = split_metrics.get("max_drawdown", 0.0) * 100
                win_rate = split_metrics.get("win_rate", 0.0) * 100
                
                print(f"{split_name.upper()}:")
                print(f"  Trades: {trades}")
                print(f"  Return: {return_pct:+.2f}%")
                print(f"  Sharpe: {sharpe:.2f}")
                print(f"  Max DD: {max_dd:.2f}%")
                print(f"  Win Rate: {win_rate:.1f}%")
                print()
        
        # Гены (краткая информация)
        genes = data.get("genes", {})
        if genes:
            print("=" * 60)
            print("НАЙДЕННЫЕ ПАРАМЕТРЫ (кратко)")
            print("=" * 60)
            
            key_params = [
                "rsi_period", "stop_loss_pct", "take_profit_pct",
                "time_filter_enabled", "time_window_start", "time_window_length",
                "long_signal_multiplier", "short_signal_multiplier"
            ]
            
            for param in key_params:
                if param in genes:
                    value = genes[param]
                    if isinstance(value, float):
                        print(f"  {param}: {value:.4f}")
                    else:
                        print(f"  {param}: {value}")
            print()
        
        # Проверка constraints
        print("=" * 60)
        print("ПРОВЕРКА CONSTRAINTS")
        print("=" * 60)
        
        if "long_signal_multiplier" in genes and "short_signal_multiplier" in genes:
            long_mult = genes["long_signal_multiplier"]
            short_mult = genes["short_signal_multiplier"]
            sum_mult = long_mult + short_mult
            
            if 0.8 <= long_mult <= 1.2 and 0.8 <= short_mult <= 1.2:
                print(f"[OK] Множители в диапазоне [0.8, 1.2]")
            else:
                print(f"[WARNING] Множители вне диапазона!")
            
            if sum_mult >= 1.6:
                print(f"[OK] Сумма множителей: {sum_mult:.2f} >= 1.6")
            else:
                print(f"[WARNING] Сумма множителей: {sum_mult:.2f} < 1.6")
        
        if "entry_threshold_long" in genes and "entry_threshold_short" in genes:
            long_thresh = genes["entry_threshold_long"]
            short_thresh = genes["entry_threshold_short"]
            
            if long_thresh >= short_thresh:
                print(f"[OK] Long порог ({long_thresh:.3f}) >= Short порог ({short_thresh:.3f})")
            else:
                print(f"[WARNING] Long порог ({long_thresh:.3f}) < Short порог ({short_thresh:.3f})")
        
        print()
        
        # Проверка на завершение
        val_metrics = metrics.get("val", {})
        val_trades = val_metrics.get("total_trades", 0)
        
        if val_trades >= 10:
            print("[OK] Val trades >= 10 (прошёл hard constraints)")
        else:
            print(f"[WARNING] Val trades = {val_trades} < 10 (не прошёл hard constraints)")
        
        print()
        print("=" * 60)
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Ошибка чтения JSON: {e}")
        print("[INFO] Файл может быть повреждён или запись ещё не завершена")
        return False
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Проверка статуса прогона ГА")
    parser.add_argument(
        "--file",
        type=str,
        default="ga_best_fixed.json",
        help="Имя файла результатов (по умолчанию: ga_best_fixed.json)"
    )
    
    args = parser.parse_args()
    
    check_ga_status(args.file)

