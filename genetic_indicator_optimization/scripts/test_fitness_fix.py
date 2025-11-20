"""
Тестовый скрипт для проверки исправлений fitness функции.

Проверяет:
1. Обработку edge cases (Infinity/NaN)
2. Жёсткие constraints
3. Применение constraints для Long/Short параметров
"""

import sys
import os
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from core.genetic_optimizer import GeneticOptimizer

def test_safe_metric_value():
    """Тест 1: Проверка обработки edge cases"""
    print("=" * 60)
    print("ТЕСТ 1: Обработка edge cases")
    print("=" * 60)
    
    # Создаём минимальный оптимизатор для тестирования
    # Используем путь к данным относительно корня проекта CRYPTO_2025
    data_path = PROJECT_ROOT.parent.parent / "dataframe" / "with_full_depth" / "df_btc_15m_complete.csv"
    optimizer = GeneticOptimizer(
        ga_config_path=str(PROJECT_ROOT / "config" / "ga_config.yaml"),
        strategy_config_path=str(PROJECT_ROOT / "config" / "mvp_strategy_config.yaml"),
        data_path=str(data_path) if data_path.exists() else None
    )
    
    # Тест обработки Infinity для profit_factor
    pf_inf = optimizer._safe_metric_value(float('inf'), "profit_factor", 0.0)
    assert pf_inf == 10.0, f"Ожидалось 10.0, получено {pf_inf}"
    print(f"[OK] profit_factor Infinity -> {pf_inf}")
    
    # Тест обработки Infinity для sharpe_ratio (положительный)
    sharpe_inf_pos = optimizer._safe_metric_value(float('inf'), "sharpe_ratio", 0.0)
    assert sharpe_inf_pos == 50.0, f"Ожидалось 50.0, получено {sharpe_inf_pos}"
    print(f"[OK] sharpe_ratio +Infinity -> {sharpe_inf_pos}")
    
    # Тест обработки Infinity для sharpe_ratio (отрицательный)
    sharpe_inf_neg = optimizer._safe_metric_value(-float('inf'), "sharpe_ratio", 0.0)
    assert sharpe_inf_neg == -50.0, f"Ожидалось -50.0, получено {sharpe_inf_neg}"
    print(f"[OK] sharpe_ratio -Infinity -> {sharpe_inf_neg}")
    
    # Тест обработки NaN
    nan_value = optimizer._safe_metric_value(float('nan'), "profit_factor", 0.0)
    assert nan_value == 10.0, f"Ожидалось 10.0, получено {nan_value}"
    print(f"[OK] profit_factor NaN -> {nan_value}")
    
    print("[OK] Все тесты обработки edge cases пройдены!\n")


def test_hard_constraints():
    """Тест 2: Проверка жёстких constraints"""
    print("=" * 60)
    print("ТЕСТ 2: Жёсткие constraints")
    print("=" * 60)
    
    # Используем путь к данным относительно корня проекта CRYPTO_2025
    data_path = PROJECT_ROOT.parent.parent / "dataframe" / "with_full_depth" / "df_btc_15m_complete.csv"
    optimizer = GeneticOptimizer(
        ga_config_path=str(PROJECT_ROOT / "config" / "ga_config.yaml"),
        strategy_config_path=str(PROJECT_ROOT / "config" / "mvp_strategy_config.yaml"),
        data_path=str(data_path) if data_path.exists() else None
    )
    
    # Тест: слишком мало сделок
    metrics_low_trades = {
        "total_trades": 5,  # < 10
        "max_drawdown": 0.10,
        "win_rate": 0.50
    }
    assert not optimizer._passes_hard_constraints(metrics_low_trades), \
        "Должно вернуть False для trades < 10"
    print("[OK] trades < 10 -> False")
    
    # Тест: слишком большая просадка
    metrics_high_dd = {
        "total_trades": 20,
        "max_drawdown": 0.25,  # > 0.20
        "win_rate": 0.50
    }
    assert not optimizer._passes_hard_constraints(metrics_high_dd), \
        "Должно вернуть False для max_drawdown > 0.20"
    print("[OK] max_drawdown > 0.20 -> False")
    
    # Тест: слишком низкий win rate
    metrics_low_wr = {
        "total_trades": 20,
        "max_drawdown": 0.10,
        "win_rate": 0.20  # < 0.25
    }
    assert not optimizer._passes_hard_constraints(metrics_low_wr), \
        "Должно вернуть False для win_rate < 0.25"
    print("[OK] win_rate < 0.25 -> False")
    
    # Тест: все constraints выполнены
    metrics_ok = {
        "total_trades": 20,
        "max_drawdown": 0.10,
        "win_rate": 0.50
    }
    assert optimizer._passes_hard_constraints(metrics_ok), \
        "Должно вернуть True для валидных метрик"
    print("[OK] Все constraints выполнены -> True")
    
    print("[OK] Все тесты жёстких constraints пройдены!\n")


def test_constraints_application():
    """Тест 3: Проверка применения constraints для Long/Short"""
    print("=" * 60)
    print("ТЕСТ 3: Применение constraints для Long/Short")
    print("=" * 60)
    
    # Используем путь к данным относительно корня проекта CRYPTO_2025
    data_path = PROJECT_ROOT.parent.parent / "dataframe" / "with_full_depth" / "df_btc_15m_complete.csv"
    optimizer = GeneticOptimizer(
        ga_config_path=str(PROJECT_ROOT / "config" / "ga_config.yaml"),
        strategy_config_path=str(PROJECT_ROOT / "config" / "mvp_strategy_config.yaml"),
        data_path=str(data_path) if data_path.exists() else None
    )
    
    # Тест: ослабленные множители (сумма < 1.6)
    weak_genes = {
        "long_signal_multiplier": 0.7,
        "short_signal_multiplier": 0.7,
        "entry_threshold_long": 0.4,
        "entry_threshold_short": 0.5
    }
    
    constrained = optimizer._apply_constraints(weak_genes)
    
    # Проверка constraint 1: сумма множителей >= 1.6
    sum_mult = constrained["long_signal_multiplier"] + constrained["short_signal_multiplier"]
    assert sum_mult >= 1.6, f"Сумма множителей должна быть >= 1.6, получено {sum_mult}"
    print(f"[OK] Сумма множителей: {sum_mult:.2f} >= 1.6")
    
    # Проверка, что множители в диапазоне [0.8, 1.2]
    assert 0.8 <= constrained["long_signal_multiplier"] <= 1.2, \
        f"long_signal_multiplier должен быть в [0.8, 1.2], получено {constrained['long_signal_multiplier']}"
    assert 0.8 <= constrained["short_signal_multiplier"] <= 1.2, \
        f"short_signal_multiplier должен быть в [0.8, 1.2], получено {constrained['short_signal_multiplier']}"
    print(f"[OK] Множители в диапазоне [0.8, 1.2]: long={constrained['long_signal_multiplier']:.2f}, short={constrained['short_signal_multiplier']:.2f}")
    
    # Проверка constraint 2: long порог >= short порога
    assert constrained["entry_threshold_long"] >= constrained["entry_threshold_short"], \
        f"entry_threshold_long должен быть >= entry_threshold_short"
    print(f"[OK] Пороги: long={constrained['entry_threshold_long']:.2f} >= short={constrained['entry_threshold_short']:.2f}")
    
    print("[OK] Все тесты применения constraints пройдены!\n")


def test_fitness_with_edge_cases():
    """Тест 4: Проверка fitness с edge cases"""
    print("=" * 60)
    print("ТЕСТ 4: Fitness с edge cases")
    print("=" * 60)
    
    # Используем путь к данным относительно корня проекта CRYPTO_2025
    data_path = PROJECT_ROOT.parent.parent / "dataframe" / "with_full_depth" / "df_btc_15m_complete.csv"
    optimizer = GeneticOptimizer(
        ga_config_path=str(PROJECT_ROOT / "config" / "ga_config.yaml"),
        strategy_config_path=str(PROJECT_ROOT / "config" / "mvp_strategy_config.yaml"),
        data_path=str(data_path) if data_path.exists() else None
    )
    
    # Тест: метрики с Infinity и малым количеством сделок
    test_metrics = {
        "val": {
            "total_return": 0.0004,  # +0.04%
            "sharpe_ratio": float('inf'),  # Бесконечный Sharpe
            "profit_factor": float('inf'),  # Бесконечный PF
            "max_drawdown": 0.0,  # Нулевая просадка
            "win_rate": 1.0,  # 100% win rate
            "total_trades": 1  # Всего 1 сделка
        },
        "train": {
            "total_return": -0.086,
            "sharpe_ratio": -57.47,
            "profit_factor": 0.25,
            "max_drawdown": 0.086,
            "win_rate": 0.167,
            "total_trades": 18
        }
    }
    
    fitness = optimizer._calculate_fitness(test_metrics)
    
    # Должно вернуть -inf из-за trades < 10
    assert fitness == -float('inf'), \
        f"Ожидалось -inf для trades < 10, получено {fitness}"
    print(f"[OK] Fitness для trades < 10: {fitness} (ожидалось -inf)")
    
    # Тест: валидные метрики
    valid_metrics = {
        "val": {
            "total_return": 0.05,
            "sharpe_ratio": 2.0,
            "profit_factor": 1.5,
            "max_drawdown": 0.10,
            "win_rate": 0.50,
            "total_trades": 30
        },
        "train": {
            "total_return": 0.04,
            "sharpe_ratio": 1.8,
            "profit_factor": 1.3,
            "max_drawdown": 0.12,
            "win_rate": 0.45,
            "total_trades": 150
        }
    }
    
    fitness_valid = optimizer._calculate_fitness(valid_metrics)
    
    # Должно быть конечное значение
    assert not (np.isinf(fitness_valid) or np.isnan(fitness_valid)), \
        f"Fitness должен быть конечным, получено {fitness_valid}"
    print(f"[OK] Fitness для валидных метрик: {fitness_valid:.2f} (конечное значение)")
    
    print("[OK] Все тесты fitness с edge cases пройдены!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ИСПРАВЛЕНИЙ FITNESS ФУНКЦИИ")
    print("=" * 60 + "\n")
    
    try:
        test_safe_metric_value()
        test_hard_constraints()
        test_constraints_application()
        test_fitness_with_edge_cases()
        
        print("=" * 60)
        print("[OK] ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] ОШИБКА В ТЕСТАХ: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

