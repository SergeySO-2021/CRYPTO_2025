"""
Детальный анализ результатов генетического алгоритма.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Добавляем src в путь
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import DataLoader
from analysis.indicator_pipeline import IndicatorPipeline
from strategies.mvp_strategy import MVPStrategy, MVPSignalConfig
from core.simple_backtester import SimpleBacktester, BacktestConfig


def load_ga_results(results_path: str) -> dict:
    """Загружает результаты ГА."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_strategy_config(genes: dict, base_config_path: str) -> dict:
    """Строит конфигурацию стратегии из генома."""
    import yaml
    
    with open(base_config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Обновляем параметры из генома
    signals_cfg = cfg.setdefault("signals", {})
    rsi_cfg = signals_cfg.setdefault("rsi", {})
    rsi_cfg["column"] = f"rsi_{genes['rsi_period']}"
    
    wobi_cfg = signals_cfg.setdefault("wobi", {})
    wobi_cfg["weight"] = genes["wobi_weight"]
    
    risk_cfg = cfg.setdefault("risk", {})
    risk_cfg["stop_loss_pct"] = genes["stop_loss_pct"]
    risk_cfg["take_profit_pct"] = genes["take_profit_pct"]
    
    atr_cfg = risk_cfg.setdefault("atr", {})
    atr_cfg["enabled"] = True
    atr_cfg["period"] = genes["atr_period"]
    atr_cfg["stop_multiplier"] = genes["atr_stop_multiplier"]
    atr_cfg["trailing_multiplier"] = genes["atr_trailing_multiplier"]
    
    # Временной фильтр
    time_cfg = cfg.setdefault("time_filter", {})
    time_cfg["enabled"] = bool(genes.get("time_filter_enabled", 0))
    if time_cfg["enabled"]:
        start = int(genes.get("time_window_start", 10))
        length = int(genes.get("time_window_length", 4))
        time_cfg["allowed_hours_start"] = start
        time_cfg["allowed_hours_end"] = min(start + length, 24)
    
    # Long/Short балансировка (если есть в генах)
    combination_cfg = cfg.setdefault("combination", {})
    if "long_signal_multiplier" in genes:
        combination_cfg["long_signal_multiplier"] = float(genes["long_signal_multiplier"])
    if "short_signal_multiplier" in genes:
        combination_cfg["short_signal_multiplier"] = float(genes["short_signal_multiplier"])
    if "entry_threshold_long" in genes:
        combination_cfg["entry_threshold_long"] = float(genes["entry_threshold_long"])
    if "entry_threshold_short" in genes:
        combination_cfg["entry_threshold_short"] = float(genes["entry_threshold_short"])
    
    return cfg


def analyze_parameters(genes: dict):
    """Анализирует найденные параметры."""
    print("=" * 80)
    print("АНАЛИЗ НАЙДЕННЫХ ПАРАМЕТРОВ")
    print("=" * 80)
    
    print(f"\n[ИНДИКАТОРЫ]")
    print(f"  RSI период: {genes['rsi_period']}")
    if 'rsi_weight' in genes:
        print(f"  RSI вес: {genes['rsi_weight']:.4f}")
    if 'rsi_buy_below_long' in genes:
        print(f"  RSI buy_below_long: {genes['rsi_buy_below_long']:.2f}")
    if 'rsi_sell_above_short' in genes:
        print(f"  RSI sell_above_short: {genes['rsi_sell_above_short']:.2f}")
    
    if 'macd_fast_period' in genes and 'macd_slow_period' in genes and 'macd_signal_period' in genes:
        print(f"  MACD: {genes['macd_fast_period']}-{genes['macd_slow_period']}-{genes['macd_signal_period']}")
    if 'macd_weight' in genes:
        print(f"  MACD вес: {genes['macd_weight']:.4f}")
    
    if 'bollinger_period' in genes and 'bollinger_std_dev' in genes:
        print(f"  Bollinger: период {genes['bollinger_period']}, std_dev {genes['bollinger_std_dev']:.2f}")
    if 'bollinger_weight' in genes:
        print(f"  Bollinger вес: {genes['bollinger_weight']:.4f}")
    
    print(f"  WOBI вес: {genes['wobi_weight']:.4f}")
    if all(f'wobi_weight_ratio{k}' in genes for k in [3, 5, 8, 60]):
        w3 = genes['wobi_weight_ratio3']
        w5 = genes['wobi_weight_ratio5']
        w8 = genes['wobi_weight_ratio8']
        w60 = genes['wobi_weight_ratio60']
        total = w3 + w5 + w8 + w60
        if total > 0:
            print(f"  WOBI веса глубин: ratio3={w3/total:.3f}, ratio5={w5/total:.3f}, ratio8={w8/total:.3f}, ratio60={w60/total:.3f}")
    
    print(f"\n[УПРАВЛЕНИЕ РИСКАМИ]")
    print(f"  Stop Loss: {genes['stop_loss_pct']*100:.2f}%")
    print(f"  Take Profit: {genes['take_profit_pct']*100:.2f}%")
    print(f"  ATR период: {genes['atr_period']}")
    print(f"  ATR stop multiplier: {genes['atr_stop_multiplier']:.4f}")
    print(f"  ATR trailing multiplier: {genes['atr_trailing_multiplier']:.4f}")
    
    print(f"\n[ВРЕМЕННОЙ ФИЛЬТР]")
    if genes.get('time_filter_enabled', 0):
        start = genes.get('time_window_start', 0)
        length = genes.get('time_window_length', 4)
        end = min(start + length, 24)
        print(f"  Включен: ДА")
        print(f"  Окно: {start:02d}:00 - {end:02d}:00 UTC ({length} часов)")
        print(f"  Это соответствует: {start+3:02d}:00 - {end+3:02d}:00 МСК")
    else:
        print(f"  Включен: НЕТ (торгуем весь день)")
    
    # Анализ соотношения SL/TP
    sl_tp_ratio = genes['stop_loss_pct'] / genes['take_profit_pct']
    print(f"\n[СООТНОШЕНИЕ SL/TP]: {sl_tp_ratio:.2f}")
    if sl_tp_ratio > 1.5:
        print("  [WARN] Stop Loss значительно больше Take Profit - консервативный подход")
    elif sl_tp_ratio < 0.8:
        print("  [WARN] Take Profit больше Stop Loss - агрессивный подход")
    else:
        print("  [OK] Сбалансированное соотношение")
    
    # Long/Short балансировка (если есть)
    if "long_signal_multiplier" in genes and "short_signal_multiplier" in genes:
        print(f"\n[LONG/SHORT БАЛАНСИРОВКА]")
        long_mult = genes["long_signal_multiplier"]
        short_mult = genes["short_signal_multiplier"]
        sum_mult = long_mult + short_mult
        print(f"  Long multiplier: {long_mult:.4f}")
        print(f"  Short multiplier: {short_mult:.4f}")
        print(f"  Сумма: {sum_mult:.4f} {'[OK]' if sum_mult >= 1.6 else '[WARNING]'}")
        
        if "entry_threshold_long" in genes and "entry_threshold_short" in genes:
            long_thresh = genes["entry_threshold_long"]
            short_thresh = genes["entry_threshold_short"]
            print(f"  Entry threshold Long: {long_thresh:.4f}")
            print(f"  Entry threshold Short: {short_thresh:.4f}")
            print(f"  Long >= Short: {'[OK]' if long_thresh >= short_thresh else '[WARNING]'}")


def analyze_metrics_comparison(metrics: dict):
    """Сравнивает метрики между срезами."""
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ МЕТРИК ПО СРЕЗАМ")
    print("=" * 80)
    
    splits = ['train', 'val', 'test']
    split_names = {'train': 'Обучение', 'val': 'Валидация', 'test': 'Тест'}
    
    # Создаём таблицу сравнения
    comparison = []
    for split in splits:
        m = metrics.get(split, {})
        comparison.append({
            'Срез': split_names[split],
            'Доходность (%)': m.get('total_return', 0) * 100,
            'Sharpe': m.get('sharpe_ratio', 0),
            'Max DD (%)': m.get('max_drawdown', 0) * 100,
            'Win Rate (%)': m.get('win_rate', 0) * 100,
            'Profit Factor': m.get('profit_factor', 0),
            'Сделок': m.get('total_trades', 0),
        })
    
    df = pd.DataFrame(comparison)
    print("\n" + df.to_string(index=False))
    
    # Анализ переобучения
    print("\n" + "-" * 80)
    print("АНАЛИЗ НА ПЕРЕОБУЧЕНИЕ")
    print("-" * 80)
    
    train_return = metrics['train'].get('total_return', 0)
    val_return = metrics['val'].get('total_return', 0)
    test_return = metrics['test'].get('total_return', 0)
    
    train_sharpe = metrics['train'].get('sharpe_ratio', 0)
    val_sharpe = metrics['val'].get('sharpe_ratio', 0)
    test_sharpe = metrics['test'].get('sharpe_ratio', 0)
    
    # Проверка на переобучение
    if train_return > val_return * 1.5:
        print("[WARN] ПЕРЕОБУЧЕНИЕ: Train доходность значительно выше Val")
    elif val_return > train_return * 1.2:
        print("[OK] ХОРОШО: Val доходность лучше Train (хорошая обобщающая способность)")
    else:
        print("[OK] НОРМАЛЬНО: Train и Val доходности близки")
    
    if train_sharpe > val_sharpe * 2:
        print("[WARN] ПЕРЕОБУЧЕНИЕ: Train Sharpe значительно выше Val")
    elif val_sharpe > train_sharpe * 1.5:
        print("[OK] ОТЛИЧНО: Val Sharpe намного лучше Train")
    else:
        print("[OK] НОРМАЛЬНО: Train и Val Sharpe близки")
    
    # Стабильность на тесте
    print("\n" + "-" * 80)
    print("СТАБИЛЬНОСТЬ НА ТЕСТЕ")
    print("-" * 80)
    
    val_test_return_diff = abs(val_return - test_return) / max(abs(val_return), 0.001)
    val_test_sharpe_diff = abs(val_sharpe - test_sharpe) / max(abs(val_sharpe), 0.001)
    
    if val_test_return_diff < 0.3:
        print(f"[OK] Стабильная доходность: разница между Val и Test {val_test_return_diff*100:.1f}%")
    else:
        print(f"[WARN] Нестабильная доходность: разница между Val и Test {val_test_return_diff*100:.1f}%")
    
    if val_test_sharpe_diff < 0.4:
        print(f"[OK] Стабильный Sharpe: разница между Val и Test {val_test_sharpe_diff*100:.1f}%")
    else:
        print(f"[WARN] Нестабильный Sharpe: разница между Val и Test {val_test_sharpe_diff*100:.1f}%")


def analyze_trades_by_split(results_path: str, data_path: str = None):
    """Анализирует сделки по каждому срезу."""
    print("\n" + "=" * 80)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ СДЕЛОК ПО СРЕЗАМ")
    print("=" * 80)
    
    results = load_ga_results(results_path)
    genes = results['genes']
    
    # Загружаем данные
    loader = DataLoader(data_path=data_path)
    raw_data = loader.load_data()
    train_df, val_df, test_df = loader.split_data()
    
    # Строим конфигурацию
    base_config_path = Path(__file__).parent.parent / "config" / "mvp_strategy_config.yaml"
    strategy_config = build_strategy_config(genes, str(base_config_path))
    
    # Вычисляем индикаторы
    # Извлекаем параметры индикаторов из конфига
    indicator_params = {
        'rsi': {'period': genes['rsi_period']},
        'atr': {'period': genes['atr_period']},
    }
    pipeline = IndicatorPipeline(raw_data, params=indicator_params)
    artifacts = pipeline.run()
    enriched = artifacts.data.dropna()
    
    # Создаём стратегию
    signal_config = MVPSignalConfig.from_dict(strategy_config)
    strategy = MVPStrategy(signal_config)
    
    # Создаём бэктестер
    backtest_config = BacktestConfig.from_dict(strategy_config.get("risk"))
    
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df,
    }
    
    for split_name, df in splits.items():
        print(f"\n{'-' * 80}")
        print(f"СРЕЗ: {split_name.upper()}")
        print(f"{'-' * 80}")
        
        # Получаем данные для среза (используем reindex, как в genetic_optimizer)
        split_data = enriched.reindex(df.index).dropna()
        if split_data.empty:
            print(f"  [WARN] Нет данных для среза {split_name}")
            continue
        signals = strategy.generate_signals(split_data)
        
        # Запускаем бэктест
        backtester = SimpleBacktester(backtest_config)
        result = backtester.run(split_data, signals)
        
        if result.trades:
            trades_df = pd.DataFrame(result.trades)
            
            # Статистика по сделкам
            long_trades = trades_df[trades_df['position_type'] == 'long']
            short_trades = trades_df[trades_df['position_type'] == 'short']
            
            print(f"\n[ОБЩАЯ СТАТИСТИКА]")
            print(f"  Всего сделок: {len(trades_df)}")
            print(f"  Long: {len(long_trades)} ({len(long_trades)/len(trades_df)*100:.1f}%)")
            print(f"  Short: {len(short_trades)} ({len(short_trades)/len(trades_df)*100:.1f}%)")
            
            winning = trades_df[trades_df['pnl'] > 0]
            losing = trades_df[trades_df['pnl'] <= 0]
            
            print(f"\n[РЕЗУЛЬТАТЫ]")
            print(f"  Прибыльных: {len(winning)} ({len(winning)/len(trades_df)*100:.1f}%)")
            print(f"  Убыточных: {len(losing)} ({len(losing)/len(trades_df)*100:.1f}%)")
            print(f"  Средний PnL: {trades_df['pnl'].mean():.2f}")
            print(f"  Медианный PnL: {trades_df['pnl'].median():.2f}")
            
            if len(winning) > 0:
                print(f"  Средний выигрыш: {winning['pnl'].mean():.2f}")
                print(f"  Максимальный выигрыш: {winning['pnl'].max():.2f}")
            
            if len(losing) > 0:
                print(f"  Средний проигрыш: {losing['pnl'].mean():.2f}")
                print(f"  Максимальный проигрыш: {losing['pnl'].min():.2f}")
            
            # Анализ по направлениям
            if len(long_trades) > 0:
                long_win_rate = (long_trades['pnl'] > 0).sum() / len(long_trades) * 100
                print(f"\n[LONG ПОЗИЦИИ]")
                print(f"  Win Rate: {long_win_rate:.1f}%")
                print(f"  Средний PnL: {long_trades['pnl'].mean():.2f}")
            
            if len(short_trades) > 0:
                short_win_rate = (short_trades['pnl'] > 0).sum() / len(short_trades) * 100
                print(f"\n[SHORT ПОЗИЦИИ]")
                print(f"  Win Rate: {short_win_rate:.1f}%")
                print(f"  Средний PnL: {short_trades['pnl'].mean():.2f}")
            
            # Длительность удержания
            trades_df['holding_hours'] = pd.to_timedelta(trades_df['holding_period']).dt.total_seconds() / 3600
            print(f"\n[ДЛИТЕЛЬНОСТЬ УДЕРЖАНИЯ]")
            print(f"  Средняя: {trades_df['holding_hours'].mean():.2f} часов")
            print(f"  Медианная: {trades_df['holding_hours'].median():.2f} часов")
            print(f"  Минимальная: {trades_df['holding_hours'].min():.2f} часов")
            print(f"  Максимальная: {trades_df['holding_hours'].max():.2f} часов")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Детальный анализ результатов ГА")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Путь к файлу результатов (по умолчанию: results/ga_best.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Префикс для выходных файлов (опционально)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Путь к файлу данных (по умолчанию: используется путь из конфига)"
    )
    
    args = parser.parse_args()
    
    # Определяем путь к файлу результатов
    if args.input:
        results_path = Path(args.input)
        if not results_path.is_absolute():
            results_path = Path(__file__).parent.parent / results_path
    else:
        results_path = Path(__file__).parent.parent / "results" / "ga_best.json"
    
    if not results_path.exists():
        print(f"[ERROR] Файл результатов не найден: {results_path}")
        return
    
    results = load_ga_results(str(results_path))
    
    print("=" * 80)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
    print("=" * 80)
    
    fitness = results.get('fitness')
    if fitness is not None:
        if fitness == float('inf'):
            print(f"\nFitness: Infinity (проблема!)")
        elif fitness == -float('inf'):
            print(f"\nFitness: -Infinity (решение не прошло constraints)")
        else:
            print(f"\nFitness: {fitness:.2f}")
    else:
        print(f"\nFitness: не найден")
    
    # Анализ параметров
    analyze_parameters(results['genes'])
    
    # Сравнение метрик
    analyze_metrics_comparison(results['metrics'])
    
    # Детальный анализ сделок
    try:
        analyze_trades_by_split(str(results_path), args.data_path)
    except Exception as e:
        print(f"\n[ERROR] Ошибка при анализе сделок: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 80)


if __name__ == "__main__":
    main()

