"""
Анализ сделок на тестовом сплите для выявления работающих паттернов.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Добавляем src в путь
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import DataLoader
from analysis.indicator_pipeline import IndicatorPipeline
from strategies.mvp_strategy import MVPStrategy, MVPSignalConfig
from core.simple_backtester import SimpleBacktester, BacktestConfig


def load_best_genome(results_path: str) -> dict:
    """Загружает лучший геном из результатов ГА."""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['genes']


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
    
    return cfg


def enrich_trades_with_indicators(trades, data: pd.DataFrame, signals: pd.DataFrame, rsi_period: int):
    """Обогащает сделки информацией об индикаторах на момент входа."""
    enriched_trades = []
    
    # Находим правильные имена колонок
    rsi_col = f'rsi_{rsi_period}'
    atr_col = 'atr'  # ATR может быть без периода в названии
    
    for trade in trades:
        entry_idx = data.index.get_loc(trade.entry_time)
        entry_row = data.iloc[entry_idx]
        entry_signals = signals.iloc[entry_idx]
        
        # Собираем информацию об индикаторах
        indicator_values = {
            'rsi': entry_row.get(rsi_col, np.nan),
            'macd': entry_row.get('macd', np.nan),
            'macd_signal': entry_row.get('macd_signal', np.nan),
            'macd_hist': entry_row.get('macd_hist', np.nan),
            'bb_upper': entry_row.get('bb_upper', np.nan),
            'bb_middle': entry_row.get('bb_middle', np.nan),
            'bb_lower': entry_row.get('bb_lower', np.nan),
            'wobi': entry_row.get('wobi', np.nan),
            'wobi_zscore': entry_row.get('wobi_zscore', np.nan),
            'atr': entry_row.get(atr_col, np.nan),
        }
        
        # Голоса индикаторов
        votes = {
            'rsi_vote': entry_signals.get('rsi_vote', 0),
            'macd_vote': entry_signals.get('macd_vote', 0),
            'bollinger_vote': entry_signals.get('bollinger_vote', 0),
            'wobi_vote': entry_signals.get('wobi_vote', 0),
        }
        
        # Combined signal
        combined_signal = entry_signals.get('combined_signal', 0)
        
        # Ценовые данные
        price_data = {
            'open': entry_row.get('open', np.nan),
            'high': entry_row.get('high', np.nan),
            'low': entry_row.get('low', np.nan),
            'close': entry_row.get('close', np.nan),
            'volume': entry_row.get('volume', np.nan),
        }
        
        # Создаём обогащённую запись сделки
        enriched_trade = {
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'position_type': trade.position_type,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'holding_period': trade.holding_period,
            'is_winning': trade.pnl > 0,
            **indicator_values,
            **votes,
            'combined_signal': combined_signal,
            **price_data,
        }
        
        enriched_trades.append(enriched_trade)
    
    return pd.DataFrame(enriched_trades)


def analyze_patterns(trades_df: pd.DataFrame):
    """Анализирует паттерны в прибыльных и убыточных сделках."""
    print("\n" + "="*80)
    print("АНАЛИЗ ПАТТЕРНОВ В СДЕЛКАХ")
    print("="*80)
    
    winning = trades_df[trades_df['is_winning'] == True]
    losing = trades_df[trades_df['is_winning'] == False]
    
    print(f"\nВсего сделок: {len(trades_df)}")
    print(f"Прибыльных: {len(winning)} ({len(winning)/len(trades_df)*100:.1f}%)")
    print(f"Убыточных: {len(losing)} ({len(losing)/len(trades_df)*100:.1f}%)")
    
    # Анализ индикаторов
    print("\n" + "-"*80)
    print("СРЕДНИЕ ЗНАЧЕНИЯ ИНДИКАТОРОВ НА ВХОДЕ")
    print("-"*80)
    
    indicators = ['rsi', 'macd', 'macd_hist', 'wobi_zscore', 'atr']
    for ind in indicators:
        if ind in trades_df.columns:
            win_mean = winning[ind].mean() if len(winning) > 0 else np.nan
            lose_mean = losing[ind].mean() if len(losing) > 0 else np.nan
            win_std = winning[ind].std() if len(winning) > 0 else np.nan
            lose_std = losing[ind].std() if len(losing) > 0 else np.nan
            if not np.isnan(win_mean) and not np.isnan(lose_mean):
                diff = win_mean - lose_mean
                print(f"{ind:15s} | Прибыльные: {win_mean:8.2f} ± {win_std:6.2f} | Убыточные: {lose_mean:8.2f} ± {lose_std:6.2f} | Разница: {diff:8.2f}")
            else:
                print(f"{ind:15s} | Прибыльные: {win_mean:8.2f} | Убыточные: {lose_mean:8.2f}")
    
    # Анализ голосов
    print("\n" + "-"*80)
    print("СРЕДНИЕ ГОЛОСА ИНДИКАТОРОВ")
    print("-"*80)
    
    votes = ['rsi_vote', 'macd_vote', 'bollinger_vote', 'wobi_vote']
    for vote in votes:
        if vote in trades_df.columns:
            win_mean = winning[vote].mean() if len(winning) > 0 else np.nan
            lose_mean = losing[vote].mean() if len(losing) > 0 else np.nan
            print(f"{vote:15s} | Прибыльные: {win_mean:8.2f} | Убыточные: {lose_mean:8.2f}")
    
    # Анализ combined_signal
    print("\n" + "-"*80)
    print("COMBINED SIGNAL")
    print("-"*80)
    win_signal = winning['combined_signal'].mean() if len(winning) > 0 else np.nan
    lose_signal = losing['combined_signal'].mean() if len(losing) > 0 else np.nan
    print(f"Прибыльные: {win_signal:.3f} | Убыточные: {lose_signal:.3f}")
    
    # Анализ по типам позиций
    print("\n" + "-"*80)
    print("АНАЛИЗ ПО ТИПАМ ПОЗИЦИЙ")
    print("-"*80)
    for pos_type in ['long', 'short']:
        pos_trades = trades_df[trades_df['position_type'] == pos_type]
        if len(pos_trades) > 0:
            win_rate = (pos_trades['is_winning'].sum() / len(pos_trades)) * 100
            avg_pnl = pos_trades['pnl'].mean()
            print(f"{pos_type:5s}: {len(pos_trades):3d} сделок, Win Rate: {win_rate:5.1f}%, Avg PnL: {avg_pnl:8.2f}")
    
    # Топ-5 прибыльных сделок
    print("\n" + "-"*80)
    print("ТОП-5 ПРИБЫЛЬНЫХ СДЕЛОК")
    print("-"*80)
    top_winners = winning.nlargest(5, 'pnl')[['entry_time', 'position_type', 'pnl', 'pnl_pct', 'rsi', 'macd_hist', 'wobi_zscore', 'combined_signal']]
    print(top_winners.to_string())
    
    # Топ-5 убыточных сделок
    print("\n" + "-"*80)
    print("ТОП-5 УБЫТОЧНЫХ СДЕЛОК")
    print("-"*80)
    top_losers = losing.nsmallest(5, 'pnl')[['entry_time', 'position_type', 'pnl', 'pnl_pct', 'rsi', 'macd_hist', 'wobi_zscore', 'combined_signal']]
    print(top_losers.to_string())
    
    # Анализ по времени суток
    print("\n" + "-"*80)
    print("АНАЛИЗ ПО ЧАСАМ ДНЯ")
    print("-"*80)
    trades_df['hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
    for hour in sorted(trades_df['hour'].unique()):
        hour_trades = trades_df[trades_df['hour'] == hour]
        if len(hour_trades) > 0:
            win_rate = (hour_trades['is_winning'].sum() / len(hour_trades)) * 100
            avg_pnl = hour_trades['pnl'].mean()
            print(f"Час {hour:2d}:00 | Сделок: {len(hour_trades):2d} | Win Rate: {win_rate:5.1f}% | Avg PnL: {avg_pnl:8.2f}")
    
    # Анализ по дням недели
    print("\n" + "-"*80)
    print("АНАЛИЗ ПО ДНЯМ НЕДЕЛИ")
    print("-"*80)
    trades_df['weekday'] = pd.to_datetime(trades_df['entry_time']).dt.day_name()
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        day_trades = trades_df[trades_df['weekday'] == day]
        if len(day_trades) > 0:
            win_rate = (day_trades['is_winning'].sum() / len(day_trades)) * 100
            avg_pnl = day_trades['pnl'].mean()
            print(f"{day:10s} | Сделок: {len(day_trades):2d} | Win Rate: {win_rate:5.1f}% | Avg PnL: {avg_pnl:8.2f}")
    
    # Анализ по длительности удержания
    print("\n" + "-"*80)
    print("АНАЛИЗ ПО ДЛИТЕЛЬНОСТИ УДЕРЖАНИЯ")
    print("-"*80)
    trades_df['holding_hours'] = pd.to_timedelta(trades_df['holding_period']).dt.total_seconds() / 3600
    short_trades = trades_df[trades_df['holding_hours'] < 2]
    medium_trades = trades_df[(trades_df['holding_hours'] >= 2) & (trades_df['holding_hours'] < 6)]
    long_trades = trades_df[trades_df['holding_hours'] >= 6]
    
    for name, trades in [('Короткие (<2ч)', short_trades), ('Средние (2-6ч)', medium_trades), ('Долгие (>6ч)', long_trades)]:
        if len(trades) > 0:
            win_rate = (trades['is_winning'].sum() / len(trades)) * 100
            avg_pnl = trades['pnl'].mean()
            print(f"{name:15s} | Сделок: {len(trades):2d} | Win Rate: {win_rate:5.1f}% | Avg PnL: {avg_pnl:8.2f}")
    
    return {
        'winning': winning,
        'losing': losing,
        'summary': {
            'total_trades': len(trades_df),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            'avg_winning_pnl': winning['pnl'].mean() if len(winning) > 0 else 0,
            'avg_losing_pnl': losing['pnl'].mean() if len(losing) > 0 else 0,
        }
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Анализ сделок на тестовом сплите')
    parser.add_argument('--results', type=str, default='results/ga_optimized_full.json',
                       help='Путь к файлу с результатами ГА')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Путь к данным (по умолчанию из DataLoader)')
    parser.add_argument('--output', type=str, default='results/test_trades_analysis.csv',
                       help='Путь для сохранения обогащённых сделок')
    parser.add_argument('--config', type=str, default='config/mvp_strategy_config.yaml',
                       help='Путь к базовому конфигу стратегии')
    
    args = parser.parse_args()
    
    # Загружаем лучший геном
    print(f"[INFO] Загрузка лучшего генома из {args.results}...")
    genes = load_best_genome(args.results)
    print(f"[OK] Загружен геном: {genes}")
    
    # Загружаем данные
    print(f"\n[INFO] Загрузка данных...")
    loader = DataLoader(data_path=args.data_path)
    raw_data = loader.load_data()
    train_df, val_df, test_df = loader.split_data()
    print(f"[OK] Загружено {len(raw_data)} строк")
    print(f"      Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Вычисляем индикаторы
    print(f"\n[INFO] Вычисление индикаторов...")
    indicator_params = {
        "rsi": {"period": genes["rsi_period"]},
        "atr": {"period": genes["atr_period"]},
    }
    pipeline = IndicatorPipeline(raw_data, params=indicator_params)
    artifacts = pipeline.run()
    enriched_data = artifacts.data.dropna()
    print(f"[OK] Индикаторы вычислены")
    
    # Подготавливаем тестовый сплит
    test_enriched = enriched_data.reindex(test_df.index).dropna()
    print(f"\n[INFO] Тестовый сплит: {len(test_enriched)} строк")
    
    # Строим стратегию
    print(f"\n[INFO] Построение стратегии...")
    strategy_config = build_strategy_config(genes, args.config)
    signal_config = MVPSignalConfig.from_dict(strategy_config)
    strategy = MVPStrategy(signal_config)
    
    # Генерируем сигналы
    print(f"[INFO] Генерация сигналов...")
    signals = strategy.generate_signals(test_enriched)
    print(f"[OK] Сигналы сгенерированы")
    
    # Запускаем бэктест
    print(f"\n[INFO] Запуск бэктеста на тестовом сплите...")
    backtest_config = BacktestConfig.from_dict(strategy_config.get("risk"))
    backtester = SimpleBacktester(backtest_config)
    result = backtester.run(test_enriched, signals)
    print(f"[OK] Бэктест завершён: {len(result.trades)} сделок")
    
    # Обогащаем сделки индикаторами
    print(f"\n[INFO] Обогащение сделок информацией об индикаторах...")
    enriched_trades = enrich_trades_with_indicators(result.trades, test_enriched, signals, genes['rsi_period'])
    print(f"[OK] Сделки обогащены")
    
    # Сохраняем
    enriched_trades.to_csv(args.output, index=False)
    print(f"\n[OK] Сделки сохранены в {args.output}")
    
    # Анализируем паттерны
    analysis = analyze_patterns(enriched_trades)
    
    # Сохраняем сводку
    summary_path = args.output.replace('.csv', '_summary.json')
    import json
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(analysis['summary'], f, indent=2, default=str)
    print(f"\n[OK] Сводка сохранена в {summary_path}")
    
    print("\n" + "="*80)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("="*80)


if __name__ == '__main__':
    main()

