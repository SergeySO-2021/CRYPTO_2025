"""
Пример использования комплексного сборщика данных Binance

Этот скрипт демонстрирует, как использовать ComprehensiveBinanceCollector
для сбора всех доступных данных и работы с ними
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent.parent))

from binance_data_collector.scripts.collect_comprehensive_data import ComprehensiveBinanceCollector
from binance_data_collector.utils.file_handler import save_data
from binance_data_collector.config import config


def example_basic_collection():
    """Пример базового сбора данных"""
    print("="*70)
    print("Пример 1: Базовый сбор данных")
    print("="*70)
    
    # Создаем коллектор
    collector = ComprehensiveBinanceCollector(symbol="BTCUSDT")
    
    # Собираем данные за последний месяц
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"\n📊 Сбор данных с {start_date.date()} по {end_date.date()}")
    
    df = collector.collect_comprehensive_data(
        start_date=start_date,
        end_date=end_date,
        interval="15m",
        target_interval="1h"
    )
    
    if not df.empty:
        print(f"\n✅ Данные собраны!")
        print(f"📊 Записей: {len(df)}")
        print(f"📅 Период: {df.index[0]} - {df.index[-1]}")
        print(f"\n📋 Первые 5 строк:")
        print(df.head())
        
        # Сохраняем данные
        filename = f"BTCUSDT_example_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        filepath = config.DATA_DIR / "historical" / filename
        save_data(df, filepath, format="csv")
        
        return df
    else:
        print("❌ Данные не собраны")
        return pd.DataFrame()


def example_multiple_symbols():
    """Пример сбора данных для нескольких символов"""
    print("\n" + "="*70)
    print("Пример 2: Сбор данных для нескольких символов")
    print("="*70)
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    all_data = {}
    
    for symbol in symbols:
        print(f"\n📊 Сбор данных для {symbol}...")
        
        collector = ComprehensiveBinanceCollector(symbol=symbol)
        
        # Собираем данные за последние 7 дней
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        df = collector.collect_comprehensive_data(
            start_date=start_date,
            end_date=end_date,
            interval="1h",
            target_interval="1h"
        )
        
        if not df.empty:
            all_data[symbol] = df
            print(f"✅ {symbol}: {len(df)} записей")
    
    return all_data


def example_data_analysis(df: pd.DataFrame):
    """Пример анализа собранных данных"""
    print("\n" + "="*70)
    print("Пример 3: Анализ данных")
    print("="*70)
    
    if df.empty:
        print("❌ Нет данных для анализа")
        return
    
    # Базовая статистика
    print("\n📊 Базовая статистика по цене:")
    print(df[['open', 'high', 'low', 'close', 'volume']].describe())
    
    # Вычисляемые метрики
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(24).std()
    
    print("\n📈 Среднее изменение цены:", df['price_change'].mean() * 100, "%")
    print("📊 Средняя волатильность:", df['volatility'].mean())
    
    # Анализ объемов
    if 'market_buy_volume' in df.columns and 'market_sell_volume' in df.columns:
        df['buy_sell_imbalance'] = (df['market_buy_volume'] - df['market_sell_volume']) / \
                                   (df['market_buy_volume'] + df['market_sell_volume'])
        print("\n📊 Средний дисбаланс покупок/продаж:", df['buy_sell_imbalance'].mean())
    
    # Анализ ликвидаций
    if 'total_liquidations' in df.columns:
        total_liq = df['total_liquidations'].sum()
        print(f"\n💥 Всего ликвидаций за период: {total_liq:.2f} BTC")
        if 'long_liquidations' in df.columns and 'short_liquidations' in df.columns:
            long_liq = df['long_liquidations'].sum()
            short_liq = df['short_liquidations'].sum()
            print(f"   Long: {long_liq:.2f} BTC ({long_liq/total_liq*100:.1f}%)")
            print(f"   Short: {short_liq:.2f} BTC ({short_liq/total_liq*100:.1f}%)")
    
    # Корреляция
    numeric_cols = ['close', 'volume', 'total_liquidations', 'open_interest']
    available_cols = [col for col in numeric_cols if col in df.columns]
    if len(available_cols) > 1:
        print("\n📊 Корреляционная матрица:")
        corr = df[available_cols].corr()
        print(corr)


def example_custom_period():
    """Пример сбора данных за произвольный период"""
    print("\n" + "="*70)
    print("Пример 4: Сбор данных за произвольный период")
    print("="*70)
    
    collector = ComprehensiveBinanceCollector(symbol="BTCUSDT")
    
    # Собираем данные за конкретный месяц
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    
    print(f"\n📊 Сбор данных с {start_date.date()} по {end_date.date()}")
    
    df = collector.collect_comprehensive_data(
        start_date=start_date,
        end_date=end_date,
        interval="5m",      # Детальный интервал
        target_interval="15m" # Финальная агрегация
    )
    
    if not df.empty:
        print(f"✅ Собрано {len(df)} записей")
        return df
    
    return pd.DataFrame()


def main():
    """Основная функция с примерами"""
    print("\n" + "="*70)
    print("🚀 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ COMPREHENSIVE BINANCE COLLECTOR")
    print("="*70)
    
    # Пример 1: Базовый сбор
    df = example_basic_collection()
    
    # Пример 2: Несколько символов
    # all_data = example_multiple_symbols()
    
    # Пример 3: Анализ данных
    if not df.empty:
        example_data_analysis(df)
    
    # Пример 4: Произвольный период
    # custom_df = example_custom_period()
    
    print("\n" + "="*70)
    print("✅ Примеры выполнены!")
    print("="*70)
    
    print("\n💡 Совет: Измените функции в main() для запуска других примеров")


if __name__ == "__main__":
    main()

