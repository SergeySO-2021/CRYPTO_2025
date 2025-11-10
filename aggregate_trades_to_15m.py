import pickle
import pandas as pd
import sys
import io
from pathlib import Path

# Устанавливаем кодировку для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Пути
pkl_path = r"C:\Users\XE\Desktop\CRYPTO_2025\binance_data_collector\BTCUSDT_15m_20251104_011229.pkl"
output_dir = Path(r"C:\Users\XE\Desktop\CRYPTO_2025\binance_data_collector")

print("=" * 80)
print("АГРЕГАЦИЯ TRADES ДАННЫХ В 15-МИНУТНЫЕ ИНТЕРВАЛЫ")
print("=" * 80)

# Загружаем данные
print("\n📥 Загрузка данных...")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

df_trades = data['trades_df']
df_ohlcv = data['ohlcv_df']

print(f"✅ Загружено:")
print(f"   - OHLCV: {len(df_ohlcv):,} строк (15-мин интервалы)")
print(f"   - Trades: {len(df_trades):,} строк (отдельные сделки)")

print(f"\n📅 Период OHLCV: {df_ohlcv.index.min()} → {df_ohlcv.index.max()}")
print(f"📅 Период Trades: {df_trades.index.min()} → {df_trades.index.max()}")

# Агрегируем trades в 15-минутные интервалы
print(f"\n📊 Агрегация trades в 15-минутные интервалы...")
print(f"   Это может занять некоторое время для {len(df_trades):,} сделок...")

# Агрегируем по 15-минутным интервалам
trades_15m = df_trades.resample('15min').agg({
    'buy_volume': 'sum',      # Сумма объемов покупок
    'sell_volume': 'sum',     # Сумма объемов продаж
    'quantity': 'sum',        # Общая сумма количеств
    'price': 'mean'           # Средняя цена за интервал
})

# Добавляем дополнительные метрики
trades_15m['total_volume'] = trades_15m['buy_volume'] + trades_15m['sell_volume']
trades_15m['buy_sell_ratio'] = trades_15m['buy_volume'] / trades_15m['sell_volume'].replace(0, 1)
trades_15m['trade_count'] = df_trades.resample('15min').size()  # Количество сделок в интервале

# Переименовываем колонки для ясности
trades_15m.columns = [
    'trades_buy_volume',
    'trades_sell_volume', 
    'trades_total_quantity',
    'trades_avg_price',
    'trades_total_volume',
    'trades_buy_sell_ratio',
    'trades_count'
]

print(f"✅ Агрегировано: {len(trades_15m):,} 15-минутных интервалов")

print(f"\n📊 Статистика агрегированных trades:")
print(trades_15m.describe())

# Объединяем с OHLCV датафреймом
print(f"\n🔗 Объединение с OHLCV данными...")

# Объединяем по индексу (timestamp)
df_combined = df_ohlcv.join(trades_15m, how='left')

print(f"✅ Объединенный датафрейм создан: {len(df_combined):,} строк")

# Проверяем, сколько строк имеют trades данные
trades_coverage = df_combined['trades_count'].notna().sum()
print(f"📊 Покрытие: {trades_coverage:,} из {len(df_combined):,} интервалов имеют trades данные")
print(f"   ({trades_coverage/len(df_combined)*100:.1f}% покрытие)")

# Показываем примеры
print(f"\n📋 Примеры объединенных данных:")
print("\nПервые 5 строк с trades данными:")
print(df_combined[df_combined['trades_count'].notna()].head())

print("\nПоследние 5 строк с trades данными:")
print(df_combined[df_combined['trades_count'].notna()].tail())

# Сохраняем результаты
print(f"\n💾 Сохранение результатов...")

# 1. Агрегированные trades (15-мин интервалы)
trades_15m_file = output_dir / "BTCUSDT_15m_TRADES_AGGREGATED.xlsx"
print(f"   1. Агрегированные trades: {trades_15m_file}")

with pd.ExcelWriter(trades_15m_file, engine='openpyxl') as writer:
    trades_15m.to_excel(writer, sheet_name='Trades_15m', index=True)

print(f"      ✅ Сохранено: {trades_15m_file.stat().st_size / 1024 / 1024:.2f} MB")

# 2. Объединенный датафрейм (OHLCV + агрегированные trades)
combined_file = output_dir / "BTCUSDT_15m_COMBINED.xlsx"
print(f"   2. Объединенный датафрейм: {combined_file}")

# Для Excel ограничим размер (первые 10000 строк для просмотра)
sample_size = min(10000, len(df_combined))
df_combined_sample = df_combined.head(sample_size)

with pd.ExcelWriter(combined_file, engine='openpyxl') as writer:
    df_combined_sample.to_excel(writer, sheet_name='OHLCV_Trades', index=True)
    # Также сохраняем полный датафрейм в отдельном листе (только структуру)
    df_combined.head(100).to_excel(writer, sheet_name='Sample_100', index=True)

print(f"      ✅ Сохранено (образец {sample_size:,} строк): {combined_file.stat().st_size / 1024 / 1024:.2f} MB")

# 3. Сохраняем полный объединенный датафрейм в pickle
combined_pkl_file = output_dir / "BTCUSDT_15m_COMBINED.pkl"
print(f"   3. Полный объединенный датафрейм (pickle): {combined_pkl_file}")

import pickle as pkl
with open(combined_pkl_file, 'wb') as f:
    pkl.dump({
        'combined_df': df_combined,
        'trades_15m_df': trades_15m,
        'ohlcv_df': df_ohlcv,
        'metadata': {
            'symbol': data.get('symbol', 'BTCUSDT'),
            'interval': '15m',
            'period_start': str(df_combined.index.min()),
            'period_end': str(df_combined.index.max()),
            'total_rows': len(df_combined),
            'trades_coverage': trades_coverage,
            'trades_coverage_percent': trades_coverage/len(df_combined)*100
        }
    }, f)

print(f"      ✅ Сохранено: {combined_pkl_file.stat().st_size / 1024 / 1024:.2f} MB")

print("\n" + "=" * 80)
print("✅ ВСЕ ДАННЫЕ СОХРАНЕНЫ!")
print("=" * 80)

print(f"\n📁 Созданные файлы:")
print(f"   1. BTCUSDT_15m_TRADES_AGGREGATED.xlsx")
print(f"      - Агрегированные trades в 15-мин интервалы")
print(f"      - {len(trades_15m):,} строк")
print(f"      - Колонки: {list(trades_15m.columns)}")
print(f"\n   2. BTCUSDT_15m_COMBINED.xlsx")
print(f"      - Объединенный датафрейм (OHLCV + агрегированные trades)")
print(f"      - Образец: {sample_size:,} строк")
print(f"      - Полный: {len(df_combined):,} строк")
print(f"\n   3. BTCUSDT_15m_COMBINED.pkl")
print(f"      - Полный объединенный датафрейм в pickle формате")
print(f"      - {len(df_combined):,} строк")
print(f"      - Все колонки: {list(df_combined.columns)}")

print(f"\n💡 Использование:")
print(f"   - Excel файлы для просмотра структуры")
print(f"   - Pickle файл для работы с полными данными в Python")

