"""
Скрипт для загрузки объединенных CSV данных (OHLCV + Trades) в InfluxDB
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Добавляем путь к проекту
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from binance_data_collector.utils.influxdb_client import InfluxDBWriter
from binance_data_collector.utils.logger import setup_logger

logger = setup_logger("load_combined_to_influxdb")

def load_combined_to_influxdb(
    csv_file: str,
    symbol: str = "BTCUSDT",
    timeframe: str = "15m",
    influxdb_url: str = "http://localhost:8086",
    influxdb_token: str = "my-super-secret-admin-token",
    org: str = "crypto",
    bucket: str = "binance_data"
):
    """
    Загрузка объединенных CSV данных (OHLCV + Trades) в InfluxDB
    
    Args:
        csv_file: Путь к CSV файлу
        symbol: Торговая пара
        timeframe: Таймфрейм данных
        influxdb_url: URL InfluxDB сервера
        influxdb_token: Токен доступа
        org: Организация
        bucket: Бакет (база данных)
    """
    logger.info("="*70)
    logger.info("📊 ЗАГРУЗКА ОБЪЕДИНЕННЫХ ДАННЫХ В INFLUXDB")
    logger.info("="*70)
    logger.info(f"📁 Файл: {csv_file}")
    logger.info(f"📊 Символ: {symbol}")
    logger.info(f"⏱️  Таймфрейм: {timeframe}")
    logger.info(f"🔗 InfluxDB: {influxdb_url}")
    logger.info("="*70)
    
    # Проверка существования файла
    csv_path = Path(csv_file)
    if not csv_path.exists():
        logger.error(f"❌ Файл не найден: {csv_file}")
        return False
    
    # Загрузка CSV
    logger.info(f"\n📖 Загрузка CSV файла...")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"✅ Загружено {len(df)} строк")
        logger.info(f"📋 Колонки: {list(df.columns)}")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки CSV: {e}")
        return False
    
    # Обработка временной метки
    logger.info(f"\n🕐 Обработка временных меток...")
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        logger.info(f"✅ Использована колонка 'time'")
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        logger.info(f"✅ Использована колонка 'timestamp'")
    else:
        logger.error(f"❌ Не найдена колонка с временной меткой!")
        return False
    
    # Сортировка по времени
    df.sort_index(inplace=True)
    
    logger.info(f"📅 Период данных: {df.index[0]} - {df.index[-1]}")
    logger.info(f"📊 Всего записей: {len(df)}")
    
    # Подключение к InfluxDB
    logger.info(f"\n🔗 Подключение к InfluxDB...")
    writer = InfluxDBWriter(
        url=influxdb_url,
        token=influxdb_token,
        org=org,
        bucket=bucket
    )
    
    if writer.client is None:
        logger.error("❌ Не удалось подключиться к InfluxDB!")
        return False
    
    # Запись данных
    logger.info(f"\n💾 Запись данных в InfluxDB...")
    try:
        from influxdb_client import Point
        
        chunk_size = 10000
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk_num = (i // chunk_size) + 1
            
            logger.info(f"   📦 Записываю чанк {chunk_num}/{total_chunks} ({len(chunk)} записей)...")
            
            points = []
            for timestamp, row in chunk.iterrows():
                point = Point("btc_combined") \
                    .tag("symbol", symbol) \
                    .tag("timeframe", timeframe) \
                    .time(pd.Timestamp(timestamp))
                
                # OHLCV поля
                if 'open' in row and pd.notna(row['open']):
                    point = point.field("open", float(row['open']))
                if 'high' in row and pd.notna(row['high']):
                    point = point.field("high", float(row['high']))
                if 'low' in row and pd.notna(row['low']):
                    point = point.field("low", float(row['low']))
                if 'close' in row and pd.notna(row['close']):
                    point = point.field("close", float(row['close']))
                if 'volume' in row and pd.notna(row['volume']):
                    point = point.field("volume", float(row['volume']))
                if 'quote_volume' in row and pd.notna(row['quote_volume']):
                    point = point.field("quote_volume", float(row['quote_volume']))
                if 'taker_buy_base' in row and pd.notna(row['taker_buy_base']):
                    point = point.field("taker_buy_base", float(row['taker_buy_base']))
                if 'taker_buy_quote' in row and pd.notna(row['taker_buy_quote']):
                    point = point.field("taker_buy_quote", float(row['taker_buy_quote']))
                
                # Trades поля
                if 'trades_buy_volume' in row and pd.notna(row['trades_buy_volume']):
                    point = point.field("trades_buy_volume", float(row['trades_buy_volume']))
                if 'trades_sell_volume' in row and pd.notna(row['trades_sell_volume']):
                    point = point.field("trades_sell_volume", float(row['trades_sell_volume']))
                if 'trades_total_volume' in row and pd.notna(row['trades_total_volume']):
                    point = point.field("trades_total_volume", float(row['trades_total_volume']))
                if 'trades_count' in row and pd.notna(row['trades_count']):
                    point = point.field("trades_count", int(row['trades_count']))
                
                points.append(point)
            
            writer.write_api.write(bucket=bucket, record=points)
            logger.info(f"   ✅ Чанк {chunk_num} записан")
        
        logger.info(f"\n✅ Все данные успешно записаны в InfluxDB!")
        logger.info(f"📊 Всего записано: {len(df)} записей")
        logger.info(f"📅 Период: {df.index[0]} - {df.index[-1]}")
        logger.info(f"\n🎨 Теперь можно открыть Grafana: http://localhost:3001")
        logger.info(f"   И использовать новый дашборд для визуализации")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка записи в InfluxDB: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        writer.close()


if __name__ == "__main__":
    csv_file = r"C:\Users\XE\Desktop\CRYPTO_2025\binance_data_collector\BTCUSDT_15m_COMBINED.csv"
    
    success = load_combined_to_influxdb(
        csv_file=csv_file,
        symbol="BTCUSDT",
        timeframe="15m"
    )
    
    sys.exit(0 if success else 1)

