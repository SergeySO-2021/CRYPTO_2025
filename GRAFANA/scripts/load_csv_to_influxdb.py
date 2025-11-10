"""
Скрипт для загрузки CSV данных в InfluxDB для визуализации в Grafana
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import argparse

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from binance_data_collector.utils.influxdb_client import InfluxDBWriter
from binance_data_collector.utils.logger import setup_logger

logger = setup_logger("load_csv_to_influxdb")

def load_csv_to_influxdb(
    csv_file: str,
    symbol: str = "BTCUSDT",
    timeframe: str = "15m",
    influxdb_url: str = "http://localhost:8086",
    influxdb_token: str = "my-super-secret-admin-token",
    org: str = "crypto",
    bucket: str = "binance_data"
):
    """
    Загрузка CSV файла в InfluxDB
    
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
    logger.info("📊 ЗАГРУЗКА CSV ДАННЫХ В INFLUXDB")
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
    
    # Проверка необходимых колонок
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"❌ Отсутствуют обязательные колонки: {missing_columns}")
        return False
    
    # Обработка временной метки
    logger.info(f"\n🕐 Обработка временных меток...")
    if 'timestamps' in df.columns:
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df.set_index('timestamps', inplace=True)
        logger.info(f"✅ Использована колонка 'timestamps'")
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        logger.info(f"✅ Использована колонка 'timestamp'")
    else:
        logger.warning(f"⚠️ Не найдена колонка с временной меткой, создаю индекс...")
        df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='15min')
    
    # Сортировка по времени
    df.sort_index(inplace=True)
    
    logger.info(f"📅 Период данных: {df.index[0]} - {df.index[-1]}")
    logger.info(f"📊 Всего записей: {len(df)}")
    
    # Проверка на дубликаты
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"⚠️ Найдено {duplicates} дубликатов, удаляю...")
        df = df[~df.index.duplicated(keep='first')]
        logger.info(f"✅ После удаления дубликатов: {len(df)} записей")
    
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
        logger.error("   Убедитесь, что:")
        logger.error("   1. InfluxDB запущен (docker-compose up -d)")
        logger.error("   2. Указан правильный токен")
        logger.error("   3. Указаны правильные org и bucket")
        return False
    
    # Запись данных
    logger.info(f"\n💾 Запись данных в InfluxDB...")
    try:
        # Обработка больших файлов по частям
        chunk_size = 10000
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk_num = (i // chunk_size) + 1
            
            logger.info(f"   📦 Записываю чанк {chunk_num}/{total_chunks} ({len(chunk)} записей)...")
            
            writer.write_ohlcv(chunk, symbol=symbol, timeframe=timeframe)
            
            logger.info(f"   ✅ Чанк {chunk_num} записан")
        
        logger.info(f"\n✅ Все данные успешно записаны в InfluxDB!")
        logger.info(f"📊 Всего записано: {len(df)} записей")
        logger.info(f"📅 Период: {df.index[0]} - {df.index[-1]}")
        logger.info(f"\n🎨 Теперь можно открыть Grafana: http://localhost:3000")
        logger.info(f"   И использовать запросы для визуализации данных")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка записи в InfluxDB: {e}")
        return False
    finally:
        writer.close()


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Загрузка CSV данных в InfluxDB')
    parser.add_argument('--csv-file', type=str, required=True, help='Путь к CSV файлу')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Торговая пара (по умолчанию: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='15m', help='Таймфрейм (по умолчанию: 15m)')
    parser.add_argument('--influxdb-url', type=str, default='http://localhost:8086', help='URL InfluxDB')
    parser.add_argument('--influxdb-token', type=str, default='my-super-secret-admin-token', help='Токен InfluxDB')
    parser.add_argument('--org', type=str, default='crypto', help='Организация')
    parser.add_argument('--bucket', type=str, default='binance_data', help='Бакет')
    
    args = parser.parse_args()
    
    success = load_csv_to_influxdb(
        csv_file=args.csv_file,
        symbol=args.symbol,
        timeframe=args.timeframe,
        influxdb_url=args.influxdb_url,
        influxdb_token=args.influxdb_token,
        org=args.org,
        bucket=args.bucket
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

