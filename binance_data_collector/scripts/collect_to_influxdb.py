"""
Скрипт для сбора данных и записи в InfluxDB для визуализации в Grafana
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

from binance_data_collector.config import config
from binance_data_collector.utils.logger import setup_logger
from binance_data_collector.utils.influxdb_client import InfluxDBWriter
from binance_data_collector.scripts.collect_advanced_btc_data import AdvancedBTCDataCollector

try:
    from binance.client import Client
except ImportError:
    print("❌ Не установлена библиотека python-binance!")
    print("   Установите: pip install python-binance")
    sys.exit(1)

logger = setup_logger("collect_to_influxdb")

class InfluxDBDataCollector:
    """Коллектор данных с записью в InfluxDB"""
    
    def __init__(
        self,
        influxdb_url: str = "http://localhost:8086",
        influxdb_token: str = "",
        influxdb_org: str = "crypto",
        influxdb_bucket: str = "binance_data"
    ):
        self.collector = AdvancedBTCDataCollector()
        self.influxdb = InfluxDBWriter(
            url=influxdb_url,
            token=influxdb_token,
            org=influxdb_org,
            bucket=influxdb_bucket
        )
    
    def collect_and_store_historical(
        self,
        start_date: datetime,
        end_date: datetime,
        batch_days: int = 30
    ):
        """
        Сбор исторических данных с записью в InfluxDB
        
        Args:
            start_date: Начальная дата
            end_date: Конечная дата
            batch_days: Количество дней в одной партии (для избежания перегрузки)
        """
        logger.info(f"🚀 Начало сбора и записи данных в InfluxDB")
        logger.info(f"📅 Период: {start_date.date()} - {end_date.date()}")
        
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=batch_days), end_date)
            
            logger.info(f"\n📊 Сбор данных за период: {current_start.date()} - {current_end.date()}")
            
            # Собираем данные
            df = self.collector.collect_historical_data(
                start_date=current_start,
                end_date=current_end,
                sample_orderbook=False
            )
            
            if not df.empty:
                # Записываем в InfluxDB
                logger.info(f"💾 Запись {len(df)} записей в InfluxDB...")
                self.influxdb.write_advanced_data(df, symbol="BTCUSDT")
                logger.info(f"✅ Данные записаны")
            else:
                logger.warning(f"⚠️ Нет данных за период {current_start.date()} - {current_end.date()}")
            
            current_start = current_end
            
            # Пауза между батчами
            time.sleep(2)
        
        logger.info("\n✅ Сбор и запись данных завершены!")
    
    def collect_and_store_realtime(self):
        """
        Сбор данных в реальном времени с записью в InfluxDB
        """
        from binance_data_collector.scripts.collect_realtime_advanced_btc import RealtimeAdvancedBTCCollector
        
        class RealtimeCollectorWithInfluxDB(RealtimeAdvancedBTCCollector):
            """Расширенный коллектор с записью в InfluxDB"""
            
            def __init__(self, influxdb_writer):
                super().__init__()
                self.influxdb_writer = influxdb_writer
            
            def aggregate_15m_interval(self):
                """Агрегация с записью в InfluxDB"""
                interval_data = super().aggregate_15m_interval()
                
                if isinstance(interval_data, pd.Series) and self.influxdb_writer:
                    # Преобразуем Series в DataFrame для записи
                    df = pd.DataFrame([interval_data.to_dict()])
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    
                    self.influxdb_writer.write_advanced_data(df, symbol="BTCUSDT")
                    logger.info(f"💾 Данные записаны в InfluxDB: {interval_data.get('timestamp', 'N/A')}")
                
                return interval_data
        
        collector = RealtimeCollectorWithInfluxDB(self.influxdb)
        collector.start()

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Сбор данных и запись в InfluxDB")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["historical", "realtime"],
        default="realtime",
        help="Режим сбора: historical или realtime"
    )
    parser.add_argument(
        "--influxdb-url",
        type=str,
        default="http://localhost:8086",
        help="URL InfluxDB сервера"
    )
    parser.add_argument(
        "--influxdb-token",
        type=str,
        default="",
        help="Токен InfluxDB (по умолчанию пустой для разработки)"
    )
    parser.add_argument(
        "--influxdb-org",
        type=str,
        default="crypto",
        help="Организация InfluxDB"
    )
    parser.add_argument(
        "--influxdb-bucket",
        type=str,
        default="binance_data",
        help="Бакет InfluxDB"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Начальная дата для исторических данных (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Конечная дата для исторических данных (YYYY-MM-DD)"
    )
    
    args = parser.parse_args()
    
    collector = InfluxDBDataCollector(
        influxdb_url=args.influxdb_url,
        influxdb_token=args.influxdb_token,
        influxdb_org=args.influxdb_org,
        influxdb_bucket=args.influxdb_bucket
    )
    
    if args.mode == "historical":
        if args.start_date and args.end_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        else:
            # По умолчанию: последний месяц
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
        
        collector.collect_and_store_historical(start_date, end_date)
    else:
        collector.collect_and_store_realtime()

if __name__ == "__main__":
    main()

