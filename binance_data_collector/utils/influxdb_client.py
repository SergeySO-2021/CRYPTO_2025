"""
Клиент для записи данных в InfluxDB для визуализации в Grafana
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from binance_data_collector.config import config
from binance_data_collector.utils.logger import setup_logger

try:
    from influxdb_client import InfluxDBClient, Point, WriteOptions
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:
    print("⚠️ InfluxDB client не установлен. Установите: pip install influxdb-client")
    InfluxDBClient = None

logger = setup_logger("influxdb_client")

class InfluxDBWriter:
    """Класс для записи данных в InfluxDB"""
    
    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: str = "",
        org: str = "crypto",
        bucket: str = "binance_data"
    ):
        """
        Инициализация клиента InfluxDB
        
        Args:
            url: URL InfluxDB сервера
            token: Токен доступа (можно использовать admin/admin для разработки)
            org: Организация
            bucket: Бакет (база данных)
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.client = None
        self.write_api = None
        
        if InfluxDBClient is None:
            logger.error("❌ InfluxDB client не установлен!")
            return
        
        try:
            self.client = InfluxDBClient(url=url, token=token, org=org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            logger.info(f"✅ Подключение к InfluxDB установлено: {url}")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к InfluxDB: {e}")
    
    def write_ohlcv(self, df: pd.DataFrame, symbol: str = "BTCUSDT", timeframe: str = "15m"):
        """
        Запись OHLCV данных
        
        Args:
            df: DataFrame с колонками open, high, low, close, volume
            symbol: Торговая пара
            timeframe: Таймфрейм
        """
        if self.write_api is None:
            logger.error("❌ InfluxDB клиент не инициализирован!")
            return
        
        try:
            points = []
            
            for timestamp, row in df.iterrows():
                point = Point("ohlcv") \
                    .tag("symbol", symbol) \
                    .tag("timeframe", timeframe) \
                    .field("open", float(row['open'])) \
                    .field("high", float(row['high'])) \
                    .field("low", float(row['low'])) \
                    .field("close", float(row['close'])) \
                    .field("volume", float(row['volume'])) \
                    .time(pd.Timestamp(timestamp))
                
                points.append(point)
            
            self.write_api.write(bucket=self.bucket, record=points)
            logger.info(f"✅ Записано {len(points)} OHLCV записей в InfluxDB")
        
        except Exception as e:
            logger.error(f"❌ Ошибка записи OHLCV: {e}")
    
    def write_advanced_data(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT"
    ):
        """
        Запись расширенных данных BTC
        
        Args:
            df: DataFrame с расширенными данными
            symbol: Торговая пара
        """
        if self.write_api is None:
            logger.error("❌ InfluxDB клиент не инициализирован!")
            return
        
        try:
            points = []
            
            for timestamp, row in df.iterrows():
                # Основные данные
                point = Point("btc_advanced") \
                    .tag("symbol", symbol) \
                    .field("close", float(row.get('close', 0))) \
                    .field("volume", float(row.get('volume', 0))) \
                    .time(pd.Timestamp(timestamp))
                
                # Рыночные объемы
                if 'market_buy_volume' in row:
                    point = point.field("market_buy_volume", float(row['market_buy_volume']))
                if 'market_sell_volume' in row:
                    point = point.field("market_sell_volume", float(row['market_sell_volume']))
                
                # Ликвидации
                if 'long_liquidations' in row:
                    point = point.field("long_liquidations", float(row['long_liquidations']))
                if 'short_liquidations' in row:
                    point = point.field("short_liquidations", float(row['short_liquidations']))
                if 'total_liquidations' in row:
                    point = point.field("total_liquidations", float(row['total_liquidations']))
                
                # Открытый интерес
                if 'open_interest' in row:
                    point = point.field("open_interest", float(row['open_interest']))
                if 'open_interest_value' in row:
                    point = point.field("open_interest_value", float(row['open_interest_value']))
                
                # Order book depths (3%, 8%, 15%, 60%)
                for depth in [3, 8, 15, 60]:
                    depth_key = f"{depth}pct"
                    if f'bid_volume_{depth_key}' in row:
                        point = point.field(f"bid_volume_{depth_key}", float(row[f'bid_volume_{depth_key}']))
                    if f'ask_volume_{depth_key}' in row:
                        point = point.field(f"ask_volume_{depth_key}", float(row[f'ask_volume_{depth_key}']))
                    if f'total_volume_{depth_key}' in row:
                        point = point.field(f"total_volume_{depth_key}", float(row[f'total_volume_{depth_key}']))
                    if f'imbalance_{depth_key}' in row:
                        point = point.field(f"imbalance_{depth_key}", float(row[f'imbalance_{depth_key}']))
                
                points.append(point)
            
            self.write_api.write(bucket=self.bucket, record=points)
            logger.info(f"✅ Записано {len(points)} расширенных записей в InfluxDB")
        
        except Exception as e:
            logger.error(f"❌ Ошибка записи расширенных данных: {e}")
    
    def write_point(
        self,
        measurement: str,
        fields: Dict,
        tags: Optional[Dict] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Запись одной точки данных
        
        Args:
            measurement: Имя измерения
            fields: Поля (значения)
            tags: Теги (метаданные)
            timestamp: Временная метка
        """
        if self.write_api is None:
            logger.error("❌ InfluxDB клиент не инициализирован!")
            return
        
        try:
            point = Point(measurement)
            
            if tags:
                for key, value in tags.items():
                    point = point.tag(key, str(value))
            
            for key, value in fields.items():
                point = point.field(key, float(value) if isinstance(value, (int, float)) else value)
            
            if timestamp:
                point = point.time(timestamp)
            
            self.write_api.write(bucket=self.bucket, record=point)
        
        except Exception as e:
            logger.error(f"❌ Ошибка записи точки: {e}")
    
    def close(self):
        """Закрытие соединения"""
        if self.client:
            self.client.close()
            logger.info("✅ Соединение с InfluxDB закрыто")


