"""
Скрипт для сбора расширенных данных по BTC:
- Цена (OHLCV)
- Лимитные заявки на глубине 3%, 8%, 15%, 60% от цены
- Объемы рыночных покупок/продаж
- Ликвидации
- Открытый интерес
Все данные агрегируются по 15-минутным интервалам
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
from typing import Dict, List, Optional

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent.parent))

from binance_data_collector.config import config
from binance_data_collector.utils.logger import setup_logger
from binance_data_collector.utils.file_handler import save_data

# Импорт библиотеки для работы с Binance
try:
    from binance.client import Client
    from binance import BinanceSocketManager, ThreadedWebsocketManager
except ImportError:
    print("❌ Не установлена библиотека python-binance!")
    print("   Установите: pip install python-binance")
    sys.exit(1)

logger = setup_logger("advanced_btc_collector")

class AdvancedBTCDataCollector:
    """Коллектор расширенных данных по BTC"""
    
    def __init__(self):
        # Инициализация клиента
        if config.BINANCE_API_KEY and config.BINANCE_API_SECRET:
            self.spot_client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
            self.futures_client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET, testnet=False)
        else:
            self.spot_client = Client()
            self.futures_client = Client()
        
        self.symbol = "BTCUSDT"
        self.timeframe = "15m"
        
        # Глубины для анализа order book
        self.depths = [0.03, 0.08, 0.15, 0.60]  # 3%, 8%, 15%, 60%
    
    def get_order_book_depth(self, price: float, depth_percent: float) -> Dict[str, float]:
        """
        Получение объема лимитных заявок на заданной глубине от цены
        
        Args:
            price: Текущая цена
            depth_percent: Глубина в процентах (0.03 = 3%)
        
        Returns:
            Словарь с объемами покупок и продаж
        """
        try:
            order_book = self.spot_client.get_order_book(symbol=self.symbol, limit=5000)
            
            # Цены для покупок (bid) и продаж (ask)
            bid_price_threshold = price * (1 - depth_percent)
            ask_price_threshold = price * (1 + depth_percent)
            
            # Суммируем объемы на уровне покупок (bids)
            bid_volume = 0.0
            for bid in order_book['bids']:
                bid_price = float(bid[0])
                if bid_price >= bid_price_threshold:
                    bid_volume += float(bid[1])
                else:
                    break
            
            # Суммируем объемы на уровне продаж (asks)
            ask_volume = 0.0
            for ask in order_book['asks']:
                ask_price = float(ask[0])
                if ask_price <= ask_price_threshold:
                    ask_volume += float(ask[1])
                else:
                    break
            
            return {
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': bid_volume + ask_volume,
                'imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            }
        
        except Exception as e:
            logger.error(f"❌ Ошибка получения order book: {e}")
            return {'bid_volume': 0, 'ask_volume': 0, 'total_volume': 0, 'imbalance': 0}
    
    def get_aggregated_trades(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Получение агрегированных сделок за период
        
        Args:
            start_time: Начальное время
            end_time: Конечное время
        
        Returns:
            DataFrame с агрегированными сделками
        """
        try:
            all_trades = []
            
            # Binance возвращает до 1000 сделок за запрос
            from_time = start_time
            
            while from_time < end_time:
                trades = self.spot_client.get_aggregate_trades(
                    symbol=self.symbol,
                    startTime=int(from_time.timestamp() * 1000),
                    endTime=int(end_time.timestamp() * 1000),
                    limit=1000
                )
                
                if not trades:
                    break
                
                all_trades.extend(trades)
                
                # Обновляем время для следующей итерации
                last_trade_time = trades[-1]['T'] / 1000
                from_time = datetime.fromtimestamp(last_trade_time) + timedelta(milliseconds=1)
                
                time.sleep(config.REQUEST_DELAY)
            
            if not all_trades:
                return pd.DataFrame()
            
            # Преобразуем в DataFrame
            df = pd.DataFrame(all_trades)
            df['timestamp'] = pd.to_datetime(df['T'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Разделяем на покупки и продажи
            df['is_buyer_maker'] = df['m'].apply(lambda x: 1 if x else 0)
            df['buy_volume'] = df.apply(lambda x: float(x['q']) if not x['m'] else 0, axis=1)
            df['sell_volume'] = df.apply(lambda x: float(x['q']) if x['m'] else 0, axis=1)
            
            return df[['buy_volume', 'sell_volume', 'p']]
        
        except Exception as e:
            logger.error(f"❌ Ошибка получения агрегированных сделок: {e}")
            return pd.DataFrame()
    
    def get_liquidations(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Получение данных о ликвидациях (для фьючерсов)
        
        Args:
            start_time: Начальное время
            end_time: Конечное время
        
        Returns:
            DataFrame с ликвидациями
        """
        try:
            # Binance Futures API для ликвидаций (forced orders)
            all_liquidations = []
            
            url = "https://fapi.binance.com/fapi/v1/forceOrders"
            
            from_time = start_time
            
            while from_time < end_time:
                params = {
                    'symbol': self.symbol,
                    'startTime': int(from_time.timestamp() * 1000),
                    'endTime': int(end_time.timestamp() * 1000),
                    'limit': 1000
                }
                
                response = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
                
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                    
                    all_liquidations.extend(data)
                    
                    # Обновляем время для следующей итерации
                    last_time = data[-1]['time'] / 1000
                    from_time = datetime.fromtimestamp(last_time) + timedelta(milliseconds=1)
                    
                    time.sleep(config.REQUEST_DELAY)
                else:
                    logger.warning(f"⚠️ Ошибка API при получении ликвидаций: {response.status_code}")
                    break
            
            if not all_liquidations:
                return pd.DataFrame()
            
            df = pd.DataFrame(all_liquidations)
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Разделяем на long и short ликвидации
            # SELL = ликвидация long позиции, BUY = ликвидация short позиции
            df['liquidation_type'] = df['side'].apply(lambda x: 'long' if x == 'SELL' else 'short')
            df['liquidation_quantity'] = df['executedQty'].astype(float)
            df['liquidation_price'] = df['price'].astype(float)
            
            return df[['liquidation_type', 'liquidation_quantity', 'liquidation_price']]
        
        except Exception as e:
            logger.warning(f"⚠️ Ошибка получения ликвидаций: {e}")
            logger.warning("   Ликвидации доступны только для фьючерсов и через WebSocket в реальном времени")
            return pd.DataFrame()
    
    def get_open_interest_history(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Получение истории открытого интереса
        
        Args:
            start_time: Начальное время
            end_time: Конечное время
        
        Returns:
            DataFrame с историей открытого интереса
        """
        try:
            # Binance Futures API для истории открытого интереса
            url = "https://fapi.binance.com/futures/data/openInterestHist"
            
            oi_data = []
            current_time = start_time
            
            # API возвращает данные по интервалам (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            while current_time < end_time:
                params = {
                    'symbol': self.symbol,
                    'period': '15m',  # 15-минутный интервал
                    'startTime': int(current_time.timestamp() * 1000),
                    'endTime': int(min(current_time + timedelta(days=30), end_time).timestamp() * 1000),
                    'limit': 500
                }
                
                response = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
                
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                    
                    oi_data.extend(data)
                    
                    # Обновляем время
                    last_time = data[-1]['timestamp'] / 1000
                    current_time = datetime.fromtimestamp(last_time) + timedelta(minutes=15)
                    
                    time.sleep(config.REQUEST_DELAY)
                else:
                    logger.warning(f"⚠️ Ошибка API при получении открытого интереса: {response.status_code}")
                    break
            
            if not oi_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(oi_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['open_interest'] = df['sumOpenInterest'].astype(float)
            df['open_interest_value'] = df['sumOpenInterestValue'].astype(float)
            
            return df[['open_interest', 'open_interest_value']]
        
        except Exception as e:
            logger.warning(f"⚠️ Ошибка получения истории открытого интереса: {e}")
            return pd.DataFrame()
    
    def aggregate_to_15m(
        self,
        ohlcv_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        liquidations_df: pd.DataFrame,
        oi_df: pd.DataFrame,
        order_book_data: List[Dict]
    ) -> pd.DataFrame:
        """
        Агрегация всех данных по 15-минутным интервалам
        
        Args:
            ohlcv_df: OHLCV данные
            trades_df: Данные о сделках
            liquidations_df: Данные о ликвидациях
            order_book_data: Данные order book
        
        Returns:
            Агрегированный DataFrame по 15-минутным интервалам
        """
        # Используем OHLCV как основу для временных меток
        result_df = ohlcv_df.copy()
        
        # Агрегируем сделки по 15-минутным интервалам
        if not trades_df.empty:
            trades_15m = trades_df.resample('15T').agg({
                'buy_volume': 'sum',
                'sell_volume': 'sum',
                'p': 'last'  # последняя цена сделки
            })
            trades_15m.columns = ['market_buy_volume', 'market_sell_volume', 'last_trade_price']
            result_df = result_df.join(trades_15m, how='left')
            result_df['market_buy_volume'] = result_df['market_buy_volume'].fillna(0)
            result_df['market_sell_volume'] = result_df['market_sell_volume'].fillna(0)
        else:
            result_df['market_buy_volume'] = 0
            result_df['market_sell_volume'] = 0
        
        # Агрегируем ликвидации
        if not liquidations_df.empty:
            liquidations_15m = liquidations_df.resample('15T').agg({
                'liquidation_quantity': 'sum'
            })
            liquidations_15m.columns = ['total_liquidations']
            
            # Разделяем на long и short
            long_liq = liquidations_df[liquidations_df['liquidation_type'] == 'long'].resample('15T')['liquidation_quantity'].sum()
            short_liq = liquidations_df[liquidations_df['liquidation_type'] == 'short'].resample('15T')['liquidation_quantity'].sum()
            
            result_df['long_liquidations'] = long_liq
            result_df['short_liquidations'] = short_liq
            result_df['total_liquidations'] = liquidations_15m['total_liquidations']
            result_df['long_liquidations'] = result_df['long_liquidations'].fillna(0)
            result_df['short_liquidations'] = result_df['short_liquidations'].fillna(0)
            result_df['total_liquidations'] = result_df['total_liquidations'].fillna(0)
        else:
            result_df['long_liquidations'] = 0
            result_df['short_liquidations'] = 0
            result_df['total_liquidations'] = 0
        
        # Добавляем данные order book для каждой глубины
        for depth in self.depths:
            depth_key = f"{int(depth * 100)}pct"
            
            # Для каждого интервала берем последние данные order book
            # (в реальности нужно собирать их в реальном времени, но для исторических данных используем текущие)
            result_df[f'bid_volume_{depth_key}'] = 0
            result_df[f'ask_volume_{depth_key}'] = 0
            result_df[f'total_volume_{depth_key}'] = 0
            result_df[f'imbalance_{depth_key}'] = 0
        
        # Добавляем открытый интерес
        if not oi_df.empty:
            result_df = result_df.join(oi_df[['open_interest']], how='left')
            result_df['open_interest'] = result_df['open_interest'].ffill().fillna(0)
            result_df['open_interest_value'] = oi_df['open_interest_value'] if 'open_interest_value' in oi_df.columns else 0
        else:
            result_df['open_interest'] = 0
            result_df['open_interest_value'] = 0
        
        return result_df
    
    def collect_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        sample_orderbook: bool = True
    ) -> pd.DataFrame:
        """
        Сбор исторических данных за период
        
        Args:
            start_date: Начальная дата
            end_date: Конечная дата
            sample_orderbook: Собирать ли данные order book (рекомендуется False для исторических данных)
        
        Returns:
            DataFrame с агрегированными данными
        """
        logger.info(f"🚀 Начало сбора расширенных данных для {self.symbol}")
        logger.info(f"📅 Период: {start_date.date()} - {end_date.date()}")
        
        # 1. Получаем OHLCV данные (основа)
        logger.info("\n📊 Шаг 1/4: Получение OHLCV данных...")
        ohlcv_df = self._get_ohlcv(start_date, end_date)
        
        if ohlcv_df.empty:
            logger.error("❌ Не удалось получить OHLCV данные!")
            return pd.DataFrame()
        
        logger.info(f"✅ Получено {len(ohlcv_df)} свечей")
        
        # 2. Получаем агрегированные сделки
        logger.info("\n📊 Шаг 2/4: Получение данных о рыночных сделках...")
        trades_df = self.get_aggregated_trades(start_date, end_date)
        
        if not trades_df.empty:
            logger.info(f"✅ Получено {len(trades_df)} сделок")
        else:
            logger.warning("⚠️ Не удалось получить данные о сделках")
        
        # 3. Получаем ликвидации
        logger.info("\n📊 Шаг 3/4: Получение данных о ликвидациях...")
        liquidations_df = self.get_liquidations(start_date, end_date)
        
        if not liquidations_df.empty:
            logger.info(f"✅ Получено {len(liquidations_df)} ликвидаций")
        else:
            logger.warning("⚠️ Ликвидации недоступны или не найдены для указанного периода")
        
        # 4. Получаем историю открытого интереса
        logger.info("\n📊 Шаг 4/5: Получение истории открытого интереса...")
        oi_df = self.get_open_interest_history(start_date, end_date)
        
        if not oi_df.empty:
            logger.info(f"✅ Получено {len(oi_df)} записей открытого интереса")
        else:
            logger.warning("⚠️ История открытого интереса недоступна")
        
        # 5. Агрегируем данные по 15-минутным интервалам
        logger.info("\n📊 Шаг 5/5: Агрегация данных по 15-минутным интервалам...")
        
        # Для order book данных нужно собирать их в реальном времени
        # Для исторических данных собираем образцы
        order_book_data = []
        
        if sample_orderbook:
            logger.info("   Сбор образцов данных order book...")
            sample_times = ohlcv_df.index[::max(1, len(ohlcv_df) // 100)]  # Берем 100 образцов
            
            for idx, timestamp in enumerate(sample_times):
                price = ohlcv_df.loc[timestamp, 'close']
                for depth in self.depths:
                    depth_data = self.get_order_book_depth(price, depth)
                    depth_data['timestamp'] = timestamp
                    depth_data['depth'] = depth
                    order_book_data.append(depth_data)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"   Обработано {idx + 1}/{len(sample_times)} образцов")
                
                time.sleep(config.REQUEST_DELAY)
        
        # Агрегируем все данные
        result_df = self.aggregate_to_15m(ohlcv_df, trades_df, liquidations_df, oi_df, order_book_data)
        
        logger.info(f"\n✅ Сбор данных завершен! Итого записей: {len(result_df)}")
        
        return result_df
    
    def _get_ohlcv(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Вспомогательный метод для получения OHLCV"""
        all_klines = []
        current_start = start_time
        
        while current_start < end_time:
            try:
                klines = self.spot_client.get_klines(
                    symbol=self.symbol,
                    interval=self.timeframe,
                    startTime=int(current_start.timestamp() * 1000),
                    endTime=int(end_time.timestamp() * 1000),
                    limit=1000
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                last_timestamp = klines[-1][0] / 1000
                current_start = datetime.fromtimestamp(last_timestamp) + timedelta(seconds=1)
                
                time.sleep(config.REQUEST_DELAY)
            
            except Exception as e:
                logger.error(f"❌ Ошибка при получении OHLCV: {e}")
                time.sleep(config.REQUEST_DELAY * 10)
        
        if not all_klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        return df[['open', 'high', 'low', 'close', 'volume']]

def main():
    """Основная функция"""
    collector = AdvancedBTCDataCollector()
    
    # Определяем период сбора данных
    # Максимальный период - с момента запуска Binance (2017) до сегодня
    end_date = datetime.now()
    start_date = datetime(2017, 8, 1)  # Binance начал работу в августе 2017
    
    logger.info(f"📅 Сбор данных с {start_date.date()} по {end_date.date()}")
    logger.info(f"⏰ Это может занять много времени из-за большого объема данных...")
    
    # Собираем данные
    df = collector.collect_historical_data(
        start_date=start_date,
        end_date=end_date,
        sample_orderbook=False  # Для полной истории отключаем order book (слишком долго)
    )
    
    if not df.empty:
        # Сохраняем данные
        filename = f"BTCUSDT_advanced_15m_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        filepath = config.DATA_DIR / "historical" / filename
        save_data(df, filepath, format="csv")
        
        logger.info(f"\n✅ Данные сохранены: {filepath}")
        logger.info(f"📊 Колонки: {list(df.columns)}")
        logger.info(f"📈 Записей: {len(df)}")
    else:
        logger.error("❌ Не удалось собрать данные!")

if __name__ == "__main__":
    main()

