"""
Комплексный сбор всех доступных исторических данных с Binance
Собирает максимально полную историю и объединяет все данные в один DataFrame
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import gc
from typing import Dict, List, Optional, Tuple

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent.parent))

from binance_data_collector.config import config
from binance_data_collector.utils.logger import setup_logger
from binance_data_collector.utils.file_handler import save_data

# Импорт библиотеки для работы с Binance
try:
    from binance.client import Client
except ImportError:
    print("❌ Не установлена библиотека python-binance!")
    print("   Установите: pip install python-binance")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("⚠️  tqdm не установлен, прогресс-бар не будет отображаться")
    print("   Установите: pip install tqdm")
    # Создаем заглушку
    def tqdm(iterable, *args, **kwargs):
        return iterable

logger = setup_logger("comprehensive_collector")

def get_memory_usage_mb() -> float:
    """Получение использования памяти процесса в MB"""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # Если psutil не установлен, возвращаем 0
        return 0
    except Exception:
        return 0

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Оптимизация использования памяти DataFrame"""
    start_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Оптимизируем числовые колонки
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Оптимизируем строковые колонки
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype('category')
            except:
                pass
    
    end_memory = df.memory_usage(deep=True).sum() / 1024**2
    if end_memory < start_memory:
        logger.info(f"   💾 Оптимизация памяти: {start_memory:.2f} MB → {end_memory:.2f} MB (экономия {start_memory - end_memory:.2f} MB)")
    
    return df


class ComprehensiveBinanceCollector:
    """Комплексный сбор всех доступных данных с Binance"""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        """
        Args:
            symbol: Торговая пара (например, BTCUSDT)
        """
        # Инициализация клиента
        if config.BINANCE_API_KEY and config.BINANCE_API_SECRET:
            self.spot_client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
            self.futures_client = Client(
                config.BINANCE_API_KEY, 
                config.BINANCE_API_SECRET, 
                testnet=False
            )
        else:
            self.spot_client = Client()
            # Для фьючерсов нужен отдельный клиент
            try:
                self.futures_client = Client()
            except:
                self.futures_client = None
                logger.warning("⚠️  Фьючерсный клиент недоступен (некоторые данные могут отсутствовать)")
        
        self.symbol = symbol
        self.symbol_futures = symbol  # Для фьючерсов
        
        # Глубины для анализа order book (в процентах)
        self.orderbook_depths = [0.03, 0.08, 0.15, 0.60]  # 3%, 8%, 15%, 60%
        
    def get_klines_batch(
        self,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Получение исторических OHLCV данных с максимальной историей
        
        Args:
            interval: Интервал (1m, 5m, 15m, 1h, 1d и т.д.)
            start_time: Начальное время
            end_time: Конечное время
            limit: Максимальное количество свечей за запрос
        
        Returns:
            DataFrame с OHLCV данными
        """
        all_klines = []
        current_start = start_time
        
        logger.info(f"📊 Сбор OHLCV для {self.symbol} ({interval}) с {start_time.date()} по {end_time.date()}")
        
        # Оцениваем количество запросов для прогресс-бара
        total_days = (end_time - start_time).days
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            estimated_requests = max(1, total_days * 24 * 60 // minutes // limit)
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            estimated_requests = max(1, total_days * 24 // hours // limit)
        elif interval.endswith('d'):
            days = int(interval[:-1]) if interval[:-1] != '' else 1
            estimated_requests = max(1, total_days // days // limit)
        else:
            estimated_requests = 100
        
        pbar = tqdm(total=estimated_requests, desc=f"OHLCV ({interval})", unit="batch")
        
        while current_start < end_time:
            try:
                klines = self.spot_client.get_klines(
                    symbol=self.symbol,
                    interval=interval,
                    startTime=int(current_start.timestamp() * 1000),
                    endTime=int(end_time.timestamp() * 1000),
                    limit=limit
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # Обновляем время начала для следующей итерации
                last_timestamp = klines[-1][0] / 1000
                current_start = datetime.fromtimestamp(last_timestamp) + timedelta(seconds=1)
                
                pbar.update(1)
                time.sleep(config.REQUEST_DELAY)
                
            except Exception as e:
                logger.error(f"❌ Ошибка при получении OHLCV: {e}")
                time.sleep(config.REQUEST_DELAY * 10)
        
        pbar.close()
        
        if not all_klines:
            logger.warning("⚠️ OHLCV данные не получены")
            return pd.DataFrame()
        
        # Преобразуем в DataFrame
        logger.info(f"   📊 Преобразование {len(all_klines)} записей в DataFrame...")
        memory_before = get_memory_usage_mb()
        
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Конвертируем типы
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'quote_volume', 'taker_buy_base', 'taker_buy_quote']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        df['trades'] = pd.to_numeric(df['trades'], downcast='integer')
        
        # Удаляем дубликаты
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        
        # Оптимизируем память
        df = optimize_dataframe_memory(df)
        
        # Очищаем список из памяти
        del all_klines
        gc.collect()
        
        memory_after = get_memory_usage_mb()
        memory_used = memory_after - memory_before
        logger.info(f"✅ Получено {len(df)} свечей OHLCV")
        if memory_used > 0:
            logger.info(f"   💾 Использовано памяти: {memory_used:.2f} MB")
        
        return df
    
    def get_aggregated_trades_batch(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Получение агрегированных сделок (рыночные объемы покупок/продаж)
        
        Args:
            start_time: Начальное время
            end_time: Конечное время
        
        Returns:
            DataFrame с агрегированными сделками
        """
        logger.info(f"📊 Сбор агрегированных сделок с {start_time.date()} по {end_time.date()}")
        
        all_trades = []
        from_time = start_time
        
        # Оценка для прогресс-бара (до 1000 сделок за запрос)
        # Примерно 1000 сделок = 1-5 минут в зависимости от активности
        estimated_duration = (end_time - start_time).total_seconds() / 60  # в минутах
        estimated_requests = max(1, int(estimated_duration / 2))  # примерная оценка
        
        # Мониторинг памяти для больших объемов
        CHUNK_SIZE = 50000  # Обрабатываем по 50k записей
        chunk_counter = 0
        temp_dfs = []
        
        pbar = tqdm(total=estimated_requests, desc="Aggregated Trades", unit="batch")
        
        while from_time < end_time:
            try:
                trades = self.spot_client.get_aggregate_trades(
                    symbol=self.symbol,
                    startTime=int(from_time.timestamp() * 1000),
                    endTime=int(end_time.timestamp() * 1000),
                    limit=1000
                )
                
                if not trades:
                    break
                
                all_trades.extend(trades)
                
                # Обработка по частям для экономии памяти
                if len(all_trades) >= CHUNK_SIZE:
                    chunk_counter += 1
                    memory_usage = get_memory_usage_mb()
                    
                    # Преобразуем в DataFrame и обрабатываем
                    chunk_df = pd.DataFrame(all_trades)
                    chunk_df['timestamp'] = pd.to_datetime(chunk_df['T'], unit='ms')
                    chunk_df.set_index('timestamp', inplace=True)
                    
                    # Предварительная обработка
                    chunk_df['price'] = pd.to_numeric(chunk_df['p'], downcast='float')
                    chunk_df['quantity'] = pd.to_numeric(chunk_df['q'], downcast='float')
                    chunk_df['buy_volume'] = chunk_df.apply(lambda x: float(x['q']) if not x['m'] else 0, axis=1)
                    chunk_df['sell_volume'] = chunk_df.apply(lambda x: float(x['q']) if x['m'] else 0, axis=1)
                    
                    temp_dfs.append(chunk_df[['buy_volume', 'sell_volume', 'price', 'quantity']])
                    
                    # Очищаем список
                    all_trades = []
                    gc.collect()
                    
                    if chunk_counter % 10 == 0:
                        logger.info(f"   💾 Обработано {chunk_counter} чанков. Использовано памяти: {memory_usage:.2f} MB")
                
                # Обновляем время для следующей итерации
                last_trade_time = trades[-1]['T'] / 1000
                from_time = datetime.fromtimestamp(last_trade_time) + timedelta(milliseconds=1)
                
                pbar.update(1)
                time.sleep(config.REQUEST_DELAY)
                
            except Exception as e:
                logger.error(f"❌ Ошибка получения агрегированных сделок: {e}")
                time.sleep(config.REQUEST_DELAY * 10)
        
        pbar.close()
        
        # Обрабатываем оставшиеся данные
        if all_trades:
            chunk_df = pd.DataFrame(all_trades)
            chunk_df['timestamp'] = pd.to_datetime(chunk_df['T'], unit='ms')
            chunk_df.set_index('timestamp', inplace=True)
            chunk_df['price'] = pd.to_numeric(chunk_df['p'], downcast='float')
            chunk_df['quantity'] = pd.to_numeric(chunk_df['q'], downcast='float')
            chunk_df['buy_volume'] = chunk_df.apply(lambda x: float(x['q']) if not x['m'] else 0, axis=1)
            chunk_df['sell_volume'] = chunk_df.apply(lambda x: float(x['q']) if x['m'] else 0, axis=1)
            temp_dfs.append(chunk_df[['buy_volume', 'sell_volume', 'price', 'quantity']])
            all_trades = []
        
        if not temp_dfs:
            logger.warning("⚠️ Агрегированные сделки не получены")
            return pd.DataFrame()
        
        # Объединяем все чанки
        logger.info(f"   📊 Объединение {len(temp_dfs)} чанков...")
        memory_before = get_memory_usage_mb()
        
        df = pd.concat(temp_dfs, ignore_index=False)
        del temp_dfs
        gc.collect()
        
        # Удаляем дубликаты
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        
        # Оптимизируем память
        df = optimize_dataframe_memory(df)
        
        memory_after = get_memory_usage_mb()
        memory_used = memory_after - memory_before
        
        logger.info(f"✅ Получено {len(df)} агрегированных сделок")
        if memory_used > 0:
            logger.info(f"   💾 Использовано памяти: {memory_used:.2f} MB")
        
        return df[['buy_volume', 'sell_volume', 'price', 'quantity']]
    
    def get_liquidations_batch(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Получение данных о ликвидациях (для фьючерсов)
        
        Args:
            start_time: Начальное время
            end_time: Конечное время
        
        Returns:
            DataFrame с ликвидациями
        """
        logger.info(f"📊 Сбор данных о ликвидациях с {start_time.date()} по {end_time.date()}")
        
        try:
            url = "https://fapi.binance.com/fapi/v1/forceOrders"
            
            all_liquidations = []
            from_time = start_time
            
            # Оценка для прогресс-бара
            estimated_days = (end_time - start_time).days
            estimated_requests = max(1, estimated_days)
            
            pbar = tqdm(total=estimated_requests, desc="Liquidations", unit="batch")
            
            while from_time < end_time:
                params = {
                    'symbol': self.symbol_futures,
                    'startTime': int(from_time.timestamp() * 1000),
                    'endTime': int(end_time.timestamp() * 1000),
                    'limit': 1000
                }
                
                try:
                    response = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if not data:
                            break
                        
                        all_liquidations.extend(data)
                        
                        # Обновляем время для следующей итерации
                        last_time = data[-1]['time'] / 1000
                        from_time = datetime.fromtimestamp(last_time) + timedelta(milliseconds=1)
                        
                        pbar.update(1)
                        time.sleep(config.REQUEST_DELAY)
                    elif response.status_code == 429:
                        logger.warning("⚠️ Rate limit, увеличение задержки...")
                        time.sleep(config.REQUEST_DELAY * 10)
                        continue
                    else:
                        logger.warning(f"⚠️ Ошибка API: {response.status_code}")
                        break
                        
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка при запросе ликвидаций: {e}")
                    break
            
            pbar.close()
            
            if not all_liquidations:
                logger.warning("⚠️ Ликвидации не найдены (возможно, недоступны для указанного периода)")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_liquidations)
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Разделяем на long и short ликвидации
            df['liquidation_type'] = df['side'].apply(lambda x: 'long' if x == 'SELL' else 'short')
            df['liquidation_quantity'] = df['executedQty'].astype(float)
            df['liquidation_price'] = df['price'].astype(float)
            
            # Удаляем дубликаты
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            logger.info(f"✅ Получено {len(df)} ликвидаций")
            
            return df[['liquidation_type', 'liquidation_quantity', 'liquidation_price']]
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка получения ликвидаций: {e}")
            return pd.DataFrame()
    
    def get_open_interest_history_batch(
        self,
        start_time: datetime,
        end_time: datetime,
        period: str = "15m"
    ) -> pd.DataFrame:
        """
        Получение истории открытого интереса
        
        Args:
            start_time: Начальное время
            end_time: Конечное время
            period: Период агрегации (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
        
        Returns:
            DataFrame с историей открытого интереса
        """
        logger.info(f"📊 Сбор истории открытого интереса с {start_time.date()} по {end_time.date()}")
        
        try:
            url = "https://fapi.binance.com/futures/data/openInterestHist"
            
            oi_data = []
            current_time = start_time
            
            # Оценка для прогресс-бара
            total_days = (end_time - start_time).days
            estimated_requests = max(1, total_days // 30)  # API возвращает до 30 дней за запрос
            
            pbar = tqdm(total=estimated_requests, desc="Open Interest", unit="batch")
            
            while current_time < end_time:
                params = {
                    'symbol': self.symbol_futures,
                    'period': period,
                    'startTime': int(current_time.timestamp() * 1000),
                    'endTime': int(min(current_time + timedelta(days=30), end_time).timestamp() * 1000),
                    'limit': 500
                }
                
                try:
                    response = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if not data:
                            break
                        
                        oi_data.extend(data)
                        
                        # Обновляем время
                        last_time = data[-1]['timestamp'] / 1000
                        current_time = datetime.fromtimestamp(last_time) + timedelta(minutes=15)
                        
                        pbar.update(1)
                        time.sleep(config.REQUEST_DELAY)
                    elif response.status_code == 429:
                        logger.warning("⚠️ Rate limit, увеличение задержки...")
                        time.sleep(config.REQUEST_DELAY * 10)
                        continue
                    else:
                        logger.warning(f"⚠️ Ошибка API: {response.status_code}")
                        break
                        
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка при запросе открытого интереса: {e}")
                    break
            
            pbar.close()
            
            if not oi_data:
                logger.warning("⚠️ История открытого интереса недоступна")
                return pd.DataFrame()
            
            df = pd.DataFrame(oi_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            df['open_interest'] = df['sumOpenInterest'].astype(float)
            df['open_interest_value'] = df['sumOpenInterestValue'].astype(float)
            
            # Удаляем дубликаты
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            logger.info(f"✅ Получено {len(df)} записей открытого интереса")
            
            return df[['open_interest', 'open_interest_value']]
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка получения открытого интереса: {e}")
            return pd.DataFrame()
    
    def get_order_book_depth(self, price: float, depth_percent: float) -> Dict[str, float]:
        """
        Получение объема лимитных заявок на заданной глубине от цены
        
        Args:
            price: Текущая цена
            depth_percent: Глубина в процентах (0.03 = 3%)
        
        Returns:
            Словарь с объемами покупок и продаж на заданной глубине
        """
        try:
            # Получаем order book с максимальной глубиной (5000 уровней)
            order_book = self.spot_client.get_order_book(symbol=self.symbol, limit=5000)
            
            # Цены для покупок (bid) и продаж (ask)
            bid_price_threshold = price * (1 - depth_percent)
            ask_price_threshold = price * (1 + depth_percent)
            
            # Суммируем объемы на уровне покупок (bids) - лимитные заявки на покупку
            bid_volume = 0.0
            for bid in order_book['bids']:
                bid_price = float(bid[0])
                if bid_price >= bid_price_threshold:
                    bid_volume += float(bid[1])
                else:
                    break
            
            # Суммируем объемы на уровне продаж (asks) - лимитные заявки на продажу
            ask_volume = 0.0
            for ask in order_book['asks']:
                ask_price = float(ask[0])
                if ask_price <= ask_price_threshold:
                    ask_volume += float(ask[1])
                else:
                    break
            
            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            return {
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': total_volume,
                'imbalance': imbalance
            }
        
        except Exception as e:
            logger.warning(f"⚠️ Ошибка получения order book: {e}")
            return {
                'bid_volume': 0.0,
                'ask_volume': 0.0,
                'total_volume': 0.0,
                'imbalance': 0.0
            }
    
    def collect_order_book_snapshots(
        self,
        ohlcv_df: pd.DataFrame,
        sample_rate: int = 10
    ) -> pd.DataFrame:
        """
        Сбор периодических снимков order book на основе исторических цен
        
        Важно: Binance не предоставляет историю order book, поэтому мы делаем
        периодические снимки текущего состояния order book. Это не точная история,
        но дает представление о глубине рынка.
        
        Args:
            ohlcv_df: DataFrame с OHLCV данными (нужен для получения цен)
            sample_rate: Каждую N-ю свечу делать снимок order book
        
        Returns:
            DataFrame с данными order book
        """
        logger.info(f"📊 Сбор периодических снимков order book (каждую {sample_rate}-ю свечу)...")
        
        if ohlcv_df.empty:
            logger.warning("⚠️ Нет данных OHLCV для сбора order book")
            return pd.DataFrame()
        
        order_book_data = []
        
        # Берем каждую sample_rate-ю свечу
        sample_indices = range(0, len(ohlcv_df), sample_rate)
        total_samples = len(sample_indices)
        
        pbar = tqdm(total=total_samples, desc="Order Book Snapshots", unit="snapshot")
        
        for idx in sample_indices:
            try:
                timestamp = ohlcv_df.index[idx]
                current_price = ohlcv_df.loc[timestamp, 'close']
                
                snapshot = {
                    'timestamp': timestamp
                }
                
                # Собираем данные для каждой глубины
                for depth in self.orderbook_depths:
                    depth_key = f"{int(depth * 100)}pct"
                    depth_data = self.get_order_book_depth(current_price, depth)
                    
                    snapshot[f'bid_volume_{depth_key}'] = depth_data['bid_volume']
                    snapshot[f'ask_volume_{depth_key}'] = depth_data['ask_volume']
                    snapshot[f'total_volume_{depth_key}'] = depth_data['total_volume']
                    snapshot[f'imbalance_{depth_key}'] = depth_data['imbalance']
                
                order_book_data.append(snapshot)
                
                pbar.update(1)
                time.sleep(config.REQUEST_DELAY)
                
            except Exception as e:
                logger.warning(f"⚠️ Ошибка при сборе снимка order book: {e}")
                continue
        
        pbar.close()
        
        if not order_book_data:
            logger.warning("⚠️ Не удалось собрать данные order book")
            return pd.DataFrame()
        
        df = pd.DataFrame(order_book_data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        logger.info(f"✅ Собрано {len(df)} снимков order book")
        
        return df
    
    def get_24h_ticker_stats(self) -> Dict:
        """Получение статистики за 24 часа"""
        try:
            ticker = self.spot_client.get_ticker(symbol=self.symbol)
            return {
                'price_change_24h': float(ticker.get('priceChange', 0)),
                'price_change_percent_24h': float(ticker.get('priceChangePercent', 0)),
                'high_24h': float(ticker.get('highPrice', 0)),
                'low_24h': float(ticker.get('lowPrice', 0)),
                'volume_24h': float(ticker.get('volume', 0)),
                'quote_volume_24h': float(ticker.get('quoteVolume', 0)),
                'count_24h': int(ticker.get('count', 0))
            }
        except Exception as e:
            logger.warning(f"⚠️ Ошибка получения 24h статистики: {e}")
            return {}
    
    def combine_all_data(
        self,
        ohlcv_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        liquidations_df: pd.DataFrame,
        oi_df: pd.DataFrame,
        order_book_df: pd.DataFrame = None,
        target_interval: str = "15m"
    ) -> pd.DataFrame:
        """
        Объединение всех данных в один DataFrame с заданным интервалом
        
        Args:
            ohlcv_df: OHLCV данные
            trades_df: Данные о сделках
            liquidations_df: Данные о ликвидациях
            oi_df: Данные об открытом интересе
            order_book_df: Данные order book (периодические снимки)
            target_interval: Целевой интервал агрегации (например, '15m', '1h', '1d')
        
        Returns:
            Объединенный DataFrame
        """
        logger.info(f"\n📊 Объединение всех данных с интервалом {target_interval}...")
        
        if ohlcv_df.empty:
            logger.error("❌ Нет OHLCV данных для объединения!")
            return pd.DataFrame()
        
        # Используем OHLCV как основу
        result_df = ohlcv_df.copy()
        
        # Ресемплируем OHLCV на целевой интервал, если нужно
        if target_interval and ohlcv_df.index[0] < ohlcv_df.index[-1]:
            # Проверяем текущий интервал
            time_diff = (ohlcv_df.index[1] - ohlcv_df.index[0]).total_seconds() / 60
            target_minutes = self._interval_to_minutes(target_interval)
            
            if target_minutes > time_diff:
                # Нужно агрегировать
                result_df = result_df.resample(target_interval).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'quote_volume': 'sum',
                    'taker_buy_base': 'sum',
                    'taker_buy_quote': 'sum',
                    'trades': 'sum'
                })
                result_df = result_df.dropna()
        
        # Добавляем агрегированные сделки
        if not trades_df.empty:
            trades_resampled = trades_df.resample(target_interval).agg({
                'buy_volume': 'sum',
                'sell_volume': 'sum',
                'quantity': 'sum',
                'price': 'last'
            })
            trades_resampled.columns = ['market_buy_volume', 'market_sell_volume', 
                                        'total_trade_quantity', 'last_trade_price']
            result_df = result_df.join(trades_resampled, how='left')
            result_df['market_buy_volume'] = result_df['market_buy_volume'].fillna(0)
            result_df['market_sell_volume'] = result_df['market_sell_volume'].fillna(0)
            result_df['total_trade_quantity'] = result_df['total_trade_quantity'].fillna(0)
        else:
            result_df['market_buy_volume'] = 0
            result_df['market_sell_volume'] = 0
            result_df['total_trade_quantity'] = 0
        
        # Добавляем ликвидации
        if not liquidations_df.empty:
            # Агрегируем ликвидации по интервалам
            liquidations_resampled = liquidations_df.resample(target_interval).agg({
                'liquidation_quantity': 'sum'
            })
            liquidations_resampled.columns = ['total_liquidations']
            
            # Разделяем на long и short
            long_liq = liquidations_df[liquidations_df['liquidation_type'] == 'long'].resample(target_interval)['liquidation_quantity'].sum()
            short_liq = liquidations_df[liquidations_df['liquidation_type'] == 'short'].resample(target_interval)['liquidation_quantity'].sum()
            
            result_df['long_liquidations'] = long_liq
            result_df['short_liquidations'] = short_liq
            result_df['total_liquidations'] = liquidations_resampled['total_liquidations']
            
            result_df['long_liquidations'] = result_df['long_liquidations'].fillna(0)
            result_df['short_liquidations'] = result_df['short_liquidations'].fillna(0)
            result_df['total_liquidations'] = result_df['total_liquidations'].fillna(0)
        else:
            result_df['long_liquidations'] = 0
            result_df['short_liquidations'] = 0
            result_df['total_liquidations'] = 0
        
        # Добавляем открытый интерес
        if not oi_df.empty:
            # Ресемплируем на целевой интервал
            oi_resampled = oi_df.resample(target_interval).agg({
                'open_interest': 'last',
                'open_interest_value': 'last'
            })
            result_df = result_df.join(oi_resampled, how='left')
            
            # Заполняем пропуски forward fill
            result_df['open_interest'] = result_df['open_interest'].ffill().fillna(0)
            result_df['open_interest_value'] = result_df['open_interest_value'].ffill().fillna(0)
        else:
            result_df['open_interest'] = 0
            result_df['open_interest_value'] = 0
        
        # Добавляем данные order book (лимитные заявки на разных глубинах)
        if order_book_df is not None and not order_book_df.empty:
            # Ресемплируем order book на целевой интервал, используя последнее значение (forward fill)
            # Это нужно, так как order book данные собираются периодически
            order_book_cols = order_book_df.columns
            
            # Агрегируем по интервалам, используя последнее доступное значение
            order_book_resampled = order_book_df.resample(target_interval).last()
            
            # Объединяем с основным DataFrame
            result_df = result_df.join(order_book_resampled, how='left')
            
            # Заполняем пропуски forward fill (используем последнее известное значение)
            for col in order_book_cols:
                if col in result_df.columns:
                    result_df[col] = result_df[col].ffill().fillna(0)
            
            logger.info(f"✅ Добавлены данные order book для {len(order_book_cols)} колонок")
        else:
            # Заполняем нулями для всех глубин order book
            for depth in self.orderbook_depths:
                depth_key = f"{int(depth * 100)}pct"
                result_df[f'bid_volume_{depth_key}'] = 0
                result_df[f'ask_volume_{depth_key}'] = 0
                result_df[f'total_volume_{depth_key}'] = 0
                result_df[f'imbalance_{depth_key}'] = 0
        
        # Добавляем дополнительные метрики
        result_df['buy_sell_ratio'] = np.where(
            result_df['market_sell_volume'] > 0,
            result_df['market_buy_volume'] / result_df['market_sell_volume'],
            0
        )
        
        result_df['liquidation_ratio'] = np.where(
            result_df['total_liquidations'] > 0,
            result_df['long_liquidations'] / (result_df['long_liquidations'] + result_df['short_liquidations']),
            0
        )
        
        # Удаляем полностью пустые строки
        result_df = result_df.dropna(how='all')
        
        # Сортируем по индексу
        result_df.sort_index(inplace=True)
        
        logger.info(f"✅ Объединено {len(result_df)} записей")
        logger.info(f"📊 Колонки: {list(result_df.columns)}")
        
        return result_df
    
    def _interval_to_minutes(self, interval: str) -> int:
        """Конвертация интервала в минуты"""
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 24 * 60
        elif interval.endswith('w'):
            return int(interval[:-1]) * 7 * 24 * 60
        elif interval.endswith('M'):
            return int(interval[:-1]) * 30 * 24 * 60
        else:
            return 15  # По умолчанию 15 минут
    
    def collect_comprehensive_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "15m",
        target_interval: str = "15m",
        include_all_timeframes: bool = False,
        skip_aggregated_trades: bool = False
    ) -> pd.DataFrame:
        """
        Комплексный сбор всех доступных данных
        
        Args:
            start_date: Начальная дата (если None, берется с начала работы Binance - 2017-08-01)
            end_date: Конечная дата (если None, текущая дата)
            interval: Интервал для сбора OHLCV (минимальный доступный, например '1m', '5m')
            target_interval: Целевой интервал для агрегации (например '15m', '1h', '1d')
            include_all_timeframes: Собирать ли данные для всех таймфреймов
        
        Returns:
            DataFrame со всеми собранными данными
        """
        # Определяем период
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            # Binance начал работу в августе 2017
            start_date = datetime(2017, 8, 1)
        
        logger.info("="*70)
        logger.info("🚀 КОМПЛЕКСНЫЙ СБОР ДАННЫХ С BINANCE")
        logger.info("="*70)
        logger.info(f"📊 Символ: {self.symbol}")
        logger.info(f"📅 Период: {start_date.date()} - {end_date.date()}")
        logger.info(f"⏱️  Интервал сбора: {interval}")
        logger.info(f"⏱️  Целевой интервал: {target_interval}")
        logger.info(f"⏰ Это может занять значительное время...")
        
        # Мониторинг памяти
        initial_memory = get_memory_usage_mb()
        if initial_memory > 0:
            logger.info(f"💾 Начальное использование памяти: {initial_memory:.2f} MB")
        logger.info("="*70)
        
        # 1. Собираем OHLCV данные (основа)
        logger.info("\n📊 Шаг 1/5: Сбор OHLCV данных...")
        ohlcv_df = self.get_klines_batch(interval, start_date, end_date)
        
        if ohlcv_df.empty:
            logger.error("❌ Не удалось получить OHLCV данные!")
            return pd.DataFrame()
        
        # 2. Собираем агрегированные сделки (рыночные покупки и продажи)
        # Пропускаем если период слишком большой (> 90 дней) или явно указано пропустить
        period_days = (end_date - start_date).days
        if skip_aggregated_trades or period_days > 90:
            if period_days > 90:
                logger.warning(f"\n⚠️ Период {period_days} дней слишком большой для агрегированных сделок.")
                logger.warning("   Сбор может занять десятки часов. Пропускаем агрегированные сделки.")
                logger.warning("   Используйте меньший период или запустите с skip_aggregated_trades=False для полного сбора.")
            else:
                logger.info("\n📊 Шаг 2/5: Пропуск агрегированных сделок (skip_aggregated_trades=True)...")
            trades_df = pd.DataFrame()
        else:
            logger.info("\n📊 Шаг 2/5: Сбор агрегированных сделок (рыночные объемы)...")
            trades_df = self.get_aggregated_trades_batch(start_date, end_date)
        
        # 3. Собираем ликвидации
        logger.info("\n📊 Шаг 3/5: Сбор данных о ликвидациях...")
        liquidations_df = self.get_liquidations_batch(start_date, end_date)
        
        # 4. Собираем историю открытого интереса
        logger.info("\n📊 Шаг 4/5: Сбор истории открытого интереса...")
        oi_period = target_interval if target_interval in ['5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'] else '15m'
        oi_df = self.get_open_interest_history_batch(start_date, end_date, period=oi_period)
        
        # 5. Собираем периодические снимки order book (лимитные заявки на глубинах 3%, 8%, 15%, 60%)
        logger.info("\n📊 Шаг 5/5: Сбор данных order book (лимитные заявки на разных глубинах)...")
        logger.info("   ⚠️  Важно: Binance не предоставляет историю order book.")
        logger.info("   📸 Собираем периодические снимки текущего состояния order book.")
        
        # Определяем sample_rate в зависимости от размера данных
        # Чем больше данных, тем реже делаем снимки
        if len(ohlcv_df) > 10000:
            sample_rate = 50  # Каждую 50-ю свечу
        elif len(ohlcv_df) > 5000:
            sample_rate = 30  # Каждую 30-ю свечу
        elif len(ohlcv_df) > 1000:
            sample_rate = 20  # Каждую 20-ю свечу
        else:
            sample_rate = 10  # Каждую 10-ю свечу
        
        order_book_df = self.collect_order_book_snapshots(ohlcv_df, sample_rate=sample_rate)
        
        # 6. Объединяем все данные
        logger.info("\n📊 Объединение всех данных...")
        result_df = self.combine_all_data(
            ohlcv_df, 
            trades_df, 
            liquidations_df, 
            oi_df,
            order_book_df=order_book_df,
            target_interval=target_interval
        )
        
        logger.info("\n" + "="*70)
        logger.info("✅ СБОР ДАННЫХ ЗАВЕРШЕН!")
        logger.info("="*70)
        logger.info(f"📊 Всего записей: {len(result_df)}")
        logger.info(f"📅 Период данных: {result_df.index[0]} - {result_df.index[-1]}")
        logger.info(f"📈 Колонок: {len(result_df.columns)}")
        logger.info(f"📋 Колонки: {', '.join(result_df.columns)}")
        
        return result_df


def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Комплексный сбор данных с Binance')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Торговая пара')
    parser.add_argument('--start-date', type=str, help='Начальная дата (YYYY-MM-DD). По умолчанию: 2017-08-01')
    parser.add_argument('--end-date', type=str, help='Конечная дата (YYYY-MM-DD). По умолчанию: сегодня')
    parser.add_argument('--interval', type=str, default='15m', help='Интервал сбора OHLCV (например, 1m, 5m, 15m, 1h, 1d)')
    parser.add_argument('--target-interval', type=str, default='15m', help='Целевой интервал агрегации (например, 15m, 1h, 1d)')
    parser.add_argument('--output', type=str, help='Путь для сохранения файла (опционально)')
    parser.add_argument('--skip-trades', action='store_true', help='Пропустить сбор агрегированных сделок (быстрее для больших периодов)')
    
    args = parser.parse_args()
    
    collector = ComprehensiveBinanceCollector(symbol=args.symbol)
    
    # Парсим даты
    start_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    
    end_date = None
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Собираем данные
    df = collector.collect_comprehensive_data(
        start_date=start_date,
        end_date=end_date,
        interval=args.interval,
        target_interval=args.target_interval,
        skip_aggregated_trades=args.skip_trades
    )
    
    if not df.empty:
        # Определяем имя файла
        if args.output:
            filepath = Path(args.output)
        else:
            start_str = df.index[0].strftime('%Y%m%d')
            end_str = df.index[-1].strftime('%Y%m%d')
            filename = f"{args.symbol}_comprehensive_{args.target_interval}_{start_str}_{end_str}.csv"
            filepath = config.DATA_DIR / "historical" / filename
        
        # Сохраняем данные
        logger.info(f"\n💾 Сохранение данных в {filepath}...")
        save_data(df, filepath, format="csv")
        
        logger.info(f"\n✅ Данные успешно сохранены!")
        logger.info(f"📁 Файл: {filepath}")
        logger.info(f"📊 Размер: {len(df)} строк × {len(df.columns)} колонок")
        logger.info(f"\n📋 Предпросмотр данных:")
        logger.info(f"\n{df.head(10)}")
        logger.info(f"\n...")
        logger.info(f"\n{df.tail(10)}")
    else:
        logger.error("❌ Не удалось собрать данные!")


if __name__ == "__main__":
    main()
