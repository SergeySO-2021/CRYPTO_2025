"""
Скрипт для сбора расширенных данных по BTC в реальном времени через WebSocket
Включает: order book depth, рыночные объемы, ликвидации, открытый интерес
Все данные агрегируются и сохраняются по 15-минутным интервалам
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from collections import deque
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent.parent))

from binance_data_collector.config import config
from binance_data_collector.utils.logger import setup_logger
from binance_data_collector.utils.file_handler import save_data

try:
    from binance import ThreadedWebsocketManager
    from binance.client import Client
except ImportError:
    print("❌ Не установлена библиотека python-binance!")
    print("   Установите: pip install python-binance")
    sys.exit(1)

logger = setup_logger("realtime_advanced_btc")

class RealtimeAdvancedBTCCollector:
    """Коллектор расширенных данных в реальном времени"""
    
    def __init__(self):
        if config.BINANCE_API_KEY and config.BINANCE_API_SECRET:
            self.client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
        else:
            self.client = Client()
        
        self.symbol = "BTCUSDT"
        self.interval = "15m"
        self.depths = [0.03, 0.08, 0.15, 0.60]  # 3%, 8%, 15%, 60%
        
        self.twm = None
        self.is_running = False
        
        # Буферы данных
        self.order_book_buffer = deque(maxlen=1000)
        self.trades_buffer = deque(maxlen=10000)
        self.liquidations_buffer = deque(maxlen=1000)
        self.oi_buffer = deque(maxlen=1000)
        
        # Текущий order book snapshot
        self.current_order_book = None
        self.current_price = None
        
        # Данные для агрегации
        self.aggregation_start = None
        self.current_15m_data = {}
    
    def process_order_book_update(self, msg):
        """Обработка обновления order book"""
        try:
            data = msg.get('data', {})
            if data.get('s') == self.symbol:
                # Обновляем текущий order book
                bids = [(float(b[0]), float(b[1])) for b in data.get('b', [])]
                asks = [(float(a[0]), float(a[1])) for a in data.get('a', [])]
                
                self.current_order_book = {
                    'bids': bids,
                    'asks': asks,
                    'timestamp': datetime.now()
                }
                
                # Вычисляем объемы на разных глубинах
                if self.current_price:
                    order_book_depths = {}
                    for depth in self.depths:
                        depth_data = self._calculate_depth_volume(depth)
                        order_book_depths[f"{int(depth * 100)}pct"] = depth_data
                    
                    self.order_book_buffer.append({
                        'timestamp': datetime.now(),
                        'price': self.current_price,
                        **order_book_depths
                    })
        
        except Exception as e:
            logger.error(f"❌ Ошибка обработки order book: {e}")
    
    def _calculate_depth_volume(self, depth_percent: float) -> Dict:
        """Вычисление объемов на заданной глубине"""
        if not self.current_order_book or not self.current_price:
            return {'bid_volume': 0, 'ask_volume': 0, 'total_volume': 0, 'imbalance': 0}
        
        bid_threshold = self.current_price * (1 - depth_percent)
        ask_threshold = self.current_price * (1 + depth_percent)
        
        bid_volume = sum(vol for price, vol in self.current_order_book['bids'] if price >= bid_threshold)
        ask_volume = sum(vol for price, vol in self.current_order_book['asks'] if price <= ask_threshold)
        
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'total_volume': total_volume,
            'imbalance': imbalance
        }
    
    def process_trade_update(self, msg):
        """Обработка обновления сделок"""
        try:
            data = msg.get('data', {})
            if data.get('s') == self.symbol:
                trade = {
                    'timestamp': pd.to_datetime(data['T'], unit='ms'),
                    'price': float(data['p']),
                    'quantity': float(data['q']),
                    'is_buyer_maker': data['m'],  # True = продажа, False = покупка
                    'buy_volume': float(data['q']) if not data['m'] else 0,
                    'sell_volume': float(data['q']) if data['m'] else 0
                }
                
                self.trades_buffer.append(trade)
                self.current_price = trade['price']
        
        except Exception as e:
            logger.error(f"❌ Ошибка обработки сделки: {e}")
    
    def process_liquidation_update(self, msg):
        """Обработка обновления ликвидаций"""
        try:
            data = msg.get('data', {})
            if data.get('s') == self.symbol:
                liquidation = {
                    'timestamp': pd.to_datetime(data['T'], unit='ms'),
                    'side': data.get('S', 'UNKNOWN'),  # LONG или SHORT
                    'quantity': float(data.get('q', 0)),
                    'price': float(data.get('p', 0))
                }
                
                self.liquidations_buffer.append(liquidation)
        
        except Exception as e:
            logger.error(f"❌ Ошибка обработки ликвидации: {e}")
    
    def process_open_interest_update(self, msg):
        """Обработка обновления открытого интереса"""
        try:
            data = msg.get('data', {})
            if data.get('symbol') == self.symbol:
                oi_update = {
                    'timestamp': datetime.now(),
                    'open_interest': float(data.get('openInterest', 0)),
                    'sum_open_interest': float(data.get('sumOpenInterest', 0)),
                    'sum_open_interest_value': float(data.get('sumOpenInterestValue', 0))
                }
                
                self.oi_buffer.append(oi_update)
        
        except Exception as e:
            logger.error(f"❌ Ошибка обработки открытого интереса: {e}")
    
    def aggregate_15m_interval(self) -> pd.Series:
        """Агрегация данных за последний 15-минутный интервал"""
        now = datetime.now()
        interval_end = now.replace(second=0, microsecond=0)
        interval_end = interval_end - timedelta(minutes=interval_end.minute % 15)
        interval_start = interval_end - timedelta(minutes=15)
        
        # Агрегируем сделки
        trades_interval = [
            t for t in self.trades_buffer
            if interval_start <= t['timestamp'] <= interval_end
        ]
        
        market_buy_volume = sum(t['buy_volume'] for t in trades_interval)
        market_sell_volume = sum(t['sell_volume'] for t in trades_interval)
        
        # Агрегируем ликвидации
        liquidations_interval = [
            l for l in self.liquidations_buffer
            if interval_start <= pd.to_datetime(l['timestamp']) <= interval_end
        ]
        
        long_liquidations = sum(l['quantity'] for l in liquidations_interval if l['side'] == 'LONG')
        short_liquidations = sum(l['quantity'] for l in liquidations_interval if l['side'] == 'SHORT')
        total_liquidations = sum(l['quantity'] for l in liquidations_interval)
        
        # Берем последние значения order book для каждого depth
        order_book_data = {}
        for depth in self.depths:
            depth_key = f"{int(depth * 100)}pct"
            # Берем последнее значение из буфера
            if self.order_book_buffer:
                latest_ob = list(self.order_book_buffer)[-1]
                if depth_key in latest_ob:
                    order_book_data[f'bid_volume_{depth_key}'] = latest_ob[depth_key]['bid_volume']
                    order_book_data[f'ask_volume_{depth_key}'] = latest_ob[depth_key]['ask_volume']
                    order_book_data[f'total_volume_{depth_key}'] = latest_ob[depth_key]['total_volume']
                    order_book_data[f'imbalance_{depth_key}'] = latest_ob[depth_key]['imbalance']
                else:
                    order_book_data[f'bid_volume_{depth_key}'] = 0
                    order_book_data[f'ask_volume_{depth_key}'] = 0
                    order_book_data[f'total_volume_{depth_key}'] = 0
                    order_book_data[f'imbalance_{depth_key}'] = 0
        
        # Берем последний открытый интерес
        open_interest = 0
        if self.oi_buffer:
            open_interest = list(self.oi_buffer)[-1]['open_interest']
        
        # Формируем результат
        result = {
            'timestamp': interval_end,
            'market_buy_volume': market_buy_volume,
            'market_sell_volume': market_sell_volume,
            'long_liquidations': long_liquidations,
            'short_liquidations': short_liquidations,
            'total_liquidations': total_liquidations,
            'open_interest': open_interest,
            **order_book_data
        }
        
        return pd.Series(result)
    
    def start(self):
        """Запуск сбора данных"""
        logger.info(f"🚀 Запуск сбора расширенных данных для {self.symbol} в реальном времени")
        
        self.twm = ThreadedWebsocketManager()
        self.twm.start()
        
        # Подписываемся на потоки
        # 1. Order book depth (20 уровней)
        self.twm.start_depth_socket(
            callback=self.process_order_book_update,
            symbol=self.symbol,
            depth=20
        )
        
        # 2. Trades (сделки)
        self.twm.start_trade_socket(
            callback=self.process_trade_update,
            symbol=self.symbol
        )
        
        # 3. Liquidations (фьючерсы)
        self.twm.start_futures_socket(
            callback=self.process_liquidation_update,
            symbol=self.symbol.lower()
        )
        
        # 4. Open Interest (фьючерсы)
        self.twm.start_futures_socket(
            callback=self.process_open_interest_update,
            symbol=self.symbol.lower()
        )
        
        self.is_running = True
        self.aggregation_start = datetime.now()
        
        logger.info("✅ Подписки активны. Начинаем сбор данных...")
        logger.info("📊 Данные будут агрегироваться и сохраняться каждые 15 минут")
        logger.info("   Нажмите Ctrl+C для остановки")
        
        # Периодически сохраняем данные
        last_save_time = datetime.now()
        collected_intervals = []
        
        try:
            while self.is_running:
                time.sleep(1)
                
                # Проверяем, прошло ли 15 минут
                now = datetime.now()
                if (now - last_save_time).total_seconds() >= 900:  # 15 минут
                    # Агрегируем данные
                    interval_data = self.aggregate_15m_interval()
                    collected_intervals.append(interval_data)
                    
                    logger.info(f"📊 Агрегирован интервал: {interval_data['timestamp']}")
                    logger.info(f"   Buy volume: {interval_data['market_buy_volume']:.2f}")
                    logger.info(f"   Sell volume: {interval_data['market_sell_volume']:.2f}")
                    logger.info(f"   Total liquidations: {interval_data['total_liquidations']:.2f}")
                    
                    # Сохраняем накопленные данные
                    if collected_intervals:
                        df = pd.DataFrame(collected_intervals)
                        df.set_index('timestamp', inplace=True)
                        
                        filename = f"{self.symbol}_advanced_15m_realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        filepath = config.DATA_DIR / "realtime" / filename
                        save_data(df, filepath, format="csv")
                        
                        logger.info(f"💾 Данные сохранены: {filepath}")
                        collected_intervals.clear()
                    
                    last_save_time = now
        
        except KeyboardInterrupt:
            logger.info("\n⚠️ Получен сигнал остановки")
        finally:
            self.stop()
    
    def stop(self):
        """Остановка сбора данных"""
        logger.info("🛑 Остановка сбора данных...")
        
        if self.twm:
            self.twm.stop()
        
        self.is_running = False
        logger.info("✅ Сбор данных остановлен")

def main():
    collector = RealtimeAdvancedBTCCollector()
    collector.start()

if __name__ == "__main__":
    main()


