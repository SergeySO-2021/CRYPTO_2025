"""
Скрипт для получения данных в реальном времени через WebSocket
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime
import pandas as pd

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent.parent))

from binance_data_collector.config import config
from binance_data_collector.utils.logger import setup_logger
from binance_data_collector.utils.file_handler import save_data

try:
    from binance import ThreadedWebsocketManager
except ImportError:
    print("❌ Не установлена библиотека python-binance!")
    print("   Установите: pip install python-binance")
    sys.exit(1)

logger = setup_logger("binance_realtime")

class BinanceRealtimeCollector:
    """Коллектор данных в реальном времени"""
    
    def __init__(self, symbol: str = None, interval: str = "1m"):
        self.symbol = symbol or config.DEFAULT_SYMBOL
        self.interval = interval
        self.twm = None
        self.data_buffer = []
        self.is_running = False
    
    def handle_socket_message(self, msg):
        """Обработка сообщения от WebSocket"""
        try:
            # Извлекаем данные свечи
            kline = msg['k']
            
            if kline['x']:  # Свеча закрылась
                data_point = {
                    'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                }
                
                self.data_buffer.append(data_point)
                logger.info(f"📊 Новая свеча: {data_point['close']:.2f} ({data_point['timestamp']})")
                
                # Сохраняем в файл каждые N свечей
                if len(self.data_buffer) >= 100:
                    self.save_buffer()
        
        except Exception as e:
            logger.error(f"❌ Ошибка обработки сообщения: {e}")
    
    def save_buffer(self):
        """Сохранение буфера данных в файл"""
        if not self.data_buffer:
            return
        
        df = pd.DataFrame(self.data_buffer)
        df.set_index('timestamp', inplace=True)
        
        filename = f"{self.symbol}_{self.interval}_realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = config.DATA_DIR / "realtime" / filename
        
        save_data(df, filepath, format="csv")
        
        logger.info(f"💾 Сохранено {len(self.data_buffer)} свечей")
        self.data_buffer.clear()
    
    def start(self):
        """Запуск сбора данных"""
        logger.info(f"🚀 Запуск сбора данных в реальном времени для {self.symbol} ({self.interval})")
        
        self.twm = ThreadedWebsocketManager()
        self.twm.start()
        
        # Подписываемся на поток свечей
        self.twm.start_kline_socket(
            callback=self.handle_socket_message,
            symbol=self.symbol,
            interval=self.interval
        )
        
        self.is_running = True
        logger.info("✅ Подписка активна. Нажмите Ctrl+C для остановки.")
    
    def stop(self):
        """Остановка сбора данных"""
        logger.info("🛑 Остановка сбора данных...")
        
        if self.twm:
            self.twm.stop()
        
        # Сохраняем оставшиеся данные
        if self.data_buffer:
            self.save_buffer()
        
        self.is_running = False
        logger.info("✅ Сбор данных остановлен")

def main():
    """Основная функция"""
    collector = BinanceRealtimeCollector(
        symbol=config.DEFAULT_SYMBOL,
        interval="1m"
    )
    
    try:
        collector.start()
        
        # Ожидание
        while collector.is_running:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ Получен сигнал остановки")
    finally:
        collector.stop()

if __name__ == "__main__":
    main()

