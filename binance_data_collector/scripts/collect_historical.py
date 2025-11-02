"""
Скрипт для сбора исторических данных с Binance
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import time

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

logger = setup_logger("binance_collector")

def get_klines(
    client: Client,
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Получение исторических данных (свечей) с Binance
    
    Args:
        client: Клиент Binance API
        symbol: Торговая пара (например, BTCUSDT)
        interval: Интервал (1m, 5m, 1h, 1d и т.д.)
        start_time: Начальное время
        end_time: Конечное время
        limit: Максимальное количество свечей за запрос (до 1000)
    
    Returns:
        DataFrame с OHLCV данными
    """
    all_klines = []
    current_start = start_time
    
    logger.info(f"📊 Сбор данных для {symbol} ({interval}) с {start_time} по {end_time}")
    
    while current_start < end_time:
        try:
            # Получаем данные
            klines = client.get_klines(
                symbol=symbol,
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
            
            logger.info(f"   Получено {len(klines)} свечей (всего: {len(all_klines)})")
            
            # Задержка для соблюдения лимитов API
            time.sleep(config.REQUEST_DELAY)
            
        except Exception as e:
            logger.error(f"❌ Ошибка при получении данных: {e}")
            time.sleep(config.REQUEST_DELAY * 10)  # Увеличенная задержка при ошибке
    
    if not all_klines:
        logger.warning("⚠️ Данные не получены")
        return pd.DataFrame()
    
    # Преобразуем в DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Конвертируем типы
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    df['trades'] = df['trades'].astype(int)
    
    # Оставляем только нужные колонки
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    logger.info(f"✅ Получено всего {len(df)} свечей")
    
    return df

def main():
    """Основная функция"""
    logger.info("🚀 Запуск сбора исторических данных с Binance")
    
    # Инициализация клиента
    if config.BINANCE_API_KEY and config.BINANCE_API_SECRET:
        client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
    else:
        client = Client()  # Публичный API (без ключа)
    
    # Параметры сбора
    symbol = config.DEFAULT_SYMBOL
    timeframes = config.DEFAULT_TIMEFRAMES
    
    # Временной диапазон (последние 2 года)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=730)
    
    logger.info(f"📅 Период: {start_time.date()} - {end_time.date()}")
    
    # Собираем данные для каждого таймфрейма
    for timeframe in timeframes:
        logger.info(f"\n{'='*50}")
        logger.info(f"📊 Таймфрейм: {timeframe}")
        logger.info(f"{'='*50}")
        
        df = get_klines(client, symbol, timeframe, start_time, end_time)
        
        if not df.empty:
            # Сохраняем данные
            filename = f"{symbol}_{timeframe}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
            filepath = config.DATA_DIR / "historical" / filename
            save_data(df, filepath, format="csv")
        else:
            logger.warning(f"⚠️ Нет данных для {symbol} {timeframe}")
    
    logger.info("\n✅ Сбор данных завершен!")

if __name__ == "__main__":
    main()


