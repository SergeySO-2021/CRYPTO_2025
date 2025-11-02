"""
Конфигурация для работы с Binance API
"""

import os
from pathlib import Path

# Базовый путь к проекту
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "binance_data_collector" / "data"

# Создаем папки, если их нет
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "historical").mkdir(exist_ok=True)
(DATA_DIR / "realtime").mkdir(exist_ok=True)
(DATA_DIR / "processed").mkdir(exist_ok=True)

# Binance API настройки
BINANCE_API_BASE_URL = "https://api.binance.com/api/v3"
BINANCE_WS_BASE_URL = "wss://stream.binance.com:9443/ws"

# API ключи (загружаются из переменных окружения)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Настройки запросов
REQUEST_TIMEOUT = 30  # секунды
REQUEST_DELAY = 0.1   # задержка между запросами (секунды)
MAX_RETRIES = 3       # максимальное количество повторов при ошибке

# Лимиты API
RATE_LIMIT_WEIGHT_PER_MINUTE = 1200
RATE_LIMIT_ORDERS_PER_SECOND = 10

# Настройки данных
SUPPORTED_TIMEFRAMES = [
    "1m", "3m", "5m", "15m", "30m", 
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M"
]

DEFAULT_TIMEFRAMES = ["15m", "1h", "4h", "1d"]
DEFAULT_SYMBOL = "BTCUSDT"

# Форматы сохранения данных
EXPORT_FORMATS = ["csv", "json", "parquet", "xlsx"]

# Настройки WebSocket
WS_RECONNECT_DELAY = 5  # секунды между переподключениями
WS_HEARTBEAT_INTERVAL = 60  # секунды между heartbeat


