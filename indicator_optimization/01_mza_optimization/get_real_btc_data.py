#!/usr/bin/env python3
"""
Скрипт для получения реальных данных BTC с Volume
Использует различные источники для получения качественных данных
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import os

class BTCDataDownloader:
    """Класс для загрузки реальных данных BTC с Volume"""
    
    def __init__(self):
        self.data_dir = "../../"  # Корневая директория проекта
        
    def download_from_binance(self, symbol="BTCUSDT", interval="15m", limit=1000):
        """
        Загрузка данных с Binance API
        """
        try:
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            print(f"📊 Загружаем данные {symbol} с Binance...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Конвертируем в DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Оставляем только нужные колонки
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Конвертируем типы данных
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Переименовываем колонки
            df.columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume']
            
            print(f"✅ Загружено {len(df)} записей")
            print(f"📅 Период: {df['timestamps'].min()} - {df['timestamps'].max()}")
            print(f"📊 Volume: {df['volume'].min():.2f} - {df['volume'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка загрузки с Binance: {e}")
            return None
    
    def download_from_yahoo(self, symbol="BTC-USD", period="1y", interval="15m"):
        """
        Загрузка данных с Yahoo Finance (если доступно)
        """
        try:
            import yfinance as yf
            
            print(f"📊 Загружаем данные {symbol} с Yahoo Finance...")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                print("❌ Данные не найдены")
                return None
            
            # Переименовываем колонки
            df.reset_index(inplace=True)
            df.columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
            df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume']]
            
            print(f"✅ Загружено {len(df)} записей")
            print(f"📅 Период: {df['timestamps'].min()} - {df['timestamps'].max()}")
            print(f"📊 Volume: {df['volume'].min():.2f} - {df['volume'].max():.2f}")
            
            return df
            
        except ImportError:
            print("❌ yfinance не установлен. Установите: pip install yfinance")
            return None
        except Exception as e:
            print(f"❌ Ошибка загрузки с Yahoo: {e}")
            return None
    
    def save_data(self, df, timeframe):
        """Сохранение данных в CSV"""
        if df is None or df.empty:
            print("❌ Нет данных для сохранения")
            return False
        
        filename = f"{self.data_dir}df_btc_{timeframe}_with_volume.csv"
        
        try:
            df.to_csv(filename, index=False)
            print(f"✅ Данные сохранены в {filename}")
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return False
    
    def download_all_timeframes(self):
        """Загрузка данных для всех таймфреймов"""
        timeframes = {
            '15m': {'interval': '15m', 'limit': 2000},
            '30m': {'interval': '30m', 'limit': 2000},
            '1h': {'interval': '1h', 'limit': 2000},
            '4h': {'interval': '4h', 'limit': 2000},
            '1d': {'interval': '1d', 'limit': 1000}
        }
        
        results = {}
        
        for tf, params in timeframes.items():
            print(f"\n🔄 Загружаем данные для {tf}...")
            
            # Пробуем Binance
            df = self.download_from_binance(**params)
            
            if df is not None:
                if self.save_data(df, tf):
                    results[tf] = True
                else:
                    results[tf] = False
            else:
                print(f"❌ Не удалось загрузить данные для {tf}")
                results[tf] = False
            
            # Пауза между запросами
            time.sleep(1)
        
        return results

def main():
    """Основная функция"""
    print("🚀 ЗАГРУЗКА РЕАЛЬНЫХ ДАННЫХ BTC С VOLUME")
    print("=" * 50)
    
    downloader = BTCDataDownloader()
    
    # Загружаем данные для всех таймфреймов
    results = downloader.download_all_timeframes()
    
    print("\n📊 РЕЗУЛЬТАТЫ ЗАГРУЗКИ:")
    print("=" * 30)
    
    for tf, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {tf}: {'Успешно' if success else 'Ошибка'}")
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"\n🎯 ИТОГО: {successful}/{total} таймфреймов загружено")
    
    if successful > 0:
        print("\n💡 Теперь можно использовать реальные данные с Volume!")
        print("📝 Обновите пути в test_complete_mza_notebook.ipynb")
    else:
        print("\n⚠️ Не удалось загрузить данные. Рассмотрите альтернативные варианты.")

if __name__ == "__main__":
    main()
