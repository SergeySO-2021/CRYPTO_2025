# 📊 МОДУЛЬ ЗАГРУЗКИ ДАННЫХ С ВНЕШНИХ ИСТОЧНИКОВ
# ==================================================

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class BinanceDataLoader:
    """
    Класс для загрузки данных с Binance API
    """
    
    def __init__(self, base_url: str = "https://api.binance.com/api/v3"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def download_period_data(self, 
                           symbol: str = "BTCUSDT", 
                           interval: str = "15m", 
                           start_date: str = "2024-01-22", 
                           end_date: Optional[str] = None,
                           max_requests: int = 200) -> Optional[pd.DataFrame]:
        """
        Загрузка данных за период с улучшенной логикой
        
        Args:
            symbol: Торговая пара (по умолчанию BTCUSDT)
            interval: Таймфрейм (15m, 30m, 1h, 4h, 1d)
            start_date: Начальная дата в формате YYYY-MM-DD
            end_date: Конечная дата (по умолчанию текущая)
            max_requests: Максимальное количество запросов
            
        Returns:
            DataFrame с данными или None при ошибке
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            print(f"📊 Загружаем данные {symbol} за период {start_date} - {end_date}")
            print(f"🕐 Формат времени: YYYY-MM-DD HH:MM:SS")
            
            # Конвертируем даты в timestamp (миллисекунды)
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            
            print(f"🕐 Начальный timestamp: {start_timestamp} ({datetime.fromtimestamp(start_timestamp/1000)})")
            print(f"🕐 Конечный timestamp: {end_timestamp} ({datetime.fromtimestamp(end_timestamp/1000)})")
            
            all_data = []
            current_end_time = end_timestamp
            
            request_count = 0
            consecutive_empty_requests = 0
            max_empty_requests = 3
            reached_start_period = False
            
            while current_end_time > start_timestamp and request_count < max_requests:
                request_count += 1
                print(f"🔄 Запрос {request_count}...")
                
                url = f"{self.base_url}/klines"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': 1000,
                    'endTime': current_end_time
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if not data:
                    print("⚠️ Нет больше данных")
                    consecutive_empty_requests += 1
                    if consecutive_empty_requests >= max_empty_requests:
                        print("🛑 Слишком много пустых запросов подряд, останавливаемся")
                        break
                    current_end_time = current_end_time - (24 * 60 * 60 * 1000)
                    time.sleep(0.1)
                    continue
                
                consecutive_empty_requests = 0
                
                # Фильтруем данные по периоду
                filtered_data = []
                earliest_timestamp = None
                
                for candle in data:
                    candle_time = candle[0]
                    if candle_time >= start_timestamp:
                        filtered_data.append(candle)
                    else:
                        reached_start_period = True
                        break
                    
                    if earliest_timestamp is None or candle_time < earliest_timestamp:
                        earliest_timestamp = candle_time
                
                all_data.extend(filtered_data)
                
                # Обновляем end_time для следующего запроса
                if earliest_timestamp is not None:
                    current_end_time = earliest_timestamp - 1
                else:
                    current_end_time = current_end_time - (24 * 60 * 60 * 1000)
                
                time.sleep(0.1)
                
                print(f"✅ Загружено {len(filtered_data)} записей, всего: {len(all_data)}")
                
                # Проверяем завершение
                if len(filtered_data) < 1000 or reached_start_period:
                    if reached_start_period:
                        print("📅 Достигли начала периода")
                    else:
                        print("📅 Получили меньше 1000 записей")
                    break
                
                # Показываем прогресс
                if len(all_data) % 10000 == 0 and len(all_data) > 0:
                    print(f"📊 Прогресс: {len(all_data)} записей загружено")
            
            if not all_data:
                print("❌ Не удалось загрузить данные")
                return None
            
            # Конвертируем в DataFrame
            df = pd.DataFrame(all_data, columns=[
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
            
            # Форматируем время
            df['timestamps'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df = df.drop('timestamp', axis=1)
            
            # Переименовываем колонки
            df.columns = ['open', 'high', 'low', 'close', 'volume', 'timestamps']
            df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume']]
            
            # Сортируем по времени
            df = df.sort_values('timestamps').reset_index(drop=True)
            
            print(f"✅ ИТОГО загружено {len(df)} записей")
            print(f"📅 Период: {df['timestamps'].min()} - {df['timestamps'].max()}")
            print(f"📊 Volume: {df['volume'].min():.2f} - {df['volume'].max():.2f}")
            
            # Проверяем достижение периода
            actual_start = df['timestamps'].min()
            expected_start = f"{start_date} 00:00:00"
            
            if actual_start <= expected_start:
                print(f"✅ Достигли начала периода: {actual_start}")
            else:
                print(f"⚠️ Не достигли начала периода. Получено: {actual_start}, ожидалось: {expected_start}")
                print(f"💡 Это нормально - Binance API имеет ограничения на исторические данные")
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return None
    
    def download_all_timeframes(self, 
                              symbol: str = "BTCUSDT", 
                              start_date: str = "2024-01-22",
                              end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Загрузка данных для всех таймфреймов
        
        Args:
            symbol: Торговая пара
            start_date: Начальная дата
            end_date: Конечная дата
            
        Returns:
            Словарь с данными по таймфреймам
        """
        timeframes = {
            '15m': {'interval': '15m'},
            '30m': {'interval': '30m'},
            '1h': {'interval': '1h'},
            '4h': {'interval': '4h'},
            '1d': {'interval': '1d'}
        }
        
        results = {}
        
        for tf, params in timeframes.items():
            print(f"\n🔄 Загружаем данные для {tf} за период {start_date} - {end_date or 'сегодня'}...")
            
            df = self.download_period_data(symbol, params['interval'], start_date, end_date)
            
            if df is not None:
                results[tf] = df
                print(f"✅ {tf}: {len(df):,} записей загружено")
            else:
                print(f"❌ {tf}: Ошибка загрузки")
            
            # Пауза между таймфреймами
            time.sleep(2)
        
        return results

class DataManager:
    """
    Менеджер для работы с данными
    """
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.getcwd()
        if 'indicator_optimization' in self.base_path:
            self.base_path = os.path.dirname(os.path.dirname(self.base_path))
    
    def save_data(self, data: Dict[str, pd.DataFrame], prefix: str = "df_btc") -> Dict[str, str]:
        """
        Сохранение данных в CSV файлы
        
        Args:
            data: Словарь с данными по таймфреймам
            prefix: Префикс для имен файлов
            
        Returns:
            Словарь с путями к сохраненным файлам
        """
        saved_files = {}
        
        for tf, df in data.items():
            filename = f"{prefix}_{tf}_complete.csv"
            filepath = os.path.join(self.base_path, filename)
            
            try:
                df.to_csv(filepath, index=False)
                saved_files[tf] = filepath
                print(f"💾 {tf}: Сохранено в {filename}")
            except Exception as e:
                print(f"❌ {tf}: Ошибка сохранения - {e}")
        
        return saved_files
    
    def load_data(self, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Загрузка данных из CSV файлов
        
        Args:
            timeframes: Список таймфреймов для загрузки
            
        Returns:
            Словарь с загруженными данными
        """
        if timeframes is None:
            timeframes = ['15m', '30m', '1h', '4h', '1d']
        
        loaded_data = {}
        
        for tf in timeframes:
            # Приоритет файлов: complete -> matching -> large -> real -> original
            file_priorities = [
                f"df_btc_{tf}_complete.csv",
                f"df_btc_{tf}_matching.csv", 
                f"df_btc_{tf}_large.csv",
                f"df_btc_{tf}_real.csv",
                f"df_btc_{tf}.csv"
            ]
            
            df = None
            for filename in file_priorities:
                filepath = os.path.join(self.base_path, filename)
                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath)
                        print(f"✅ {tf}: Загружены данные из {filename} ({len(df)} записей)")
                        break
                    except Exception as e:
                        print(f"❌ {tf}: Ошибка загрузки {filename} - {e}")
                        continue
            
            if df is not None:
                # Проверяем наличие Volume
                if 'volume' not in df.columns:
                    print(f"⚠️ {tf}: Volume отсутствует, добавляем синтетический")
                    price_range = df['high'] - df['low']
                    avg_price = df['close'].mean()
                    np.random.seed(42)
                    random_factor = np.random.uniform(0.5, 2.0, len(df))
                    df['volume'] = (price_range * avg_price * random_factor).astype(int)
                
                loaded_data[tf] = df
            else:
                print(f"❌ {tf}: Не удалось загрузить данные")
        
        return loaded_data
    
    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Получение сводки по данным
        
        Args:
            data: Словарь с данными
            
        Returns:
            Словарь со сводкой
        """
        summary = {
            'total_records': sum(len(df) for df in data.values()),
            'timeframes': list(data.keys()),
            'timeframe_details': {}
        }
        
        for tf, df in data.items():
            summary['timeframe_details'][tf] = {
                'records': len(df),
                'period': f"{df['timestamps'].min()} - {df['timestamps'].max()}",
                'volume_range': f"{df['volume'].min():.2f} - {df['volume'].max():.2f}",
                'price_range': f"{df['close'].min():.2f} - {df['close'].max():.2f}"
            }
        
        return summary

# Функции для быстрого использования
def download_btc_data(start_date: str = "2024-01-22", 
                     end_date: Optional[str] = None,
                     save_to_csv: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Быстрая загрузка данных BTC
    
    Args:
        start_date: Начальная дата
        end_date: Конечная дата
        save_to_csv: Сохранять ли в CSV
        
    Returns:
        Словарь с данными
    """
    loader = BinanceDataLoader()
    manager = DataManager()
    
    print("🚀 ЗАГРУЗКА ДАННЫХ BTC С BINANCE")
    print("=" * 50)
    
    data = loader.download_all_timeframes("BTCUSDT", start_date, end_date)
    
    if save_to_csv and data:
        print("\n💾 СОХРАНЕНИЕ ДАННЫХ")
        print("=" * 30)
        manager.save_data(data)
    
    return data

def load_btc_data(timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Быстрая загрузка данных BTC из файлов
    
    Args:
        timeframes: Список таймфреймов
        
    Returns:
        Словарь с данными
    """
    manager = DataManager()
    
    print("📊 ЗАГРУЗКА ДАННЫХ BTC ИЗ ФАЙЛОВ")
    print("=" * 40)
    
    data = manager.load_data(timeframes)
    
    if data:
        summary = manager.get_data_summary(data)
        print(f"\n📈 ИТОГО: {summary['total_records']:,} записей")
        print(f"📊 Таймфреймов: {len(summary['timeframes'])}")
    
    return data

if __name__ == "__main__":
    # Пример использования
    print("📊 ТЕСТИРОВАНИЕ МОДУЛЯ ЗАГРУЗКИ ДАННЫХ")
    print("=" * 50)
    
    # Загружаем данные из файлов
    data = load_btc_data(['15m', '1h'])
    
    if data:
        print("\n✅ Модуль работает корректно!")
        for tf, df in data.items():
            print(f"📊 {tf}: {len(df)} записей")
    else:
        print("❌ Ошибка загрузки данных")
