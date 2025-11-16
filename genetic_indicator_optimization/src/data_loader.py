"""
Модуль для загрузки и подготовки данных для генетической оптимизации
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import yaml


class DataLoader:
    """
    Класс для загрузки и подготовки данных
    """
    
    def __init__(self, data_path: str = None, config_path: str = None):
        """
        Инициализация загрузчика данных
        
        Args:
            data_path: Путь к файлу данных
            config_path: Путь к конфигурационному файлу
        """
        if data_path is None:
            # Путь по умолчанию
            project_root = Path(__file__).parent.parent
            data_path = project_root.parent / "dataframe" / "with_full_depth" / "df_btc_15m_complete.csv"
        
        self.data_path = Path(data_path)
        self.config_path = config_path
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Загрузка данных из CSV файла
        
        Returns:
            pd.DataFrame: Загруженные данные
        """
        print(f"📁 Загружаем данные из {self.data_path}...")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Файл данных не найден: {self.data_path}")
        
        # Загрузка данных
        df = pd.read_csv(self.data_path)
        
        # Преобразование timestamps
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df.set_index('timestamps', inplace=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Сортировка по времени
        df.sort_index(inplace=True)
        
        # Проверка наличия необходимых колонок
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Отсутствуют необходимые колонки: {missing_cols}")
        
        # Проверка на пропуски
        missing_count = df[required_cols].isnull().sum().sum()
        if missing_count > 0:
            print(f"⚠️  Обнаружено {missing_count} пропусков в данных")
            print("   Заполняем пропуски...")
            df[required_cols] = df[required_cols].fillna(method='ffill').fillna(method='bfill')
        
        self.data = df
        
        print(f"✅ Загружено {len(df)} записей")
        print(f"📅 Период: {df.index.min()} - {df.index.max()}")
        print(f"📊 Колонки: {list(df.columns)}")
        
        return df
    
    def split_data(self, train_split: float = 0.7, val_split: float = 0.15, 
                   test_split: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разделение данных на train/validation/test по времени
        
        Args:
            train_split: Доля данных для обучения (по умолчанию 0.7)
            val_split: Доля данных для валидации (по умолчанию 0.15)
            test_split: Доля данных для тестирования (по умолчанию 0.15)
        
        Returns:
            Tuple: (train_data, val_data, test_data)
        """
        if self.data is None:
            raise ValueError("Данные не загружены. Сначала вызовите load_data()")
        
        # Проверка суммы долей
        if abs(train_split + val_split + test_split - 1.0) > 0.01:
            raise ValueError("Сумма train_split + val_split + test_split должна быть равна 1.0")
        
        total_len = len(self.data)
        train_end = int(total_len * train_split)
        val_end = int(total_len * (train_split + val_split))
        
        # Разделение по времени (без перемешивания!)
        self.train_data = self.data.iloc[:train_end].copy()
        self.val_data = self.data.iloc[train_end:val_end].copy()
        self.test_data = self.data.iloc[val_end:].copy()
        
        print(f"\n📊 Разделение данных:")
        print(f"   Train:  {len(self.train_data)} записей ({len(self.train_data)/total_len*100:.1f}%)")
        print(f"            Период: {self.train_data.index.min()} - {self.train_data.index.max()}")
        print(f"   Val:    {len(self.val_data)} записей ({len(self.val_data)/total_len*100:.1f}%)")
        print(f"            Период: {self.val_data.index.min()} - {self.val_data.index.max()}")
        print(f"   Test:   {len(self.test_data)} записей ({len(self.test_data)/total_len*100:.1f}%)")
        print(f"            Период: {self.test_data.index.min()} - {self.test_data.index.max()}")
        
        return self.train_data, self.val_data, self.test_data
    
    def get_data_info(self) -> Dict:
        """
        Получение информации о данных
        
        Returns:
            Dict: Словарь с информацией о данных
        """
        if self.data is None:
            return {}
        
        info = {
            'total_records': len(self.data),
            'period_start': self.data.index.min(),
            'period_end': self.data.index.max(),
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Статистика по числовым колонкам
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        info['statistics'] = self.data[numeric_cols].describe().to_dict()
        
        return info
    
    def print_data_info(self):
        """
        Вывод информации о данных
        """
        info = self.get_data_info()
        
        print("\n" + "="*60)
        print("📊 ИНФОРМАЦИЯ О ДАННЫХ")
        print("="*60)
        print(f"Всего записей: {info.get('total_records', 'N/A')}")
        print(f"Период: {info.get('period_start', 'N/A')} - {info.get('period_end', 'N/A')}")
        print(f"Колонок: {len(info.get('columns', []))}")
        print(f"Использование памяти: {info.get('memory_usage_mb', 0):.2f} MB")
        
        print("\n📋 Колонки:")
        for col in info.get('columns', []):
            missing = info.get('missing_values', {}).get(col, 0)
            dtype = info.get('data_types', {}).get(col, 'unknown')
            print(f"   - {col}: {dtype} (пропусков: {missing})")
        
        print("\n📈 Статистика по основным колонкам:")
        stats = info.get('statistics', {})
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in stats:
                print(f"\n   {col.upper()}:")
                for stat, value in stats[col].items():
                    print(f"      {stat}: {value:.2f}")


def load_config(config_path: str = None) -> Dict:
    """
    Загрузка конфигурации из YAML файла
    
    Args:
        config_path: Путь к конфигурационному файлу
    
    Returns:
        Dict: Словарь с конфигурацией
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "ga_config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


if __name__ == "__main__":
    # Пример использования
    loader = DataLoader()
    
    # Загрузка данных
    data = loader.load_data()
    
    # Вывод информации
    loader.print_data_info()
    
    # Разделение данных
    train, val, test = loader.split_data()
    
    print("\n✅ Данные готовы к использованию!")

