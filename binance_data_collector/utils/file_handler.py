"""
Утилиты для работы с файлами и экспорта данных
"""

import pandas as pd
import json
from pathlib import Path
from typing import Union, Optional

def save_data(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    format: str = "csv",
    **kwargs
) -> None:
    """
    Сохранение данных в различных форматах
    
    Args:
        df: DataFrame для сохранения
        filepath: Путь к файлу
        format: Формат сохранения (csv, json, parquet, xlsx)
        **kwargs: Дополнительные параметры для методов сохранения
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "csv":
        df.to_csv(filepath, index=True, encoding='utf-8', **kwargs)
    elif format.lower() == "json":
        df.to_json(filepath, orient='records', date_format='iso', **kwargs)
    elif format.lower() == "parquet":
        df.to_parquet(filepath, **kwargs)
    elif format.lower() == "xlsx":
        df.to_excel(filepath, index=True, **kwargs)
    else:
        raise ValueError(f"Неподдерживаемый формат: {format}")
    
    print(f"✅ Данные сохранены: {filepath}")

def load_data(
    filepath: Union[str, Path],
    format: Optional[str] = None
) -> pd.DataFrame:
    """
    Загрузка данных из файла
    
    Args:
        filepath: Путь к файлу
        format: Формат файла (определяется автоматически, если None)
    
    Returns:
        DataFrame с данными
    """
    filepath = Path(filepath)
    
    if format is None:
        format = filepath.suffix[1:].lower()  # Без точки
    
    if format == "csv":
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    elif format == "json":
        df = pd.read_json(filepath, orient='records')
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
    elif format == "parquet":
        df = pd.read_parquet(filepath)
    elif format == "xlsx":
        df = pd.read_excel(filepath, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Неподдерживаемый формат: {format}")
    
    return df


