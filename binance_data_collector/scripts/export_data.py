"""
Скрипт для экспорта собранных данных в различные форматы
"""

import sys
from pathlib import Path
import pandas as pd

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent.parent))

from binance_data_collector.config import config
from binance_data_collector.utils.logger import setup_logger
from binance_data_collector.utils.file_handler import load_data, save_data

logger = setup_logger("binance_export")

def convert_timeframe_data(
    input_file: Path,
    output_format: str = "csv",
    output_dir: Path = None
) -> None:
    """
    Конвертация данных в другой формат
    
    Args:
        input_file: Путь к исходному файлу
        output_format: Целевой формат (csv, json, parquet, xlsx)
        output_dir: Директория для сохранения (по умолчанию processed/)
    """
    if output_dir is None:
        output_dir = config.DATA_DIR / "processed"
    
    logger.info(f"📤 Конвертация {input_file.name} -> {output_format}")
    
    # Определяем формат исходного файла
    source_format = input_file.suffix[1:].lower()
    
    # Загружаем данные
    df = load_data(input_file, format=source_format)
    
    if df.empty:
        logger.warning(f"⚠️ Файл пуст: {input_file}")
        return
    
    # Создаем имя выходного файла
    output_file = output_dir / f"{input_file.stem}.{output_format}"
    
    # Сохраняем в новом формате
    save_data(df, output_file, format=output_format)
    
    logger.info(f"✅ Сохранено: {output_file}")

def batch_convert(
    input_dir: Path,
    output_format: str = "parquet",
    output_dir: Path = None
) -> None:
    """
    Пакетная конвертация всех файлов в директории
    
    Args:
        input_dir: Директория с исходными файлами
        output_format: Целевой формат
        output_dir: Директория для сохранения
    """
    if output_dir is None:
        output_dir = config.DATA_DIR / "processed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📦 Пакетная конвертация из {input_dir} в {output_format}")
    
    # Находим все CSV файлы
    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"⚠️ Не найдено CSV файлов в {input_dir}")
        return
    
    logger.info(f"📊 Найдено {len(csv_files)} файлов")
    
    for csv_file in csv_files:
        try:
            convert_timeframe_data(csv_file, output_format, output_dir)
        except Exception as e:
            logger.error(f"❌ Ошибка при конвертации {csv_file.name}: {e}")
    
    logger.info("✅ Пакетная конвертация завершена")

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Экспорт данных Binance в различные форматы")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(config.DATA_DIR / "historical"),
        help="Директория с исходными файлами"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["csv", "json", "parquet", "xlsx"],
        default="parquet",
        help="Целевой формат"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.DATA_DIR / "processed"),
        help="Директория для сохранения"
    )
    
    args = parser.parse_args()
    
    batch_convert(
        Path(args.input_dir),
        args.output_format,
        Path(args.output_dir)
    )

if __name__ == "__main__":
    main()


