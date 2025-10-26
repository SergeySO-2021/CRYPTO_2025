"""
Пакет классификаторов рыночных зон
Версия 2.0.0 - Обновлено после архивирования устаревших файлов
"""

from .base_classifier import BaseMarketZoneClassifier
from .mza_classifier_vectorized import VectorizedMZAClassifier
from .ml_classifier_optimized import OptimizedMarketRegimeMLClassifier
from .trend_classifier import TrendClassifier

__all__ = [
    'BaseMarketZoneClassifier',
    'VectorizedMZAClassifier',
    'OptimizedMarketRegimeMLClassifier', 
    'TrendClassifier'
]

__version__ = '2.0.0'

# Информация об архивированных файлах:
# Устаревшие файлы перемещены в archive/classifiers/:
# - mza_classifier.py (заменен на mza_classifier_vectorized.py)
# - mza_classifier_proper.py (заменен на mza_classifier_vectorized.py)
# - ml_classifier.py (заменен на ml_classifier_optimized.py)
# - trading_classifier.py (не используется в финальных исследованиях)