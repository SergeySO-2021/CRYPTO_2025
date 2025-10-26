"""
ML-классификатор рыночных режимов
Адаптация из 11_adaptive_market_regime_ml_classifier.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class AdaptiveMarketRegimeMLClassifier:
    """Адаптивный ML-классификатор рыночных режимов с автоматической кластеризацией"""
    
    def __init__(self, n_clusters=4, method='kmeans'):
        self.n_clusters = n_clusters
        self.method = method
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.cluster_labels = []
        self.regime_names = {}
        
        # Инициализация модели в зависимости от метода
        if method == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=50)
        elif method == 'gmm':
            self.model = GaussianMixture(n_components=n_clusters, random_state=42)
        
        print(f"✅ AdaptiveMarketRegimeMLClassifier инициализирован!")
        print(f"🤖 Метод кластеризации: {method.upper()}")
        print(f"📊 Количество кластеров: {n_clusters}")
    
    def extract_classification_features(self, data):
        """Извлечение классификационных признаков для ML"""
        print("🔍 Извлекаем классификационные признаки...")
        
        # Убеждаемся, что data - это DataFrame
        if not isinstance(data, pd.DataFrame):
            print("⚠️ Конвертируем данные в DataFrame")
            data = pd.DataFrame(data)
        
        # Дополнительная проверка: убеждаемся, что у нас есть нужные колонки
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            print(f"⚠️ Отсутствуют нужные колонки. Доступные: {list(data.columns)}")
            # Если это numpy array, создаем DataFrame с правильными колонками
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
            else:
                # Пытаемся использовать первые 4 колонки как OHLC
                data = data.iloc[:, :4]
                data.columns = ['open', 'high', 'low', 'close']
        
        features = pd.DataFrame(index=data.index)
        
        # ===== ГРУППА 1: ТРЕНД И СИЛА =====
        
        # ADX - сила тренда (упрощенная версия)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        # Конвертируем numpy array в pandas Series
        true_range = pd.Series(true_range, index=data.index)
        atr = true_range.rolling(14).mean()
        
        # Упрощенный ADX
        plus_dm = np.where((data['high'].diff() > data['low'].diff().abs()) & (data['high'].diff() > 0), data['high'].diff(), 0)
        minus_dm = np.where((data['low'].diff().abs() > data['high'].diff()) & (data['low'].diff() < 0), data['low'].diff().abs(), 0)
        
        # Конвертируем numpy arrays обратно в pandas Series
        plus_dm = pd.Series(plus_dm, index=data.index)
        minus_dm = pd.Series(minus_dm, index=data.index)
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        # Конвертируем numpy array в pandas Series
        dx = pd.Series(dx, index=data.index)
        adx = dx.rolling(14).mean()
        
        features['adx_value'] = adx
        features['adx_trend'] = (adx > 25).astype(int)
        
        # ===== ГРУППА 2: ВОЛАТИЛЬНОСТЬ =====
        
        # Bollinger Bands Width
        bb_period = 20
        bb_std = 2
        bb_middle = data['close'].rolling(bb_period).mean()
        bb_std_val = data['close'].rolling(bb_period).std()
        bb_upper = bb_middle + (bb_std_val * bb_std)
        bb_lower = bb_middle - (bb_std_val * bb_std)
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        features['bb_width'] = bb_width
        features['bb_squeeze'] = (bb_width < bb_width.rolling(20).mean() * 0.8).astype(int)
        
        # ATR
        features['atr_value'] = atr
        features['atr_ratio'] = atr / atr.rolling(20).mean()
        
        # ===== ГРУППА 3: ИМПУЛЬС И ЦИКЛЫ =====
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        features['rsi_value'] = rsi
        features['rsi_trend'] = rsi.diff(5)
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - signal_line
        
        features['macd_histogram'] = macd_histogram
        features['macd_histogram_slope'] = macd_histogram.diff(5)
        
        # ===== ГРУППА 4: ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ =====
        
        # Позиция цены относительно скользящих средних
        ma_50 = data['close'].rolling(50).mean()
        ma_200 = data['close'].rolling(200).mean()
        
        features['price_vs_ma50'] = (data['close'] / ma_50 - 1) * 100
        features['price_vs_ma200'] = (data['close'] / ma_200 - 1) * 100
        features['ma50_vs_ma200'] = (ma_50 / ma_200 - 1) * 100
        
        # Динамика объема
        if 'volume' in data.columns:
            volume_ma = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / volume_ma
            features['volume_trend'] = data['volume'].rolling(5).mean().diff(5)
        else:
            features['volume_ratio'] = 1.0
            features['volume_trend'] = 0
        
        # Убираем пропуски
        features = features.fillna(0)
        
        print(f"✅ Извлечено {len(features.columns)} признаков")
        return features
    
    def fit(self, data):
        """Обучение классификатора"""
        print("🤖 Обучение ML-классификатора...")
        
        # Извлекаем признаки
        features = self.extract_classification_features(data)
        
        # Нормализуем признаки
        features_scaled = self.scaler.fit_transform(features)
        
        # Обучаем модель
        if self.method == 'dbscan':
            self.model.fit(features_scaled)
            self.cluster_labels = self.model.labels_
        else:
            self.model.fit(features_scaled)
            self.cluster_labels = self.model.predict(features_scaled)
        
        # Интерпретируем кластеры
        self.regime_names = self.interpret_clusters(features, self.cluster_labels)
        
        print(f"✅ ML-классификатор обучен!")
        print(f"📊 Количество кластеров: {len(np.unique(self.cluster_labels))}")
        
        return self
    
    def predict(self, data):
        """Предсказание режимов"""
        print("🔮 Предсказание режимов...")
        
        # Извлекаем признаки
        features = self.extract_classification_features(data)
        
        # Нормализуем признаки
        features_scaled = self.scaler.transform(features)
        
        # Предсказываем кластеры
        if self.method == 'dbscan':
            cluster_labels = self.model.fit_predict(features_scaled)
        else:
            cluster_labels = self.model.predict(features_scaled)
        
        # Конвертируем в режимы (-1, 0, 1)
        predictions = []
        for label in cluster_labels:
            if label in self.regime_names:
                regime = self.regime_names[label]
                if 'bull' in regime.lower():
                    predictions.append(1)
                elif 'bear' in regime.lower():
                    predictions.append(-1)
                else:
                    predictions.append(0)
            else:
                predictions.append(0)
        
        print(f"✅ Предсказано {len(predictions)} режимов")
        return np.array(predictions)
    
    def interpret_clusters(self, features, cluster_labels):
        """Интерпретация кластеров"""
        regime_names = {}
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Шум в DBSCAN
                regime_names[label] = "Noise"
                continue
                
            cluster_mask = cluster_labels == label
            cluster_features = features[cluster_mask]
            
            # Анализируем характеристики кластера
            avg_rsi = cluster_features['rsi_value'].mean()
            avg_adx = cluster_features['adx_value'].mean()
            avg_bb_width = cluster_features['bb_width'].mean()
            avg_price_vs_ma50 = cluster_features['price_vs_ma50'].mean()
            
            # Определяем режим на основе характеристик
            if avg_rsi > 60 and avg_adx > 25 and avg_price_vs_ma50 > 0:
                regime_names[label] = "Strong Bull"
            elif avg_rsi < 40 and avg_adx > 25 and avg_price_vs_ma50 < 0:
                regime_names[label] = "Strong Bear"
            elif avg_bb_width < cluster_features['bb_width'].quantile(0.3):
                regime_names[label] = "Low Volatility Sideways"
            elif avg_bb_width > cluster_features['bb_width'].quantile(0.7):
                regime_names[label] = "High Volatility Sideways"
            else:
                regime_names[label] = "Sideways"
        
        print(f"📊 Интерпретация кластеров:")
        for label, regime in regime_names.items():
            print(f"   Кластер {label}: {regime}")
        
        return regime_names
