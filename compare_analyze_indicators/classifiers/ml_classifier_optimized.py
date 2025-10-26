"""
Оптимизированная версия ML-классификатора рыночных режимов
Улучшенные алгоритмы кластеризации и интерпретации
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


class OptimizedMarketRegimeMLClassifier:
    """Оптимизированный ML-классификатор рыночных режимов с улучшенными алгоритмами"""
    
    def __init__(self, n_clusters=4, method='kmeans'):
        self.n_clusters = n_clusters
        self.method = method
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.cluster_labels = []
        self.regime_names = {}
        
        # Оптимизированная инициализация модели
        if method == 'kmeans':
            # Улучшенный KMeans с лучшей инициализацией
            self.model = KMeans(
                n_clusters=n_clusters, 
                random_state=42, 
                n_init=20,  # Больше инициализаций
                max_iter=300,  # Больше итераций
                init='k-means++'  # Лучшая инициализация
            )
        elif method == 'dbscan':
            # Адаптивный DBSCAN - параметры будут настроены автоматически
            self.model = None  # Будет создан позже с оптимальными параметрами
        elif method == 'gmm':
            # Улучшенный GMM
            self.model = GaussianMixture(
                n_components=n_clusters, 
                random_state=42,
                covariance_type='full',  # Полная ковариационная матрица
                max_iter=200,
                init_params='kmeans'  # K-means инициализация
            )
        
        print(f"✅ OptimizedMarketRegimeMLClassifier инициализирован!")
        print(f"🤖 Метод кластеризации: {method.upper()}")
        print(f"📊 Количество кластеров: {n_clusters}")
    
    def extract_classification_features(self, data):
        """Извлечение классификационных признаков для ML (улучшенная версия)"""
        print("🔍 Извлекаем классификационные признаки (оптимизированная версия)...")
        
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
        
        # ===== ГРУППА 1: ТРЕНД И СИЛА (УЛУЧШЕННАЯ) =====
        
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
        
        # ===== ГРУППА 2: ВОЛАТИЛЬНОСТЬ (УЛУЧШЕННАЯ) =====
        
        # Bollinger Bands Width
        bb_period = 20
        bb_std = 2
        bb_middle = data['close'].rolling(bb_period).mean()
        bb_std_val = data['close'].rolling(bb_period).std()
        bb_upper = bb_middle + (bb_std_val * bb_std)
        bb_lower = bb_middle - (bb_std_val * bb_std)
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        features['bb_width'] = bb_width
        features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR (уже рассчитан выше)
        features['atr_value'] = atr
        features['atr_ratio'] = atr / data['close']
        
        # ===== ГРУППА 3: МОМЕНТУМ (УЛУЧШЕННАЯ) =====
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        features['rsi_value'] = rsi
        features['rsi_zone'] = pd.cut(rsi, bins=[0, 30, 70, 100], labels=[0, 1, 2])  # 0=oversold, 1=neutral, 2=overbought
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        features['macd_value'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_histogram
        
        # ===== ГРУППА 4: ЦЕНОВОЕ ДВИЖЕНИЕ (НОВАЯ) =====
        
        # Price momentum
        features['price_momentum_5'] = data['close'].pct_change(5)
        features['price_momentum_10'] = data['close'].pct_change(10)
        features['price_momentum_20'] = data['close'].pct_change(20)
        
        # Volume (если доступен)
        if 'volume' in data.columns:
            volume_ma = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / volume_ma
        else:
            features['volume_ratio'] = 1.0  # Заглушка
        
        # ===== ГРУППА 5: ТЕХНИЧЕСКИЕ УРОВНИ (НОВАЯ) =====
        
        # Support/Resistance levels
        features['high_20'] = data['high'].rolling(20).max()
        features['low_20'] = data['low'].rolling(20).min()
        features['close_vs_high'] = (data['close'] - features['low_20']) / (features['high_20'] - features['low_20'])
        
        # Заполняем пропуски
        features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        print(f"✅ Извлечено {len(features.columns)} признаков (улучшенная версия)")
        return features
    
    def _optimize_dbscan_parameters(self, features):
        """Оптимизация параметров DBSCAN на основе данных"""
        print("🔧 Оптимизируем параметры DBSCAN...")
        
        # Масштабируем признаки
        features_scaled = self.scaler.fit_transform(features)
        
        # Находим оптимальные параметры
        from sklearn.neighbors import NearestNeighbors
        
        # Анализируем расстояния до k-го соседа
        k = min(20, len(features) // 10)  # Адаптивный k
        nbrs = NearestNeighbors(n_neighbors=k).fit(features_scaled)
        distances, indices = nbrs.kneighbors(features_scaled)
        
        # Сортируем расстояния
        distances = np.sort(distances[:, k-1])
        
        # Находим "локоть" в кривой расстояний
        # Используем 75-й процентиль как eps
        eps = np.percentile(distances, 75)
        
        # Минимальное количество образцов
        min_samples = max(5, len(features) // 100)
        
        print(f"📊 Оптимальные параметры DBSCAN: eps={eps:.4f}, min_samples={min_samples}")
        
        return DBSCAN(eps=eps, min_samples=min_samples)
    
    def _improved_cluster_interpretation(self, features, cluster_labels):
        """Улучшенная интерпретация кластеров"""
        print("📊 Улучшенная интерпретация кластеров...")
        
        regime_names = {}
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise для DBSCAN
                regime_names[label] = "Noise"
                continue
            
            # Получаем данные для кластера
            cluster_mask = cluster_labels == label
            cluster_features = features[cluster_mask]
            
            if len(cluster_features) == 0:
                regime_names[label] = "Empty"
                continue
            
            # Анализируем ключевые признаки
            avg_rsi = cluster_features['rsi_value'].mean() if 'rsi_value' in cluster_features.columns else 50
            avg_momentum = cluster_features['price_momentum_5'].mean() if 'price_momentum_5' in cluster_features.columns else 0
            avg_adx = cluster_features['adx_value'].mean() if 'adx_value' in cluster_features.columns else 0
            avg_bb_position = cluster_features['bb_position'].mean() if 'bb_position' in cluster_features.columns else 0.5
            
            # Улучшенная логика классификации
            regime_score = 0
            
            # RSI анализ
            if avg_rsi > 70:
                regime_score += 2  # Сильный бычий сигнал
            elif avg_rsi > 60:
                regime_score += 1  # Слабый бычий сигнал
            elif avg_rsi < 30:
                regime_score -= 2  # Сильный медвежий сигнал
            elif avg_rsi < 40:
                regime_score -= 1  # Слабый медвежий сигнал
            
            # Momentum анализ
            if avg_momentum > 0.02:  # 2% рост
                regime_score += 2
            elif avg_momentum > 0.01:  # 1% рост
                regime_score += 1
            elif avg_momentum < -0.02:  # 2% падение
                regime_score -= 2
            elif avg_momentum < -0.01:  # 1% падение
                regime_score -= 1
            
            # ADX анализ (сила тренда)
            if avg_adx > 30:
                regime_score *= 1.5  # Усиливаем сигнал при сильном тренде
            elif avg_adx < 15:
                regime_score *= 0.5  # Ослабляем сигнал при слабом тренде
            
            # Bollinger Bands позиция
            if avg_bb_position > 0.8:
                regime_score += 1  # Близко к верхней полосе
            elif avg_bb_position < 0.2:
                regime_score -= 1  # Близко к нижней полосе
            
            # Определяем режим
            if regime_score >= 3:
                regime_names[label] = "Strong Bull"
            elif regime_score >= 1:
                regime_names[label] = "Bull"
            elif regime_score <= -3:
                regime_names[label] = "Strong Bear"
            elif regime_score <= -1:
                regime_names[label] = "Bear"
            else:
                regime_names[label] = "Sideways"
        
        return regime_names
    
    def fit(self, data):
        """Обучение оптимизированного ML-классификатора"""
        print("🤖 Обучение оптимизированного ML-классификатора...")
        
        # Извлекаем признаки
        features = self.extract_classification_features(data)
        self.feature_names = features.columns.tolist()
        
        # Масштабируем признаки
        features_scaled = self.scaler.fit_transform(features)
        
        # Создаем модель в зависимости от метода
        if self.method == 'dbscan':
            self.model = self._optimize_dbscan_parameters(features)
        
        # Обучаем модель
        if self.method == 'gmm':
            self.model.fit(features_scaled)
            self.cluster_labels = self.model.predict(features_scaled)
        else:
            self.cluster_labels = self.model.fit_predict(features_scaled)
        
        # Интерпретируем кластеры
        self.regime_names = self._improved_cluster_interpretation(features, self.cluster_labels)
        
        print(f"✅ Оптимизированный ML-классификатор обучен!")
        print(f"📊 Количество кластеров: {len(np.unique(self.cluster_labels))}")
        
        # Выводим интерпретацию кластеров
        for label, regime in self.regime_names.items():
            print(f"   Кластер {label}: {regime}")
        
        return self
    
    def predict(self, data):
        """Предсказание режимов"""
        print("🔮 Предсказание режимов...")
        
        # Извлекаем признаки
        features = self.extract_classification_features(data)
        
        # Масштабируем признаки
        features_scaled = self.scaler.transform(features)
        
        # Предсказываем кластеры
        if self.method == 'dbscan':
            # DBSCAN не имеет метода predict, используем fit_predict для новых данных
            cluster_predictions = self.model.fit_predict(features_scaled)
        else:
            cluster_predictions = self.model.predict(features_scaled)
        
        # Конвертируем кластеры в режимы
        predictions = []
        for cluster in cluster_predictions:
            if cluster in self.regime_names:
                regime = self.regime_names[cluster]
                if regime in ["Strong Bull", "Bull"]:
                    predictions.append(1)
                elif regime in ["Strong Bear", "Bear"]:
                    predictions.append(-1)
                else:
                    predictions.append(0)
            else:
                predictions.append(0)
        
        print(f"✅ Предсказано {len(predictions)} режимов")
        return np.array(predictions)
    
    def fit_predict(self, data):
        """Обучает классификатор и сразу делает предсказания"""
        self.fit(data)
        return self.predict(data)
