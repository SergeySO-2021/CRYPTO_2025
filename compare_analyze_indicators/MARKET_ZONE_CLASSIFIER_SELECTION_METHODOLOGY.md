# Методология выбора классификатора рыночных зон

## 🎯 Цель проекта
**Этап 1:** Сравнить и выбрать лучший алгоритм для классификации рыночных зон из трех вариантов.

**Общая цель проекта:** Создание системы автоматического поиска оптимальных параметров технических индикаторов для торговли криптовалютой с учетом классификации рыночных зон.

## 📊 Три варианта классификации рыночных зон

### 1. Market Zone Analyzer (MZA) - Четырехуровневая система
**Описание:** Комплексный Pine Script индикатор для TradingView с адаптивными весами

**Архитектура:**
```
MZA = Trend Strength + Momentum + Price Action + Market Activity
```

**Компоненты:**
- **Trend Strength (40% веса):** ADX/DMI, Moving Averages, Ichimoku
- **Momentum (30% веса):** RSI, Stochastic, MACD  
- **Price Action (30% веса):** HH/LL, Heikin-Ashi, Candle Range
- **Market Activity (10% веса):** Bollinger Bands, ATR, Volume

**Преимущества:**
- Комплексный подход с 4 категориями анализа
- Адаптивные веса в зависимости от волатильности
- Готовые настройки для BTC
- Высокая интерпретируемость результатов

### 2. trend_classifier_iziceros - Сегментация временных рядов
**Описание:** Python библиотека для автоматической сегментации временных рядов

**Алгоритм:**
- Скользящие окна с перекрытием
- Линейная регрессия для каждого окна
- Проверка изменения тренда по наклону и смещению

**Параметры:**
- `N` - размер окна (по умолчанию 24)
- `alpha` - порог изменения наклона (по умолчанию 2.0)
- `beta` - порог изменения смещения (по умолчанию 2.0)
- `overlap_ratio` - коэффициент перекрытия (по умолчанию 0.33)

**Проблемы:**
- Переоптимизация (R² = -5.7%)
- Сложная настройка (4+ параметра)
- Избыточная сегментация

### 3. trading_classifier.pine - Полосы тренда
**Описание:** Pine Script индикатор с системой полос тренда

**Алгоритм:**
- SMEMA (Smooth EMA) - комбинация SMA и EMA
- 6 уровней полос (3 вверх, 3 вниз)
- Сила тренда по пересечениям

**Параметры:**
- `length` - период для SMEMA (по умолчанию 10)

**Результаты:**
- 79% разность в доходности
- Простая настройка (1 параметр)
- Стабильная работа

## 🔬 Методология отбора

### Этап 1: Подготовка данных и валидация

#### 1.1 Загрузка и подготовка данных
```python
# Загрузка данных BTC для всех таймфреймов
timeframes = ['15m', '30m', '1h', '4h', '1d']
data = {}

for tf in timeframes:
    file_path = f'../../indicators/data_frames/df_btc_{tf}.csv'
    df = pd.read_csv(file_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df.set_index('timestamps', inplace=True)
    data[tf] = df
```

#### 1.2 Анализ качества данных
```python
def analyze_data_quality(data):
    """Анализ качества данных"""
    quality_report = {}
    
    for tf, df in data.items():
        # Проверка пропусков
        missing_values = df.isnull().sum().sum()
        
        # Проверка дубликатов
        duplicates = df.index.duplicated().sum()
        
        # Проверка выбросов
        price_mean = df['close'].mean()
        price_std = df['close'].std()
        outliers = df[abs(df['close'] - price_mean) > 3 * price_std]
        
        quality_report[tf] = {
            'records': len(df),
            'missing_values': missing_values,
            'duplicates': duplicates,
            'outliers': len(outliers)
        }
    
    return quality_report
```

#### 1.3 Валидация данных
```python
def validate_data(data):
    """Валидация входных данных"""
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Проверяем наличие всех необходимых колонок
    if not all(col in data.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in data.columns]
        raise ValueError(f"Отсутствуют колонки: {missing_cols}")
    
    # Проверяем логичность данных
    if not (data['high'] >= data['low']).all():
        raise ValueError("Некорректные данные: high < low")
    
    return True
```

### Этап 2: Реализация классификаторов

#### 2.1 Базовый класс для всех классификаторов
```python
class BaseMarketZoneClassifier(ABC):
    """Базовый класс для классификаторов рыночных зон"""
    
    def __init__(self, name: str, parameters: Dict = None):
        self.name = name
        self.parameters = parameters or {}
        self.is_fitted = False
        self.classes_ = None
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseMarketZoneClassifier':
        """Обучение классификатора на данных"""
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Предсказание рыночных зон"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Получение важности признаков"""
        pass
```

#### 2.2 Market Zone Analyzer (MZA)
```python
class MZAClassifier(BaseMarketZoneClassifier):
    """Market Zone Analyzer - четырехуровневая система"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            # Trend Indicators
            'adx_length': 21,
            'adx_threshold': 18,
            'fast_ma_length': 21,
            'slow_ma_length': 55,
            
            # Momentum Indicators
            'rsi_length': 21,
            'stoch_length': 21,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Price Action Indicators
            'hh_ll_range': 21,
            'ha_doji_range': 8,
            'candle_range_length': 13,
            
            # Market Activity Indicators
            'bb_length': 20,
            'bb_multiplier': 2.2,
            'atr_length': 14,
            'volume_ma_length': 20,
            
            # Weights (адаптивные веса)
            'trend_weight': 0.4,
            'momentum_weight': 0.3,
            'price_action_weight': 0.3
        }
        
        super().__init__("Market Zone Analyzer", default_params)
    
    def _calculate_indicators(self, data: pd.DataFrame) -> None:
        """Вычисление всех технических индикаторов"""
        # Trend Indicators
        self._calculate_adx_dmi(data)
        self._calculate_moving_averages(data)
        
        # Momentum Indicators
        self._calculate_rsi(data)
        self._calculate_stochastic(data)
        self._calculate_macd(data)
        
        # Price Action Indicators
        self._calculate_hh_ll(data)
        self._calculate_heikin_ashi(data)
        self._calculate_candle_range(data)
        
        # Market Activity Indicators
        self._calculate_bollinger_bands(data)
        self._calculate_atr(data)
        self._calculate_volume_indicators(data)
    
    def _calculate_adaptive_weights(self, data: pd.DataFrame) -> None:
        """Вычисление адаптивных весов на основе волатильности"""
        volatility = self.calculate_volatility(data, window=20)
        
        high_vol_threshold = self.parameters['high_volatility_threshold']
        low_vol_threshold = self.parameters['low_volatility_threshold']
        
        # Адаптивные веса в зависимости от волатильности
        high_vol_mask = volatility > high_vol_threshold
        low_vol_mask = volatility < low_vol_threshold
        
        self.parameters['trend_weight'] = np.where(high_vol_mask, 0.5, 
                                                   np.where(low_vol_mask, 0.25, 0.4))
        self.parameters['momentum_weight'] = np.where(high_vol_mask, 0.35,
                                                       np.where(low_vol_mask, 0.20, 0.30))
        self.parameters['price_action_weight'] = np.where(high_vol_mask, 0.15,
                                                          np.where(low_vol_mask, 0.55, 0.30))
```

#### 2.3 trend_classifier_iziceros
```python
class TrendClassifier(BaseMarketZoneClassifier):
    """Сегментация временных рядов"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'N': 24,           # размер окна
            'alpha': 2.0,      # порог изменения наклона
            'beta': 2.0,       # порог изменения смещения
            'overlap_ratio': 0.33  # коэффициент перекрытия
        }
        
        super().__init__("Trend Classifier", default_params)
    
    def _calculate_segments(self, data: pd.DataFrame) -> None:
        """Вычисление сегментов временного ряда"""
        n = self.parameters['N']
        overlap_ratio = self.parameters['overlap_ratio']
        alpha = self.parameters['alpha']
        beta = self.parameters['beta']
        
        offset = max(1, int(n * overlap_ratio))
        
        segments = []
        current_segment = {
            'start': 0,
            'slopes': [],
            'offsets': [],
            'starts': []
        }
        
        prev_fit = None
        
        for start in range(0, len(data) - n, offset):
            end = start + n
            
            x_window = np.array(range(start, end))
            y_window = np.array(data['close'][start:end])
            fit = np.polyfit(x_window, y_window, 1)
            
            current_segment['slopes'].append(fit[0])
            current_segment['offsets'].append(fit[1])
            current_segment['starts'].append(start)
            
            if prev_fit is not None:
                slope_change = abs(fit[0] - prev_fit[0])
                offset_change = abs(fit[1] - prev_fit[1])
                
                if slope_change > alpha or offset_change > beta:
                    # Создаем новый сегмент
                    segment = {
                        'start': current_segment['start'],
                        'stop': start + offset // 2,
                        'slope': np.mean(current_segment['slopes']),
                        'offset': np.mean(current_segment['offsets'])
                    }
                    segments.append(segment)
                    
                    current_segment = {
                        'start': start + offset // 2,
                        'slopes': [],
                        'offsets': [],
                        'starts': []
                    }
            
            prev_fit = fit
        
        self.segments = segments
```

#### 2.4 trading_classifier.pine
```python
class TradingClassifier(BaseMarketZoneClassifier):
    """Полосы тренда"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'length': 10  # период для SMEMA
        }
        
        super().__init__("Trading Classifier", default_params)
    
    def _calculate_smema(self, data: pd.DataFrame) -> pd.Series:
        """Smooth EMA - комбинация SMA и EMA"""
        length = self.parameters['length']
        ema_val = data['close'].ewm(span=length).mean()
        smema_val = ema_val.rolling(window=length).mean()
        return smema_val
    
    def _calculate_trend_bands(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Вычисление полос тренда"""
        smema_line = self._calculate_smema(data)
        
        # Вычисляем step
        step = self._calculate_smema(data[['high', 'low']].diff().abs())
        
        # Создаем полосы
        bands = {
            'up3': smema_line + step * 3,
            'up2': smema_line + step * 2,
            'up1': smema_line + step,
            'dn1': smema_line - step,
            'dn2': smema_line - step * 2,
            'dn3': smema_line - step * 3
        }
        
        return bands
    
    def _calculate_trend_strength(self, data: pd.DataFrame, bands: Dict[str, pd.Series]) -> np.ndarray:
        """Вычисление силы тренда"""
        close = data['close']
        
        # Определяем пересечения с полосами
        above3 = close > bands['up3']
        above2 = close > bands['up2']
        above1 = close > bands['up1']
        
        below1 = close < bands['dn1']
        below2 = close < bands['dn2']
        below3 = close < bands['dn3']
        
        # Вычисляем силу бычьего и медвежьего тренда
        bull_strength = (above1.astype(int) + above2.astype(int) + above3.astype(int))
        bear_strength = (below1.astype(int) + below2.astype(int) + below3.astype(int))
        
        # Определяем направление тренда
        trend = smema_line > smema_line.shift(1)
        
        # Классификация
        predictions = np.where(
            trend & (bull_strength >= 1), 1,      # Бычий
            np.where(~trend & (bear_strength >= 1), -1, 0)  # Медвежий, Боковой
        )
        
        return predictions
```

### Этап 3: Метрики оценки качества

#### 3.1 Базовые метрики классификации
```python
def calculate_classification_metrics(y_true, y_pred):
    """Вычисление метрик классификации"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics
```

#### 3.2 Метрики для временных рядов
```python
def calculate_time_series_metrics(data, predictions):
    """Вычисление метрик для временных рядов"""
    # Вычисляем доходность
    returns = data['close'].pct_change().dropna()
    
    # Разделяем по предсказаниям
    bull_mask = predictions == 1
    bear_mask = predictions == -1
    sideways_mask = predictions == 0
    
    # Вычисляем среднюю доходность в каждом режиме
    bull_return = returns[bull_mask].mean() if bull_mask.any() else 0
    bear_return = returns[bear_mask].mean() if bear_mask.any() else 0
    sideways_return = returns[sideways_mask].mean() if sideways_mask.any() else 0
    
    # Вычисляем волатильность
    bull_vol = returns[bull_mask].std() if bull_mask.any() else 0
    bear_vol = returns[bear_mask].std() if bear_mask.any() else 0
    sideways_vol = returns[sideways_mask].std() if sideways_mask.any() else 0
    
    metrics = {
        'bull_return': bull_return,
        'bear_return': bear_return,
        'sideways_return': sideways_return,
        'bull_volatility': bull_vol,
        'bear_volatility': bear_vol,
        'sideways_volatility': sideways_vol,
        'return_spread': bull_return - bear_return
    }
    
    return metrics
```

#### 3.3 Анализ look-ahead bias
```python
def analyze_look_ahead_bias(data, classifier, n_splits=5):
    """Анализ look-ahead bias с помощью walk-forward анализа"""
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    
    for train_idx, test_idx in tscv.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Обучаем на train_data
        classifier.fit(train_data)
        
        # Предсказываем на test_data
        predictions = classifier.predict(test_data)
        
        # Вычисляем метрики
        metrics = calculate_time_series_metrics(test_data, predictions)
        results.append(metrics)
    
    return results
```

### Этап 4: Мультитаймфреймовый анализ

#### 4.1 Анализ согласованности сигналов
```python
def analyze_signal_consistency(data_dict, classifier):
    """Анализ согласованности сигналов между таймфреймами"""
    predictions_dict = {}
    
    # Получаем предсказания для каждого таймфрейма
    for tf, df in data_dict.items():
        classifier.fit(df)
        predictions = classifier.predict(df)
        predictions_dict[tf] = predictions
    
    # Вычисляем корреляции между таймфреймами
    correlation_matrix = pd.DataFrame(index=data_dict.keys(), 
                                   columns=data_dict.keys())
    
    for tf1 in data_dict.keys():
        for tf2 in data_dict.keys():
            if tf1 != tf2:
                # Находим общие периоды
                common_periods = data_dict[tf1].index.intersection(data_dict[tf2].index)
                
                if len(common_periods) > 0:
                    pred1 = predictions_dict[tf1][data_dict[tf1].index.isin(common_periods)]
                    pred2 = predictions_dict[tf2][data_dict[tf2].index.isin(common_periods)]
                    
                    correlation = np.corrcoef(pred1, pred2)[0, 1]
                    correlation_matrix.loc[tf1, tf2] = correlation
    
    return correlation_matrix
```

#### 4.2 Lead/Lag анализ
```python
def analyze_lead_lag(data_dict, classifier):
    """Анализ опережающих и запаздывающих сигналов"""
    lead_lag_results = {}
    
    # Сравниваем каждый таймфрейм с дневным
    daily_data = data_dict['1d']
    daily_classifier = classifier.fit(daily_data)
    daily_predictions = daily_classifier.predict(daily_data)
    
    for tf, df in data_dict.items():
        if tf != '1d':
            # Получаем предсказания для текущего таймфрейма
            tf_classifier = classifier.fit(df)
            tf_predictions = tf_classifier.predict(df)
            
            # Вычисляем корреляции с различными лагами
            lags = range(-5, 6)  # от -5 до +5 периодов
            correlations = []
            
            for lag in lags:
                if lag > 0:
                    # Текущий таймфрейм опережает дневной
                    corr = np.corrcoef(tf_predictions[:-lag], daily_predictions[lag:])[0, 1]
                elif lag < 0:
                    # Дневной опережает текущий таймфрейм
                    corr = np.corrcoef(tf_predictions[-lag:], daily_predictions[:lag])[0, 1]
                else:
                    # Без лага
                    corr = np.corrcoef(tf_predictions, daily_predictions)[0, 1]
                
                correlations.append(corr)
            
            lead_lag_results[tf] = {
                'lags': lags,
                'correlations': correlations,
                'best_lag': lags[np.argmax(correlations)],
                'max_correlation': max(correlations)
            }
    
    return lead_lag_results
```

### Этап 5: Оптимизация параметров

#### 5.1 Grid Search
```python
def grid_search_optimization(data, classifier_class, param_grid):
    """Grid Search для оптимизации параметров"""
    from sklearn.model_selection import ParameterGrid
    
    best_score = -np.inf
    best_params = None
    best_classifier = None
    
    for params in ParameterGrid(param_grid):
        classifier = classifier_class(parameters=params)
        
        # Обучаем и тестируем
        classifier.fit(data)
        predictions = classifier.predict(data)
        
        # Вычисляем скор
        score = calculate_classification_score(data, predictions)
        
        if score > best_score:
            best_score = score
            best_params = params
            best_classifier = classifier
    
    return best_classifier, best_params, best_score
```

#### 5.2 Bayesian Optimization
```python
def bayesian_optimization(data, classifier_class, param_bounds, n_iterations=50):
    """Bayesian Optimization для оптимизации параметров"""
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    
    def objective(params):
        # Преобразуем параметры в словарь
        param_dict = dict(zip(param_bounds.keys(), params))
        
        classifier = classifier_class(parameters=param_dict)
        classifier.fit(data)
        predictions = classifier.predict(data)
        
        score = calculate_classification_score(data, predictions)
        return -score  # Минимизируем отрицательный скор
    
    # Определяем пространство параметров
    space = []
    for param_name, (low, high) in param_bounds.items():
        if isinstance(low, int):
            space.append(Integer(low, high))
        else:
            space.append(Real(low, high))
    
    # Запускаем оптимизацию
    result = gp_minimize(objective, space, n_calls=n_iterations)
    
    best_params = dict(zip(param_bounds.keys(), result.x))
    best_score = -result.fun
    
    return best_params, best_score
```

### Этап 6: Сравнительный анализ

#### 6.1 Статистические тесты
```python
def statistical_comparison(results_dict):
    """Статистическое сравнение результатов"""
    from scipy import stats
    
    # Извлекаем метрики для сравнения
    metrics = ['accuracy', 'f1_score', 'return_spread']
    
    comparison_results = {}
    
    for metric in metrics:
        values = [results[metric] for results in results_dict.values()]
        classifiers = list(results_dict.keys())
        
        # t-test для сравнения средних
        t_stat, p_value = stats.ttest_ind(values[0], values[1])
        
        # Mann-Whitney U test для непараметрического сравнения
        u_stat, u_p_value = stats.mannwhitneyu(values[0], values[1])
        
        comparison_results[metric] = {
            't_statistic': t_stat,
            't_p_value': p_value,
            'u_statistic': u_stat,
            'u_p_value': u_p_value,
            'significant': p_value < 0.05
        }
    
    return comparison_results
```

#### 6.2 Критерии выбора лучшего решения
```python
def select_best_classifier(results_dict):
    """Выбор лучшего классификатора на основе критериев"""
    
    # Веса критериев
    weights = {
        'accuracy': 0.25,
        'f1_score': 0.25,
        'return_spread': 0.20,
        'stability': 0.15,
        'speed': 0.10,
        'interpretability': 0.05
    }
    
    scores = {}
    
    for classifier_name, results in results_dict.items():
        score = 0
        
        # Нормализуем метрики (0-1)
        normalized_accuracy = results['accuracy']
        normalized_f1 = results['f1_score']
        normalized_return = min(results['return_spread'] / 0.1, 1)  # Нормализуем к 10%
        
        # Вычисляем общий скор
        score = (
            normalized_accuracy * weights['accuracy'] +
            normalized_f1 * weights['f1_score'] +
            normalized_return * weights['return_spread'] +
            results['stability'] * weights['stability'] +
            results['speed'] * weights['speed'] +
            results['interpretability'] * weights['interpretability']
        )
        
        scores[classifier_name] = score
    
    # Выбираем лучший
    best_classifier = max(scores, key=scores.get)
    best_score = scores[best_classifier]
    
    return best_classifier, best_score, scores
```

## 📊 Ожидаемые результаты

### 1. Технические результаты
- **Сравнительная таблица** всех классификаторов
- **Оптимальные параметры** для каждого
- **Метрики качества** с учетом look-ahead bias
- **Рекомендации** по использованию

### 2. Практические результаты
- **Готовый код** для всех классификаторов
- **Jupyter notebooks** с примерами использования
- **Документация** по настройке и применению
- **TradingView интеграция** для лучшего решения

### 3. Научные результаты
- **Статистический анализ** эффективности
- **Методология** оценки классификаторов рынка
- **Best practices** для избежания переоптимизации
- **Рекомендации** для дальнейших исследований

## ⚠️ Важные замечания

### Избежание look-ahead bias
- **Строгое разделение** train/validation/test
- **Временная последовательность** данных
- **Out-of-sample тестирование**
- **Walk-forward анализ**

### Статистическая значимость
- **Множественные тесты** корректировка
- **Bootstrap confidence intervals**
- **Effect size** анализ
- **Practical significance** оценка

### Воспроизводимость
- **Фиксированные random seeds**
- **Документирование** всех параметров
- **Version control** для кода
- **Детальное логирование** экспериментов

---

**Дата создания:** 24.10.2025  
**Автор:** AI Assistant  
**Статус:** В разработке  
**Приоритет:** Высокий
