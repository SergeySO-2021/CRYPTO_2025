ПРЕДЛАГАЕМЫЕ ИНДИКАТОРЫ
1. 📊 ИНДИКАТОРЫ ДИСБАЛАНСА
1.1. Weighted Order Book Imbalance (WOBI)
python

WOBI = (w1*ratio3 + w2*ratio5 + w3*ratio8 + w4*ratio60) / (w1 + w2 + w3 + w4)

    Экономический смысл: Взвешенный дисбаланс стакана с учетом разных глубин

    Диапазон: -100% до +100%

    Параметры для оптимизации: w1, w2, w3, w4 (веса для глубин 3%, 5%, 8%, 60%)

    Интерпретация: Положительные значения - давление покупателей, отрицательные - продавцов

1.2. Depth Imbalance Gradient (DIG)
python

DIG = (ratio3 - ratio60) / 3  # градиент между ближней и дальней глубиной

    Экономический смысл: Скорость изменения дисбаланса от ближних к дальним глубинам

    Диапазон: -66% до +66%

    Параметры: Можно добавить веса для промежуточных глубин

    Интерпретация: Положительное значение - дисбаланс усиливается у цены (агрессивные участники)

1.3. Imbalance Momentum (IM)
python

IM = WOBI - WOBI_shift(1)  # изменение за 1 бар

    Экономический смысл: Темп изменения дисбаланса

    Диапазон: -200% до +200%

    Параметры: Период сдвига (1, 2, 3 бара)

    Интерпретация: Рост - усиление давления, падение - ослабление

2. 💧 ИНДИКАТОРЫ ЛИКВИДНОСТИ
2.1. Total Liquidity Index (TLI)
python

TLI = (bid3 + ask3 + bid5 + ask5 + bid8 + ask8 + bid60 + ask60) / 8
TLI_normalized = (TLI - TLI.rolling(100).min()) / (TLI.rolling(100).max() - TLI.rolling(100).min())

    Экономический смысл: Общий уровень ликвидности в стакане

    Диапазон: 0-1 (после нормализации)

    Параметры: Период нормализации (50, 100, 200 баров)

    Интерпретация: Высокие значения - высокая ликвидность, низкие - низкая

2.2. Liquidity Concentration Ratio (LCR)
python

LCR = (bid3 + ask3) / (bid60 + ask60)

    Экономический смысл: Концентрация ликвидности near-the-money

    Диапазон: 0.1 до 0.5 (типично)

    Параметры: Соотношения глубин (3%/60%, 5%/60%, 8%/60%)

    Интерпретация: Высокие значения - ликвидность сконцентрирована у цены

2.3. Liquidity Asymmetry Index (LAI)
python

LAI = abs(ratio3) + abs(ratio5) + abs(ratio8) + abs(ratio60)

    Экономический смысл: Общая асимметрия стакана (независимо от направления)

    Диапазон: 0% до 400%

    Параметры: Веса для разных глубин

    Интерпретация: Высокие значения - сильный дисбаланс в одну сторону

3. 📈 ИНДИКАТОРЫ ДИНАМИКИ
3.1. Order Book Flow (OBF)
python

OBF = (d3.diff(1) + d5.diff(1) + d8.diff(1) + d60.diff(1)) / 4

    Экономический смысл: Чистый поток ликвидности в стакан

    Диапазон: Зависит от масштаба

    Параметры: Период дифференцирования (1, 2, 3 бара)

    Интерпретация: Положительные значения - приток ликвидности в bids

3.2. Liquidity Momentum (LM)
python

LM = (TLI / TLI.shift(3) - 1) * 100  # изменение за 3 бара (45 минут)

    Экономический смысл: Темп изменения общей ликвидности

    Диапазон: -100% до +∞

    Параметры: Период сдвига (1-6 баров)

    Интерпретация: Рост - увеличение ликвидности, падение - уменьшение

3.3. Imbalance Volatility (IV)
python

IV = WOBI.rolling(6).std()  # волатильность за 1.5 часа (6 баров)

    Экономический смысл: Стабильность/нестабильность дисбаланса

    Диапазон: 0% до 100%

    Параметры: Период скользящего окна (4, 6, 8 баров)

    Интерпретация: Высокие значения - нестабильный стакан

4. 🔄 КОМПОЗИТНЫЕ ИНДИКАТОРЫ
4.1. Smart Money Pressure (SMP)
python

SMP = (ratio3 * 2 - ratio60) * LCR  # комбинация дисбаланса и концентрации

    Экономический смысл: Давление "умных денег" (ближние глубины против дальних)

    Диапазон: -200% до +200%

    Параметры: Коэффициенты для ratio3 и ratio60

    Интерпретация: Положительные значения - умные деньги покупают

4.2. Flow-Order Book Alignment (FOBA)
python

FOBA = (flow_trade_imbalance / volume) * WOBI  # согласованность flow и стакана

    Экономический смысл: Согласованность агрессивных сделок и лимитного стакана

    Диапазон: -1 до +1

    Параметры: Веса для flow и WOBI

    Интерпретация: Положительные значения - согласованное давление

4.3. Liquidity Quality Index (LQI)
python

LQI = (TLI_normalized + (1 - IV_normalized) + LCR) / 3

    Экономический смысл: Качество ликвидности (объем + стабильность + концентрация)

    Диапазон: 0-1

    Параметры: Веса для компонентов

    Интерпретация: Высокие значения - качественная ликвидность

5. 🎚️ ИНДИКАТОРЫ ДЛЯ ГЕНЕТИЧЕСКОГО АЛГОРИТМА
5.1. Multi-Timeframe WOBI
python

WOBI_fast = WOBI.rolling(2).mean()    # быстрая версия (30 минут)
WOBI_slow = WOBI.rolling(6).mean()    # медленная версия (1.5 часа)

    Экономический смысл: Дисбаланс на разных таймфреймах

    Параметры: Периоды для fast и slow

5.2. Order Book Regime Detection
python

# 4 режима стакана
regime_conditions = [
    (WOBI > 0.1) & (IV < 0.05),    # 0: Сильный бычий, стабильный
    (WOBI > 0.1) & (IV >= 0.05),   # 1: Сильный бычий, нестабильный  
    (WOBI < -0.1) & (IV < 0.05),   # 2: Сильный медвежий, стабильный
    (WOBI < -0.1) & (IV >= 0.05),  # 3: Сильный медвежий, нестабильный
    (abs(WOBI) <= 0.1) & (IV < 0.05), # 4: Сбалансированный, стабильный
    (abs(WOBI) <= 0.1) & (IV >= 0.05) # 5: Сбалансированный, нестабильный
]

    Экономический смысл: Классификация рыночного режима по стакану

    Параметры: Пороги для WOBI и IV

💻 ПРИМЕР РЕАЛИЗАЦИИ
python

import pandas as pd
import numpy as np

def calculate_order_book_indicators(df, params=None):
    """
    Расчет Order Book индикаторов для генетического алгоритма
    
    Parameters:
    df: DataFrame с исходными данными
    params: словарь параметров для оптимизации
    
    Returns:
    DataFrame с добавленными индикаторами
    """
    
    if params is None:
        params = {
            'w1': 0.4, 'w2': 0.3, 'w3': 0.2, 'w4': 0.1,  # веса для WOBI
            'normalization_period': 100,
            'momentum_period': 3,
            'volatility_period': 6
        }
    
    result_df = df.copy()
    
    # 1. Weighted Order Book Imbalance
    w1, w2, w3, w4 = params['w1'], params['w2'], params['w3'], params['w4']
    result_df['WOBI'] = (w1*df['ratio3'] + w2*df['ratio5'] + 
                         w3*df['ratio8'] + w4*df['ratio60']) / (w1 + w2 + w3 + w4)
    
    # 2. Depth Imbalance Gradient
    result_df['DIG'] = (df['ratio3'] - df['ratio60']) / 3
    
    # 3. Imbalance Momentum
    result_df['IM'] = result_df['WOBI'].diff(params['momentum_period'])
    
    # 4. Total Liquidity Index (нормализованный)
    result_df['TLI_raw'] = (df['bid3'] + df['ask3'] + df['bid5'] + df['ask5'] + 
                           df['bid8'] + df['ask8'] + df['bid60'] + df['ask60']) / 8
    min_tli = result_df['TLI_raw'].rolling(params['normalization_period']).min()
    max_tli = result_df['TLI_raw'].rolling(params['normalization_period']).max()
    result_df['TLI'] = (result_df['TLI_raw'] - min_tli) / (max_tli - min_tli + 1e-8)
    
    # 5. Liquidity Concentration Ratio
    result_df['LCR'] = (df['bid3'] + df['ask3']) / (df['bid60'] + df['ask60'] + 1e-8)
    
    # 6. Liquidity Asymmetry Index
    result_df['LAI'] = (abs(df['ratio3']) + abs(df['ratio5']) + 
                       abs(df['ratio8']) + abs(df['ratio60']))
    
    # 7. Order Book Flow
    result_df['OBF'] = (df['d3'].diff(1) + df['d5'].diff(1) + 
                       df['d8'].diff(1) + df['d60'].diff(1)) / 4
    
    # 8. Liquidity Momentum
    result_df['LM'] = (result_df['TLI_raw'] / result_df['TLI_raw'].shift(params['momentum_period']) - 1) * 100
    
    # 9. Imbalance Volatility
    result_df['IV'] = result_df['WOBI'].rolling(params['volatility_period']).std()
    
    # 10. Smart Money Pressure
    result_df['SMP'] = (df['ratio3'] * 2 - df['ratio60']) * result_df['LCR']
    
    # 11. Flow-Order Book Alignment
    flow_imbalance = df['flow_trade_imbalance'] / (df['volume'] + 1e-8)
    result_df['FOBA'] = flow_imbalance * result_df['WOBI']
    
    # 12. Multi-Timeframe WOBI
    result_df['WOBI_fast'] = result_df['WOBI'].rolling(2).mean()
    result_df['WOBI_slow'] = result_df['WOBI'].rolling(6).mean()
    
    # Заполнение пропусков
    result_df = result_df.fillna(method='bfill').fillna(method='ffill')
    
    return result_df

# Пример использования
# df_with_indicators = calculate_order_book_indicators(df, params)

🎯 РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ
Приоритетные индикаторы для старта:

    WOBI - базовый индикатор дисбаланса

    SMP - давление умных денег

    FOBA - согласованность flow и стакана

    LCR - концентрация ликвидности

Комбинации индикаторов:

    Трендовые стратегии: WOBI + IM + LM

    Среднесрочные: SMP + LCR + TLI

    Короткие позиции: FOBA + IV + DIG

Предупреждения:

    Волатильность: IV > 0.1 может указывать на ненадежность сигналов

    Низкая ликвидность: TLI < 0.2 - осторожность с крупными позициями

    Расхождения: Negative FOBA - противоречивые сигналы

⚙️ ПАРАМЕТРЫ ДЛЯ ОПТИМИЗАЦИИ
python

# Диапазоны параметров для генетического алгоритма
parameter_ranges = {
    'WOBI_weights': {'w1': [0.1, 0.6], 'w2': [0.1, 0.5], 'w3': [0.1, 0.4], 'w4': [0.05, 0.3]},
    'periods': {
        'normalization_period': [50, 200],
        'momentum_period': [1, 6], 
        'volatility_period': [4, 10]
    },
    'thresholds': {
        'WOBI_threshold': [0.05, 0.2],
        'IV_threshold': [0.03, 0.1],
        'LCR_threshold': [0.15, 0.35]
    }
}

Эти индикаторы обеспечат богатый набор фич для генетического алгоритма, учитывающих как статическое состояние стакана, так и его динамику, с акцентом на интерпретируемость и вычислительную эффективность.

Вот более глубокое размышление на эту тему:

ГЛУБОКИЙ АНАЛИЗ ORDER BOOK ИНДИКАТОРОВ ДЛЯ ГЕНЕТИЧЕСКОЙ ОПТИМИЗАЦИИ
📚 ТЕОРЕТИЧЕСКИЕ ОСНОВЫ
Экономическая интерпретация Order Book данных

Order Book как отражение рыночных намерений:

    Ближние глубины (3-8%): Отображают краткосрочные намерения маркет-мейкеров и активных трейдеров

    Дальние глубины (60%): Отражают стратегические позиции институциональных игроков и долгосрочные уровни поддержки/сопротивления

Микроструктура рынка BTCUSDT:

    Высокая волатильность: Быстрое изменение стакана требует индикаторов с адаптивными параметрами

    24/7 торговля: Отсутствие гэпов, но наличие циклических паттернов ликвидности

    Маркет-мейкеры: Особое внимание к асимметрии между агрессивными и пассивными ордерами

🎯 РАСШИРЕННАЯ КЛАССИФИКАЦИЯ ИНДИКАТОРОВ
1. 🎪 ИНДИКАТОРЫ СТРУКТУРЫ СТАКАНА
1.1. Order Book Slope (OBS)
python

def calculate_obs(ratios):
    """Наклон кривой стакана через линейную регрессию по глубинам"""
    depths = np.array([3, 5, 8, 60])
    ratios_array = np.array([ratios['ratio3'], ratios['ratio5'], ratios['ratio8'], ratios['ratio60']])
    
    # Исключаем выбросы
    valid_mask = ~np.isnan(ratios_array)
    if np.sum(valid_mask) < 2:
        return 0
    
    slope, intercept = np.polyfit(depths[valid_mask], ratios_array[valid_mask], 1)
    return slope

# Экономическая интерпретация:
# Положительный OBS: дисбаланс усиливается с удалением от цены (стратегические позиции)
# Отрицательный OBS: дисбаланс ослабевает с удалением от цены (тактические позиции)

1.2. Liquidity Distribution Entropy (LDE)
python

def liquidity_entropy(bids, asks):
    """Энтропия распределения ликвидности по глубинам"""
    depths = [3, 5, 8, 60]
    total_bid = sum([bids[f'bid{d}'] for d in depths])
    total_ask = sum([asks[f'ask{d}'] for d in depths])
    
    # Вероятности для каждой глубины
    p_bid = [bids[f'bid{d}'] / total_bid for d in depths]
    p_ask = [asks[f'ask{d}'] / total_ask for d in depths]
    
    # Энтропия Шеннона
    entropy_bid = -sum([p * np.log(p + 1e-8) for p in p_bid])
    entropy_ask = -sum([p * np.log(p + 1e-8) for p in p_ask])
    
    return (entropy_bid + entropy_ask) / 2

# Интерпретация:
# Высокая энтропия: ликвидность равномерно распределена (стабильный рынок)
# Низкая энтропия: ликвидность сконцентрирована (потенциальная волатильность)

1.3. Price Impact Coefficient (PIC)
python

def price_impact_coefficient(bids, asks, price_levels=[3, 5, 8]):
    """Коэффициент ценового воздействия"""
    impacts = []
    for level in price_levels:
        # Оцениваем воздействие на цену при прохождении заданного объема
        bid_liq = bids[f'bid{level}']
        ask_liq = asks[f'ask{level}']
        
        # Упрощенная модель линейного воздействия
        impact_bid = level / 100 * (1 / (bid_liq + 1e-8))
        impact_ask = level / 100 * (1 / (ask_liq + 1e-8))
        
        impacts.append((impact_bid + impact_ask) / 2)
    
    return np.mean(impacts)

# Экономический смысл: Ожидаемое движение цены при агрессивной сделке

2. 🔄 ДИНАМИЧЕСКИЕ ИНДИКАТОРЫ
2.1. Order Book Momentum Spectrum
python

class OrderBookMomentum:
    def __init__(self, periods=[1, 3, 6, 12]):
        self.periods = periods  # в 15-минутных барах
    
    def calculate_spectrum(self, df, column_template):
        """Спектр моментума для разных периодов"""
        spectrum = {}
        for period in self.periods:
            for depth in [3, 5, 8, 60]:
                col = column_template.format(depth)
                momentum = df[col].pct_change(period)
                spectrum[f'momentum_{depth}_{period}'] = momentum
        
        return spectrum

# Применение:
# momentum_calculator = OrderBookMomentum()
# ratio_spectrum = momentum_calculator.calculate_spectrum(df, 'ratio{}')
# volume_spectrum = momentum_calculator.calculate_spectrum(df, 'bid{}')

2.2. Regime Change Detection
python

def detect_regime_changes(df, window=20, threshold=2.0):
    """Обнаружение смены режима стакана"""
    regimes = []
    
    for i in range(window, len(df)):
        window_data = df.iloc[i-window:i]
        current = df.iloc[i]
        
        # Статистические тесты на аномалии
        zscore_ratio3 = (current['ratio3'] - window_data['ratio3'].mean()) / window_data['ratio3'].std()
        zscore_volume = (current['volume'] - window_data['volume'].mean()) / window_data['volume'].std()
        zscore_imbalance = (current['d3'] - window_data['d3'].mean()) / window_data['d3'].std()
        
        # Композитный показатель аномалии
        anomaly_score = np.sqrt(zscore_ratio3**2 + zscore_volume**2 + zscore_imbalance**2)
        
        if anomaly_score > threshold:
            regimes.append(1)  # смена режима
        else:
            regimes.append(0)  # нормальный режим
    
    return pd.Series(regimes, index=df.index[window:])

2.3. Liquidity Flow Vector
python

def liquidity_flow_vector(df, lookback=4):
    """Вектор потока ликвидности между глубинами"""
    flows = []
    
    for i in range(lookback, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i-lookback]
        
        # Изменения объемов на разных глубинах
        flow_vector = []
        for depth in [3, 5, 8, 60]:
            bid_flow = current[f'bid{depth}'] - previous[f'bid{depth}']
            ask_flow = current[f'ask{depth}'] - previous[f'ask{depth}']
            net_flow = bid_flow - ask_flow
            flow_vector.append(net_flow)
        
        # Нормализация вектора
        flow_magnitude = np.linalg.norm(flow_vector)
        if flow_magnitude > 0:
            normalized_vector = flow_vector / flow_magnitude
        else:
            normalized_vector = flow_vector
            
        flows.append(normalized_vector)
    
    return flows

3. 📊 ПРОСТРАНСТВЕННО-ВРЕМЕННЫЕ ИНДИКАТОРЫ
3.1. Multi-Scale Order Book Imbalance
python

class MultiScaleImbalance:
    def __init__(self, time_scales=[1, 4, 12, 24]):  # 15min, 1h, 3h, 6h
        self.time_scales = time_scales
    
    def calculate_ms_imbalance(self, df):
        """Многомасштабный дисбаланс с вейвлет-анализом"""
        imbalances = {}
        
        for scale in self.time_scales:
            # Скользящее среднее для сглаживания
            smoothed_ratio3 = df['ratio3'].rolling(scale).mean()
            smoothed_ratio5 = df['ratio5'].rolling(scale).mean()
            smoothed_ratio8 = df['ratio8'].rolling(scale).mean()
            
            # Композитный индикатор для масштаба
            composite = (smoothed_ratio3 * 0.5 + smoothed_ratio5 * 0.3 + smoothed_ratio8 * 0.2)
            imbalances[f'imbalance_scale_{scale}'] = composite
        
        return imbalances

3.2. Order Book Fractal Dimension
python

def fractal_dimension_orderbook(df, window=50):
    """Фрактальная размерность стакана (мера сложности)"""
    fractal_dims = []
    
    for i in range(window, len(df)):
        window_data = df.iloc[i-window:i]
        
        # Анализ временных рядов дисбалансов
        ratios = window_data[['ratio3', 'ratio5', 'ratio8', 'ratio60']].values.flatten()
        
        # Упрощенный расчет фрактальной размерности через R/S анализ
        n = len(ratios)
        r_scores = []
        
        for chunk_size in [10, 20, 25]:
            chunks = n // chunk_size
            if chunks == 0:
                continue
                
            # R/S статистика для каждого чанка
            chunk_r_s = []
            for j in range(chunks):
                chunk_data = ratios[j*chunk_size:(j+1)*chunk_size]
                if len(chunk_data) < 2:
                    continue
                    
                mean_val = np.mean(chunk_data)
                deviations = chunk_data - mean_val
                cumulative_dev = np.cumsum(deviations)
                r = np.max(cumulative_dev) - np.min(cumulative_dev)
                s = np.std(chunk_data)
                
                if s > 0:
                    chunk_r_s.append(r / s)
            
            if chunk_r_s:
                r_scores.append(np.mean(chunk_r_s))
        
        if len(r_scores) >= 2:
            # Оценка фрактальной размерности через наклон в log-log пространстве
            x = np.log([10, 20, 25][:len(r_scores)])
            y = np.log(r_scores)
            slope = np.polyfit(x, y, 1)[0]
            fractal_dim = 2 - slope
        else:
            fractal_dim = 1.5  # значение по умолчанию
            
        fractal_dims.append(fractal_dim)
    
    return pd.Series(fractal_dims, index=df.index[window:])

4. 🎭 ИНДИКАТОРЫ ВЗАИМОДЕЙСТВИЯ С FLOW ДАННЫМИ
4.1. Aggressive Flow Absorption
python

def flow_absorption_ratio(df, window=10):
    """Коэффициент поглощения агрессивного потока"""
    absorption_ratios = []
    
    for i in range(window, len(df)):
        window_data = df.iloc[i-window:i]
        
        # Корреляция между flow imbalance и изменениями стакана
        flow_imbalance = window_data['flow_trade_imbalance']
        book_changes = window_data['d3'].diff()
        
        # Удаляем NaN
        valid_mask = ~(flow_imbalance.isna() | book_changes.isna())
        flow_imbalance = flow_imbalance[valid_mask]
        book_changes = book_changes[valid_mask]
        
        if len(flow_imbalance) > 3:
            # Коэффициент корреляции как мера поглощения
            correlation = np.corrcoef(flow_imbalance, book_changes)[0,1]
            if np.isnan(correlation):
                correlation = 0
        else:
            correlation = 0
        
        absorption_ratios.append(correlation)
    
    return pd.Series(absorption_ratios, index=df.index[window:])

4.2. Market Impact Efficiency
python

def market_impact_efficiency(df, impact_window=4):
    """Эффективность рыночного воздействия"""
    efficiency_scores = []
    
    for i in range(impact_window, len(df)):
        # Текущий flow imbalance
        current_flow = df.iloc[i]['flow_trade_imbalance']
        
        # Последующее движение цены
        future_returns = (df.iloc[i+1:i+impact_window+1]['close'].pct_change().mean()
                         if i + impact_window < len(df) else 0)
        
        # Эффективность = направленность воздействия
        if abs(current_flow) > 1e-8:  # избегаем деления на 0
            efficiency = (future_returns * np.sign(current_flow)) / abs(current_flow)
        else:
            efficiency = 0
            
        efficiency_scores.append(efficiency)
    
    return pd.Series(efficiency_scores, index=df.index[impact_window:len(df)-impact_window])

🔧 РАСШИРЕННАЯ РЕАЛИЗАЦИЯ С КЛАССАМИ
python

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class AdvancedOrderBookAnalytics:
    """
    Продвинутый анализ Order Book данных для генетической оптимизации
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self._validate_config()
    
    def _default_config(self):
        return {
            'imbalance_weights': [0.4, 0.3, 0.2, 0.1],
            'time_scales': [1, 4, 12, 24],
            'regime_threshold': 2.0,
            'fractal_window': 50,
            'efficiency_window': 4,
            'correlation_window': 10
        }
    
    def _validate_config(self):
        assert len(self.config['imbalance_weights']) == 4
        assert sum(self.config['imbalance_weights']) == 1.0
    
    def calculate_comprehensive_indicators(self, df):
        """
        Расчет полного набора расширенных индикаторов
        """
        results = {}
        
        # Базовые индикаторы
        results.update(self._basic_imbalance_indicators(df))
        results.update(self._liquidity_structure_indicators(df))
        results.update(self._dynamic_indicators(df))
        results.update(self._regime_indicators(df))
        results.update(self._advanced_composite_indicators(df))
        
        # Создаем DataFrame с результатами
        indicator_df = pd.DataFrame(results, index=df.index)
        
        # Заполняем пропуски
        indicator_df = indicator_df.ffill().bfill()
        
        return indicator_df
    
    def _basic_imbalance_indicators(self, df):
        """Базовые индикаторы дисбаланса"""
        w = self.config['imbalance_weights']
        
        # Weighted Order Book Imbalance
        wobi = (w[0]*df['ratio3'] + w[1]*df['ratio5'] + 
                w[2]*df['ratio8'] + w[3]*df['ratio60'])
        
        # Depth Gradient
        depth_gradient = self._calculate_depth_gradient(df)
        
        # Imbalance Momentum
        imbalance_momentum = wobi.diff(3)
        
        return {
            'WOBI': wobi,
            'depth_gradient': depth_gradient,
            'imbalance_momentum_3': imbalance_momentum,
            'imbalance_momentum_6': wobi.diff(6)
        }
    
    def _calculate_depth_gradient(self, df):
        """Градиент по глубинам с robust regression"""
        gradients = []
        
        for i in range(len(df)):
            depths = np.array([3, 5, 8, 60])
            ratios = np.array([
                df.iloc[i]['ratio3'],
                df.iloc[i]['ratio5'], 
                df.iloc[i]['ratio8'],
                df.iloc[i]['ratio60']
            ])
            
            # Robust linear regression
            try:
                slope, intercept = self._robust_linear_fit(depths, ratios)
                gradients.append(slope)
            except:
                gradients.append(0)
        
        return pd.Series(gradients, index=df.index)
    
    def _robust_linear_fit(self, x, y):
        """Устойчивая линейная регрессия"""
        # Исключаем выбросы
        if len(x) != len(y) or len(x) < 2:
            return 0, 0
            
        try:
            # Theil-Sen estimator для устойчивости к выбросам
            slopes = []
            for i in range(len(x)):
                for j in range(i+1, len(x)):
                    if x[j] != x[i]:
                        slope = (y[j] - y[i]) / (x[j] - x[i])
                        slopes.append(slope)
            
            if slopes:
                robust_slope = np.median(slopes)
                robust_intercept = np.median(y - robust_slope * x)
                return robust_slope, robust_intercept
            else:
                return 0, 0
        except:
            return 0, 0
    
    def _liquidity_structure_indicators(self, df):
        """Индикаторы структуры ликвидности"""
        # Total Liquidity Index
        tli_raw = (df['bid3'] + df['ask3'] + df['bid5'] + df['ask5'] + 
                  df['bid8'] + df['ask8'] + df['bid60'] + df['ask60']) / 8
        
        # Нормализация TLI
        tli_norm = self._rolling_normalize(tli_raw, 100)
        
        # Liquidity Concentration
        lcr = (df['bid3'] + df['ask3']) / (df['bid60'] + df['ask60'] + 1e-8)
        
        # Liquidity Entropy
        entropy = self._calculate_liquidity_entropy(df)
        
        return {
            'TLI_raw': tli_raw,
            'TLI_normalized': tli_norm,
            'LCR': lcr,
            'liquidity_entropy': entropy
        }
    
    def _calculate_liquidity_entropy(self, df):
        """Расчет энтропии ликвидности"""
        entropies = []
        
        for i in range(len(df)):
            bids = [df.iloc[i]['bid3'], df.iloc[i]['bid5'], 
                   df.iloc[i]['bid8'], df.iloc[i]['bid60']]
            asks = [df.iloc[i]['ask3'], df.iloc[i]['ask5'],
                   df.iloc[i]['ask8'], df.iloc[i]['ask60']]
            
            total_bid = sum(bids)
            total_ask = sum(asks)
            
            if total_bid > 0 and total_ask > 0:
                p_bid = [b / total_bid for b in bids]
                p_ask = [a / total_ask for a in asks]
                
                entropy_bid = -sum([p * np.log(p + 1e-8) for p in p_bid])
                entropy_ask = -sum([p * np.log(p + 1e-8) for p in p_ask])
                
                avg_entropy = (entropy_bid + entropy_ask) / 2
            else:
                avg_entropy = 0
                
            entropies.append(avg_entropy)
        
        return pd.Series(entropies, index=df.index)
    
    def _dynamic_indicators(self, df):
        """Динамические индикаторы"""
        # Order Book Flow
        obf = (df['d3'].diff() + df['d5'].diff() + 
               df['d8'].diff() + df['d60'].diff()) / 4
        
        # Liquidity Momentum
        tli_raw = (df['bid3'] + df['ask3'] + df['bid5'] + df['ask5'] + 
                  df['bid8'] + df['ask8'] + df['bid60'] + df['ask60']) / 8
        liq_momentum = tli_raw.pct_change(3) * 100
        
        # Imbalance Volatility
        wobi = (self.config['imbalance_weights'][0]*df['ratio3'] + 
                self.config['imbalance_weights'][1]*df['ratio5'] + 
                self.config['imbalance_weights'][2]*df['ratio8'] + 
                self.config['imbalance_weights'][3]*df['ratio60'])
        imb_volatility = wobi.rolling(6).std()
        
        return {
            'OBF': obf,
            'liquidity_momentum_3': liq_momentum,
            'imbalance_volatility_6': imb_volatility,
            'flow_absorption': self._flow_absorption_ratio(df)
        }
    
    def _flow_absorption_ratio(self, df):
        """Коэффициент поглощения flow"""
        absorption = []
        window = self.config['correlation_window']
        
        for i in range(window, len(df)):
            flow_data = df.iloc[i-window:i]['flow_trade_imbalance']
            book_changes = df.iloc[i-window:i]['d3'].diff()
            
            valid_mask = ~(flow_data.isna() | book_changes.isna())
            if valid_mask.sum() > 3:
                corr = np.corrcoef(flow_data[valid_mask], book_changes[valid_mask])[0,1]
                absorption.append(0 if np.isnan(corr) else corr)
            else:
                absorption.append(0)
        
        # Выравнивание индекса
        absorption_series = pd.Series(absorption, index=df.index[window:])
        return absorption_series.reindex(df.index, method='ffill')
    
    def _regime_indicators(self, df):
        """Индикаторы рыночных режимов"""
        regimes = self._detect_regime_changes(df)
        fractal_dims = self._fractal_dimension_approximation(df)
        
        return {
            'regime_change': regimes,
            'fractal_dimension': fractal_dims
        }
    
    def _detect_regime_changes(self, df):
        """Обнаружение смены режима"""
        regimes = [0] * min(20, len(df))
        window = 20
        
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]
            current = df.iloc[i]
            
            # Z-скоры ключевых метрик
            z_ratio = (current['ratio3'] - window_data['ratio3'].mean()) / (window_data['ratio3'].std() + 1e-8)
            z_volume = (current['volume'] - window_data['volume'].mean()) / (window_data['volume'].std() + 1e-8)
            z_imbalance = (current['d3'] - window_data['d3'].mean()) / (window_data['d3'].std() + 1e-8)
            
            anomaly_score = np.sqrt(z_ratio**2 + z_volume**2 + z_imbalance**2)
            
            if anomaly_score > self.config['regime_threshold']:
                regimes.append(1)
            else:
                regimes.append(0)
        
        return pd.Series(regimes, index=df.index)
    
    def _fractal_dimension_approximation(self, df):
        """Приближенный расчет фрактальной размерности"""
        fractal_dims = [1.5] * min(self.config['fractal_window'], len(df))
        window = self.config['fractal_window']
        
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]
            ratios = window_data[['ratio3', 'ratio5', 'ratio8', 'ratio60']].values.flatten()
            
            # Упрощенный R/S анализ
            try:
                chunk_sizes = [10, 20]
                r_s_ratios = []
                
                for size in chunk_sizes:
                    if len(ratios) >= size:
                        chunks = len(ratios) // size
                        chunk_rs = []
                        
                        for j in range(chunks):
                            chunk = ratios[j*size:(j+1)*size]
                            if len(chunk) > 1:
                                r = np.ptp(np.cumsum(chunk - np.mean(chunk)))
                                s = np.std(chunk)
                                if s > 0:
                                    chunk_rs.append(r / s)
                        
                        if chunk_rs:
                            r_s_ratios.append(np.mean(chunk_rs))
                
                if len(r_s_ratios) >= 2:
                    # Логарифмическая регрессия
                    log_sizes = np.log(chunk_sizes[:len(r_s_ratios)])
                    log_rs = np.log(r_s_ratios)
                    slope = np.polyfit(log_sizes, log_rs, 1)[0]
                    fractal_dim = 2 - slope
                else:
                    fractal_dim = 1.5
                    
            except:
                fractal_dim = 1.5
                
            fractal_dims.append(fractal_dim)
        
        return pd.Series(fractal_dims, index=df.index)
    
    def _advanced_composite_indicators(self, df):
        """Продвинутые композитные индикаторы"""
        # Smart Money Pressure
        smp = (df['ratio3'] * 2 - df['ratio60']) * self._liquidity_concentration(df)
        
        # Flow-Book Alignment
        flow_imb = df['flow_trade_imbalance'] / (df['volume'] + 1e-8)
        wobi = (self.config['imbalance_weights'][0]*df['ratio3'] + 
                self.config['imbalance_weights'][1]*df['ratio5'] + 
                self.config['imbalance_weights'][2]*df['ratio8'] + 
                self.config['imbalance_weights'][3]*df['ratio60'])
        foba = flow_imb * wobi
        
        # Liquidity Quality Index
        lqi = self._liquidity_quality_index(df)
        
        return {
            'SMP': smp,
            'FOBA': foba,
            'LQI': lqi,
            'market_efficiency': self._market_efficiency_score(df)
        }
    
    def _liquidity_concentration(self, df):
        """Концентрация ликвидности"""
        return (df['bid3'] + df['ask3']) / (df['bid60'] + df['ask60'] + 1e-8)
    
    def _liquidity_quality_index(self, df):
        """Индекс качества ликвидности"""
        # Нормализованные компоненты
        tli_norm = self._rolling_normalize(
            (df['bid3'] + df['ask3'] + df['bid5'] + df['ask5'] + 
             df['bid8'] + df['ask8'] + df['bid60'] + df['ask60']) / 8, 100
        )
        
        wobi = (self.config['imbalance_weights'][0]*df['ratio3'] + 
                self.config['imbalance_weights'][1]*df['ratio5'] + 
                self.config['imbalance_weights'][2]*df['ratio8'] + 
                self.config['imbalance_weights'][3]*df['ratio60'])
        vol_norm = 1 - self._rolling_normalize(wobi.rolling(6).std(), 100)
        
        lcr_norm = self._rolling_normalize(self._liquidity_concentration(df), 100)
        
        return (tli_norm + vol_norm + lcr_norm) / 3
    
    def _market_efficiency_score(self, df):
        """Оценка эффективности рынка"""
        efficiency_scores = [0] * self.config['efficiency_window']
        
        for i in range(self.config['efficiency_window'], len(df)):
            current_flow = df.iloc[i]['flow_trade_imbalance']
            
            if i + self.config['efficiency_window'] < len(df):
                future_returns = df.iloc[i+1:i+self.config['efficiency_window']+1]['close'].pct_change().mean()
            else:
                future_returns = 0
                
            if abs(current_flow) > 1e-8:
                efficiency = (future_returns * np.sign(current_flow)) / abs(current_flow)
            else:
                efficiency = 0
                
            efficiency_scores.append(efficiency)
        
        return pd.Series(efficiency_scores, index=df.index)
    
    def _rolling_normalize(self, series, window):
        """Скользящая нормализация 0-1"""
        rolling_min = series.rolling(window, min_periods=1).min()
        rolling_max = series.rolling(window, min_periods=1).max()
        return (series - rolling_min) / (rolling_max - rolling_min + 1e-8)

# Пример использования
def create_advanced_order_book_features(df, config=None):
    """
    Создание продвинутых фич Order Book для генетического алгоритма
    """
    analyzer = AdvancedOrderBookAnalytics(config)
    indicators = analyzer.calculate_comprehensive_indicators(df)
    
    # Добавляем многомасштабные версии ключевых индикаторов
    for col in ['WOBI', 'TLI_raw', 'SMP']:
        for period in [2, 4, 8]:
            indicators[f'{col}_sma_{period}'] = indicators[col].rolling(period).mean()
            indicators[f'{col}_momentum_{period}'] = indicators[col].pct_change(period)
    
    return indicators

🎯 СТРАТЕГИИ ИСПОЛЬЗОВАНИЯ В ГЕНЕТИЧЕСКОМ АЛГОРИТМЕ
1. Иерархическая оптимизация параметров
python

genetic_parameter_ranges = {
    # Веса для WOBI
    'WOBI_weights': {
        'w1': [0.1, 0.6],  # вес глубины 3%
        'w2': [0.1, 0.5],  # вес глубины 5%
        'w3': [0.1, 0.4],  # вес глубины 8%
        'w4': [0.05, 0.3]  # вес глубины 60%
    },
    
    # Периоды для анализа
    'time_parameters': {
        'momentum_periods': [1, 12],      # 15min - 3h
        'volatility_window': [4, 24],     # 1h - 6h
        'regime_window': [10, 50],        # 2.5h - 12.5h
        'normalization_period': [50, 200] # 12.5h - 50h
    },
    
    # Пороговые значения
    'threshold_parameters': {
        'regime_threshold': [1.5, 3.0],
        'imbalance_threshold': [0.05, 0.3],
        'liquidity_threshold': [0.1, 0.4],
        'efficiency_threshold': [-0.1, 0.1]
    },
    
    # Коэффициенты композитных индикаторов
    'composite_parameters': {
        'SMP_beta': [1.5, 3.0],     # коэффициент для ratio3 в SMP
        'SMP_gamma': [0.5, 1.5],    # коэффициент для ratio60 в SMP
        'LQI_alpha': [0.2, 0.5],    # вес ликвидности в LQI
        'LQI_beta': [0.2, 0.5],     # вес стабильности в LQI
        'LQI_gamma': [0.2, 0.5]     # вес концентрации в LQI
    }
}

2. Адаптивные стратегии для разных рыночных условий
python

class AdaptiveTradingStrategy:
    """
    Адаптивная стратегия, использующая Order Book индикаторы
    """
    
    def __init__(self, genetic_params):
        self.params = genetic_params
        self.current_regime = None
        
    def calculate_signals(self, indicators, price_data):
        """Расчет торговых сигналов с учетом режима"""
        signals = []
        
        for i in range(len(indicators)):
            current_indicators = indicators.iloc[i]
            regime = self._classify_regime(current_indicators)
            
            if regime == 'trending_bullish':
                signal = self._trending_bullish_strategy(current_indicators)
            elif regime == 'trending_bearish':
                signal = self._trending_bearish_strategy(current_indicators)
            elif regime == 'ranging_high_vol':
                signal = self._ranging_high_vol_strategy(current_indicators)
            elif regime == 'ranging_low_vol':
                signal = self._ranging_low_vol_strategy(current_indicators)
            else:
                signal = 0
                
            signals.append(signal)
            
        return pd.Series(signals, index=indicators.index)
    
    def _classify_regime(self, indicators):
        """Классификация рыночного режима"""
        wobi = indicators.get('WOBI', 0)
        volatility = indicators.get('imbalance_volatility_6', 0)
        liquidity = indicators.get('TLI_normalized', 0)
        efficiency = indicators.get('market_efficiency', 0)
        
        # Логика классификации на основе генетически оптимизированных параметров
        if abs(wobi) > self.params['imbalance_threshold']:
            if wobi > 0:
                return 'trending_bullish'
            else:
                return 'trending_bearish'
        elif volatility > self.params['volatility_threshold']:
            return 'ranging_high_vol'
        else:
            return 'ranging_low_vol'
    
    def _trending_bullish_strategy(self, indicators):
        """Стратегия для бычьего тренда"""
        # Используем SMP и FOBA для подтверждения тренда
        smp_signal = 1 if indicators.get('SMP', 0) > 0.1 else 0
        foba_signal = 1 if indicators.get('FOBA', 0) > 0.05 else 0
        momentum_signal = 1 if indicators.get('imbalance_momentum_3', 0) > 0 else 0
        
        return (smp_signal + foba_signal + momentum_signal) / 3
    
    # ... аналогичные методы для других режимов

📊 ВАЛИДАЦИЯ И ТЕСТИРОВАНИЕ
1. Статистическая значимость индикаторов
python

def validate_indicators_significance(indicators, future_returns, min_correlation=0.05):
    """
    Валидация статистической значимости индикаторов
    """
    significant_indicators = {}
    
    for col in indicators.columns:
        # Корреляция с будущими returns (1, 3, 6 баров вперед)
        correlations = []
        for horizon in [1, 3, 6]:
            if horizon < len(future_returns):
                corr = np.corrcoef(indicators[col].iloc[:-horizon], 
                                 future_returns.iloc[horizon:])[0,1]
                correlations.append(abs(corr))
        
        avg_correlation = np.mean(correlations) if correlations else 0
        
        if avg_correlation > min_correlation:
            significant_indicators[col] = {
                'avg_correlation': avg_correlation,
                'max_correlation': max(correlations) if correlations else 0,
                'stability': np.std(correlations) if len(correlations) > 1 else 0
            }
    
    return significant_indicators

2. Анализ устойчивости во времени
python

def temporal_stability_analysis(indicators, returns, rolling_window=1000):
    """
    Анализ устойчивости индикаторов во времени
    """
    stability_metrics = {}
    
    for col in indicators.columns:
        rolling_correlations = []
        
        for i in range(rolling_window, len(indicators), rolling_window//4):
            chunk_indicators = indicators[col].iloc[i-rolling_window:i]
            chunk_returns = returns.iloc[i-rolling_window:i]
            
            if len(chunk_indicators) > 10:
                corr = np.corrcoef(chunk_indicators, chunk_returns)[0,1]
                rolling_correlations.append(corr if not np.isnan(corr) else 0)
        
        if rolling_correlations:
            stability = 1 - np.std(rolling_correlations) / (np.mean(np.abs(rolling_correlations)) + 1e-8)
            stability_metrics[col] = stability
    
    return stability_metrics

Этот расширенный подход обеспечивает:

    Глубину анализа: От базовых дисбалансов до фрактальной размерности

    Адаптивность: Индикаторы, работающие в разных рыночных режимах

    Статистическую обоснованность: Валидация значимости и устойчивости

    Интеграцию с генетическим алгоритмом: Параметризация и оптимизация

    Экономическую интерпретируемость: Каждый индикатор имеет четкий экономический смысл

Такой комплексный подход позволит генетическому алгоритму находить действительно эффективные торговые стратегии, основанные на микроструктуре рынка.