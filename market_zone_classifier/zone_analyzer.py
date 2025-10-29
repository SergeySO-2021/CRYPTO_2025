"""
Market Zone Classifier - Python Analyzer
Простой и надежный анализатор рыночных зон
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MarketZoneClassifier:
    """Классификатор рыночных зон на основе Price Action"""
    
    def __init__(self, lookback_period=20, trend_threshold=0.5, volatility_threshold=1.5):
        self.lookback_period = lookback_period
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        
    def calculate_price_action(self, df):
        """Расчет Price Action метрик"""
        # Ценовые экстремумы
        df['highest_high'] = df['high'].rolling(window=self.lookback_period).max()
        df['lowest_low'] = df['low'].rolling(window=self.lookback_period).min()
        df['price_range'] = df['highest_high'] - df['lowest_low']
        
        # Скользящие средние
        df['fast_ma'] = df['close'].rolling(window=10).mean()
        df['slow_ma'] = df['close'].rolling(window=20).mean()
        df['ma_slope'] = df['fast_ma'] - df['slow_ma']
        
        # Позиция цены в диапазоне
        df['price_position'] = (df['close'] - df['lowest_low']) / df['price_range']
        
        return df
    
    def calculate_volatility(self, df, volatility_period=14):
        """Расчет волатильности"""
        # ATR
        df['tr'] = np.maximum(df['high'] - df['low'], 
                             np.maximum(abs(df['high'] - df['close'].shift(1)),
                                       abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(window=volatility_period).mean()
        df['atr_ma'] = df['atr'].rolling(window=volatility_period).mean()
        df['volatility_ratio'] = df['atr'] / df['atr_ma']
        
        # Bollinger Bands
        df['bb_basis'] = df['close'].rolling(window=volatility_period).mean()
        df['bb_dev'] = df['close'].rolling(window=volatility_period).std()
        df['bb_upper'] = df['bb_basis'] + 2.0 * df['bb_dev']
        df['bb_lower'] = df['bb_basis'] - 2.0 * df['bb_dev']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_basis']
        
        return df
    
    def classify_zones(self, df):
        """Классификация рыночных зон"""
        # Тренд
        df['trend_up'] = (df['ma_slope'] > self.trend_threshold) & (df['close'] > df['fast_ma'])
        df['trend_down'] = (df['ma_slope'] < -self.trend_threshold) & (df['close'] < df['fast_ma'])
        df['sideways'] = ~df['trend_up'] & ~df['trend_down']
        
        # Волатильность
        df['high_volatility'] = (df['volatility_ratio'] > self.volatility_threshold) | (df['bb_width'] > 0.1)
        df['low_volatility'] = (df['volatility_ratio'] < (1 / self.volatility_threshold)) & (df['bb_width'] < 0.05)
        df['normal_volatility'] = ~df['high_volatility'] & ~df['low_volatility']
        
        # Ценовая позиция
        df['price_near_high'] = df['price_position'] > 0.8
        df['price_near_low'] = df['price_position'] < 0.2
        df['price_in_middle'] = ~df['price_near_high'] & ~df['price_near_low']
        
        # Определение зон
        df['primary_zone'] = np.where(df['trend_up'], 1, 
                                    np.where(df['trend_down'], -1, 0))
        df['secondary_zone'] = np.where(df['high_volatility'], 2,
                                       np.where(df['low_volatility'], -2, 0))
        df['zone'] = df['primary_zone'] + df['secondary_zone']
        
        # Названия зон
        zone_names = {
            3: "Strong Bull + High Vol",
            2: "Bull + High Vol", 
            1: "Bull + Normal Vol",
            0: "Sideways",
            -1: "Bear + Normal Vol",
            -2: "Bear + High Vol",
            -3: "Strong Bear + High Vol"
        }
        df['zone_name'] = df['zone'].map(zone_names).fillna("Unknown")
        
        return df
    
    def analyze_zones(self, df):
        """Анализ качества классификации зон"""
        # Статистика по зонам
        zone_stats = df.groupby('zone_name').agg({
            'close': ['count', 'mean', 'std'],
            'volume': 'mean',
            'volatility_ratio': 'mean',
            'bb_width': 'mean'
        }).round(4)
        
        # Стабильность зон
        zone_changes = (df['zone'] != df['zone'].shift(1)).sum()
        total_periods = len(df)
        stability = 1 - (zone_changes / total_periods)
        
        # Распределение зон
        zone_distribution = df['zone_name'].value_counts(normalize=True) * 100
        
        return {
            'zone_stats': zone_stats,
            'stability': stability,
            'zone_distribution': zone_distribution,
            'zone_changes': zone_changes
        }
    
    def plot_zones(self, df, title="Market Zone Classification"):
        """Визуализация классификации зон"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # График цены с зонами
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1)
        ax1.plot(df.index, df['fast_ma'], label='Fast MA', alpha=0.7)
        ax1.plot(df.index, df['slow_ma'], label='Slow MA', alpha=0.7)
        ax1.set_title(f'{title} - Price Action')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График зон
        ax2 = axes[1]
        colors = {3: 'green', 2: 'lime', 1: 'aqua', 0: 'gray', 
                 -1: 'orange', -2: 'red', -3: 'maroon'}
        for zone in df['zone'].unique():
            if not pd.isna(zone):
                mask = df['zone'] == zone
                ax2.scatter(df.index[mask], df['zone'][mask], 
                           c=colors.get(zone, 'black'), alpha=0.6, s=20)
        ax2.set_title('Zone Classification')
        ax2.set_ylabel('Zone')
        ax2.grid(True, alpha=0.3)
        
        # График волатильности
        ax3 = axes[2]
        ax3.plot(df.index, df['volatility_ratio'], label='Volatility Ratio', alpha=0.7)
        ax3.axhline(y=self.volatility_threshold, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=1/self.volatility_threshold, color='blue', linestyle='--', alpha=0.5)
        ax3.set_title('Volatility Analysis')
        ax3.set_ylabel('Volatility Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def optimize_parameters(self, df, param_ranges):
        """Оптимизация параметров классификатора"""
        best_params = None
        best_score = 0
        
        for lookback in param_ranges['lookback_period']:
            for trend_thresh in param_ranges['trend_threshold']:
                for vol_thresh in param_ranges['volatility_threshold']:
                    # Создаем новый классификатор с тестовыми параметрами
                    test_classifier = MarketZoneClassifier(lookback, trend_thresh, vol_thresh)
                    
                    # Применяем классификацию
                    test_df = df.copy()
                    test_df = test_classifier.calculate_price_action(test_df)
                    test_df = test_classifier.calculate_volatility(test_df)
                    test_df = test_classifier.classify_zones(test_df)
                    
                    # Анализируем качество
                    analysis = test_classifier.analyze_zones(test_df)
                    
                    # Оценка качества (комбинация стабильности и распределения)
                    score = analysis['stability'] * 0.7 + (1 - analysis['zone_distribution'].std() / 100) * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'lookback_period': lookback,
                            'trend_threshold': trend_thresh,
                            'volatility_threshold': vol_thresh,
                            'score': score
                        }
        
        return best_params

def load_test_data():
    """Загрузка тестовых данных"""
    # Здесь можно загрузить реальные данные BTC
    # Пока создадим синтетические данные для демонстрации
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='4H')
    np.random.seed(42)
    
    # Создаем синтетические данные с трендами и флэтами
    data = []
    price = 50000
    
    for i, date in enumerate(dates):
        # Добавляем тренды и флэты
        if i % 100 < 30:  # Восходящий тренд
            price += np.random.normal(100, 50)
        elif i % 100 < 60:  # Флэт
            price += np.random.normal(0, 20)
        else:  # Нисходящий тренд
            price += np.random.normal(-80, 40)
        
        # Создаем OHLC данные
        high = price + abs(np.random.normal(0, 100))
        low = price - abs(np.random.normal(0, 100))
        open_price = price + np.random.normal(0, 50)
        close = price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

if __name__ == "__main__":
    print("🎯 Market Zone Classifier - Python Analyzer")
    print("=" * 50)
    
    # Загружаем тестовые данные
    print("📊 Загружаем тестовые данные...")
    df = load_test_data()
    print(f"✅ Загружено {len(df)} записей")
    
    # Создаем классификатор
    print("\n🔧 Создаем классификатор...")
    classifier = MarketZoneClassifier()
    
    # Применяем классификацию
    print("🧠 Применяем классификацию зон...")
    df = classifier.calculate_price_action(df)
    df = classifier.calculate_volatility(df)
    df = classifier.classify_zones(df)
    
    # Анализируем результаты
    print("📊 Анализируем результаты...")
    analysis = classifier.analyze_zones(df)
    
    print(f"\n📈 СТАТИСТИКА КЛАССИФИКАЦИИ:")
    print(f"   Стабильность зон: {analysis['stability']:.3f}")
    print(f"   Количество смен зон: {analysis['zone_changes']}")
    print(f"   Всего периодов: {len(df)}")
    
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ ЗОН:")
    for zone, percentage in analysis['zone_distribution'].items():
        print(f"   {zone}: {percentage:.1f}%")
    
    # Оптимизация параметров
    print(f"\n🚀 Оптимизация параметров...")
    param_ranges = {
        'lookback_period': [15, 20, 25],
        'trend_threshold': [0.3, 0.5, 0.7],
        'volatility_threshold': [1.2, 1.5, 1.8]
    }
    
    best_params = classifier.optimize_parameters(df, param_ranges)
    print(f"🏆 Лучшие параметры:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    # Визуализация
    print(f"\n📊 Создаем визуализацию...")
    classifier.plot_zones(df)
    
    print(f"\n✅ Анализ завершен!")
