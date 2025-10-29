"""
🔍 АНАЛИЗ ВАШЕГО ТЕКУЩЕГО MZA
Анализ качества определения рыночных зон на 15m и 1h
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к модулям
current_dir = os.getcwd()
if 'indicator_optimization' in current_dir:
    sys.path.append('.')
else:
    sys.path.append('./indicator_optimization/01_mza_optimization')

try:
    from data_loader import load_btc_data
    from accurate_mza_classifier import AccurateMZAClassifier
    from mza_quality_analyzer import MZAQualityAnalyzer
    print("✅ Модули импортированы")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")

def analyze_current_mza():
    """Анализ текущего MZA на разных таймфреймах"""
    print("🔍 АНАЛИЗ ВАШЕГО ТЕКУЩЕГО MZA")
    print("=" * 50)
    
    # Загружаем данные
    print("📊 Загружаем данные...")
    btc_data = load_btc_data()
    
    # Параметры из вашей оптимизации 15m
    params_15m = {
        'adxLength': 16,
        'adxThreshold': 25,
        'fastMALength': 18,
        'slowMALength': 45,
        'rsiLength': 12,
        'stochKLength': 16,
        'macdFast': 10,
        'macdSlow': 24,
        'macdSignal': 8,
        'hhllRange': 22,
        'haDojiRange': 6,
        'candleRangeLength': 10,
        'bbLength': 18,
        'bbMultiplier': 2.2,
        'atrLength': 12,
        'kcLength': 18,
        'kcMultiplier': 1.8,
        'volumeMALength': 18,
        'trendWeightBase': 42,
        'momentumWeightBase': 32,
        'priceActionWeightBase': 26,
        'useSmoothing': True,
        'useHysteresis': True
    }
    
    # Параметры для 1h (из вашей оптимизации)
    params_1h = {
        'adxLength': 15,
        'adxThreshold': 22,
        'fastMALength': 19,
        'slowMALength': 47,
        'rsiLength': 13,
        'stochKLength': 15,
        'macdFast': 11,
        'macdSlow': 25,
        'macdSignal': 9,
        'hhllRange': 21,
        'haDojiRange': 7,
        'candleRangeLength': 11,
        'bbLength': 19,
        'bbMultiplier': 2.1,
        'atrLength': 13,
        'kcLength': 19,
        'kcMultiplier': 1.7,
        'volumeMALength': 19,
        'trendWeightBase': 41,
        'momentumWeightBase': 33,
        'priceActionWeightBase': 26,
        'useSmoothing': True,
        'useHysteresis': True
    }
    
    analyzer = MZAQualityAnalyzer()
    results_list = []
    
    # Анализируем каждый таймфрейм
    timeframes = ['15m', '1h', '4h']
    params_dict = {'15m': params_15m, '1h': params_1h, '4h': params_1h}  # Для 4h используем параметры 1h
    
    for tf in timeframes:
        if tf in btc_data:
            print(f"\n{'='*60}")
            print(f"🔍 АНАЛИЗ ТАЙМФРЕЙМА: {tf}")
            print(f"{'='*60}")
            
            data = btc_data[tf]
            params = params_dict[tf]
            
            # Создаем классификатор
            classifier = AccurateMZAClassifier(params)
            
            # Получаем предсказания зон
            print(f"🧠 Вычисляем зоны для {tf}...")
            zones = classifier.predict(data)
            
            # Анализируем качество
            results = analyzer.analyze_mza_quality(data, zones, params, tf)
            results_list.append(results)
            
            # Сохраняем результаты
            results['data'] = data
            results['zones'] = zones
            results['params'] = params
            
        else:
            print(f"⚠️ Данные для {tf} не найдены")
    
    # Сравниваем таймфреймы
    if len(results_list) > 1:
        print(f"\n{'='*60}")
        print("📊 СРАВНЕНИЕ ПО ТАЙМФРЕЙМАМ")
        print(f"{'='*60}")
        
        comparison_df = analyzer.compare_timeframes(results_list)
        
        # Рекомендации
        print_recommendations(results_list)
    
    return results_list

def print_recommendations(results_list):
    """Выводит рекомендации по улучшению MZA"""
    print(f"\n🎯 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ MZA:")
    print("=" * 50)
    
    # Находим лучший результат
    best_result = max(results_list, key=lambda x: x['composite_score'])
    worst_result = min(results_list, key=lambda x: x['composite_score'])
    
    print(f"🏆 ЛУЧШИЙ ТАЙМФРЕЙМ: {best_result['timeframe']}")
    print(f"   Composite Score: {best_result['composite_score']:.6f}")
    print(f"   Economic Value: {best_result['economic_value']:.6f}")
    
    print(f"\n⚠️ ПРОБЛЕМНЫЙ ТАЙМФРЕЙМ: {worst_result['timeframe']}")
    print(f"   Composite Score: {worst_result['composite_score']:.6f}")
    print(f"   Economic Value: {worst_result['economic_value']:.6f}")
    
    # Анализируем проблемы
    print(f"\n🔍 АНАЛИЗ ПРОБЛЕМ:")
    
    for result in results_list:
        tf = result['timeframe']
        ev = result['economic_value']
        zs = result['zone_stability']
        sc = result['signal_consistency']
        
        print(f"\n📊 {tf}:")
        
        if ev < 0.001:
            print(f"   ❌ Economic Value слишком низкий ({ev:.6f})")
            print(f"      💡 Рекомендация: Увеличьте adxThreshold или используйте другой таймфрейм")
        
        if zs < 0.4:
            print(f"   ❌ Зоны нестабильны ({zs:.3f})")
            print(f"      💡 Рекомендация: Включите useSmoothing=True и useHysteresis=True")
        
        if sc < 0.5:
            print(f"   ❌ Низкая точность сигналов ({sc:.3f})")
            print(f"      💡 Рекомендация: Переоптимизируйте параметры или используйте другой таймфрейм")
        
        if ev > 0.005 and zs > 0.6 and sc > 0.5:
            print(f"   ✅ Качество хорошее!")
    
    # Общие рекомендации
    print(f"\n🚀 ОБЩИЕ РЕКОМЕНДАЦИИ:")
    print("   1. 📊 Используйте лучший таймфрейм для торговли")
    print("   2. 🔄 Регулярно переоптимизируйте параметры (раз в месяц)")
    print("   3. 📈 Не полагайтесь только на MZA - используйте дополнительные индикаторы")
    print("   4. ⚠️ Тестируйте на разных активах и рыночных условиях")
    print("   5. 🎯 Фокусируйтесь на Economic Value > 0.005 для хорошего качества")

def create_visual_analysis(results_list):
    """Создает визуальный анализ результатов"""
    print(f"\n📊 СОЗДАЕМ ВИЗУАЛЬНЫЙ АНАЛИЗ...")
    
    # Подготавливаем данные для графиков
    timeframes = [r['timeframe'] for r in results_list]
    economic_values = [r['economic_value'] for r in results_list]
    zone_stabilities = [r['zone_stability'] for r in results_list]
    signal_consistencies = [r['signal_consistency'] for r in results_list]
    composite_scores = [r['composite_score'] for r in results_list]
    
    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Анализ качества MZA по таймфреймам', fontsize=16)
    
    # Economic Value
    axes[0, 0].bar(timeframes, economic_values, color=['green', 'blue', 'orange'])
    axes[0, 0].set_title('Economic Value')
    axes[0, 0].set_ylabel('Economic Value')
    axes[0, 0].axhline(y=0.005, color='red', linestyle='--', label='Хорошее качество')
    axes[0, 0].legend()
    
    # Zone Stability
    axes[0, 1].bar(timeframes, zone_stabilities, color=['green', 'blue', 'orange'])
    axes[0, 1].set_title('Zone Stability')
    axes[0, 1].set_ylabel('Zone Stability')
    axes[0, 1].axhline(y=0.6, color='red', linestyle='--', label='Стабильные зоны')
    axes[0, 1].legend()
    
    # Signal Consistency
    axes[1, 0].bar(timeframes, signal_consistencies, color=['green', 'blue', 'orange'])
    axes[1, 0].set_title('Signal Consistency')
    axes[1, 0].set_ylabel('Signal Consistency')
    axes[1, 0].axhline(y=0.5, color='red', linestyle='--', label='Средняя точность')
    axes[1, 0].legend()
    
    # Composite Score
    axes[1, 1].bar(timeframes, composite_scores, color=['green', 'blue', 'orange'])
    axes[1, 1].set_title('Composite Score')
    axes[1, 1].set_ylabel('Composite Score')
    axes[1, 1].axhline(y=0.005, color='red', linestyle='--', label='Хорошее качество')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('indicator_optimization/01_mza_optimization/mza_quality_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ График сохранен как mza_quality_analysis.png")

if __name__ == "__main__":
    # Запускаем анализ
    results = analyze_current_mza()
    
    # Создаем визуальный анализ
    if len(results) > 1:
        create_visual_analysis(results)
    
    print(f"\n🎉 АНАЛИЗ ЗАВЕРШЕН!")
    print("📊 Проверьте результаты выше для понимания качества вашего MZA")
