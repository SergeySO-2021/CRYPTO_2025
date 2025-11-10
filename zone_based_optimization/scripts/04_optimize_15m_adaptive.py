"""
Скрипт 04: Адаптивная оптимизация параметров
Адаптирует целевое количество сегментов под реальные данные
"""

import sys
import io
import os
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

# Исправление кодировки для Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# Импорт Trading Classifier
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
classifier_path = os.path.join(project_root, 'indicators', 'trading_classifier_iziceros', 'src')
sys.path.insert(0, classifier_path)

try:
    from trend_classifier import Segmenter, Config
    from trend_classifier.models import Metrics
except ImportError as e:
    print(f"[ERROR] Не удалось импортировать Trading Classifier: {e}")
    sys.exit(1)

def evaluate_segmentation_quality_adaptive(seg, df):
    """Адаптивная оценка качества - учитывает реальное количество сегментов"""
    days_in_data = (df.index[-1] - df.index[0]).days
    segments_count = len(seg.segments)
    
    # Оценка по длине сегментов
    segment_lengths = [s.stop - s.start for s in seg.segments]
    avg_length = np.mean(segment_lengths) if segment_lengths else 0
    min_length = np.min(segment_lengths) if segment_lengths else 0
    max_length = np.max(segment_lengths) if segment_lengths else 0
    
    # Оценка длины (не слишком короткие, не слишком длинные)
    if 20 <= avg_length <= 500:
        length_score = 1.0
    elif 10 <= avg_length <= 1000:
        length_score = 0.8
    else:
        length_score = 0.5
    
    # Оценка стабильности
    stability_scores = []
    for segment in seg.segments:
        try:
            slope = segment.slope
            slope_std = getattr(segment, 'slopes_std', 0) or 0
            if abs(slope) > 0.0001 and slope_std is not None:
                cv = slope_std / abs(slope)
                stability_scores.append(1.0 if cv < 0.5 else 0.8 if cv < 0.8 else 0.5)
            else:
                stability_scores.append(0.5)
        except:
            stability_scores.append(0.5)
    
    stability_score = np.mean(stability_scores) if stability_scores else 0.5
    
    # Оценка разброса длин (меньше разброс = лучше)
    if segment_lengths:
        cv_length = np.std(segment_lengths) / avg_length if avg_length > 0 else 1.0
        consistency_score = 1.0 if cv_length < 0.8 else 0.7 if cv_length < 1.2 else 0.5
    else:
        consistency_score = 0.5
    
    # Проверка на экстремальные значения
    extreme_short = sum(1 for l in segment_lengths if l < 5)
    extreme_long = sum(1 for l in segment_lengths if l > 2000)
    extreme_ratio = (extreme_short + extreme_long) / len(segment_lengths) if segment_lengths else 1.0
    extreme_score = 1.0 - extreme_ratio
    
    # Общий score - НЕ штрафуем за количество сегментов, если они хорошего качества
    total_score = (
        length_score * 0.3 +
        stability_score * 0.3 +
        consistency_score * 0.2 +
        extreme_score * 0.2
    )
    
    quality = 'excellent' if total_score > 0.85 else \
              'very_good' if total_score > 0.75 else \
              'good' if total_score > 0.65 else \
              'acceptable' if total_score > 0.55 else 'poor'
    
    return {
        'total_score': total_score,
        'quality': quality,
        'segments_count': segments_count,
        'avg_length': avg_length,
        'min_length': min_length,
        'max_length': max_length,
        'stability': stability_score,
        'consistency': consistency_score
    }

def optimize_adaptive(df, verbose=True):
    """Адаптивная оптимизация - фокус на качество сегментов, а не на их количество"""
    
    # Очень широкие диапазоны для поиска
    N_range = range(40, 120, 5)
    overlap_range = [0.2, 0.25, 0.33, 0.4, 0.5]
    # Очень высокие значения alpha/beta для получения разумного количества сегментов
    alpha_range = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0]
    beta_range = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0]
    
    best_config = None
    best_score = 0
    best_evaluation = None
    results = []
    
    total_combinations = len(list(N_range)) * len(overlap_range) * len(alpha_range) * len(beta_range)
    
    if verbose:
        print(f"\n[SEARCH] Адаптивная оптимизация параметров")
        print(f"   Фокус: качество сегментов (длина, стабильность, консистентность)")
        print(f"   Проверяем {total_combinations} комбинаций...")
        print(f"   alpha/beta: 15.0-50.0 (очень высокие значения)\n")
    
    checked = 0
    for N, overlap, alpha, beta in itertools.product(N_range, overlap_range, alpha_range, beta_range):
        try:
            config = Config(
                N=N,
                overlap_ratio=overlap,
                alpha=alpha,
                beta=beta,
                metrics_for_alpha=Metrics.RELATIVE_ABSOLUTE_ERROR,
                metrics_for_beta=Metrics.RELATIVE_ABSOLUTE_ERROR
            )
            
            seg = Segmenter(df=df, column="close", config=config)
            seg.calculate_segments()
            
            if not seg.segments or len(seg.segments) < 3:
                continue
            
            # Фильтр: разумное количество сегментов (не меньше 10, не больше 500)
            segments_count = len(seg.segments)
            if segments_count < 10 or segments_count > 500:
                continue
            
            evaluation = evaluate_segmentation_quality_adaptive(seg, df)
            score = evaluation['total_score']
            
            results.append({
                'N': N,
                'overlap_ratio': overlap,
                'alpha': alpha,
                'beta': beta,
                'score': score,
                'segments_count': segments_count,
                'quality': evaluation['quality'],
                'avg_length': evaluation['avg_length'],
                'stability': evaluation['stability']
            })
            
            if score > best_score:
                best_score = score
                best_config = config
                best_evaluation = evaluation
                
                if verbose:
                    print(f"   [NEW BEST] Score: {score:.3f} ({evaluation['quality']}) | "
                          f"N={N}, overlap={overlap}, alpha={alpha}, beta={beta}")
                    print(f"              Сегментов: {segments_count}, "
                          f"Средняя длина: {evaluation['avg_length']:.1f}, "
                          f"Stability: {evaluation['stability']:.3f}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Прервано пользователем")
            break
        except Exception:
            continue
        
        checked += 1
        if verbose and checked % 150 == 0:
            print(f"   [PROGRESS] {checked}/{total_combinations} | "
                  f"Найдено: {len(results)} | Лучший: {best_score:.3f}")
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        'best_config': {
            'N': best_config.N if best_config else None,
            'overlap_ratio': best_config.overlap_ratio if best_config else None,
            'alpha': best_config.alpha if best_config else None,
            'beta': best_config.beta if best_config else None
        },
        'best_score': best_score,
        'best_evaluation': best_evaluation,
        'top_20_results': results[:20] if results else [],
        'checked_combinations': checked
    }

def main():
    try:
        print("=" * 80)
        print("АДАПТИВНАЯ ОПТИМИЗАЦИЯ ПАРАМЕТРОВ TRADING CLASSIFIER")
        print("Фокус: качество сегментов, а не их точное количество")
        print("=" * 80)
        
        # Загрузка данных (ИСПОЛЬЗУЕМ ТОЛЬКО TRAIN SET!)
        data_file = 'data/train/df_btc_15m_train.csv'
        if not os.path.exists(data_file):
            # Fallback на полные данные (если train set еще не создан)
            data_file = 'data/prepared/df_btc_15m_prepared.csv'
            if not os.path.exists(data_file):
                print(f"[ERROR] Файл данных не найден: {data_file}")
                print("   Сначала запустите: py scripts/03b_split_data.py")
                return
            else:
                print(f"[WARNING] Используются полные данные вместо train set!")
                print(f"   Рекомендуется: py scripts/03b_split_data.py")
        
        print(f"\n[LOAD] Загрузка данных...")
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        print(f"   [OK] Загружено {len(df)} записей (TRAIN SET для обучения)")
        
        result = optimize_adaptive(df, verbose=True)
        
        # Сохранение
        output_file = Path('results') / 'optimization_15m_adaptive.json'
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\n[SAVE] Результаты сохранены: {output_file}")
        
        if result['top_20_results']:
            print("\n" + "=" * 80)
            print("ТОП-10 РЕЗУЛЬТАТОВ:")
            print("=" * 80)
            for i, res in enumerate(result['top_20_results'][:10], 1):
                print(f"\n{i}. Score: {res['score']:.3f} ({res['quality']})")
                print(f"   N={res['N']}, overlap={res['overlap_ratio']}, "
                      f"alpha={res['alpha']}, beta={res['beta']}")
                print(f"   Сегментов: {res['segments_count']}, "
                      f"Средняя длина: {res['avg_length']:.1f}, "
                      f"Stability: {res['stability']:.3f}")
        else:
            print("\n[WARNING] Не найдено результатов!")
        
        print("\n" + "=" * 80)
        print("[SUCCESS] ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
        print("=" * 80)
        
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

