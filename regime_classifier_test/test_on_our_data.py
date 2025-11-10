"""
Скрипт для тестирования CryptoMarket Regime Classifier на наших данных
Использует multi-timeframe: 15m, 30m, 1h
С разделением на train/test выборки
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Импорты из проекта
from compute_features import build_features
from regime_label import get_hmm_features, train_hmm, map_states_to_regimes
from lstm_model import create_sequences, build_lstm_model

def load_multi_timeframe_data():
    """
    Загружает multi-timeframe данные: 15m, 30m, 1h
    """
    print("=" * 80)
    print("ЗАГРУЗКА MULTI-TIMEFRAME ДАННЫХ")
    print("=" * 80)
    
    base_path = Path(r'..')
    
    # Пути к файлам
    files = {
        '15m': base_path / 'df_btc_15m_complete.csv',
        '30m': base_path / 'df_btc_30m_complete.csv',
        '1h': base_path / 'df_btc_1h_complete.csv'
    }
    
    dataframes = {}
    
    for tf, file_path in files.items():
        if not file_path.exists():
            print(f"[ERROR] Файл не найден: {file_path}")
            raise FileNotFoundError(f"Не найден файл: {file_path}")
        
        print(f"\n[LOAD] Загрузка {tf} данных из {file_path.name}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"   [OK] Загружено {len(df)} записей")
        print(f"   Период: {df.index[0]} - {df.index[-1]}")
        
        # Проверяем наличие необходимых колонок
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"   [WARNING] Отсутствуют колонки: {missing}")
        
        dataframes[tf] = df
    
    return dataframes

def prepare_multi_timeframe_data(dataframes):
    """
    Подготавливает multi-timeframe данные в формате проекта
    
    Проект ожидает:
    - close_15m, open_15m, high_15m, low_15m, volume_15m (main timeframe)
    - close_30m, open_30m, high_30m, low_30m, volume_30m (context)
    - close_1h, open_1h, high_1h, low_1h, volume_1h (context)
    - timestamp колонка
    """
    print(f"\n" + "=" * 80)
    print("ПОДГОТОВКА MULTI-TIMEFRAME ДАННЫХ")
    print("=" * 80)
    
    df_15m = dataframes['15m'].copy()
    df_30m = dataframes['30m'].copy()
    df_1h = dataframes['1h'].copy()
    
    # Подготовка 15m (main timeframe)
    print(f"\n[PREPARE] Подготовка 15m данных (main timeframe)...")
    df_15m = df_15m.reset_index()
    
    # Определяем timestamp колонку
    if 'time' in df_15m.columns:
        df_15m['timestamp'] = pd.to_datetime(df_15m['time'])
    elif df_15m.index.name and 'time' in str(df_15m.index.name):
        df_15m['timestamp'] = df_15m.index
    else:
        # Используем индекс как timestamp
        df_15m['timestamp'] = pd.to_datetime(df_15m.iloc[:, 0]) if len(df_15m.columns) > 0 else pd.to_datetime(df_15m.index)
    
    # Переименовываем колонки для 15m
    rename_map_15m = {
        'open': 'open_15m',
        'high': 'high_15m',
        'low': 'low_15m',
        'close': 'close_15m',
        'volume': 'volume_15m'
    }
    
    for old, new in rename_map_15m.items():
        if old in df_15m.columns:
            df_15m[new] = pd.to_numeric(df_15m[old], errors='coerce')
        else:
            print(f"   [WARNING] Колонка '{old}' не найдена в 15m данных")
    
    df_15m = df_15m[['timestamp'] + [c for c in rename_map_15m.values() if c in df_15m.columns]]
    df_15m = df_15m.sort_values('timestamp').reset_index(drop=True)
    
    print(f"   [OK] 15m: {len(df_15m)} записей")
    
    # Подготовка 30m (context timeframe)
    print(f"\n[PREPARE] Подготовка 30m данных (context timeframe)...")
    df_30m = df_30m.reset_index()
    
    if 'time' in df_30m.columns:
        df_30m['timestamp'] = pd.to_datetime(df_30m['time'])
    else:
        df_30m['timestamp'] = pd.to_datetime(df_30m.iloc[:, 0]) if len(df_30m.columns) > 0 else pd.to_datetime(df_30m.index)
    
    rename_map_30m = {
        'open': 'open_30m',
        'high': 'high_30m',
        'low': 'low_30m',
        'close': 'close_30m',
        'volume': 'volume_30m'
    }
    
    for old, new in rename_map_30m.items():
        if old in df_30m.columns:
            df_30m[new] = pd.to_numeric(df_30m[old], errors='coerce')
    
    df_30m = df_30m[['timestamp'] + [c for c in rename_map_30m.values() if c in df_30m.columns]]
    df_30m = df_30m.sort_values('timestamp').reset_index(drop=True)
    
    print(f"   [OK] 30m: {len(df_30m)} записей")
    
    # Подготовка 1h (context timeframe)
    print(f"\n[PREPARE] Подготовка 1h данных (context timeframe)...")
    df_1h = df_1h.reset_index()
    
    if 'time' in df_1h.columns:
        df_1h['timestamp'] = pd.to_datetime(df_1h['time'])
    else:
        df_1h['timestamp'] = pd.to_datetime(df_1h.iloc[:, 0]) if len(df_1h.columns) > 0 else pd.to_datetime(df_1h.index)
    
    rename_map_1h = {
        'open': 'open_1h',
        'high': 'high_1h',
        'low': 'low_1h',
        'close': 'close_1h',
        'volume': 'volume_1h'
    }
    
    for old, new in rename_map_1h.items():
        if old in df_1h.columns:
            df_1h[new] = pd.to_numeric(df_1h[old], errors='coerce')
    
    df_1h = df_1h[['timestamp'] + [c for c in rename_map_1h.values() if c in df_1h.columns]]
    df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)
    
    print(f"   [OK] 1h: {len(df_1h)} записей")
    
    # Объединение данных
    print(f"\n[MERGE] Объединение multi-timeframe данных...")
    
    # Начинаем с 15m (main)
    df_merged = df_15m.copy()
    
    # Добавляем 30m через merge_asof
    df_merged = pd.merge_asof(
        df_merged.sort_values('timestamp'),
        df_30m.sort_values('timestamp'),
        on='timestamp',
        direction='backward',
        tolerance=pd.Timedelta('30min')
    )
    
    # Добавляем 1h через merge_asof
    df_merged = pd.merge_asof(
        df_merged.sort_values('timestamp'),
        df_1h.sort_values('timestamp'),
        on='timestamp',
        direction='backward',
        tolerance=pd.Timedelta('1h')
    )
    
    # Удаляем строки с пропусками в основных колонках
    required_main_cols = ['open_15m', 'high_15m', 'low_15m', 'close_15m', 'volume_15m']
    df_merged = df_merged.dropna(subset=required_main_cols)
    
    print(f"   [OK] Объединено {len(df_merged)} записей")
    print(f"   Колонки: {list(df_merged.columns)}")
    
    return df_merged

def split_data_temporal(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Разделяет данные на train/val/test по времени (temporal split)
    
    Args:
        df: DataFrame с данными
        train_ratio: Доля train данных
        val_ratio: Доля validation данных
        test_ratio: Доля test данных
    
    Returns:
        train_df, val_df, test_df
    """
    print(f"\n" + "=" * 80)
    print("РАЗДЕЛЕНИЕ ДАННЫХ НА ВЫБОРКИ")
    print("=" * 80)
    
    # Проверяем, что сумма = 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Сумма долей должна быть 1.0, получено {total}")
    
    # Сортируем по времени
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Вычисляем индексы разделения
    n_total = len(df)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"\n[SPLIT] Разделение данных:")
    print(f"   Train: {len(train_df)} записей ({len(train_df)/n_total:.1%})")
    print(f"      Период: {train_df['timestamp'].min()} - {train_df['timestamp'].max()}")
    print(f"   Validation: {len(val_df)} записей ({len(val_df)/n_total:.1%})")
    print(f"      Период: {val_df['timestamp'].min()} - {val_df['timestamp'].max()}")
    print(f"   Test: {len(test_df)} записей ({len(test_df)/n_total:.1%})")
    print(f"      Период: {test_df['timestamp'].min()} - {test_df['timestamp'].max()}")
    
    return train_df, val_df, test_df

def test_feature_engineering(df, main_tf='15m', context_tfs=['30m', '1h']):
    """Тестирует вычисление признаков"""
    print(f"\n" + "=" * 80)
    print("ВЫЧИСЛЕНИЕ ПРИЗНАКОВ")
    print("=" * 80)
    print(f"Main timeframe: {main_tf}, Context timeframes: {context_tfs}")
    
    try:
        df_features = build_features(
            df,
            main_tf=main_tf,
            context_tfs=context_tfs,
            use_robust_volume_z=False,
            dropna=True
        )
        
        print(f"\n[OK] Вычислено признаков: {len(df_features.columns)}")
        print(f"   Записей после dropna: {len(df_features)}")
        
        # Показываем пример признаков
        feature_cols = [c for c in df_features.columns 
                       if c not in ['timestamp', 'open_15m', 'high_15m', 'low_15m', 'close_15m', 'volume_15m']]
        print(f"   Количество признаков: {len(feature_cols)}")
        print(f"   Примеры признаков: {feature_cols[:10]}")
        
        return df_features
        
    except Exception as e:
        print(f"   [ERROR] Ошибка при вычислении признаков: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_hmm_labeling(df_features, main_tf='15m'):
    """Тестирует HMM разметку"""
    print(f"\n" + "=" * 80)
    print("ОБУЧЕНИЕ HMM МОДЕЛИ")
    print("=" * 80)
    
    try:
        # Выбираем признаки для HMM
        feature_list = [c for c in df_features.columns 
                       if c not in ['timestamp', 'open_15m', 'high_15m', 'low_15m', 'close_15m', 'volume_15m']]
        
        # Ограничиваем количество признаков для теста
        if len(feature_list) > 30:
            # Берем основные признаки
            priority_features = [f for f in feature_list if any(x in f for x in 
                ['atr_norm', 'adx', 'rsi', 'bb_width', 'ema_ratio', 'macd_hist', 'log_ret'])]
            feature_list = priority_features[:30] if len(priority_features) >= 30 else feature_list[:30]
        
        print(f"   Используется {len(feature_list)} признаков")
        
        # Получаем признаки для HMM
        X, scaler, pca_model = get_hmm_features(
            df_features,
            feature_list,
            n_components=4,  # Как в проекте
            scale=True,
            use_pca=True
        )
        
        print(f"   [OK] Признаки подготовлены: {X.shape}")
        if pca_model:
            explained_var = np.sum(pca_model.explained_variance_ratio_)
            print(f"   PCA объясненная дисперсия: {explained_var:.2%}")
        
        # Обучаем HMM
        n_states = 6  # Как в проекте
        print(f"\n   [TRAIN] Обучение HMM с {n_states} состояниями...")
        hmm_model = train_hmm(X, n_states=n_states, n_iter=150, random_state=42)
        
        # Получаем метки
        labels = hmm_model.predict(X)
        
        print(f"   [OK] HMM обучена")
        print(f"   Распределение состояний:")
        state_counts = pd.Series(labels).value_counts().sort_index()
        for state, count in state_counts.items():
            print(f"      State {state}: {count} ({count/len(labels):.1%})")
        
        # Маппинг состояний в режимы
        regime_mapping = map_states_to_regimes(df_features, labels, main_tf=main_tf)
        print(f"\n   Маппинг состояний в режимы:")
        for state, regime in sorted(regime_mapping.items()):
            count = (labels == state).sum()
            print(f"      State {state} -> {regime}: {count} ({count/len(labels):.1%})")
        
        # Добавляем метки в датафрейм
        df_labeled = df_features.copy()
        df_labeled['hmm_state'] = labels
        df_labeled['regime'] = df_labeled['hmm_state'].map(regime_mapping)
        
        print(f"\n   Распределение режимов:")
        regime_counts = df_labeled['regime'].value_counts()
        for regime, count in regime_counts.items():
            print(f"      {regime}: {count} ({count/len(df_labeled):.1%})")
        
        return df_labeled, hmm_model, scaler, pca_model, feature_list
        
    except Exception as e:
        print(f"   [ERROR] Ошибка при HMM разметке: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def test_lstm_training(df_train_labeled, df_val_labeled, feature_list):
    """Тестирует обучение LSTM"""
    print(f"\n" + "=" * 80)
    print("ОБУЧЕНИЕ LSTM МОДЕЛИ")
    print("=" * 80)
    
    try:
        # Подготавливаем данные для train
        print(f"\n[PREPARE] Подготовка данных для LSTM...")
        feature_cols = [c for c in feature_list if c in df_train_labeled.columns]
        
        X_train = df_train_labeled[feature_cols].values
        y_train = df_train_labeled['hmm_state'].values
        
        X_val = df_val_labeled[feature_cols].values
        y_val = df_val_labeled['hmm_state'].values
        
        # Нормализация
        scaler_lstm = StandardScaler()
        X_train_scaled = scaler_lstm.fit_transform(X_train)
        X_val_scaled = scaler_lstm.transform(X_val)
        
        # Создаем последовательности
        time_steps = 64
        print(f"   Создание последовательностей (time_steps={time_steps})...")
        
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, time_steps=time_steps)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, time_steps=time_steps)
        
        print(f"   [OK] Train последовательностей: {len(X_train_seq)}")
        print(f"   [OK] Val последовательностей: {len(X_val_seq)}")
        print(f"   Форма X_seq: {X_train_seq.shape}")
        
        # Преобразуем в категории
        from tensorflow.keras.utils import to_categorical
        num_classes = len(np.unique(y_train))
        y_train_cat = to_categorical(y_train_seq, num_classes=num_classes)
        y_val_cat = to_categorical(y_val_seq, num_classes=num_classes)
        
        print(f"   Количество классов: {num_classes}")
        
        # Строим модель
        print(f"\n[MODEL] Создание LSTM модели...")
        input_shape = (time_steps, X_train_seq.shape[2])
        model = build_lstm_model(
            input_shape=input_shape,
            num_classes=num_classes,
            lstm_units=64,
            dense_units=32,
            dropout_rate=0.3
        )
        
        print(f"   [OK] LSTM модель создана")
        print(f"   Параметров: {model.count_params():,}")
        
        # Обучаем
        print(f"\n[TRAIN] Обучение LSTM...")
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        history = model.fit(
            X_train_seq, y_train_cat,
            validation_data=(X_val_seq, y_val_cat),
            epochs=20,
            batch_size=32,
            verbose=1,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
            ]
        )
        
        # Оценка
        val_loss, val_accuracy = model.evaluate(X_val_seq, y_val_cat, verbose=0)
        print(f"\n   [RESULTS] Точность на validation: {val_accuracy:.2%}")
        print(f"   [RESULTS] Loss на validation: {val_loss:.4f}")
        
        return model, scaler_lstm, val_accuracy, history
        
    except Exception as e:
        print(f"   [ERROR] Ошибка при обучении LSTM: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0.0, None

def evaluate_on_test(df_test_labeled, model, scaler_lstm, feature_list, time_steps=64):
    """Оценивает модель на test set"""
    print(f"\n" + "=" * 80)
    print("ОЦЕНКА НА TEST SET")
    print("=" * 80)
    
    try:
        feature_cols = [c for c in feature_list if c in df_test_labeled.columns]
        X_test = df_test_labeled[feature_cols].values
        y_test = df_test_labeled['hmm_state'].values
        
        # Нормализация
        X_test_scaled = scaler_lstm.transform(X_test)
        
        # Создаем последовательности
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, time_steps=time_steps)
        
        # Преобразуем в категории
        from tensorflow.keras.utils import to_categorical
        num_classes = len(np.unique(y_test))
        y_test_cat = to_categorical(y_test_seq, num_classes=num_classes)
        
        # Предсказания
        y_pred = model.predict(X_test_seq, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Метрики
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(y_test_seq, y_pred_classes)
        
        print(f"\n[RESULTS] Результаты на test set:")
        print(f"   Точность: {accuracy:.2%}")
        print(f"   Количество предсказаний: {len(y_pred_classes)}")
        
        print(f"\n   Classification Report:")
        print(classification_report(y_test_seq, y_pred_classes))
        
        print(f"\n   Confusion Matrix:")
        cm = confusion_matrix(y_test_seq, y_pred_classes)
        print(cm)
        
        return accuracy
        
    except Exception as e:
        print(f"   [ERROR] Ошибка при оценке: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def main():
    """Основная функция тестирования"""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ CRYPTOMARKET REGIME CLASSIFIER")
    print("На наших данных: 15m, 30m, 1h")
    print("=" * 80)
    
    try:
        # 1. Загрузка multi-timeframe данных
        dataframes = load_multi_timeframe_data()
        
        # 2. Подготовка multi-timeframe
        df_multi = prepare_multi_timeframe_data(dataframes)
        
        # 3. Разделение на train/val/test
        df_train, df_val, df_test = split_data_temporal(df_multi, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        # 4. Вычисление признаков для каждой выборки
        print(f"\n" + "=" * 80)
        print("ОБРАБОТКА TRAIN SET")
        print("=" * 80)
        df_train_features = test_feature_engineering(df_train, main_tf='15m', context_tfs=['30m', '1h'])
        if df_train_features is None:
            return
        
        print(f"\n" + "=" * 80)
        print("ОБРАБОТКА VALIDATION SET")
        print("=" * 80)
        df_val_features = test_feature_engineering(df_val, main_tf='15m', context_tfs=['30m', '1h'])
        if df_val_features is None:
            return
        
        print(f"\n" + "=" * 80)
        print("ОБРАБОТКА TEST SET")
        print("=" * 80)
        df_test_features = test_feature_engineering(df_test, main_tf='15m', context_tfs=['30m', '1h'])
        if df_test_features is None:
            return
        
        # 5. HMM разметка (только на train)
        df_train_labeled, hmm_model, scaler_hmm, pca_model, feature_list = test_hmm_labeling(df_train_features, main_tf='15m')
        if df_train_labeled is None:
            return
        
        # Применяем HMM к val и test
        print(f"\n[APPLY] Применение HMM к validation и test sets...")
        X_val, _, _ = get_hmm_features(df_val_features, feature_list, n_components=4, scale=True, use_pca=True)
        X_test, _, _ = get_hmm_features(df_test_features, feature_list, n_components=4, scale=True, use_pca=True)
        
        labels_val = hmm_model.predict(X_val)
        labels_test = hmm_model.predict(X_test)
        
        regime_mapping = map_states_to_regimes(df_train_features, df_train_labeled['hmm_state'].values, main_tf='15m')
        
        df_val_labeled = df_val_features.copy()
        df_val_labeled['hmm_state'] = labels_val
        df_val_labeled['regime'] = df_val_labeled['hmm_state'].map(regime_mapping)
        
        df_test_labeled = df_test_features.copy()
        df_test_labeled['hmm_state'] = labels_test
        df_test_labeled['regime'] = df_test_labeled['hmm_state'].map(regime_mapping)
        
        print(f"   [OK] HMM применена ко всем выборкам")
        
        # 6. LSTM обучение
        lstm_model, scaler_lstm, val_accuracy, history = test_lstm_training(df_train_labeled, df_val_labeled, feature_list)
        if lstm_model is None:
            return
        
        # 7. Оценка на test set
        test_accuracy = evaluate_on_test(df_test_labeled, lstm_model, scaler_lstm, feature_list)
        
        # Итоги
        print(f"\n" + "=" * 80)
        print("ИТОГИ ТЕСТИРОВАНИЯ")
        print("=" * 80)
        print(f"[OK] HMM модель обучена: {hmm_model is not None}")
        print(f"[OK] LSTM модель обучена: {lstm_model is not None}")
        print(f"[OK] Точность на validation: {val_accuracy:.2%}")
        print(f"[OK] Точность на test: {test_accuracy:.2%}")
        print(f"\nСравнение с нашим методом:")
        print(f"   Наш метод (индикаторы): 4.22% точность")
        print(f"   HMM + LSTM (validation): {val_accuracy:.2%} точность")
        print(f"   HMM + LSTM (test): {test_accuracy:.2%} точность")
        
        if test_accuracy > 0.0422:
            improvement = ((test_accuracy - 0.0422) / 0.0422) * 100
            print(f"\n[SUCCESS] HMM + LSTM подход показывает ЛУЧШУЮ точность!")
            print(f"   Улучшение: +{improvement:.1f}%")
        else:
            print(f"\n[WARNING] HMM + LSTM подход показывает похожую или худшую точность")
        
        # Сохранение результатов
        results = {
            'val_accuracy': float(val_accuracy),
            'test_accuracy': float(test_accuracy),
            'our_method_accuracy': 0.0422,
            'improvement_pct': float(((test_accuracy - 0.0422) / 0.0422) * 100) if test_accuracy > 0.0422 else 0.0
        }
        
        import json
        results_file = Path('results_hmm_lstm.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] Результаты сохранены: {results_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
