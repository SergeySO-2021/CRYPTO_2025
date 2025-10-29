"""
Поэтапная оптимизация параметров MZA
Разбивает процесс на этапы для экономии памяти с сохранением качества
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from mza_parameter_optimizer import MZAParameterOptimizer
import gc

class StagedOptimization:
    """
    Поэтапная оптимизация с сохранением прогресса
    """
    
    def __init__(self, mza_system, results_dir='optimization_results'):
        self.mza_system = mza_system
        self.optimizer = MZAParameterOptimizer()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Конфигурация этапов
        self.stages = {
            'stage1_quick_test': {
                'name': '🔄 Быстрый тест (низкое качество, высокая скорость)',
                'timeframes': ['1h'],
                'data_samples': 2000,
                'combinations': 50,
                'all_params': False,
                'description': 'Быстрая проверка работы системы'
            },
            'stage2_balanced_1h': {
                'name': '⚖️ Сбалансированная оптимизация для 1h',
                'timeframes': ['1h'],
                'data_samples': 3000,
                'combinations': 150,
                'all_params': True,
                'description': 'Оптимальный баланс для 1h'
            },
            'stage3_balanced_4h': {
                'name': '⚖️ Сбалансированная оптимизация для 4h',
                'timeframes': ['4h'],
                'data_samples': 3000,
                'combinations': 150,
                'all_params': True,
                'description': 'Оптимальный баланс для 4h'
            },
            'stage4_deep_1h': {
                'name': '🔬 Глубокая оптимизация для 1h',
                'timeframes': ['1h'],
                'data_samples': 4000,
                'combinations': 200,
                'all_params': True,
                'description': 'Максимальное качество для 1h'
            },
            'stage5_deep_4h': {
                'name': '🔬 Глубокая оптимизация для 4h',
                'timeframes': ['4h'],
                'data_samples': 4000,
                'combinations': 200,
                'all_params': True,
                'description': 'Максимальное качество для 4h'
            },
            'stage6_additional_timeframes': {
                'name': '📊 Дополнительные таймфреймы (15m, 30m, 1d)',
                'timeframes': ['15m', '30m', '1d'],
                'data_samples': 3000,
                'combinations': 100,
                'all_params': True,
                'description': 'Быстрая оптимизация для остальных таймфреймов'
            }
        }
        
    def run_stage(self, stage_id, clear_memory=True):
        """
        Запуск одного этапа оптимизации
        
        Args:
            stage_id: ID этапа для запуска
            clear_memory: Очищать ли память после этапа
        """
        if stage_id not in self.stages:
            print(f"❌ Этап {stage_id} не найден")
            return None
            
        stage = self.stages[stage_id]
        print(f"\n{'='*70}")
        print(f"🎯 ЗАПУСК ЭТАПА: {stage['name']}")
        print(f"📝 {stage['description']}")
        print(f"{'='*70}\n")
        
        results = {}
        
        for tf in stage['timeframes']:
            if tf not in self.mza_system.data:
                print(f"❌ Данные для {tf} не найдены, пропускаем")
                continue
                
            print(f"\n📊 Таймфрейм: {tf}")
            print(f"   📈 Всего записей: {len(self.mza_system.data[tf]):,}")
            print(f"   📊 Будет использовано: {stage['data_samples']:,} записей")
            print(f"   🔧 Комбинаций: {stage['combinations']}")
            print(f"   ⚙️ Параметров: {'16 (все)' if stage['all_params'] else '12 (ключевые)'}")
            
            # Подготовка данных
            data_sample = self.mza_system.data[tf].tail(stage['data_samples']).copy()
            
            # Добавляем фиктивную колонку volume если ее нет (требуется для MZA)
            if 'volume' not in data_sample.columns:
                data_sample['volume'] = (data_sample['high'] + data_sample['low']) / 2  # Простая аппроксимация
            
            try:
                # Настройка параметров
                original_params = self.optimizer.optimizable_params.copy()
                if not stage['all_params']:
                    # Оставляем только ключевые параметры
                    self.optimizer.optimizable_params = dict(list(original_params.items())[:12])
                
                # Запуск оптимизации
                result = self.optimizer.optimize_parameters(
                    data_sample, 
                    max_combinations=stage['combinations']
                )
                
                results[tf] = result
                
                print(f"   ✅ Оптимизация завершена")
                print(f"   🏆 Economic Value: {result['best_score']:.6f}")
                print(f"   📊 Тестов: {result['successful_tests']}/{result['total_tested']}")
                
                # Восстанавливаем параметры
                self.optimizer.optimizable_params = original_params
                
                # Очистка памяти
                del data_sample
                gc.collect()
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()
                results[tf] = {'error': str(e)}
                
                del data_sample
                gc.collect()
        
        # Сохранение результатов этапа
        self._save_stage_results(stage_id, results, stage)
        
        print(f"\n✅ Этап {stage_id} завершен")
        
        return results
    
    def _save_stage_results(self, stage_id, results, stage_config):
        """Сохранение результатов этапа"""
        save_data = {
            'stage_id': stage_id,
            'stage_name': stage_config['name'],
            'timestamp': datetime.now().isoformat(),
            'config': stage_config,
            'results': results
        }
        
        # Сохраняем JSON
        json_file = self.results_dir / f"{stage_id}_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"💾 Результаты сохранены: {json_file}")
    
    def load_all_results(self):
        """Загрузка всех сохраненных результатов"""
        all_results = {}
        
        for stage_id in self.stages.keys():
            json_file = self.results_dir / f"{stage_id}_results.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    all_results[stage_id] = json.load(f)
        
        return all_results
    
    def get_summary_report(self):
        """Создание сводного отчета по всем этапам"""
        print("\n" + "="*70)
        print("📊 СВОДНЫЙ ОТЧЕТ ОБ ОПТИМИЗАЦИИ")
        print("="*70)
        
        all_results = self.load_all_results()
        
        if not all_results:
            print("\n❌ Нет сохраненных результатов")
            return
        
        # Собираем данные по таймфреймам
        timeframe_results = {}
        
        for stage_id, stage_data in all_results.items():
            stage_name = stage_data['stage_name']
            results = stage_data['results']
            
            for tf, result in results.items():
                if 'error' not in result:
                    if tf not in timeframe_results:
                        timeframe_results[tf] = []
                    
                    timeframe_results[tf].append({
                        'stage': stage_name,
                        'economic_value': result['best_score'],
                        'total_tested': result['total_tested'],
                        'successful_tests': result['successful_tests']
                    })
        
        # Выводим сводку по таймфреймам
        print(f"\n{'Таймфрейм':<8} {'Этап':<40} {'Economic Value':<15} {'Тестов':<15}")
        print("-" * 70)
        
        for tf, results_list in timeframe_results.items():
            print(f"\n📊 {tf}")
            for result in sorted(results_list, key=lambda x: x['economic_value'], reverse=True):
                print(f"   {result['stage'][:40]:<40} {result['economic_value']:.6f}     {result['successful_tests']}/{result['total_tested']}")
        
        # Находим лучшие результаты
        print(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ ПО ТАЙМФРЕЙМАМ:")
        print("-" * 70)
        
        for tf, results_list in timeframe_results.items():
            best = max(results_list, key=lambda x: x['economic_value'])
            print(f"{tf:>8}: {best['economic_value']:.6f} (этап: {best['stage'][:40]})")
        
    def run_all_stages(self, start_from_stage=None):
        """
        Запуск всех этапов последовательно
        
        Args:
            start_from_stage: С какого этапа начать (None = с начала)
        """
        stage_ids = list(self.stages.keys())
        
        if start_from_stage:
            try:
                start_idx = stage_ids.index(start_from_stage)
                stage_ids = stage_ids[start_idx:]
                print(f"🚀 Продолжаем с этапа: {start_from_stage}")
            except ValueError:
                print(f"❌ Этап {start_from_stage} не найден")
                return
        
        print(f"\n🎯 БУДЕТ ВЫПОЛНЕНО {len(stage_ids)} ЭТАПОВ")
        
        for i, stage_id in enumerate(stage_ids, 1):
            print(f"\n{'='*70}")
            print(f"📌 ЭТАП {i}/{len(stage_ids)}: {self.stages[stage_id]['name']}")
            print(f"{'='*70}")
            
            try:
                results = self.run_stage(stage_id)
                
                if results:
                    print(f"✅ Этап {i} завершен успешно")
                else:
                    print(f"⚠️ Этап {i} завершен с ошибками")
                    
            except KeyboardInterrupt:
                print(f"\n⚠️ Прервано пользователем")
                print(f"💾 Прогресс сохранен. Можно продолжить с этапа {stage_id}")
                break
                
            except Exception as e:
                print(f"\n❌ Ошибка на этапе {i}: {e}")
                import traceback
                traceback.print_exc()
        
        # Финальный отчет
        self.get_summary_report()
        
        print(f"\n✅ ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ")
        print(f"📁 Результаты сохранены в: {self.results_dir}")

