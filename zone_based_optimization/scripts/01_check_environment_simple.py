"""
Скрипт 01: Проверка окружения (версия без эмодзи)
Проверяет наличие всех необходимых библиотек и зависимостей
"""

import sys
import io

# Исправление кодировки для Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

def check_package(package_name, import_name=None):
    """Проверяет наличие пакета"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"[OK] {package_name} установлен")
        return True
    except ImportError:
        print(f"[FAIL] {package_name} НЕ установлен")
        print(f"   Установите: pip install {package_name}")
        return False

def check_trading_classifier():
    """Проверяет наличие Trading Classifier"""
    try:
        # Путь относительно корня проекта
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        classifier_path = os.path.join(project_root, 'indicators', 'trading_classifier_iziceros', 'src')
        
        if os.path.exists(classifier_path):
            sys.path.insert(0, classifier_path)
            from trend_classifier import Segmenter, Config
            print("[OK] Trading Classifier установлен")
            return True
        else:
            print(f"[FAIL] Trading Classifier НЕ найден")
            print(f"   Путь не существует: {classifier_path}")
            return False
    except ImportError as e:
        print(f"[FAIL] Trading Classifier НЕ найден: {e}")
        print("   Проверьте путь: ../indicators/trading_classifier_iziceros/")
        return False
    except Exception as e:
        print(f"[FAIL] Ошибка при проверке Trading Classifier: {e}")
        return False

def main():
    try:
        print("=" * 80)
        print("ПРОВЕРКА ОКРУЖЕНИЯ")
        print("=" * 80)
        print(f"Python версия: {sys.version}")
        print(f"Python путь: {sys.executable}")
        print()
        
        # Основные библиотеки
        packages = [
            ('pandas', 'pandas'),
            ('numpy', 'numpy'),
            ('matplotlib', 'matplotlib'),
            ('seaborn', 'seaborn'),
            ('scikit-learn', 'sklearn'),
        ]
        
        all_ok = True
        for package, import_name in packages:
            if not check_package(package, import_name):
                all_ok = False
        
        print()
        # Trading Classifier
        if not check_trading_classifier():
            all_ok = False
        
        print()
        print("=" * 80)
        if all_ok:
            print("[SUCCESS] ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
            print("   Окружение готово к работе")
        else:
            print("[WARNING] ЕСТЬ ПРОБЛЕМЫ!")
            print("   Установите недостающие пакеты перед продолжением")
        print("=" * 80)
        
        return all_ok
    except Exception as e:
        print(f"[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[ERROR] Ошибка выполнения: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

