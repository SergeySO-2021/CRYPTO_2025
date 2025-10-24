"""
API для интеграции TrendClassifier с TradingView
"""

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from trend_classifier import TrendClassifier
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)

# Инициализация классификатора
classifier = TrendClassifier()

@app.route('/predict', methods=['POST'])
def predict_market_zones():
    """
    API endpoint для получения предсказаний рыночных зон
    
    Expected JSON:
    {
        "symbol": "BTC-USD",
        "timeframe": "1h",
        "period": "30d"
    }
    """
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC-USD')
        timeframe = data.get('timeframe', '1h')
        period = data.get('period', '30d')
        
        # Загружаем данные
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=timeframe)
        
        if df.empty:
            return jsonify({'error': 'No data available'}), 400
        
        # Подготавливаем данные
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        
        # Обучаем классификатор
        classifier.fit(df)
        
        # Получаем предсказания
        predictions = classifier.predict(df)
        
        # Формируем результат
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'period': period,
            'predictions': predictions.tolist(),
            'current_classification': int(predictions[-1]),
            'classification_name': get_classification_name(int(predictions[-1])),
            'confidence': calculate_confidence(predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка состояния API"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

def get_classification_name(classification):
    """Получение названия классификации"""
    names = {-1: 'Bearish', 0: 'Sideways', 1: 'Bullish'}
    return names.get(classification, 'Unknown')

def calculate_confidence(predictions):
    """Расчет уверенности в предсказании"""
    recent_predictions = predictions[-10:]  # Последние 10 предсказаний
    most_common = np.bincount(recent_predictions + 1).argmax() - 1
    confidence = np.bincount(recent_predictions + 1).max() / len(recent_predictions)
    return float(confidence)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
