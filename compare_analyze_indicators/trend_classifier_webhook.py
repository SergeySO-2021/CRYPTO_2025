"""
Webhook для интеграции TrendClassifier с TradingView
"""

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from trend_classifier import TrendClassifier
import json
from datetime import datetime

app = Flask(__name__)

# Инициализация классификатора
classifier = TrendClassifier()

@app.route('/webhook', methods=['POST'])
def tradingview_webhook():
    """
    Webhook для получения данных от TradingView
    
    Expected JSON from TradingView:
    {
        "time": "2024-01-01T00:00:00Z",
        "close": 50000,
        "high": 51000,
        "low": 49000,
        "open": 49500,
        "volume": 1000
    }
    """
    try:
        data = request.get_json()
        
        # Проверяем формат данных
        if not all(key in data for key in ['time', 'close', 'high', 'low', 'open']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Создаем DataFrame из данных
        df = pd.DataFrame([{
            'timestamp': data['time'],
            'close': data['close'],
            'high': data['high'],
            'low': data['low'],
            'open': data['open'],
            'volume': data.get('volume', 1000)
        }])
        
        # Обучаем классификатор (в реальном сценарии нужно накопить больше данных)
        classifier.fit(df)
        
        # Получаем предсказание
        prediction = classifier.predict(df)
        current_classification = int(prediction[0])
        
        # Формируем ответ для TradingView
        response = {
            'classification': current_classification,
            'classification_name': get_classification_name(current_classification),
            'confidence': 0.8,  # В реальном сценарии нужно рассчитать
            'timestamp': datetime.now().isoformat(),
            'recommendation': get_trading_recommendation(current_classification)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_classification_name(classification):
    """Получение названия классификации"""
    names = {-1: 'Bearish', 0: 'Sideways', 1: 'Bullish'}
    return names.get(classification, 'Unknown')

def get_trading_recommendation(classification):
    """Получение торговых рекомендаций"""
    recommendations = {
        -1: 'Consider short positions or exit long positions',
        0: 'Wait for clearer trend direction',
        1: 'Consider long positions or exit short positions'
    }
    return recommendations.get(classification, 'No recommendation')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
