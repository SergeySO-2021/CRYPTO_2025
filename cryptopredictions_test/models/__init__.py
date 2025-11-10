# Базовые модели (всегда доступны)
from .random_forest import RandomForest
from .sarimax import Sarimax
from .xgboost import MyXGboost

MODELS = {
    'random_forest': RandomForest,
    'sarimax': Sarimax,
    'xgboost': MyXGboost
}

# Опциональные модели (загружаем только если доступны)
try:
    from .orbit import Orbit
    MODELS['orbit'] = Orbit
except ImportError:
    pass

try:
    from .LSTM import MyLSTM
    MODELS['lstm'] = MyLSTM
except ImportError:
    pass

try:
    from .GRU import MyGRU
    MODELS['gru'] = MyGRU
except ImportError:
    pass

try:
    from .arima import MyARIMA
    MODELS['arima'] = MyARIMA
except ImportError:
    pass

try:
    from .prophet import MyProphet
    MODELS['prophet'] = MyProphet
except ImportError:
    pass

try:
    from .neural_prophet import Neural_Prophet
    MODELS['neural_prophet'] = Neural_Prophet
except ImportError:
    pass
