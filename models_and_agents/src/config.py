# Configuration parameters
SEQUENCE_LENGTH = 10
NUM_ROUNDS = 5
LOCAL_EPOCHS = 2
INITIAL_BALANCE = 100000
HIDDEN_SIZE = 64
NUM_CLASSES = 3
NUM_LAYERS = 2

# Model parameters
LSTM_PARAMS = {
    'hidden_size': HIDDEN_SIZE,
    'num_classes': NUM_CLASSES,
    'num_layers': NUM_LAYERS
}

# API endpoints
API_ENDPOINTS = {
    'upload_local_weights': 'http://localhost:3000/upload-local-training-weights',
    'retrieve_local_weights': 'http://localhost:3000/retrieve-local-training-weights',
    'upload_global_weights': 'http://localhost:3000/upload-global-weights',
    'retrieve_global_weights': 'http://localhost:3000/retrieve-global-model-weights'
}

# Technical indicators
FEATURE_COLUMNS = [
    'rsi', 'macd', 'macd_signal', 'bollinger_high', 'bollinger_low',
    'bollinger_mid', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'volume_sma_20'
]