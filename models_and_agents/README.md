# Federated Learning Trading System

This project implements a federated learning system for cryptocurrency trading, combining multiple trading strategies with an AI agent for decision making.

## Structure

The project is organized into several key components:
- `data/`: Data loading and dataset management
- `models/`: LSTM model implementation
- `traders/`: Trading strategy implementations
- `utils/`: Technical indicators and training utilities
- `agents/`: AI trading agent implementation

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the system:
```bash
python main.py
```

## Components

- Multiple trading strategies (Momentum, Mean Reversion, etc.)
- Federated learning implementation
- Technical analysis indicators
- AI trading agent using OpenAI's GPT-4