import torch
from torch.utils.data import DataLoader
import requests
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
from ..models.lstm_model import LSTMModel
from ..config import API_ENDPOINTS, SEQUENCE_LENGTH
from ..data.dataset import TradeAction, TradingDataset

def backtest_trader(trader_class, coin_data_dict: Dict[str, pd.DataFrame], 
                   training_end_idx: int) -> Tuple[List[TradeAction], pd.DataFrame]:
    """Backtest a trading strategy."""
    trader = trader_class(initial_balance=100000)
    timestamps = list(coin_data_dict.values())[0]['timestamp'].tolist()[:training_end_idx]
    
    performance_data = []
    for idx in tqdm(range(len(timestamps)), desc=f"Backtesting {trader_class.__name__}"):
        timestamp = timestamps[idx]
        daily_portfolio = trader.balance
        
        for coin, df in coin_data_dict.items():
            if idx >= len(df):
                continue
            current_data = df.iloc[idx]

            action, position_size = trader.should_trade(
                coin=coin, 
                df=df,
                idx=idx
            )
            if action != 0:
                features = trader.get_technical_features(df, idx)
                trader.execute_trade(
                    timestamp=timestamp,
                    coin=coin,
                    price=current_data['price'],
                    action=action,
                    position_size=position_size,
                    features=features
                )
            
            if coin in trader.positions:
                daily_portfolio += trader.positions[coin] * current_data['price']

        performance_data.append({
            'timestamp': timestamp,
            'portfolio_value': daily_portfolio
        })
    
    performance_df = pd.DataFrame(performance_data)
    return trader.trades, performance_df

def train_local_model(trader_name: str, dataset: TradingDataset, num_epochs: int) -> Dict:
    """Train a local model for a specific trading strategy."""
    input_size = len(dataset.features[0][0])
    model = LSTMModel(input_size=input_size, hidden_size=64, num_classes=3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Trader {trader_name}, Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model.state_dict()

def prepare_test_dataset(coin_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, torch.Tensor]:
    """Prepare test dataset for model inference."""
    test_sequences = {}
    feature_cols = ['rsi', 'macd', 'macd_signal', 'bollinger_high', 'bollinger_low', 
                   'bollinger_mid', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'volume_sma_20']
    
    for coin, df in coin_data_dict.items():
        features = df[feature_cols].values
        sequences = [features[i:i + SEQUENCE_LENGTH] 
                    for i in range(len(features) - SEQUENCE_LENGTH)]
        test_sequences[coin] = torch.tensor(sequences, dtype=torch.float32)
    
    return test_sequences

def plot_performance(agent_portfolio: List[float], all_performance: Dict[str, pd.DataFrame]):
    """Plot performance comparison of different strategies."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(agent_portfolio, label='AI Agent')
    for trader_name, performance_df in all_performance.items():
        plt.plot(performance_df['portfolio_value'].values, label=trader_name)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()