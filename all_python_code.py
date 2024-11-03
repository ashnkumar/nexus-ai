import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import copy
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from datetime import datetime
import ta
from dataclasses import dataclass
import requests
import openai

@dataclass
class TradeAction:
    timestamp: int
    coin: str
    price: float
    action: int  # -1: sell, 0: hold, 1: buy
    position_size: float
    technical_features: Dict[str, float]

def load_coin_data(filename: str) -> Dict[str, List[Tuple]]:
    with open(filename) as f:
        top_coin_data = json.load(f)
    return transform_coin_data(top_coin_data)

def transform_coin_data(top_coin_data):
    transformed_data = {}
    for coin, data in top_coin_data.items():
        prices = data['prices']
        volumes = data['volumes']
        combined_data = [
            (price[0], price[1], volume[1])
            for price, volume in zip(prices, volumes)
        ]
        transformed_data[coin] = combined_data
    return transformed_data

class TechnicalIndicators:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ['timestamp', 'price', 'volume']
        assert all(col in df.columns for col in required_cols)
        
        if len(df) < 50:
            raise ValueError("Need at least 50 data points for technical indicators")
        
        try:
            df['rsi'] = ta.momentum.RSIIndicator(df['price'], window=14).rsi()
            
            macd = ta.trend.MACD(df['price'],
                                window_slow=26, 
                                window_fast=12,
                                window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            bollinger = ta.volatility.BollingerBands(df['price'], window=20)
            df['bollinger_high'] = bollinger.bollinger_hband()
            df['bollinger_low'] = bollinger.bollinger_lband()  
            df['bollinger_mid'] = bollinger.bollinger_mavg()
            
            df['sma_20'] = ta.trend.sma_indicator(df['price'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['price'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['price'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['price'], window=26)
            df['volume_sma_20'] = ta.trend.sma_indicator(df['volume'], window=20)
            
            df = df.fillna(method='ffill').fillna(method='bfill')
            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            raise

def prepare_coin_data(coin_data: List[Tuple]) -> pd.DataFrame:
    if len(coin_data) < 50:
        return None
    df = pd.DataFrame(coin_data, columns=['timestamp', 'price', 'volume'])
    return TechnicalIndicators.calculate_indicators(df)

class BaseTrader:
    def __init__(self, initial_balance: float = 100000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: Dict[str, float] = {}
        self.trades: List[TradeAction] = []

    def get_technical_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        feature_cols = ['rsi', 'macd', 'macd_signal', 'bollinger_high', 'bollinger_low', 
                       'bollinger_mid', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'volume_sma_20']
        return {col: df[col].iloc[idx] for col in feature_cols}

    def execute_trade(self, timestamp: int, coin: str, price: float, 
                     action: int, position_size: float, features: Dict[str, float]):
        if action == 1:  # Buy
            cost = position_size * price
            if cost <= self.balance:
                self.balance -= cost
                self.positions[coin] = self.positions.get(coin, 0) + position_size
        elif action == -1:  # Sell
            if coin in self.positions and self.positions[coin] >= position_size:
                self.balance += position_size * price
                self.positions[coin] -= position_size
                if self.positions[coin] == 0:
                    del self.positions[coin]

        self.trades.append(TradeAction(
            timestamp=timestamp,
            coin=coin,
            price=price,
            action=action,
            position_size=position_size,
            technical_features=features
        ))

class MomentumTrader(BaseTrader):
    def __init__(self, momentum_window: int = 5, 
                 buy_threshold: float = 0.02,
                 sell_threshold: float = -0.01,
                 initial_balance: float = 10000):
        super().__init__(initial_balance=initial_balance)
        self.momentum_window = momentum_window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
    def should_trade(self, coin: str, df: pd.DataFrame, idx: int) -> Tuple[int, float]:
        if idx < self.momentum_window:
            return 0, 0
        
        price = df['price'].iloc[idx]
        past_price = df['price'].iloc[idx - self.momentum_window]
        momentum = (price - past_price) / past_price
        
        if momentum > self.buy_threshold:
            if self.balance <= 0:
                return 0, 0
            position_size = (self.balance * 0.1) / price
            return 1, position_size
        elif momentum < self.sell_threshold:
            position = self.positions.get(coin, 0)
            position_size = position * 0.5 if position > 0 else 0
            return -1, position_size
        
        return 0, 0

class MeanReversionTrader(BaseTrader):
    def __init__(self, initial_balance: float = 10000):
        super().__init__(initial_balance=initial_balance)

    def should_trade(self, coin: str, df: pd.DataFrame, idx: int) -> Tuple[int, float]:  
        if idx < 50:
            return 0, 0

        price = df['price'].iloc[idx]
        sma_20 = df['sma_20'].iloc[idx]
        sma_50 = df['sma_50'].iloc[idx]
        
        if price < sma_20 and sma_20 < sma_50:
            if self.balance <= 0:
                return 0, 0
            position_size = (self.balance * 0.1) / price
            return 1, position_size
        elif price > sma_20 and sma_20 > sma_50:
            position = self.positions.get(coin, 0)
            position_size = position * 0.5 if position > 0 else 0
            return -1, position_size
        
        return 0, 0

class BreakoutTrader(BaseTrader):
    def __init__(self, initial_balance: float = 10000):
        super().__init__(initial_balance=initial_balance)

    def should_trade(self, coin: str, df: pd.DataFrame, idx: int) -> Tuple[int, float]:
        if idx < 20:
            return 0, 0
        
        price = df['price'].iloc[idx]
        upper_band = df['bollinger_high'].iloc[idx]
        lower_band = df['bollinger_low'].iloc[idx]
        
        if price > upper_band:
            if self.balance <= 0:
                return 0, 0
            position_size = (self.balance * 0.1) / price
            return 1, position_size
        elif price < lower_band:
            position = self.positions.get(coin, 0)
            position_size = position * 0.5 if position > 0 else 0
            return -1, position_size
        
        return 0, 0

class TrendFollowingTrader(BaseTrader):  
    def __init__(self, initial_balance: float = 10000):
        super().__init__(initial_balance=initial_balance)

    def should_trade(self, coin: str, df: pd.DataFrame, idx: int) -> Tuple[int, float]:
        if idx < 26:
            return 0, 0
        
        ema_12 = df['ema_12'].iloc[idx]
        ema_26 = df['ema_26'].iloc[idx]
        prev_ema_12 = df['ema_12'].iloc[idx-1] 
        prev_ema_26 = df['ema_26'].iloc[idx-1]
        
        if prev_ema_12 <= prev_ema_26 and ema_12 > ema_26:
            if self.balance <= 0:
                return 0, 0
            position_size = (self.balance * 0.1) / df['price'].iloc[idx] 
            return 1, position_size
        elif prev_ema_12 >= prev_ema_26 and ema_12 < ema_26:
            position = self.positions.get(coin, 0)
            position_size = position * 0.5 if position > 0 else 0
            return -1, position_size
        
        return 0, 0
    
class RSITrader(BaseTrader):
    def __init__(self, initial_balance: float = 10000):
        super().__init__(initial_balance=initial_balance)

    def should_trade(self, coin: str, df: pd.DataFrame, idx: int) -> Tuple[int, float]:
        if idx < 14:
            return 0, 0
        
        rsi = df['rsi'].iloc[idx]
        price = df['price'].iloc[idx]
        
        if rsi < 30:
            if self.balance <= 0:
                return 0, 0
            position_size = (self.balance * 0.1) / price
            return 1, position_size
        elif rsi > 70:
            position = self.positions.get(coin, 0)
            position_size = position * 0.5 if position > 0 else 0
            return -1, position_size

        return 0, 0
    
class VolumeBasedTrader(BaseTrader):
    def __init__(self, initial_balance: float = 10000):
        super().__init__(initial_balance=initial_balance)

    def should_trade(self, coin: str, df: pd.DataFrame, idx: int) -> Tuple[int, float]:
        if idx < 20:
            return 0, 0
        
        volume = df['volume'].iloc[idx]  
        volume_sma = df['volume_sma_20'].iloc[idx]
        price_change = df['price'].iloc[idx] / df['price'].iloc[idx-1] - 1

        if volume > volume_sma * 1.5 and price_change > 0:
            if self.balance <= 0:
                return 0, 0
            position_size = (self.balance * 0.1) / df['price'].iloc[idx]
            return 1, position_size
        elif volume > volume_sma * 1.5 and price_change < 0:
            position = self.positions.get(coin, 0)
            position_size = position * 0.5 if position > 0 else 0
            return -1, position_size

        return 0, 0

def prepare_coin_data_dict(coin_data_dict: Dict[str, List[Tuple]]) -> Dict[str, pd.DataFrame]:
    processed_data = {}
    for coin, data in coin_data_dict.items():
        df = prepare_coin_data(data)
        if df is not None:
            processed_data[coin] = df
    return processed_data

def backtest_trader(trader_class, coin_data_dict: Dict[str, pd.DataFrame], 
                    training_end_idx: int) -> Tuple[List[TradeAction], pd.DataFrame]:
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

def generate_training_data(coin_data_dict: Dict[str, List[Tuple]]) -> Tuple[Dict, Dict]:
    first_coin_data = next(iter(coin_data_dict.values()))
    training_end_idx = int(len(first_coin_data) * 0.8)
    
    if training_end_idx < 50:
        raise ValueError("Training period must be at least 50 days")

    traders = {
        'momentum': MomentumTrader,
        'mean_reversion': MeanReversionTrader,
        'breakout': BreakoutTrader,
        'trend_following': TrendFollowingTrader,
        'rsi': RSITrader,
        'volume': VolumeBasedTrader
    }
    all_trades = {}
    all_performance = {}
    processed_coin_data = prepare_coin_data_dict(coin_data_dict)
    
    for name, trader_class in traders.items():
        print(f"Backtesting for {name} trader...")
        try:
            trades, performance = backtest_trader(
                trader_class, 
                processed_coin_data,
                training_end_idx
            )
            all_trades[name] = trades
            all_performance[name] = performance
            print(f"Generated {len(trades)} trades for {name}")
            
            # Upload local training weights
            response = requests.post("http://localhost:3000/upload-local-training-weights", json={
                "trader_name": name,
                "trades": [trade.__dict__ for trade in trades]
            })
            if response.status_code == 200:
                print(f"Successfully uploaded local training weights for {name} trader")
            else:
                print(f"Error uploading local training weights for {name} trader: {response.text}")

        except Exception as e:
            print(f"Error backtesting {name}: {e}")
            continue
    
    return all_trades, all_performance

class TradingDataset(Dataset):
    def __init__(self, trades: List[TradeAction], sequence_length: int):
        self.sequence_length = sequence_length
        self.features = []
        self.labels = []
        self.prepare_dataset(trades)

    def prepare_dataset(self, trades: List[TradeAction]):
        trades.sort(key=lambda x: x.timestamp)
        feature_cols = list(trades[0].technical_features.keys())
        
        for i in range(len(trades) - self.sequence_length):
            sequence = trades[i:i + self.sequence_length]
            feature_sequence = [list(trade.technical_features.values()) for trade in sequence]
            label = trades[i + self.sequence_length].action + 1
            
            self.features.append(feature_sequence)
            self.labels.append(label)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_local_model(trader_name: str, dataset: TradingDataset, num_epochs: int):
    input_size = len(dataset.features[0][0])
    model = LSTMModel(input_size=input_size, hidden_size=64, num_classes=3)
    criterion = nn.CrossEntropyLoss()
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

def aggregate_models(local_weights_list):
    global_weights = copy.deepcopy(local_weights_list[0])
    for key in global_weights.keys():
        for i in range(1, len(local_weights_list)):
            global_weights[key] += local_weights_list[i][key]
        global_weights[key] = torch.div(global_weights[key], len(local_weights_list))
    return global_weights

def federated_training(traders_datasets, num_rounds=5, local_epochs=2):
    input_size = len(next(iter(traders_datasets.values())).features[0][0])  
    hidden_size = 64
    num_classes = 3
    num_layers = 2
    
    global_model = LSTMModel(input_size=input_size, hidden_size=hidden_size, 
                             num_classes=num_classes, num_layers=num_layers)

    for round_num in range(num_rounds):
        print(f"\n--- Federated Learning Round {round_num+1} ---")
        
        # Retrieve all local weights from server
        response = requests.get("http://localhost:3000/retrieve-local-training-weights")
        if response.status_code != 200:
            print(f"Error retrieving local weights: {response.text}")
            continue
            
        local_weights_list = response.json()["local_weights"]
        
        # Aggregate the weights
        global_weights = aggregate_models(local_weights_list)
        global_model.load_state_dict(global_weights)

        # Upload the aggregated global weights
        response = requests.post("http://localhost:3000/upload-global-weights", json={
            "global_weights": global_weights
        })
        if response.status_code == 200:
            print("Successfully uploaded global model weights")
        else:
            print(f"Error uploading global model weights: {response.text}")

    return global_model

def prepare_test_dataset(coin_data_dict: Dict[str, pd.DataFrame], sequence_length: int) -> Dict[str, torch.Tensor]:
    test_sequences = {}
    feature_cols = ['rsi', 'macd', 'macd_signal', 'bollinger_high', 'bollinger_low', 
                   'bollinger_mid', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'volume_sma_20']
    
    for coin, df in coin_data_dict.items():
        features = df[feature_cols].values
        sequences = [features[i:i + sequence_length] for i in range(len(features) - sequence_length)]
        test_sequences[coin] = torch.tensor(sequences, dtype=torch.float32)
    
    return test_sequences

def plot_performance(agent_portfolio, all_performance):
    plt.figure(figsize=(12,6))
    plt.plot(agent_portfolio, label='AI Agent')
    for trader_name, performance_df in all_performance.items():
        plt.plot(performance_df['portfolio_value'].values, label=trader_name)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

def run_agent(timestep: int, coin_list: List[str], global_model: LSTMModel, 
             test_sequences: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """Generate trading predictions for each coin."""
    model_predictions = {}
    global_model.eval()
    
    with torch.no_grad():
        for coin in coin_list:
            if coin not in test_sequences or timestep >= len(test_sequences[coin]):
                continue
                
            sequence = test_sequences[coin][timestep].unsqueeze(0)
            output = global_model(sequence)
            probs = torch.softmax(output, dim=1)[0]
            
            model_predictions[coin] = {
                'price_change_pct': (probs[2] - probs[0]) * 100,  # Buy - Sell probability
                'confidence_score': torch.max(probs).item()
            }
    
    return model_predictions

def construct_prompt(timestep, predictions):
    system_prompt = f"""
    Assistant, you are an advanced AI trading agent. Your objective is to maximize profits by making optimal trades based on real-time market conditions and a set of model predictions from a collaborative LSTM model.
    At each timestep, you will evaluate market conditions, interpret predictive signals from the model, and decide whether to buy, sell, or hold based on the following:

    Instructions:
    - Analyze the predicted price change percentages and confidence scores.
    - Use these probabilities to assess the likelihood of price movement in each asset.
    - Make decisions that maximize potential profits while minimizing risk.

    Function Call Instructions:
    - To execute a trade, use the following functions:
      - buy(coin: string, amount: number) - Buys the specified coin at the current price.
      - sell(coin: string, amount: number) - Sells the specified coin at the current price.
    - Respond in the following format:
      {{
          "action": "buy",
          "coin": "<coin_name>",
          "amount": <number>
      }}
    - If no trade is deemed necessary, respond with {{"action": "hold"}}.

    Examples:
    - Example 1: When prediction confidence is high for price increase, you might decide to buy.
    - Example 2: If predictions indicate a likely drop in price, you may decide to sell.

    Model Predictions for Coins:
    Below is a dynamically generated section containing the predictions from the trained LSTM model. Use this information to guide your trading decisions.

    Market Data and Model Predictions:
    """

    for coin, pred in predictions.items():
        system_prompt += f"{coin.capitalize()}: Predicted price change: {pred['price_change_pct']:.2f}%, Confidence: {pred['confidence_score']*100:.1f}%\n"

    user_prompt = f"""
    Based on the provided predictions, select an appropriate action (buy, sell, or hold) for each asset, and provide a brief reasoning for your decision.
    Your decision-making should be based on maximizing the portfolio value by predicting market trends effectively.
    """

    return system_prompt, user_prompt

def call_openai_api(system_prompt, user_prompt, functions):
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        functions=functions,
        function_call="auto"
    )

    return response

def process_api_response(api_response, portfolio, current_prices):
    if api_response.choices[0].function_call:
        function_name = api_response.choices[0].function_call.name
        function_args = json.loads(api_response.choices[0].function_call.arguments)
        
        coin = function_args["coin"]
        usd_amount = function_args["amount"]
        
        if function_name == "buy" and portfolio.get("USD", 0) >= usd_amount:
            coin_price = current_prices[coin]
            coin_amount = usd_amount / coin_price
            portfolio["USD"] = portfolio.get("USD", 0) - usd_amount
            portfolio[coin] = portfolio.get(coin, 0) + coin_amount
            print(f"Bought ${usd_amount} worth of {coin} ({coin_amount} coins)")
            
        elif function_name == "sell":
            coin_price = current_prices[coin]
            if coin in portfolio:
                coin_amount = min(usd_amount / coin_price, portfolio[coin])
                portfolio[coin] -= coin_amount
                portfolio["USD"] = portfolio.get("USD", 0) + (coin_amount * coin_price)
                print(f"Sold {coin_amount} {coin} for ${coin_amount * coin_price}")
                if portfolio[coin] == 0:
                    del portfolio[coin]
    
    return portfolio


def simulate_trading(timestep: int, coin_list: List[str], global_model: LSTMModel, 
                    portfolio: Dict[str, float], test_sequences: Dict[str, torch.Tensor],
                    current_prices: Dict[str, float]) -> Dict[str, float]:
    """Simulate trading decisions based on model predictions."""
    predictions = run_agent(timestep, coin_list, global_model, test_sequences)
    system_prompt, user_prompt = construct_prompt(timestep, predictions)

    functions = [
        {
            "name": "buy",
            "description": "Buy the specified coin at the current price",
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {"type": "string", "description": "The coin to buy"},
                    "amount": {"type": "number", "description": "The amount in USD to spend"}
                },
                "required": ["coin", "amount"]
            }
        },
        {
            "name": "sell",
            "description": "Sell the specified coin at the current price",
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {"type": "string", "description": "The coin to sell"},
                    "amount": {"type": "number", "description": "The amount in USD worth to sell"}
                },
                "required": ["coin", "amount"]
            }
        }
    ]

    api_response = call_openai_api(system_prompt, user_prompt, functions)
    return process_api_response(api_response, portfolio, current_prices)

def main():
    # Configuration
    filename = 'top_coin_data.json'
    sequence_length = 10
    num_rounds = 5
    local_epochs = 2
    initial_balance = 100000

    # Load and process data
    coin_data_dict = load_coin_data(filename)
    processed_data = prepare_coin_data_dict(coin_data_dict)
    
    # Generate trading data from different strategies and upload local weights
    all_trades, all_performance = generate_training_data(coin_data_dict)
    
    # Prepare datasets for each trader
    traders_datasets = {
        name: TradingDataset(trades, sequence_length)
        for name, trades in all_trades.items()
    }
    
    # Train global model and handle weight aggregation
    global_model = federated_training(traders_datasets, num_rounds, local_epochs)
    
    # Prepare test data
    test_sequences = prepare_test_dataset(processed_data, sequence_length)
    timesteps = len(next(iter(test_sequences.values())))
    coin_list = list(test_sequences.keys())
    portfolio = {"USD": initial_balance}
    
    # Retrieve global model weights for inference
    response = requests.get("http://localhost:3000/retrieve-global-model-weights")
    if response.status_code == 200:
        global_weights = response.json()["global_weights"]
        input_size = len(next(iter(test_sequences.values()))[0][0])
        global_model = LSTMModel(input_size=input_size, hidden_size=64, 
                                num_classes=3, num_layers=2)
        global_model.load_state_dict(global_weights)
        print("Successfully loaded global model weights for inference")
    else:
        print(f"Error retrieving global model weights: {response.text}")
        return
    
    # Get current prices for each coin
    current_prices = {
        coin: processed_data[coin]['price'].iloc[-1]
        for coin in coin_list
    }
    
    # Run trading simulation
    print("\nStarting trading simulation...")
    for timestep in tqdm(range(timesteps)):
        portfolio = simulate_trading(
            timestep=timestep,
            coin_list=coin_list,
            global_model=global_model,
            portfolio=portfolio,
            test_sequences=test_sequences,
            current_prices=current_prices
        )
        
        total_value = portfolio.get("USD", 0)
        for coin, amount in portfolio.items():
            if coin != "USD":
                total_value += amount * current_prices[coin]
        
        print(f"\nTimestep {timestep}")
        print(f"Portfolio: {portfolio}")
        print(f"Total Value: ${total_value:,.2f}")

if __name__ == "__main__":
    main()