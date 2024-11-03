import requests
from tqdm import tqdm
from src.config import (
    SEQUENCE_LENGTH, NUM_ROUNDS, LOCAL_EPOCHS, 
    INITIAL_BALANCE, LSTM_PARAMS, API_ENDPOINTS
)
from src.data.data_loader import load_coin_data, prepare_coin_data_dict
from src.data.dataset import TradingDataset
from src.models.lstm_model import LSTMModel, aggregate_models
from src.traders.strategy_traders import TRADER_CLASSES
from src.utils.training_utils import (
    backtest_trader, train_local_model, 
    prepare_test_dataset, plot_performance
)
from src.agents.trading_agent import simulate_trading

def generate_training_data(coin_data_dict, training_end_idx):
    """Generate training data from different trading strategies."""
    all_trades = {}
    all_performance = {}
    processed_coin_data = prepare_coin_data_dict(coin_data_dict)
    
    for name, trader_class in TRADER_CLASSES.items():
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
            response = requests.post(
                API_ENDPOINTS['upload_local_weights'],
                json={
                    "trader_name": name,
                    "trades": [trade.__dict__ for trade in trades]
                }
            )
            if response.status_code == 200:
                print(f"Successfully uploaded local training weights for {name} trader")
            else:
                print(f"Error uploading local training weights for {name} trader: {response.text}")

        except Exception as e:
            print(f"Error backtesting {name}: {e}")
            continue
    
    return all_trades, all_performance

def federated_training(traders_datasets):
    """Perform federated learning across trading strategies."""
    input_size = len(next(iter(traders_datasets.values())).features[0][0])
    global_model = LSTMModel(
        input_size=input_size,
        **LSTM_PARAMS
    )

    for round_num in range(NUM_ROUNDS):
        print(f"\n--- Federated Learning Round {round_num+1} ---")
        
        # Retrieve all local weights from server
        response = requests.get(API_ENDPOINTS['retrieve_local_weights'])
        if response.status_code != 200:
            print(f"Error retrieving local weights: {response.text}")
            continue
            
        local_weights_list = response.json()["local_weights"]
        global_weights = aggregate_models(local_weights_list)
        global_model.load_state_dict(global_weights)

        # Upload the aggregated global weights
        response = requests.post(
            API_ENDPOINTS['upload_global_weights'],
            json={"global_weights": global_weights}
        )
        if response.status_code == 200:
            print("Successfully uploaded global model weights")
        else:
            print(f"Error uploading global model weights: {response.text}")

    return global_model

def main():
    # Load and process data
    filename = 'top_coin_data.json'
    coin_data_dict = load_coin_data(filename)
    processed_data = prepare_coin_data_dict(coin_data_dict)
    
    # Calculate training end index
    first_coin_data = next(iter(coin_data_dict.values()))
    training_end_idx = int(len(first_coin_data) * 0.8)
    
    # Generate training data from different strategies
    all_trades, all_performance = generate_training_data(coin_data_dict, training_end_idx)
    
    # Prepare datasets for each trader
    traders_datasets = {
        name: TradingDataset(trades, SEQUENCE_LENGTH)
        for name, trades in all_trades.items()
    }
    
    # Train global model through federated learning
    global_model = federated_training(traders_datasets)
    
    # Prepare test data
    test_sequences = prepare_test_dataset(processed_data)
    timesteps = len(next(iter(test_sequences.values())))
    coin_list = list(test_sequences.keys())
    portfolio = {"USD": INITIAL_BALANCE}
    
    # Get current prices for each coin
    current_prices = {
        coin: processed_data[coin]['price'].iloc[-1]
        for coin in coin_list
    }
    
    # Run trading simulation
    print("\nStarting trading simulation...")
    portfolio_values = []
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
        
        portfolio_values.append(total_value)
        print(f"\nTimestep {timestep}")
        print(f"Portfolio: {portfolio}")
        print(f"Total Value: ${total_value:,.2f}")
    
    # Plot final performance comparison
    plot_performance(portfolio_values, all_performance)

if __name__ == "__main__":
    main()