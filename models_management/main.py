# main.py
import json
import torch
from torch.utils.data import random_split
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from data.preprocessing import transform_coin_data, prepare_coin_data
from data.dataset import TradingDataset
from models.lstm import TradingLSTM
from models.trading_system import TradingSystem
from training.federated import FederatedTraining
from training.visualization import FederatedTrainingVisualizer
from utils.types import TradeAction

def load_data(data_path: str) -> Dict[str, List[Tuple]]:
    """Load and transform raw trading data."""
    with open(data_path) as f:
        raw_data = json.load(f)
    return transform_coin_data(raw_data)

def prepare_datasets(
    coin_data_dict: Dict[str, List[Tuple]],
    sequence_length: int = 20,
    batch_size: int = 4096
) -> Tuple[Dict[str, Tuple[DataLoader, DataLoader]], DataLoader]:
    """Prepare training datasets for each trader and global validation."""
    
    # Process data for each trader
    trader_datasets = {}
    all_sequences = []
    
    for trader_name, data in coin_data_dict.items():
        processed_data = prepare_coin_data(data)
        if processed_data is None:
            continue
            
        dataset = TradingDataset(processed_data.sequences)
        
        # Split into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=0)
        
        trader_datasets[trader_name] = (train_loader, val_loader)
        all_sequences.extend(processed_data.sequences)
    
    # Create global validation dataset
    global_dataset = TradingDataset(all_sequences)
    _, global_val_dataset = random_split(
        global_dataset,
        [int(0.8 * len(global_dataset)), len(global_dataset) - int(0.8 * len(global_dataset))]
    )
    global_val_loader = DataLoader(global_val_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=0)
    
    return trader_datasets, global_val_loader

def main(args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print("Loading data...")
    coin_data_dict = load_data(args.data_path)
    
    print("Preparing datasets...")
    trader_datasets, global_val_loader = prepare_datasets(
        coin_data_dict,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size
    )
    
    # Initialize model and training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingLSTM(
        feature_size=args.feature_size,
        num_coins=len(coin_data_dict),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        sequence_length=args.sequence_length
    ).to(device)
    
    federated_training = FederatedTraining(
        model=model,
        device=device,
        checkpoint_dir=output_dir / 'checkpoints'
    )
    
    # Train
    print("Starting federated training...")
    federated_training.train_federated(
        trader_datasets=trader_datasets,
        global_val_loader=global_val_loader,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs
    )
    
    # Visualize results
    visualizer = FederatedTrainingVisualizer(
        federated_training.global_metrics,
        federated_training.trader_metrics
    )
    visualizer.plot_training_curves()
    visualizer.plot_contribution_evolution()
    visualizer.plot_coin_performance()
    
    # Save metrics
    metrics = {
        'global_metrics': dict(federated_training.global_metrics),
        'trader_metrics': {k: dict(v) for k, v in federated_training.trader_metrics.items()},
        'training_params': vars(args)
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--sequence_length', type=int, default=20)
    parser.add_argument('--feature_size', type=int, default=11)
    parser.add_argument('--hidden_size', type=int, default=