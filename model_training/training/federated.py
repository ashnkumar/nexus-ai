# training/federated.py
import torch
from typing import Dict, Tuple, List
from collections import defaultdict
import copy
from torch.utils.data import DataLoader
import os

class FederatedTraining:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Metrics storage
        self.global_metrics = defaultdict(list)
        self.trader_metrics = defaultdict(lambda: defaultdict(list))
        
        # Store coin performance metrics
        self.coin_performance = defaultdict(lambda: defaultdict(list))

    def train_federated(
        self,
        trader_datasets: Dict[str, Tuple[DataLoader, DataLoader]],
        global_val_loader: DataLoader,
        num_rounds: int,
        local_epochs: int,
        start_round: int = 0
    ):
        for round_idx in range(start_round, start_round + num_rounds):
            print(f"\nFederated Training Round {round_idx + 1}")
            
            # Train local models
            local_models = {}
            for trader_name, (train_loader, val_loader) in trader_datasets.items():
                local_state_dict, metrics = self.train_trader_model(
                    trader_name=trader_name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    local_epochs=local_epochs
                )
                local_models[trader_name] = local_state_dict
                self.trader_metrics[trader_name].update(metrics)
            
            # Aggregate models
            self.aggregate_models(local_models)
            
            # Evaluate global model
            trading_system = TradingSystem(self.model, device=self.device)
            global_metrics = trading_system.evaluate(global_val_loader)
            
            # Calculate contributions
            contributions = self.calculate_contributions(
                trader_datasets=trader_datasets,
                global_val_loader=global_val_loader
            )
            
            # Update metrics
            self.global_metrics['round'].append(round_idx)
            for k, v in global_metrics.items():
                self.global_metrics[k].append(v)
            self.global_metrics['contributions'].append(contributions)
            
            # Save checkpoint
            self.save_checkpoint(round_idx)

    def aggregate_models(self, local_models: Dict[str, Dict[str, torch.Tensor]]):
        """FedAvg algorithm for model aggregation."""
        global_state = self.model.state_dict()
        
        for key in global_state.keys():
            # Initialize accumulator
            global_state[key] = torch.zeros_like(global_state[key])
            
            # Average local models
            for local_state in local_models.values():
                global_state[key] += local_state[key]
            
            global_state[key] = torch.div(global_state[key], len(local_models))
        
        self.model.load_state_dict(global_state)

    def save_checkpoint(self, round_idx: int):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'global_metrics': dict(self.global_metrics),
            'trader_metrics': {k: dict(v) for k, v in self.trader_metrics.items()},
            'round': round_idx
        }
        path = f"{self.checkpoint_dir}/checkpoint_round{round_idx}.pt"
        torch.save(checkpoint, path)