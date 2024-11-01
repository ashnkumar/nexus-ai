# training/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

class FederatedTrainingVisualizer:
    def __init__(self, global_metrics: Dict, trader_metrics: Dict):
        self.global_metrics = global_metrics
        self.trader_metrics = trader_metrics

    def plot_training_curves(self):
        """Plot training curves for global model and individual traders."""
        metrics_to_plot = [
            ('loss', 'Loss'),
            ('accuracy', 'Accuracy'),
            ('avg_profit', 'Average Profit %'),
            ('f1', 'F1 Score')
        ]

        fig, axes = plt.subplots(len(metrics_to_plot), 2, 
                                figsize=(20, 6*len(metrics_to_plot)))
        fig.suptitle('Training Metrics - Global vs Individual Traders', 
                    fontsize=16)

        for idx, (metric_name, metric_title) in enumerate(metrics_to_plot):
            # Global model metrics
            ax = axes[idx, 0]
            if f'val_{metric_name}' in self.global_metrics:
                values = self.global_metrics[f'val_{metric_name}']
                ax.plot(values, label='Global Model', linewidth=2)
                ax.set_title(f'Global Model - {metric_title}')
                ax.set_xlabel('Round')
                ax.set_ylabel(metric_title)
                ax.grid(True)
                ax.legend()

            # Individual trader metrics
            ax = axes[idx, 1]
            for trader_name, metrics in self.trader_metrics.items():
                if f'val_{metric_name}' in metrics:
                    values = metrics[f'val_{metric_name}']
                    ax.plot(values, label=trader_name, alpha=0.7)
            ax.set_title(f'Individual Traders - {metric_title}')
            ax.set_xlabel('Local Epoch')
            ax.set_ylabel(metric_title)
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_contribution_evolution(self):
        """Plot trader contribution evolution."""
        contributions = self.global_metrics['contributions']
        plt.figure(figsize=(15, 8))

        trader_names = contributions[0].keys()
        rounds = range(1, len(contributions) + 1)

        for trader in trader_names:
            trader_contributions = [round_contrib[trader] 
                                  for round_contrib in contributions]
            plt.plot(rounds, trader_contributions, marker='o', 
                    label=trader, linewidth=2)

        plt.title('Evolution of Trader Contributions Over Training Rounds')
        plt.xlabel('Round')
        plt.ylabel('Contribution Percentage')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_coin_performance(self):
        """Plot coin-specific performance metrics."""
        coin_metrics = {}
        for trader, metrics in self.trader_metrics.items():
            for key, value in metrics.items():
                if key.startswith('coin_'):
                    coin = key.split('_')[1]
                    if coin not in coin_metrics:
                        coin_metrics[coin] = {}
                    coin_metrics[coin][trader] = value

        plt.figure(figsize=(15, 8))
        num_coins = len(coin_metrics)
        num_traders = len(self.trader_metrics)
        
        positions = np.arange(num_traders)
        width = 0.8 / num_coins

        for i, (coin, performances) in enumerate(coin_metrics.items()):
            values = [performances.get(trader, 0) for trader in self.trader_metrics.keys()]
            plt.bar(positions + i * width, values, width, label=coin)

        plt.xlabel('Traders')
        plt.ylabel('Performance Metric')
        plt.title('Coin Performance by Trader')
        plt.xticks(positions + width * (num_coins-1)/2, 
                  list(self.trader_metrics.keys()), rotation=45)
        plt.legend(title='Coins')
        plt.tight_layout()
        plt.show()