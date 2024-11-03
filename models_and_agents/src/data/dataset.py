import torch
from torch.utils.data import Dataset
from typing import List
from dataclasses import dataclass
from typing import Dict

@dataclass
class TradeAction:
    timestamp: int
    coin: str
    price: float
    action: int  # -1: sell, 0: hold, 1: buy
    position_size: float
    technical_features: Dict[str, float]

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