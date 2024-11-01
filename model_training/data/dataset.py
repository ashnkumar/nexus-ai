# data/dataset.py
import torch
from torch.utils.data import Dataset
from typing import List
import numpy as np
from tqdm import tqdm
from utils.types import TradeSequence

class TradingDataset(Dataset):
    def __init__(self, sequences: List[TradeSequence]):
        self.sequences = sequences
        self.scaler = StandardScaler()
        
        # Normalize features
        all_features = np.vstack([seq.features.reshape(-1, seq.features.shape[-1]) 
                                for seq in sequences])
        self.scaler.fit(all_features)
        
        # Cache preprocessed items
        self.cached_items = []
        for sequence in tqdm(sequences, desc="Preprocessing sequences"):
            flattened = sequence.features.reshape(-1, sequence.features.shape[-1])
            normalized = self.scaler.transform(flattened)
            reshaped = normalized.reshape(sequence.features.shape)
            
            self.cached_items.append({
                'features': torch.FloatTensor(reshaped),
                'profit_margin': torch.FloatTensor([sequence.profit_margin]),
                'trader_action': torch.LongTensor([sequence.trader_action]),
                'timestamp': torch.LongTensor([sequence.timestamp]),
                'coin': sequence.coin,
                'price': torch.FloatTensor([sequence.price])
            })

    def __len__(self):
        return len(self.cached_items)

    def __getitem__(self, idx):
        return self.cached_items[idx]