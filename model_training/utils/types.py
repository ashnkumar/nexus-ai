# utils/types.py
from dataclasses import dataclass
import numpy as np
from typing import Dict

@dataclass
class TradeAction:
    timestamp: int
    coin: str
    price: float
    action: int  # -1 (sell), 0 (hold), 1 (buy)
    position_size: float
    technical_features: Dict[str, float]

@dataclass
class TradeSequence:
    features: np.ndarray  # Shape: [sequence_length, n_features]
    profit_margin: float  # Actual profit/loss percentage from this trade
    trader_action: int    # What the trader did (-1, 0, 1)
    timestamp: int
    coin: str
    price: float