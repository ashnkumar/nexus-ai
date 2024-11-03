from typing import Dict, List, Tuple
from ..data.dataset import TradeAction
from ..config import FEATURE_COLUMNS

class BaseTrader:
    def __init__(self, initial_balance: float = 100000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: Dict[str, float] = {}
        self.trades: List[TradeAction] = []

    def get_technical_features(self, df, idx: int) -> Dict[str, float]:
        """Extract technical features from DataFrame at given index."""
        return {col: df[col].iloc[idx] for col in FEATURE_COLUMNS}

    def execute_trade(self, timestamp: int, coin: str, price: float,
                     action: int, position_size: float, features: Dict[str, float]):
        """Execute a trade and update positions and balance."""
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

    def should_trade(self, coin: str, df, idx: int) -> Tuple[int, float]:
        """Abstract method to be implemented by child classes."""
        raise NotImplementedError