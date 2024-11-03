from typing import Tuple
from .base_trader import BaseTrader

class MomentumTrader(BaseTrader):
    def __init__(self, momentum_window: int = 5, 
                 buy_threshold: float = 0.02,
                 sell_threshold: float = -0.01,
                 initial_balance: float = 10000):
        super().__init__(initial_balance=initial_balance)
        self.momentum_window = momentum_window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
    def should_trade(self, coin: str, df, idx: int) -> Tuple[int, float]:
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
    def should_trade(self, coin: str, df, idx: int) -> Tuple[int, float]:
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
    def should_trade(self, coin: str, df, idx: int) -> Tuple[int, float]:
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
    def should_trade(self, coin: str, df, idx: int) -> Tuple[int, float]:
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
    def should_trade(self, coin: str, df, idx: int) -> Tuple[int, float]:
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
    def should_trade(self, coin: str, df, idx: int) -> Tuple[int, float]:
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

# Dictionary mapping strategy names to trader classes
TRADER_CLASSES = {
    'momentum': MomentumTrader,
    'mean_reversion': MeanReversionTrader,
    'breakout': BreakoutTrader,
    'trend_following': TrendFollowingTrader,
    'rsi': RSITrader,
    'volume': VolumeBasedTrader
}