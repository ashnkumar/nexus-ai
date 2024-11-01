# data/preprocessing.py
from typing import Dict, List, Tuple
import pandas as pd
from utils.types import TradeAction
from utils.technical_indicators import TechnicalIndicators

def transform_coin_data(raw_data: Dict) -> Dict[str, List[Tuple]]:
    """Convert raw coin data to format needed for processing."""
    transformed_data = {}
    
    for coin, data in raw_data.items():
        prices = data['prices']
        volumes = data['volumes']
        
        combined_data = [
            (price[0], price[1], volume[1])
            for price, volume in zip(prices, volumes)
        ]
        
        transformed_data[coin] = combined_data
    
    return transformed_data

def prepare_coin_data(coin_data: List[Tuple]) -> pd.DataFrame:
    """Convert raw coin data to DataFrame with indicators."""
    if len(coin_data) < 50:
        return None

    df = pd.DataFrame(coin_data, columns=['timestamp', 'price', 'volume'])
    return TechnicalIndicators.calculate_indicators(df)

def filter_unstable_trades(trades: List[TradeAction], min_periods: int = 50) -> List[TradeAction]:
    """Filter out trades that occur before indicators have stabilized."""
    if not trades:
        return []

    trades = sorted(trades, key=lambda x: x.timestamp)
    return trades[min_periods:]