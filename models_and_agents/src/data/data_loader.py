import json
import pandas as pd
from typing import Dict, List, Tuple
from ..utils.technical_indicators import TechnicalIndicators

def load_coin_data(filename: str) -> Dict[str, List[Tuple]]:
    """Load coin data from JSON file."""
    with open(filename) as f:
        top_coin_data = json.load(f)
    return transform_coin_data(top_coin_data)

def transform_coin_data(top_coin_data: Dict) -> Dict[str, List[Tuple]]:
    """Transform raw coin data into the required format."""
    transformed_data = {}
    for coin, data in top_coin_data.items():
        prices = data['prices']
        volumes = data['volumes']
        combined_data = [
            (price[0], price[1], volume[1])
            for price, volume in zip(prices, volumes)
        ]
        transformed_data[coin] = combined_data
    return transformed_data

def prepare_coin_data(coin_data: List[Tuple]) -> pd.DataFrame:
    """Prepare coin data with technical indicators."""
    if len(coin_data) < 50:
        return None
    df = pd.DataFrame(coin_data, columns=['timestamp', 'price', 'volume'])
    return TechnicalIndicators.calculate_indicators(df)

def prepare_coin_data_dict(coin_data_dict: Dict[str, List[Tuple]]) -> Dict[str, pd.DataFrame]:
    """Prepare dictionary of coin data with technical indicators."""
    processed_data = {}
    for coin, data in coin_data_dict.items():
        df = prepare_coin_data(data)
        if df is not None:
            processed_data[coin] = df
    return processed_data