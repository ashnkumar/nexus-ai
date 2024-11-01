# data/augmentation.py
import numpy as np
from typing import List
from utils.types import TradeSequence

def augment_trade_sequence(sequence: TradeSequence, noise_level: float = 0.05) -> List[TradeSequence]:
    """Generate augmented versions of a trade sequence."""
    augmented_sequences = [sequence]  # Original sequence

    # Add random noise
    noisy_features = sequence.features + np.random.normal(0, noise_level, sequence.features.shape)
    augmented_sequences.append(TradeSequence(
        features=noisy_features,
        profit_margin=sequence.profit_margin,
        trader_action=sequence.trader_action,
        timestamp=sequence.timestamp,
        coin=sequence.coin,
        price=sequence.price
    ))

    # Time-based scaling
    for scale in [0.9, 1.1]:
        scaled_features = sequence.features * scale
        augmented_sequences.append(TradeSequence(
            features=scaled_features,
            profit_margin=sequence.profit_margin * scale,
            trader_action=sequence.trader_action,
            timestamp=sequence.timestamp,
            coin=sequence.coin,
            price=sequence.price * scale
        ))

    # Feature mixing
    if len(sequence.features) > 1:
        mixed_features = (sequence.features[:-1] + sequence.features[1:]) / 2
        padded_mixed = np.pad(mixed_features, ((0,1), (0,0), (0,0)), mode='edge')
        augmented_sequences.append(TradeSequence(
            features=padded_mixed,
            profit_margin=sequence.profit_margin,
            trader_action=sequence.trader_action,
            timestamp=sequence.timestamp,
            coin=sequence.coin,
            price=sequence.price
        ))

    return augmented_sequences