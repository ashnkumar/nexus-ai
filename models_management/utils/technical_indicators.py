# utils/technical_indicators.py
import pandas as pd
import ta
from typing import Optional

class TechnicalIndicators:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate all technical indicators for a given dataframe."""
        required_cols = ['timestamp', 'price', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None

        if len(df) < 50:
            return None

        try:
            # RSI (14 periods)
            df['rsi'] = ta.momentum.RSIIndicator(df['price'], window=14).rsi()

            # MACD
            macd = ta.trend.MACD(df['price'], window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()

            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['price'], window=20)
            df['bollinger_high'] = bollinger.bollinger_hband()
            df['bollinger_low'] = bollinger.bollinger_lband()
            df['bollinger_mid'] = bollinger.bollinger_mavg()

            # Moving Averages
            df['sma_20'] = ta.trend.sma_indicator(df['price'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['price'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['price'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['price'], window=26)

            # Volume
            df['volume_sma_20'] = ta.trend.sma_indicator(df['volume'], window=20)

            # Handle NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df

        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return None