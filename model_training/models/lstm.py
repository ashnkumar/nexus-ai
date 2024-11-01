# models/lstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TradingLSTM(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_coins: int,
        hidden_size: int = 512,
        num_layers: int = 4,
        dropout: float = 0.3,
        sequence_length: int = 20,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.feature_size = feature_size
        self.num_coins = num_coins
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Combined feature size
        self.combined_feature_size = feature_size * num_coins
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.combined_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output size of LSTM
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size),
            nn.LayerNorm(lstm_out_size),
            nn.ReLU(),
            nn.Linear(lstm_out_size, lstm_out_size),
            nn.Tanh(),
            nn.Linear(lstm_out_size, 1)
        )
        
        # Prediction layers
        self.prediction_layers = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size),
            nn.LayerNorm(lstm_out_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, lstm_out_size // 2),
            nn.LayerNorm(lstm_out_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size // 2, self.num_coins * 3)
        )
        
        # Skip connection
        self.skip_connection = nn.Linear(self.combined_feature_size, lstm_out_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended_out = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Skip connection
        skip_out = self.skip_connection(x[:, -1, :])
        
        # Combine and predict
        combined = attended_out + skip_out
        logits = self.prediction_layers(combined)
        logits = logits.view(-1, self.num_coins, 3)
        
        return logits, attention_weights