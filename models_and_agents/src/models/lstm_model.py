import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def aggregate_models(local_weights_list):
    """Aggregate local model weights into global weights."""
    global_weights = {}
    for key in local_weights_list[0].keys():
        global_weights[key] = sum(weights[key] for weights in local_weights_list) / len(local_weights_list)
    return global_weights