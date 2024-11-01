# models/trading_system.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
import math

class TradingSystem:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate * math.sqrt(4096/32)  # Scale with batch size
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.action_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.grad_accumulation_steps = 2
        self.grad_clip_val = 1.0
        
        # Track initial losses
        self.initial_action_loss = None
        self.initial_profit_loss = None
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=1000,
            epochs=5,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0
        )

    def train_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        if batch_idx % self.grad_accumulation_steps == 0:
            self.optimizer.zero_grad()

        features = batch['features'].to(self.device)
        batch_size, seq_len, num_coins, num_features = features.size()
        features = features.view(batch_size, seq_len, -1)
        
        trader_action = batch['trader_action'].to(self.device)
        profit_margin = batch['profit_margin'].to(self.device)

        with torch.cuda.amp.autocast():
            logits, _ = self.model(features)
            logits_flat = logits.view(-1, 3)
            trader_action_flat = trader_action.view(-1)
            
            # Calculate losses
            action_loss = self.action_criterion(logits_flat, trader_action_flat)
            predicted_action = torch.argmax(logits_flat, dim=1)
            current_accuracy = (predicted_action == trader_action_flat).float().mean()
            
            # Store initial losses
            if batch_idx == 0 and self.initial_action_loss is None:
                self.initial_action_loss = action_loss.item()
                self.initial_profit_loss = -torch.mean(profit_margin).item()
            
            # Calculate profit loss
            normalized_profit = torch.clamp(profit_margin / 100, -1, 1)
            profit_loss = -torch.mean(normalized_profit * (predicted_action == trader_action_flat).float())
            
            # Normalize and combine losses
            norm_action_loss = action_loss / (self.initial_action_loss or 1.0)
            norm_profit_loss = profit_loss / (self.initial_profit_loss or 1.0)
            
            # Dynamic weighting
            action_weight = max(0.2, min(0.8, current_accuracy))
            profit_weight = 1 - action_weight
            
            total_loss = (action_weight * norm_action_loss + 
                         profit_weight * norm_profit_loss)

        # Scale loss and backward pass
        scaled_loss = total_loss / self.grad_accumulation_steps
        self.scaler.scale(scaled_loss).backward()

        if (batch_idx + 1) % self.grad_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

        return {
            'total_loss': total_loss.item(),
            'action_loss': action_loss.item(),
            'profit_loss': profit_loss.item(),
            'accuracy': current_accuracy.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        total_metrics = {
            'total_profit': 0,
            'predictions': [],
            'targets': [],
            'profits': [],
            'total_loss': 0
        }
        num_samples = 0

        for batch in dataloader:
            features = batch['features'].to(self.device)
            features = features.view(features.size(0), features.size(1), -1)
            target = batch['trader_action'].to(self.device)
            profit_margin = batch['profit_margin'].to(self.device)

            with torch.cuda.amp.autocast():
                logits, _ = self.model(features)
                logits_flat = logits.view(-1, 3)
                target_flat = target.view(-1)
                
                loss = self.action_criterion(logits_flat, target_flat)
                predicted_action = torch.argmax(logits_flat, dim=1)
                
                # Convert to numpy
                predictions = predicted_action.cpu().numpy()
                targets = target_flat.cpu().numpy()
                profits = profit_margin.view(-1).cpu().numpy()
                
                # Accumulate metrics
                total_metrics['predictions'].extend(predictions)
                total_metrics['targets'].extend(targets)
                total_metrics['profits'].extend(profits)
                total_metrics['total_loss'] += loss.item() * len(target_flat)
                num_samples += len(target_flat)

        # Calculate final metrics
        metrics = {
            'accuracy': accuracy_score(total_metrics['targets'], 
                                    total_metrics['predictions']),
            'avg_loss': total_metrics['total_loss'] / num_samples,
            'avg_profit': np.mean(total_metrics['profits']),
            'profit_std': np.std(total_metrics['profits'])
        }

        return metrics