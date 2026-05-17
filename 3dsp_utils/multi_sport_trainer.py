import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional
import wandb
from tqdm import tqdm

from multi_sport_dataloader import MultiSportDataset, create_multi_sport_dataloaders

class MultiSportPoseEstimator(nn.Module):
    """
    Enhanced pose estimator that can handle multiple sports
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 hidden_size: int = 256,
                 num_sports: int = 5,
                 num_keypoints: int = 17,
                 sequence_length: int = 20):
        
        super().__init__()
        
        self.num_sports = num_sports
        self.num_keypoints = num_keypoints
        self.sequence_length = sequence_length
        
        # Sport classification head
        self.sport_classifier = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_sports)
        )
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((sequence_length, 8, 8)),
        )
        
        # Sport-specific pose estimation heads
        self.pose_heads = nn.ModuleDict({
            'soccer': self._create_pose_head(128, hidden_size),
            'tennis': self._create_pose_head(128, hidden_size),
            'hi_jump': self._create_pose_head(128, hidden_size),
            'throw_baseball': self._create_pose_head(128, hidden_size),
            'volley': self._create_pose_head(128, hidden_size)
        })
        
    def _create_pose_head(self, input_dim: int, hidden_size: int) -> nn.Module:
        """Create sport-specific pose estimation head"""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * self.sequence_length * 8 * 8, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.num_keypoints * 3)  # x, y, confidence
        )
    
    def forward(self, x: torch.Tensor, sport: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        x: (batch_size, sequence_length, channels, height, width)
        """
        # Reshape for 3D conv: (batch, channels, sequence, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Sport classification
        sport_logits = self.sport_classifier(x)
        
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Pose estimation
        if sport is not None:
            # Use specified sport head
            pose_output = self.pose_heads[sport](features)
        else:
            # Use predicted sport (for inference)
            predicted_sport_idx = torch.argmax(sport_logits, dim=1)
            sport_names = ['soccer', 'tennis', 'hi_jump', 'throw_baseball', 'volley']
            
            # For now, use soccer head as default (would need batch-wise selection)
            pose_output = self.pose_heads['soccer'](features)
        
        return {
            'sport_logits': sport_logits,
            'pose_output': pose_output.view(-1, self.num_keypoints, 3)
        }

class MultiSportTrainer:
    """
    Trainer for multi-sport pose estimation
    """
    
    def __init__(self, 
                 model: MultiSportPoseEstimator,
                 device: str = 'cuda',
                 learning_rate: float = 1e-3,
                 use_wandb: bool = False):
        
        self.model = model.to(device)
        self.device = device
        self.use_wandb = use_wandb
        
        # Optimizers
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss functions
        self.sport_criterion = nn.CrossEntropyLoss()
        self.pose_criterion = nn.MSELoss()
        
        # Sport mapping
        self.sport_to_idx = {
            'soccer': 0, 'tennis': 1, 'hi_jump': 2, 
            'throw_baseball': 3, 'volley': 4
        }
        
        if use_wandb:
            wandb.init(project="multi-sport-pose-estimation")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        sport_loss_total = 0
        pose_loss_total = 0
        correct_sport_predictions = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            frames = batch['frames'].to(self.device)  # (B, T, C, H, W)
            sports = batch['sport']
            
            # Convert sport names to indices
            sport_indices = torch.tensor([
                self.sport_to_idx.get(sport, 0) for sport in sports
            ]).to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(frames)
            
            # Sport classification loss
            sport_loss = self.sport_criterion(outputs['sport_logits'], sport_indices)
            
            # Pose estimation loss (dummy for now - would use actual pose annotations)
            pose_loss = torch.tensor(0.0, device=self.device)
            if 'annotations' in batch and batch['annotations']:
                # Would compute actual pose loss here
                pass
            
            # Total loss
            loss = sport_loss + pose_loss
            
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            sport_loss_total += sport_loss.item()
            pose_loss_total += pose_loss.item()
            
            # Sport accuracy
            predicted_sports = torch.argmax(outputs['sport_logits'], dim=1)
            correct_sport_predictions += (predicted_sports == sport_indices).sum().item()
            total_samples += len(sports)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Sport Acc': f"{correct_sport_predictions/total_samples:.3f}"
            })
        
        return {
            'total_loss': total_loss / len(train_loader),
            'sport_loss': sport_loss_total / len(train_loader),
            'pose_loss': pose_loss_total / len(train_loader),
            'sport_accuracy': correct_sport_predictions / total_samples
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        correct_sport_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                frames = batch['frames'].to(self.device)
                sports = batch['sport']
                
                sport_indices = torch.tensor([
                    self.sport_to_idx.get(sport, 0) for sport in sports
                ]).to(self.device)
                
                outputs = self.model(frames)
                
                # Sport classification loss
                sport_loss = self.sport_criterion(outputs['sport_logits'], sport_indices)
                total_loss += sport_loss.item()
                
                # Sport accuracy
                predicted_sports = torch.argmax(outputs['sport_logits'], dim=1)
                correct_sport_predictions += (predicted_sports == sport_indices).sum().item()
                total_samples += len(sports)
        
        return {
            'val_loss': total_loss / len(val_loader),
            'val_sport_accuracy': correct_sport_predictions / total_samples
        }
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              num_epochs: int = 10,
              save_path: str = "multi_sport_model.pth"):
        """Full training loop"""
        
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            print(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                  f"Train Sport Acc: {train_metrics['sport_accuracy']:.3f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Sport Acc: {val_metrics['val_sport_accuracy']:.3f}")
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **train_metrics,
                    **val_metrics
                })
            
            # Save best model
            if val_metrics['val_sport_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['val_sport_accuracy']
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model with val accuracy: {best_val_acc:.3f}")

def main():
    """Main training function"""
    print("🏃‍♂️ Starting Multi-Sport Training...")
    
    # Configuration
    config = {
        'batch_size': 4,
        'num_epochs': 20,
        'learning_rate': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    try:
        # Create dataloaders
        train_loader, val_loader = create_multi_sport_dataloaders(
            batch_size=config['batch_size']
        )
        
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        
        # Create model and trainer
        model = MultiSportPoseEstimator()
        trainer = MultiSportTrainer(
            model=model,
            device=config['device'],
            learning_rate=config['learning_rate'],
            use_wandb=False  # Set to True if you want to use wandb
        )
        
        # Train
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['num_epochs']
        )
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("This is expected if dataset is not accessible from workspace")

if __name__ == "__main__":
    main()