"""
HTR Model Training Improvements and Utilities
This script provides improved training strategies and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from collections import defaultdict
import sys
sys.path.append('.')

from model.HTR_ME import HTRModel
from train_iam import IAMDataset
from utils.utils import CTCLabelConverter
import editdistance


class ImprovedHTRTrainer:
    def __init__(self, model, train_loader, val_loader, converter, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.converter = converter
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_cer': [],
            'val_wer': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_cer = float('inf')
        self.best_model_state = None
        
    def setup_training(self, learning_rate=1e-3, weight_decay=1e-4, 
                      use_scheduler=True, scheduler_type='cosine'):
        """Setup optimizer and scheduler with improved strategies"""
        
        # Use AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Setup learning rate scheduler
        if use_scheduler:
            if scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
                )
            elif scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=20, gamma=0.5
                )
            elif scheduler_type == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
                )
        else:
            self.scheduler = None
        
        # Setup loss function with label smoothing
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        # Gradient clipping
        self.max_grad_norm = 1.0
        
        print(f"Training setup complete:")
        print(f"  Optimizer: AdamW (lr={learning_rate}, wd={weight_decay})")
        print(f"  Scheduler: {scheduler_type if use_scheduler else 'None'}")
        print(f"  Gradient clipping: {self.max_grad_norm}")

    def train_epoch(self, epoch):
        """Train for one epoch with improved monitoring"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        # Progress tracking
        losses = []
        start_time = time.time()
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # Handle collate_fn output format
            if len(batch_data) == 5:  # collate_fn format
                images, targets_tensor, text_lengths_tensor, texts, image_names = batch_data
                images = images.to(self.device)
                text_for_loss = targets_tensor.to(self.device)
                length_for_loss = text_lengths_tensor.to(self.device)
            else:  # Standard format
                images, labels = batch_data
                images = images.to(self.device)
                text_for_loss, length_for_loss = self.converter.encode(labels)
                text_for_loss = text_for_loss.to(self.device)
                length_for_loss = length_for_loss.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, logit_lengths = self.model(images)
            
            # Reshape for CTC loss
            logits = logits.permute(1, 0, 2)  # [seq_len, batch, vocab_size]
            logits = logits.log_softmax(2)
            
            # Calculate loss
            loss = self.criterion(logits, text_for_loss, logit_lengths, length_for_loss)
            
            # Backward pass with gradient clipping
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            losses.append(loss.item())
            
            # Progress reporting
            if batch_idx % 50 == 0:
                elapsed = time.time() - start_time
                progress = 100. * batch_idx / num_batches
                avg_loss = np.mean(losses[-50:])
                current_lr = self.optimizer.param_groups[0]['lr']
                
                print(f'Epoch {epoch} [{batch_idx:4d}/{num_batches}] ({progress:5.1f}%) | '
                      f'Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | '
                      f'Time: {elapsed:.1f}s')
        
        avg_loss = total_loss / num_batches
        
        # Update scheduler (except ReduceLROnPlateau)
        if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        
        return avg_loss

    def validate_epoch(self, epoch):
        """Validate with detailed metrics"""
        self.model.eval()
        total_loss = 0
        total_cer = 0
        total_wer = 0
        total_chars = 0
        total_words = 0
        num_samples = 0
        
        all_predictions = []
        all_ground_truths = []
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                # Handle collate_fn output format
                if len(batch_data) == 5:  # collate_fn format
                    images, targets_tensor, text_lengths_tensor, texts, image_names = batch_data
                    images = images.to(self.device)
                    text_for_loss = targets_tensor.to(self.device)
                    length_for_loss = text_lengths_tensor.to(self.device)
                    labels = texts  # Use texts for metric calculation
                else:  # Standard format
                    images, labels = batch_data
                    images = images.to(self.device)
                    text_for_loss, length_for_loss = self.converter.encode(labels)
                    text_for_loss = text_for_loss.to(self.device)
                    length_for_loss = length_for_loss.to(self.device)
                
                batch_size = images.size(0)
                
                # Forward pass
                logits, logit_lengths = self.model(images)
                
                # Calculate loss
                logits_for_loss = logits.permute(1, 0, 2).log_softmax(2)
                loss = self.criterion(logits_for_loss, text_for_loss, logit_lengths, length_for_loss)
                total_loss += loss.item()
                
                # Decode predictions
                for i in range(batch_size):
                    seq_len = logit_lengths[i].item()
                    pred_logits = logits[:seq_len, i, :]
                    
                    # Greedy decoding
                    _, pred_indices = pred_logits.max(1)
                    pred_text = self.converter.decode(pred_indices.cpu().numpy(), [seq_len])
                    
                    if isinstance(pred_text, list):
                        pred_text = pred_text[0] if pred_text else ""
                    
                    ground_truth = labels[i]
                    
                    # Calculate CER
                    cer_distance = editdistance.eval(pred_text, ground_truth)
                    total_cer += cer_distance
                    total_chars += len(ground_truth)
                    
                    # Calculate WER
                    pred_words = pred_text.split()
                    gt_words = ground_truth.split()
                    wer_distance = editdistance.eval(pred_words, gt_words)
                    total_wer += wer_distance
                    total_words += len(gt_words)
                    
                    num_samples += 1
                    
                    # Store for detailed analysis
                    all_predictions.append(pred_text)
                    all_ground_truths.append(ground_truth)
        
        # Calculate average metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_cer = total_cer / total_chars if total_chars > 0 else 0
        avg_wer = total_wer / total_words if total_words > 0 else 0
        
        # Update scheduler if using ReduceLROnPlateau
        if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_cer)
        
        # Track best model
        if avg_cer < self.best_cer:
            self.best_cer = avg_cer
            self.best_model_state = self.model.state_dict().copy()
            print(f"New best model! CER: {avg_cer:.4f}")
        
        print(f'Validation - Loss: {avg_loss:.4f} | CER: {avg_cer:.4f} | WER: {avg_wer:.4f}')
        
        return avg_loss, avg_cer, avg_wer, all_predictions, all_ground_truths

    def train(self, num_epochs, save_dir='improved_checkpoints', 
              save_frequency=5, early_stopping_patience=15):
        """Main training loop with improvements"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Checkpoints will be saved to: {save_dir}")
        
        no_improvement_count = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_cer, val_wer, predictions, ground_truths = self.validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_cer'].append(val_cer)
            self.history['val_wer'].append(val_wer)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Early stopping check
            if val_cer < self.best_cer:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Save checkpoint
            if epoch % save_frequency == 0 or epoch == num_epochs:
                self.save_checkpoint(save_dir / f"checkpoint_epoch_{epoch}.pth", epoch)
            
            # Save training plots
            if epoch % 10 == 0:
                self.plot_training_history(save_dir / f"training_history_epoch_{epoch}.png")
        
        # Save final best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            self.save_checkpoint(save_dir / "best_model.pth", epoch, is_best=True)
        
        print(f"Training completed! Best CER: {self.best_cer:.4f}")
        
        return self.history

    def save_checkpoint(self, path, epoch, is_best=False):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_cer': self.best_cer,
            'history': self.history,
            'vocab': self.converter.character
        }
        
        torch.save(checkpoint, path)
        if is_best:
            print(f"Best model saved to {path}")
        else:
            print(f"Checkpoint saved to {path}")

    def plot_training_history(self, save_path):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # CER plot
        axes[0, 1].plot(epochs, self.history['val_cer'], 'g-', label='Val CER')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('CER')
        axes[0, 1].set_title('Validation Character Error Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # WER plot
        axes[1, 0].plot(epochs, self.history['val_wer'], 'm-', label='Val WER')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('WER')
        axes[1, 0].set_title('Validation Word Error Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        axes[1, 1].plot(epochs, self.history['learning_rates'], 'c-', label='Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def analyze_failure_cases(self, predictions, ground_truths, num_worst=10):
        """Analyze worst performing samples"""
        errors = []
        for pred, gt in zip(predictions, ground_truths):
            cer = editdistance.eval(pred, gt) / len(gt) if len(gt) > 0 else 0
            errors.append((pred, gt, cer))
        
        # Sort by CER (worst first)
        errors.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nWorst {num_worst} predictions:")
        print("-" * 80)
        for i, (pred, gt, cer) in enumerate(errors[:num_worst]):
            print(f"{i+1:2d}. CER: {cer:.3f}")
            print(f"    Pred: '{pred}'")
            print(f"    GT:   '{gt}'")
            print()
        
        return errors


def create_improved_training_script():
    """Create a complete training script with improvements"""
    script_content = '''
"""
Improved HTR Training Script
Usage: python improved_train.py --data_dir data/iam --checkpoint_dir improved_checkpoints
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from train_iam import IAMDataset
from data.transform import get_transform
from model.HTR_ME import HTRModel
from utils.utils import CTCLabelConverter
from improved_trainer import ImprovedHTRTrainer


def main():
    parser = argparse.ArgumentParser(description='Improved HTR Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to IAM dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='improved_checkpoints', 
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'step', 'plateau', 'none'], help='LR scheduler type')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    transform = get_transform(augment=True)  # Enable augmentation for training
    
    train_dataset = IAMDataset(
        root=args.data_dir,
        split='train',
        transform=transform,
        target_height=40,
        chunk_width=320
    )
    
    val_dataset = IAMDataset(
        root=args.data_dir,
        split='val',
        transform=get_transform(augment=False),  # No augmentation for validation
        target_height=40,
        chunk_width=320
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    vocab = train_dataset.vocab
    model = HTRModel(
        vocab_size=len(vocab),
        max_length=256,
        target_height=40,
        chunk_width=320,
        stride=240,
        embed_dims=[64, 192, 384],
        num_heads=[1, 3, 6],
        depths=[1, 2, 10]
    ).to(device)
    
    # Create converter
    converter = CTCLabelConverter(vocab)
    
    # Create trainer
    trainer = ImprovedHTRTrainer(model, train_loader, val_loader, converter, device)
    
    # Setup training
    trainer.setup_training(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_scheduler=args.scheduler != 'none',
        scheduler_type=args.scheduler
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        trainer.best_cer = checkpoint.get('best_cer', float('inf'))
        trainer.history = checkpoint.get('history', trainer.history)
        print(f"Resumed from epoch {start_epoch-1}")
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        save_dir=args.checkpoint_dir,
        save_frequency=5,
        early_stopping_patience=15
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
'''
    
    with open('improved_train.py', 'w') as f:
        f.write(script_content)
    
    print("Created improved_train.py")


if __name__ == "__main__":
    create_improved_training_script()
