
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
