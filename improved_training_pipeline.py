
"""
Improved HTR Training Pipeline
Run this to start training with all improvements
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

# Import your modules
from model.HTR_ME import HTRModel
from improved_trainer import ImprovedHTRTrainer
from data_augmentation import HTRDataAugmentation, create_augmented_dataset

def create_improved_model(vocab_size):
    """Create model with improvements"""
    model = HTRModel(
        vocab_size=vocab_size,
        max_length=256,
        target_height=40,
        chunk_width=320,
        stride=240,
        embed_dims=[64, 192, 384],
        num_heads=[1, 3, 6], 
        depths=[1, 2, 10]
    )
    
    # Add dropout to classifier
    model.classifier = nn.Sequential(
        nn.Linear(384, 256),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.BatchNorm1d(256),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, vocab_size)
    )
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='improved_checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load datasets with augmentation
    print("Loading datasets with augmentation...")
    
    # You'll need to modify this based on your dataset loading
    # train_dataset = YourDatasetClass(args.data_dir, split='train')
    # val_dataset = YourDatasetClass(args.data_dir, split='val')
    
    # Apply augmentation
    # augmented_train = create_augmented_dataset(train_dataset, augmentation_factor=3)
    
    # Create data loaders
    # train_loader = DataLoader(augmented_train, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create improved model
    # vocab = train_dataset.vocab
    # model = create_improved_model(len(vocab)).to(device)
    
    # Create trainer with improvements
    # trainer = ImprovedHTRTrainer(model, train_loader, val_loader, converter, device)
    
    # Setup training with all improvements
    # trainer.setup_training(
    #     learning_rate=1e-3,
    #     weight_decay=1e-4,
    #     use_scheduler=True,
    #     scheduler_type='cosine'
    # )
    
    # Train with early stopping and checkpointing
    # history = trainer.train(
    #     num_epochs=args.epochs,
    #     save_dir=args.output_dir,
    #     save_frequency=5,
    #     early_stopping_patience=15
    # )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
