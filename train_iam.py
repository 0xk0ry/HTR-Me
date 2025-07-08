"""
Training script for HTR model with IAM dataset
Custom dataloader for image.png + image.txt pairs
"""

from utils.sam import SAM
from model.HTR_ME import HTRModel, CTCDecoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pickle
import json
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class IAMDataset(Dataset):
    """
    Custom dataset for IAM format with image.png and image.txt pairs
    Supports train/valid/test splits
    """

    def __init__(self, data_dir, split='train', transform=None, target_height=40):
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_height = target_height

        # Load vocabulary
        self.vocab, self.char_to_idx = self._load_vocabulary()
        print(f"Loaded vocabulary with {len(self.vocab)} characters")

        # Get all image files for the split
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples for {split} split")

        # Set up transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def _load_vocabulary(self):
        """Load vocabulary from labels.pkl or create default"""
        labels_path = self.data_dir / 'labels.pkl'

        if labels_path.exists():
            try:
                with open(labels_path, 'rb') as f:
                    data = pickle.load(f)
                    charset = data.get('charset', [])
                    vocab = ['<blank>'] + sorted(charset)
                    print(f"Loaded vocabulary from {labels_path}")
            except Exception as e:
                print(f"Error loading {labels_path}: {e}")
                vocab = self._create_default_vocab()
        else:
            vocab = self._create_default_vocab()

        char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        return vocab, char_to_idx

    def _create_default_vocab(self):
        """Create default vocabulary"""
        print("Creating default vocabulary...")
        # Basic English characters, digits, and punctuation
        chars = (
            ' !"#$%&\'()*+,-./'
            '0123456789'
            ':;<=>?@'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '[\\]^_`'
            'abcdefghijklmnopqrstuvwxyz'
            '{|}~'
        )
        return ['<blank>'] + list(chars)

    def _load_samples(self):
        """Load all samples for the specified split"""
        samples = []

        # Find all image files for this split
        pattern = f"{self.split}_*.png"
        image_files = list(self.data_dir.glob(pattern))

        for img_path in image_files:
            # Check if corresponding text file exists
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                samples.append({
                    'image_path': img_path,
                    'text_path': txt_path,
                    'image_name': img_path.name
                })
            else:
                print(f"Warning: No text file found for {img_path}")

        return sorted(samples, key=lambda x: x['image_name'])

    def _preprocess_image(self, image):
        """Preprocess image to target height while preserving aspect ratio"""
        w, h = image.size

        # Calculate new width maintaining aspect ratio
        aspect_ratio = w / h
        new_width = int(self.target_height * aspect_ratio)

        # Resize image
        image = image.resize((new_width, self.target_height), Image.LANCZOS)
        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            image = self._preprocess_image(image)
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (100, self.target_height), color='white')

        # Load text
        try:
            with open(sample['text_path'], 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception as e:
            print(f"Error loading text {sample['text_path']}: {e}")
            text = ""

        # Convert text to indices
        text_indices = []
        for char in text:
            if char in self.char_to_idx:
                text_indices.append(self.char_to_idx[char])
            else:
                # Handle unknown characters
                text_indices.append(self.char_to_idx.get('<blank>', 0))

        # Apply image transforms
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'text_indices': text_indices,
            'text': text,
            'text_length': len(text_indices),
            'image_name': sample['image_name']
        }


def collate_fn(batch):
    """
    Custom collate function for variable length sequences
    Handles batching of images and text sequences
    """
    images = []
    text_indices = []
    text_lengths = []
    texts = []
    image_names = []

    for item in batch:
        images.append(item['image'])
        text_indices.append(item['text_indices'])
        text_lengths.append(item['text_length'])
        texts.append(item['text'])
        image_names.append(item['image_name'])

    # Pad images to same width
    max_width = max(img.size(2) for img in images)
    padded_images = []

    for img in images:
        c, h, w = img.shape
        padded = torch.zeros(c, h, max_width)
        padded[:, :, :w] = img
        padded_images.append(padded)

    images_tensor = torch.stack(padded_images, 0)

    # Pad text sequences to same length
    if text_lengths:
        max_text_len = max(text_lengths)
        padded_texts = torch.zeros(
            len(text_indices), max_text_len, dtype=torch.long)

        for i, text in enumerate(text_indices):
            if len(text) > 0:
                padded_texts[i, :len(text)] = torch.tensor(
                    text, dtype=torch.long)
    else:
        padded_texts = torch.zeros(len(text_indices), 1, dtype=torch.long)

    text_lengths_tensor = torch.tensor(text_lengths, dtype=torch.long)

    return images_tensor, padded_texts, text_lengths_tensor, texts, image_names


def train_epoch(model, dataloader, optimizer, device, use_sam=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        images, targets, target_lengths, texts, image_names = batch
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        if use_sam:
            # SAM training step
            def closure():
                optimizer.zero_grad()
                logits, loss = model(images, targets, target_lengths)
                loss.backward()
                return loss

            # First forward-backward pass
            loss = closure()
            optimizer.first_step(zero_grad=True)

            # Second forward-backward pass
            closure()
            optimizer.second_step(zero_grad=True)
        else:
            # Standard training step
            optimizer.zero_grad()
            logits, loss = model(images, targets, target_lengths)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # Ensure loss is floating point for item()
        if not loss.dtype.is_floating_point:
            loss = loss.float()
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def validate(model, dataloader, device, decoder):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    correct_chars = 0
    total_chars = 0

    progress_bar = tqdm(dataloader, desc="Validation")

    with torch.no_grad():
        for batch in progress_bar:
            images, targets, target_lengths, texts, image_names = batch
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits, loss = model(images, targets, target_lengths)
            # Ensure loss is scalar and floating point
            if loss.numel() > 1:
                loss = loss.float().mean()
            elif not loss.dtype.is_floating_point:
                loss = loss.float()
            total_loss += loss.item()
            num_batches += 1

            # Calculate character accuracy
            for i in range(logits.size(1)):  # batch dimension
                pred_logits = logits[:, i, :]  # [seq_len, vocab_size]
                predicted = decoder.greedy_decode(pred_logits)

                # Get ground truth
                target_seq = targets[i][:target_lengths[i]].cpu().numpy()

                # Character-level accuracy
                min_len = min(len(predicted), len(target_seq))
                if min_len > 0:
                    correct_chars += sum(1 for j in range(min_len)
                                         if predicted[j] == target_seq[j])
                total_chars += max(len(predicted), len(target_seq))

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    return total_loss / num_batches, char_accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Train HTR model on IAM dataset')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to IAM lines directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_iam',
                        help='Output directory for checkpoints')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')

    # Model arguments
    parser.add_argument('--target_height', type=int, default=40,
                        help='Target image height')
    parser.add_argument('--chunk_width', type=int, default=256,
                        help='Chunk width for processing')
    parser.add_argument('--stride', type=int, default=192,
                        help='Stride for chunking')
    parser.add_argument('--padding', type=int, default=32,
                        help='Padding for chunks')

    # Optimizer arguments
    parser.add_argument('--use_sam', action='store_true',
                        help='Use SAM optimizer')
    parser.add_argument('--sam_rho', type=float, default=0.05,
                        help='SAM rho parameter')
    parser.add_argument('--base_optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd'],
                        help='Base optimizer for SAM')

    # Other arguments
    parser.add_argument('--resume', type=str,
                        help='Resume training from checkpoint')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    print("Creating datasets...")
    train_dataset = IAMDataset(
        data_dir=args.data_dir,
        split='train',
        target_height=args.target_height
    )

    valid_dataset = IAMDataset(
        data_dir=args.data_dir,
        split='valid',
        target_height=args.target_height
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Vocabulary size: {len(train_dataset.vocab)}")

    # Save vocabulary
    vocab_path = os.path.join(args.output_dir, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(train_dataset.vocab, f, indent=2)
    print(f"Vocabulary saved to {vocab_path}")

    # Create model
    model = HTRModel(
        vocab_size=len(train_dataset.vocab),
        max_length=256,
        target_height=args.target_height,
        chunk_width=args.chunk_width,
        stride=args.stride,
        padding=args.padding,
        embed_dims=[64, 192, 384],
        num_heads=[1, 3, 6],
        depths=[1, 2, 10]
    )

    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Create optimizer
    if args.use_sam:
        print(f"Using SAM optimizer with {args.base_optimizer} base")
        base_optimizers = {
            'adamw': optim.AdamW,
            'adam': optim.Adam,
            'sgd': optim.SGD
        }
        base_optimizer_class = base_optimizers[args.base_optimizer]

        optimizer = SAM(
            model.parameters(),
            base_optimizer_class,
            lr=args.lr,
            weight_decay=args.weight_decay,
            rho=args.sam_rho
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Create decoder
    decoder = CTCDecoder(train_dataset.vocab)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, use_sam=args.use_sam)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_accuracy = validate(model, valid_loader, device, decoder)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr:.6f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'best_val_loss': best_val_loss,
            'vocab': train_dataset.vocab,
            'args': vars(args)
        }

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, os.path.join(
                args.output_dir, 'best_model.pth'))
            print(f"🎉 New best model saved! Val Loss: {val_loss:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint, os.path.join(
                args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"Checkpoint saved at epoch {epoch+1}")

    print("\n🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
