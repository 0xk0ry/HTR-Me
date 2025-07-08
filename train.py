"""
Training script for HTR model with CvT backbone
"""

from utils.sam import SAM
import numpy as np
from PIL import Image
from model.HTR_ME import HTRModel, CTCDecoder, train_epoch, validate
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import json
from pathlib import Path
import argparse
import sys
sys.path.append('.')


class HTRDataset(Dataset):
    """Dataset class for HTR training"""

    def __init__(self, data_dir, vocab, max_length=256, target_height=40):
        self.data_dir = Path(data_dir)
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.max_length = max_length
        self.target_height = target_height

        # Load dataset annotations
        self.samples = self._load_samples()

    def _load_samples(self):
        """Load image paths and corresponding texts"""
        samples = []

        # Look for annotation file
        annotation_file = self.data_dir / "annotations.json"
        if annotation_file.exists():
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
                for item in annotations:
                    samples.append({
                        'image_path': self.data_dir / item['image'],
                        'text': item['text']
                    })
        else:
            # Fallback: look for paired .jpg/.txt files
            for img_file in self.data_dir.glob("*.jpg"):
                txt_file = img_file.with_suffix('.txt')
                if txt_file.exists():
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    samples.append({
                        'image_path': img_file,
                        'text': text
                    })

        print(f"Loaded {len(samples)} samples from {self.data_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        # Resize to target height while maintaining aspect ratio
        w, h = image.size
        aspect_ratio = w / h
        new_width = int(self.target_height * aspect_ratio)
        image = image.resize((new_width, self.target_height), Image.LANCZOS)

        # Convert to tensor and normalize
        image_array = np.array(image) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        # Encode text
        text = sample['text']
        encoded_text = [self.char_to_idx.get(
            char, self.char_to_idx.get('<unk>', 1)) for char in text]
        encoded_text = encoded_text[:self.max_length]  # Truncate if too long

        return {
            'image': image_tensor,
            'text': encoded_text,
            'text_length': len(encoded_text),
            'original_text': text
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    text_lengths = [item['text_length'] for item in batch]

    # Pad images to same width
    max_width = max(img.size(2) for img in images)
    padded_images = []
    for img in images:
        padded = torch.zeros(3, img.size(1), max_width)
        padded[:, :, :img.size(2)] = img
        padded_images.append(padded)

    images = torch.stack(padded_images)

    # Pad texts to same length
    max_text_len = max(text_lengths)
    padded_texts = []
    for text in texts:
        padded = text + [0] * (max_text_len - len(text)
                               )  # Pad with blank token
        padded_texts.append(padded)

    texts = torch.tensor(padded_texts)
    text_lengths = torch.tensor(text_lengths)

    return images, texts, text_lengths


def create_vocab():
    """Create vocabulary for the model"""
    # Common characters for handwritten text recognition
    chars = (
        ' !\"#$%&\'()*+,-./'
        '0123456789'
        ':;<=>?@'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '[\\]^_`'
        'abcdefghijklmnopqrstuvwxyz'
        '{|}~'
    )

    vocab = ['<blank>', '<unk>'] + list(chars)
    return vocab


def main():
    parser = argparse.ArgumentParser(description='Train HTR model')
    parser.add_argument('--data_dir', type=str,
                        required=True, help='Path to training data')
    parser.add_argument('--val_data_dir', type=str,
                        help='Path to validation data')
    parser.add_argument('--output_dir', type=str,
                        default='./checkpoints', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--lm_path', type=str,
                        help='Path to KenLM language model')

    # SAM optimizer options
    parser.add_argument('--use_sam', action='store_true',
                        help='Use SAM (Sharpness-Aware Minimization) optimizer')
    parser.add_argument('--sam_rho', type=float, default=0.05,
                        help='SAM rho parameter (neighborhood size)')
    parser.add_argument('--sam_adaptive', action='store_true',
                        help='Use adaptive SAM')
    parser.add_argument('--base_optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd'],
                        help='Base optimizer for SAM')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create vocabulary
    vocab = create_vocab()
    print(f"Vocabulary size: {len(vocab)}")

    # Save vocabulary
    with open(os.path.join(args.output_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=2)

    # Create model
    model = HTRModel(
        vocab_size=len(vocab),
        max_length=256,
        target_height=40,
        chunk_width=256,
        stride=192,
        padding=32,
        embed_dims=[64, 192, 384],
        num_heads=[1, 3, 6],
        depths=[1, 2, 10]
    )

    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create decoder
    decoder = CTCDecoder(vocab, lm_path=args.lm_path)

    # Create datasets
    train_dataset = HTRDataset(args.data_dir, vocab)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    val_loader = None
    if args.val_data_dir:
        val_dataset = HTRDataset(args.val_data_dir, vocab)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )

    # Optimizer and scheduler
    if args.use_sam:
        print(f"Using SAM optimizer with {args.base_optimizer.upper()} base")
        print(f"SAM rho: {args.sam_rho}, adaptive: {args.sam_adaptive}")

        # Create base optimizer class
        base_optimizers = {
            'adamw': optim.AdamW,
            'adam': optim.Adam,
            'sgd': optim.SGD
        }
        base_optimizer_class = base_optimizers[args.base_optimizer]

        # Create SAM optimizer
        optimizer = SAM(
            model.parameters(),
            base_optimizer_class,
            lr=args.lr,
            weight_decay=0.01,
            rho=args.sam_rho,
            adaptive=args.sam_adaptive
        )
    else:
        print("Using standard AdamW optimizer")
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        print(f"\\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, vocab, use_sam=args.use_sam)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        if val_loader:
            val_loss, char_acc = validate(model, val_loader, device, decoder)
            print(f"Val Loss: {val_loss:.4f}, Char Accuracy: {char_acc:.4f}")
        else:
            val_loss = train_loss

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab': vocab
        }

        torch.save(checkpoint, os.path.join(
            args.output_dir, f'checkpoint_epoch_{epoch}.pth'))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(
                args.output_dir, 'best_model.pth'))
            print(f"New best model saved with val loss: {val_loss:.4f}")

        # Update learning rate
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")


if __name__ == "__main__":
    main()
