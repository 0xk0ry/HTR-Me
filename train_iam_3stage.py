"""
Training script for 3-Stage HTR model with IAM dataset
Custom dataloader for image.png + image.txt pairs
"""

from model.HTR_ME_3Stage import HTRModel, CTCDecoder
from utils.sam import SAM
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Constants
DEFAULT_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DEFAULT_CHARSET = (
    ' !"#$%&\'()*+,-./'
    '0123456789'
    ':;<=>?@'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '[\\]^_`'
    'abcdefghijklmnopqrstuvwxyz'
    '{|}~'
)

OPTIMIZER_REGISTRY = {
    'adamw': optim.AdamW,
    'adam': optim.Adam,
    'sgd': optim.SGD
}


class IAMDataset(Dataset):
    """Custom dataset for IAM format with image.png and image.txt pairs"""

    def __init__(self, data_dir, split='train', transform=None, target_height=40):
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_height = target_height
        self.transform = transform or DEFAULT_TRANSFORMS

        # Load vocabulary and samples
        self.vocab, self.char_to_idx = self._load_vocabulary()
        self.samples = self._load_samples()

        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"Vocabulary size: {len(self.vocab)} characters")

    def _load_vocabulary(self):
        """Load vocabulary from labels.pkl or create default"""
        labels_path = self.data_dir / 'labels.pkl'

        vocab = self._try_load_from_pickle(
            labels_path) or self._create_default_vocab()
        char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        return vocab, char_to_idx

    def _try_load_from_pickle(self, labels_path):
        """Try to load vocabulary from pickle file"""
        if not labels_path.exists():
            return None

        try:
            with open(labels_path, 'rb') as f:
                data = pickle.load(f)
                charset = data.get('charset', [])
                if charset:
                    print(f"Loaded vocabulary from {labels_path}")
                    return ['<blank>'] + sorted(charset)
        except Exception as e:
            print(f"Error loading {labels_path}: {e}")
        return None

    def _create_default_vocab(self):
        """Create default vocabulary"""
        print("Creating default vocabulary...")
        return ['<blank>'] + list(DEFAULT_CHARSET)

    def _load_samples(self):
        """Load all samples for the specified split"""
        pattern = f"{self.split}_*.png"
        image_files = list(self.data_dir.glob(pattern))

        samples = []
        for img_path in image_files:
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
        """Resize image to target height while preserving aspect ratio"""
        w, h = image.size
        aspect_ratio = w / h
        new_width = int(self.target_height * aspect_ratio)
        return image.resize((new_width, self.target_height), Image.LANCZOS)

    def _load_image(self, image_path):
        """Load and preprocess image with error handling"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self._preprocess_image(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return Image.new('RGB', (100, self.target_height), color='white')

    def _load_text(self, text_path):
        """Load text with error handling"""
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading text {text_path}: {e}")
            return ""

    def _text_to_indices(self, text):
        """Convert text to character indices"""
        return [self.char_to_idx.get(char, 0) for char in text]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and process data
        image = self._load_image(sample['image_path'])
        text = self._load_text(sample['text_path'])
        text_indices = self._text_to_indices(text)

        # Apply transforms
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
    """Custom collate function for variable length sequences"""
    # Extract batch components
    images = [item['image'] for item in batch]
    text_indices = [item['text_indices'] for item in batch]
    text_lengths = [len(item['text_indices']) for item in batch]
    texts = [item['text'] for item in batch]
    image_names = [item['image_name'] for item in batch]

    # Pad images to same width
    images_tensor = _pad_images(images)

    # Create concatenated targets for CTC
    targets_tensor, text_lengths_tensor = _prepare_ctc_targets(
        text_indices, text_lengths)

    return images_tensor, targets_tensor, text_lengths_tensor, texts, image_names


def _pad_images(images):
    """Pad images to the same width"""
    max_width = max(img.size(2) for img in images)
    padded_images = []

    for img in images:
        c, h, w = img.shape
        padded = torch.zeros(c, h, max_width)
        padded[:, :, :w] = img
        padded_images.append(padded)

    return torch.stack(padded_images, 0)


def _prepare_ctc_targets(text_indices, text_lengths):
    """Prepare targets for CTC loss"""
    # Concatenate all text sequences for CTC
    concatenated_targets = []
    for text in text_indices:
        concatenated_targets.extend(text)

    targets_tensor = torch.tensor(concatenated_targets, dtype=torch.long)
    text_lengths_tensor = torch.tensor(text_lengths, dtype=torch.long)

    # Validate consistency
    total_expected = sum(text_lengths)
    actual_size = len(concatenated_targets)
    if total_expected != actual_size:
        raise ValueError(
            f"CTC target length mismatch: {total_expected} != {actual_size}")

    return targets_tensor, text_lengths_tensor


def train_epoch(model, dataloader, optimizer, device, use_sam=False):
    """Train for one epoch with improved error handling"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        try:
            loss = _process_training_batch(
                batch, model, optimizer, device, use_sam)

            if not _is_valid_loss(loss):
                print(
                    f"Warning: Invalid loss {loss.item()}, skipping batch {batch_idx}")
                continue

            total_loss += loss.item()
            num_batches += 1

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")

        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue

    return total_loss / num_batches if num_batches > 0 else float('inf')


def _process_training_batch(batch, model, optimizer, device, use_sam):
    """Process a single training batch"""
    images, targets, target_lengths, texts, image_names = batch
    images, targets, target_lengths = _move_to_device(
        images, targets, target_lengths, device)

    if use_sam:
        return _sam_training_step(model, optimizer, images, targets, target_lengths)
    else:
        return _standard_training_step(model, optimizer, images, targets, target_lengths)


def _move_to_device(images, targets, target_lengths, device):
    """Move tensors to device"""
    return (
        images.to(device),
        targets.to(device),
        target_lengths.to(device)
    )


def _sam_training_step(model, optimizer, images, targets, target_lengths):
    """SAM training step"""
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
    return loss


def _standard_training_step(model, optimizer, images, targets, target_lengths):
    """Standard training step"""
    optimizer.zero_grad()
    logits, loss = model(images, targets, target_lengths)
    
    # Check for problematic loss values
    if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0.0:
        print(f"Warning: Problematic loss detected: {loss.item()}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Target lengths: {target_lengths}")
        print(f"  Input lengths computed in model")
        # Skip this batch
        return torch.tensor(0.0, device=loss.device, requires_grad=True)
    
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss


def _is_valid_loss(loss):
    """Check if loss is valid (not NaN or Inf)"""
    if not loss.dtype.is_floating_point:
        loss = loss.float()
    return not (torch.isnan(loss) or torch.isinf(loss))


def validate(model, dataloader, device, decoder):
    """Validate the model with improved accuracy calculation"""
    model.eval()
    total_loss = 0
    num_batches = 0
    correct_chars = 0
    total_chars = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                images, targets, target_lengths, texts, image_names = batch
                images, targets, target_lengths = _move_to_device(
                    images, targets, target_lengths, device)

                # Get model output (inference mode)
                logits, input_lengths = model(images)

                # Calculate CTC loss
                loss = _calculate_ctc_loss(
                    logits, targets, input_lengths, target_lengths)

                if not _is_valid_loss(loss):
                    continue

                total_loss += loss.item()
                num_batches += 1

                # Calculate character accuracy
                batch_correct, batch_total = _calculate_batch_accuracy(
                    logits, targets, target_lengths, input_lengths, decoder
                )
                correct_chars += batch_correct
                total_chars += batch_total

            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue

    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss, char_accuracy


def _calculate_ctc_loss(logits, targets, input_lengths, target_lengths):
    """Calculate CTC loss for validation"""
    log_probs = F.log_softmax(logits, dim=-1)
    loss = F.ctc_loss(
        log_probs, targets, input_lengths, target_lengths,
        blank=0, reduction='mean', zero_infinity=True
    )

    # Ensure loss is scalar and floating point
    if loss.numel() > 1:
        loss = loss.float().mean()
    elif not loss.dtype.is_floating_point:
        loss = loss.float()

    return loss


def _calculate_batch_accuracy(logits, targets, target_lengths, input_lengths, decoder):
    """Calculate character accuracy for a batch"""
    correct_chars = 0
    total_chars = 0
    target_start = 0
    batch_size = logits.size(1)

    for i in range(batch_size):
        if i >= len(input_lengths):
            break

        # Get prediction
        seq_len = input_lengths[i].item()
        pred_logits = logits[:seq_len, i, :]  # [seq_len, vocab_size]
        predicted = decoder.greedy_decode(pred_logits)

        # Get ground truth
        target_length = target_lengths[i].item()
        target_end = target_start + target_length
        target_seq = targets[target_start:target_end].cpu().numpy()
        target_start = target_end

        # Calculate accuracy
        min_len = min(len(predicted), len(target_seq))
        if min_len > 0:
            correct_chars += sum(1 for j in range(min_len)
                                 if predicted[j] == target_seq[j])
        total_chars += max(len(predicted), len(target_seq))

    return correct_chars, total_chars


def create_datasets(args):
    """Create training and validation datasets"""
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

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Vocabulary size: {len(train_dataset.vocab)}")

    return train_dataset, valid_dataset


def create_dataloaders(train_dataset, valid_dataset, args):
    """Create data loaders"""
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

    return train_loader, valid_loader


def create_model(vocab_size, args):
    """Create 3-Stage HTR model"""
    model = HTRModel(
        vocab_size=vocab_size,
        max_length=256,
        target_height=args.target_height,
        chunk_width=args.chunk_width,
        stride=args.stride,
        embed_dims=args.embed_dims,
        num_heads=args.num_heads,
        depths=args.depths
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"3-Stage Model parameters: {param_count:,}")
    print(f"Model configuration:")
    print(f"  - Embed dims: {args.embed_dims}")
    print(f"  - Num heads: {args.num_heads}")
    print(f"  - Depths: {args.depths}")
    print(f"  - Target height: {args.target_height}")
    print(f"  - Chunk width: {args.chunk_width}")
    print(f"  - Stride: {args.stride}")
    
    return model


def create_optimizer(model, args):
    """Create optimizer"""
    if args.use_sam:
        print(f"Using SAM optimizer with {args.base_optimizer} base")
        base_optimizer_class = OPTIMIZER_REGISTRY[args.base_optimizer]

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

    return optimizer


def save_vocabulary(vocab, output_dir):
    """Save vocabulary to JSON file"""
    vocab_path = os.path.join(output_dir, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary saved to {vocab_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load checkpoint and return start epoch and best validation loss"""
    print(f"Resuming from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    return start_epoch, best_val_loss


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_accuracy,
                    best_val_loss, vocab, args, output_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'best_val_loss': best_val_loss,
        'vocab': vocab,
        'args': vars(args),
        'model_type': '3-stage'
    }

    if is_best:
        torch.save(checkpoint, os.path.join(output_dir, 'best_model_3stage.pth'))
        print(f"🎉 New best 3-stage model saved! Val Loss: {val_loss:.4f}")

    # Save periodic checkpoint
    if (epoch + 1) % args.save_every == 0:
        torch.save(checkpoint, os.path.join(
            output_dir, f'checkpoint_3stage_epoch_{epoch+1}.pth'))
        print(f"3-stage checkpoint saved at epoch {epoch+1}")


def training_loop(model, train_loader, valid_loader, optimizer, decoder, device, args, vocab):
    """Main training loop"""
    start_epoch = 0
    best_val_loss = float('inf')

    # Resume from checkpoint if specified
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, args.resume, device)

    print(f"Starting 3-stage model training for {args.epochs} epochs...")

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
        print(f"Learning Rate: {args.lr:.6f}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(
            model, optimizer, epoch, train_loss, val_loss, val_accuracy,
            best_val_loss, vocab, args, args.output_dir, is_best
        )

    return best_val_loss


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description='Train 3-Stage HTR model on IAM dataset')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to IAM lines directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_iam_3stage',
                        help='Output directory for checkpoints')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.01, help='Weight decay')

    # Model arguments
    parser.add_argument('--target_height', type=int,
                        default=40, help='Target image height')
    parser.add_argument('--chunk_width', type=int,
                        default=320, help='Chunk width for processing')
    parser.add_argument('--stride', type=int, default=240,
                        help='Stride for chunking')
    
    # 3-Stage specific arguments
    parser.add_argument('--embed_dims', type=int, nargs=3, default=[64, 192, 384],
                        help='Embedding dimensions for each stage')
    parser.add_argument('--num_heads', type=int, nargs=3, default=[1, 3, 6],
                        help='Number of attention heads for each stage')
    parser.add_argument('--depths', type=int, nargs=3, default=[1, 2, 10],
                        help='Number of blocks for each stage')

    # Optimizer arguments
    parser.add_argument('--use_sam', action='store_true',
                        help='Use SAM optimizer')
    parser.add_argument('--sam_rho', type=float,
                        default=0.05, help='SAM rho parameter')
    parser.add_argument('--base_optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd'], help='Base optimizer for SAM')

    # Other arguments
    parser.add_argument('--resume', type=str,
                        help='Resume training from checkpoint')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training 3-Stage CvT model")

    # Create datasets and data loaders
    train_dataset, valid_dataset = create_datasets(args)
    train_loader, valid_loader = create_dataloaders(
        train_dataset, valid_dataset, args)

    # Save vocabulary
    save_vocabulary(train_dataset.vocab, args.output_dir)

    # Create model, optimizer, and decoder
    model = create_model(len(train_dataset.vocab), args)
    model.to(device)

    optimizer = create_optimizer(model, args)
    decoder = CTCDecoder(train_dataset.vocab)

    # Training
    best_val_loss = training_loop(
        model, train_loader, valid_loader, optimizer, decoder,
        device, args, train_dataset.vocab
    )

    print("\n🎉 3-Stage model training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
