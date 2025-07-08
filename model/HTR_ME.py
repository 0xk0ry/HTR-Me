import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import CTCLoss
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import kenlm
    from pyctcdecode import build_ctcdecoder
    KENLM_AVAILABLE = True
except ImportError:
    KENLM_AVAILABLE = False
    print("Warning: KenLM and pyctcdecode not available. Install with: pip install kenlm pyctcdecode")

# CvT (Convolutional Vision Transformer) Implementation


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with convolution"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=None):
        super().__init__()
        if stride is None:
            stride = patch_size

        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H', W']
        _, _, H_new, W_new = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        x = self.norm(x)
        return x, (H_new, W_new)


class ConvolutionalAttention(nn.Module):
    """Convolutional Multi-head Attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., kernel_size=3):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Convolutional projection for positional encoding
        self.conv_proj_q = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.conv_proj_k = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.conv_proj_v = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        # Reshape for conv operations
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)

        # Apply convolutional projections
        q_conv = self.conv_proj_q(x_2d).flatten(2).transpose(1, 2)
        k_conv = self.conv_proj_k(x_2d).flatten(2).transpose(1, 2)
        v_conv = self.conv_proj_v(x_2d).flatten(2).transpose(1, 2)

        # Linear projections
        qkv = self.qkv(x + q_conv + k_conv + v_conv).reshape(B, N, 3,
                                                             self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CvTBlock(nn.Module):
    """CvT Transformer Block"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., kernel_size=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConvolutionalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop, proj_drop=drop, kernel_size=kernel_size)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class CvTStage(nn.Module):
    """CvT Stage with multiple blocks"""

    def __init__(self, patch_embed, blocks, norm=None):
        super().__init__()
        self.patch_embed = patch_embed
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm

    def forward(self, x):
        x, (H, W) = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, H, W)
        if self.norm is not None:
            x = self.norm(x)
        return x, (H, W)


class CvT(nn.Module):
    """Convolutional Vision Transformer - Simplified version for HTR"""

    def __init__(self, img_size=256, in_chans=3, num_classes=1000, embed_dims=[64, 192, 384],
                 num_heads=[1, 3, 6], depths=[1, 2, 10], patch_sizes=[7, 3, 3],
                 strides=[4, 2, 2], kernel_sizes=[3, 3, 3], mlp_ratios=[4, 4, 4],
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes

        # For HTR, we'll use a simplified single-stage CvT
        # First stage with patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_sizes[0],
            in_chans=in_chans,
            embed_dim=embed_dims[-1],  # Use final embedding dimension
            stride=strides[0]
        )

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(depths[-1]):  # Use final depth
            self.blocks.append(CvTBlock(
                embed_dims[-1],
                num_heads[-1],
                mlp_ratios[-1],
                qkv_bias,
                drop_rate,
                attn_drop_rate,
                kernel_sizes[-1]
            ))

        self.norm = nn.LayerNorm(embed_dims[-1])

        # Classification head (will be replaced in HTR model)
        self.head = nn.Linear(embed_dims[-1], num_classes)

    def forward(self, x):
        x, (H, W) = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, H, W)

        # Global average pooling
        x_pooled = x.mean(dim=1)
        x_cls = self.head(x_pooled)
        return x_cls

    def forward_features(self, x):
        """Return features without classification head"""
        x, (H, W) = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, H, W)
        x = self.norm(x)
        return x, (H, W)


class ImageChunker:
    """Handles image chunking with overlapping and padding"""

    def __init__(self, target_height=40, chunk_width=256, stride=192, padding=32):
        self.target_height = target_height
        self.chunk_width = chunk_width
        self.stride = stride
        self.padding = padding

    def preprocess_image(self, image):
        """Resize image to target height while preserving aspect ratio"""
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:  # PIL Image
            w, h = image.size

        # Calculate new width maintaining aspect ratio
        aspect_ratio = w / h
        new_width = int(self.target_height * aspect_ratio)

        if isinstance(image, np.ndarray):
            if CV2_AVAILABLE:
                image = cv2.resize(image, (new_width, self.target_height))
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Fallback to PIL
                image = Image.fromarray(image)
                image = image.resize(
                    (new_width, self.target_height), Image.LANCZOS)
        else:
            image = image.resize(
                (new_width, self.target_height), Image.LANCZOS)

        return image

    def create_chunks(self, image):
        """Create overlapping chunks with bidirectional padding"""
        # Convert to tensor if needed
        if isinstance(image, (np.ndarray, Image.Image)):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image_tensor = transform(image)
        else:
            image_tensor = image

        C, H, W = image_tensor.shape

        chunks = []
        chunk_positions = []

        # Calculate number of chunks needed
        if W <= self.chunk_width:
            # Single chunk case
            padded_chunk = F.pad(
                image_tensor, (self.padding, self.padding, 0, 0), mode='reflect')
            # Ensure consistent width
            target_width = self.chunk_width + 2 * self.padding
            if padded_chunk.size(2) < target_width:
                # Pad to target width, but be careful with reflect mode
                extra_pad = target_width - padded_chunk.size(2)
                max_pad = min(extra_pad, padded_chunk.size(2) - 1)
                if max_pad > 0:
                    padded_chunk = F.pad(
                        padded_chunk, (0, max_pad, 0, 0), mode='reflect')
                # If still not enough, pad with zeros
                if padded_chunk.size(2) < target_width:
                    remaining_pad = target_width - padded_chunk.size(2)
                    padded_chunk = F.pad(
                        padded_chunk, (0, remaining_pad, 0, 0), mode='constant', value=0)
            elif padded_chunk.size(2) > target_width:
                # Crop to target width
                padded_chunk = padded_chunk[:, :, :target_width]

            chunks.append(padded_chunk.unsqueeze(0))
            chunk_positions.append((0, W, self.padding, self.padding + W))
        else:
            # Multiple chunks case
            start = 0
            while start < W:
                end = min(start + self.chunk_width, W)

                # Extract chunk
                chunk = image_tensor[:, :, start:end]

                # Add bidirectional padding
                left_pad = self.padding if start > 0 else 0
                right_pad = self.padding if end < W else 0

                # Pad the chunk
                padded_chunk = F.pad(
                    chunk, (left_pad, right_pad, 0, 0), mode='reflect')

                # Ensure all chunks have the same width
                target_width = self.chunk_width + 2 * self.padding
                current_width = padded_chunk.size(2)

                if current_width < target_width:
                    # Pad to target width, but limit padding to available space
                    extra_pad = target_width - current_width
                    # Can't pad more than input size - 1
                    max_pad = min(extra_pad, current_width - 1)
                    if max_pad > 0:
                        padded_chunk = F.pad(
                            padded_chunk, (0, max_pad, 0, 0), mode='reflect')
                    # If still not enough, pad with zeros
                    if padded_chunk.size(2) < target_width:
                        remaining_pad = target_width - padded_chunk.size(2)
                        padded_chunk = F.pad(
                            padded_chunk, (0, remaining_pad, 0, 0), mode='constant', value=0)
                elif current_width > target_width:
                    # Crop to target width
                    padded_chunk = padded_chunk[:, :, :target_width]

                chunks.append(padded_chunk.unsqueeze(0))
                chunk_positions.append(
                    (start, end, left_pad, left_pad + (end - start)))

                if end >= W:
                    break
                start += self.stride

        return torch.cat(chunks, dim=0), chunk_positions


class HTRModel(nn.Module):
    """Handwritten Text Recognition Model with CvT backbone"""

    def __init__(self, vocab_size, max_length=256, target_height=40, chunk_width=256,
                 stride=192, padding=32, embed_dims=[64, 192, 384], num_heads=[1, 3, 6],
                 depths=[1, 2, 10]):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.chunker = ImageChunker(
            target_height, chunk_width, stride, padding)

        # CvT backbone for feature extraction
        self.cvt = CvT(
            img_size=chunk_width + 2*padding,  # Account for padding
            in_chans=3,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths,
            patch_sizes=[7, 3, 3],
            strides=[4, 2, 2],
            kernel_sizes=[3, 3, 3],
            mlp_ratios=[4, 4, 4],
            qkv_bias=True,
            drop_rate=0.1,
            attn_drop_rate=0.1
        )

        # MLP head for character prediction
        self.feature_dim = embed_dims[-1]
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim // 2, vocab_size)
        )

        # CTC Loss
        self.ctc_loss = CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    def forward(self, images, targets=None, target_lengths=None):
        """Forward pass through the model"""
        batch_size = images.size(0)
        all_logits = []
        all_lengths = []

        for i in range(batch_size):
            image = images[i]

            # Create chunks
            chunks, chunk_positions = self.chunker.create_chunks(image)

            # Process each chunk through CvT
            chunk_features = []
            for chunk in chunks:
                # chunk has shape [C, H, W], need to add batch dimension
                chunk = chunk.unsqueeze(0)  # [1, C, H, W]
                features, (H, W) = self.cvt.forward_features(chunk)
                chunk_features.append(features)

            # Merge chunks by removing padding and concatenating
            merged_features = self._merge_chunk_features(
                chunk_features, chunk_positions)

            # Apply classifier
            logits = self.classifier(merged_features)  # [seq_len, vocab_size]
            all_logits.append(logits)
            all_lengths.append(logits.size(0))

        # Pad sequences to same length
        max_seq_len = max(all_lengths)
        padded_logits = torch.zeros(
            batch_size, max_seq_len, self.vocab_size, device=images.device)

        for i, (logits, length) in enumerate(zip(all_logits, all_lengths)):
            padded_logits[i, :length] = logits

        # Transpose for CTC: [seq_len, batch_size, vocab_size]
        padded_logits = padded_logits.transpose(0, 1)

        if self.training and targets is not None:
            # Calculate CTC loss
            input_lengths = torch.tensor(all_lengths, device=images.device)
            loss = self.ctc_loss(F.log_softmax(
                padded_logits, dim=-1), targets, input_lengths, target_lengths)
            return padded_logits, loss
        else:
            return padded_logits, torch.tensor(all_lengths, device=images.device)

    def _merge_chunk_features(self, chunk_features, chunk_positions):
        """Merge features from multiple chunks, removing padded regions"""
        merged_features = []

        for i, (features, (start, end, left_pad, right_boundary)) in enumerate(zip(chunk_features, chunk_positions)):
            # features shape: [1, seq_len, feature_dim]
            features = features.squeeze(0)  # [seq_len, feature_dim]

            # Calculate which tokens correspond to valid (non-padded) regions
            # This depends on the CvT's downsampling factor
            total_width = features.size(0)

            # Approximate mapping from feature positions to image positions
            # This is a simplified approach - in practice, you might need to be more precise
            if len(chunk_positions) == 1:
                # Single chunk - remove padding from both sides
                start_idx = max(0, left_pad * total_width //
                                (right_boundary + self.chunker.padding - left_pad))
                end_idx = min(total_width, (left_pad + (end - start)) * total_width //
                              (right_boundary + self.chunker.padding - left_pad))
                valid_features = features[start_idx:end_idx]
            else:
                if i == 0:
                    # First chunk - remove right padding
                    end_idx = total_width - \
                        (self.chunker.padding * total_width //
                         (self.chunker.chunk_width + self.chunker.padding))
                    valid_features = features[:end_idx]
                elif i == len(chunk_positions) - 1:
                    # Last chunk - remove left padding
                    start_idx = left_pad * \
                        total_width // (right_boundary +
                                        self.chunker.padding - left_pad)
                    valid_features = features[start_idx:]
                else:
                    # Middle chunk - remove both paddings
                    start_idx = left_pad * \
                        total_width // (right_boundary +
                                        self.chunker.padding - left_pad)
                    end_idx = total_width - \
                        (self.chunker.padding * total_width //
                         (right_boundary + self.chunker.padding - left_pad))
                    valid_features = features[start_idx:end_idx]

            merged_features.append(valid_features)

        # Concatenate all valid features
        if merged_features:
            return torch.cat(merged_features, dim=0)
        else:
            return torch.empty(0, chunk_features[0].size(-1), device=chunk_features[0].device)


class CTCDecoder:
    """CTC Decoder with KenLM language model support"""

    def __init__(self, vocab, lm_path=None, alpha=0.5, beta=1.0):
        self.vocab = vocab
        self.blank_id = 0
        self.alpha = alpha
        self.beta = beta

        if KENLM_AVAILABLE and lm_path is not None:
            try:
                self.decoder = build_ctcdecoder(
                    labels=vocab,
                    kenlm_model_path=lm_path,
                    alpha=alpha,
                    beta=beta
                )
                self.use_lm = True
            except Exception as e:
                print(f"Failed to load language model: {e}")
                self.use_lm = False
        else:
            self.use_lm = False

    def greedy_decode(self, logits):
        """Simple greedy CTC decoding"""
        # logits: [seq_len, vocab_size]
        predictions = torch.argmax(logits, dim=-1)

        # Remove blanks and consecutive duplicates
        decoded = []
        prev = -1
        for pred in predictions:
            if pred != self.blank_id and pred != prev:
                decoded.append(pred.item())
            prev = pred

        return decoded

    def beam_search_decode(self, logits, beam_width=100):
        """Beam search decoding with optional language model"""
        if self.use_lm:
            # Use pyctcdecode with language model
            logits_np = F.softmax(logits, dim=-1).cpu().numpy()
            text = self.decoder.decode(logits_np, beam_width=beam_width)
            return text
        else:
            # Simple beam search without language model
            return self._simple_beam_search(logits, beam_width)

    def _simple_beam_search(self, logits, beam_width):
        """Simple beam search without language model"""
        seq_len, vocab_size = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)

        # Initialize beam
        beams = [([], 0.0)]  # (sequence, log_prob)

        for t in range(seq_len):
            new_beams = []

            for sequence, log_prob in beams:
                for c in range(vocab_size):
                    new_log_prob = log_prob + log_probs[t, c].item()

                    if c == self.blank_id:
                        # Blank token - don't extend sequence
                        new_sequence = sequence
                    elif len(sequence) > 0 and sequence[-1] == c:
                        # Same character as previous - don't extend
                        new_sequence = sequence
                    else:
                        # New character
                        new_sequence = sequence + [c]

                    new_beams.append((new_sequence, new_log_prob))

            # Keep top beam_width beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        # Return best sequence
        best_sequence = beams[0][0]
        return ''.join([self.vocab[i] for i in best_sequence if i < len(self.vocab)])

# Training utilities


def train_epoch(model, dataloader, optimizer, device, vocab, use_sam=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        images, targets, target_lengths = batch
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

        total_loss += loss.item()
        num_batches += 1

        if num_batches % 100 == 0:
            print(f"Batch {num_batches}, Loss: {loss.item():.4f}")

    return total_loss / num_batches


def validate(model, dataloader, device, decoder):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    correct_chars = 0
    total_chars = 0

    with torch.no_grad():
        for batch in dataloader:
            images, targets, target_lengths = batch
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits, loss = model(images, targets, target_lengths)
            total_loss += loss.item()
            num_batches += 1

            # Decode predictions
            for i in range(logits.size(1)):  # batch dimension
                pred_logits = logits[:, i, :]  # [seq_len, vocab_size]
                predicted = decoder.greedy_decode(pred_logits)

                # Compare with ground truth (simplified)
                target_seq = targets[i][:target_lengths[i]].cpu().numpy()

                # Character-level accuracy (simplified)
                min_len = min(len(predicted), len(target_seq))
                correct_chars += sum(1 for j in range(min_len)
                                     if predicted[j] == target_seq[j])
                total_chars += max(len(predicted), len(target_seq))

    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    return total_loss / num_batches, char_accuracy

# Example usage and inference


def inference_example(model, image_path, decoder, device):
    """Example inference on a single image"""
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = model.chunker.preprocess_image(image)

    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, lengths = model(image_tensor)

        # Decode
        pred_logits = logits[:lengths[0], 0, :]  # [seq_len, vocab_size]

        # Greedy decoding
        greedy_result = decoder.greedy_decode(pred_logits)
        greedy_text = ''.join([decoder.vocab[i]
                              for i in greedy_result if i < len(decoder.vocab)])

        # Beam search decoding
        beam_result = decoder.beam_search_decode(pred_logits, beam_width=100)

        return greedy_text, beam_result

# Example of how to initialize and use the model


def create_model_example():
    """Example of how to create and initialize the model"""

    # Define vocabulary (example)
    vocab = ['<blank>'] + \
        list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;: ')
    vocab_size = len(vocab)

    # Create model
    model = HTRModel(
        vocab_size=vocab_size,
        max_length=256,
        target_height=40,
        chunk_width=256,
        stride=192,
        padding=32,
        embed_dims=[64, 192, 384],
        num_heads=[1, 3, 6],
        depths=[1, 2, 10]
    )

    # Create decoder
    decoder = CTCDecoder(vocab, lm_path=None)  # Set lm_path for KenLM

    return model, decoder, vocab


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, decoder, vocab = create_model_example()
    model.to(device)

    print(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Vocabulary size: {len(vocab)}")

    # Example forward pass
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 40, 512).to(device)

    with torch.no_grad():
        logits, lengths = model(dummy_images)
        print(f"Output shape: {logits.shape}")
        print(f"Sequence lengths: {lengths}")
