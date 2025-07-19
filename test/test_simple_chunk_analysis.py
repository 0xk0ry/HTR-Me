"""
Simple test script to analyze HTR model chunking process without visualization dependencies.
This script shows all the technical details about how images are processed.
"""

from model.HTR_ME import HTRModel, ImageChunker
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import torch.nn.functional as F
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Import your model


def create_simple_test_image(width=800, height=40):
    """Create a simple test image with gradient pattern"""
    # Create a gradient image for testing
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Create horizontal gradient
    for x in range(width):
        intensity = int(255 * (x / width))
        img_array[:, x, :] = [intensity, intensity, intensity]

    # Add some vertical stripes for pattern
    for x in range(0, width, 50):
        img_array[:, x:x+5, :] = [0, 0, 0]  # Black stripes

    return Image.fromarray(img_array)


def analyze_chunking_process():
    """Comprehensive analysis of the chunking process"""

    print("=== HTR Model Chunking Analysis ===\n")

    # Initialize model
    vocab_size = 100
    model = HTRModel(vocab_size=vocab_size)
    chunker = model.chunker

    print("Model Configuration:")
    print(f"  Target height: {chunker.target_height}px")
    print(f"  Chunk width: {chunker.chunk_width}px")
    print(f"  Stride: {chunker.stride}px")
    print(f"  Padding: {chunker.padding}px")
    print(f"  Overlap size: {chunker.chunk_width - chunker.stride}px")

    # CvT Configuration
    patch_stride = model.cvt.patch_embed.stride
    patch_size = model.cvt.patch_embed.patch_size
    embed_dim = model.feature_dim

    print(f"\nCvT Configuration:")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Patch stride: {patch_stride}")
    print(f"  Embedding dimension: {embed_dim}")

    # Test different image widths
    test_widths = [200, 400, 600, 800, 1000]

    for width in test_widths:
        print(f"\n{'='*60}")
        print(f"TESTING IMAGE WIDTH: {width}px")
        print(f"{'='*60}")

        # Create test image
        test_image = create_simple_test_image(width=width, height=40)
        print(f"1. Original image: {test_image.size} (W x H)")

        # Preprocess
        processed_image = chunker.preprocess_image(test_image)
        print(f"2. After preprocessing: {processed_image.size}")

        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        image_tensor = transform(processed_image)
        print(f"3. Tensor shape: {image_tensor.shape} (C x H x W)")

        # Create chunks
        chunks, chunk_positions = chunker.create_chunks(image_tensor)
        print(f"4. Number of chunks: {len(chunks)}")
        print(f"5. Each chunk shape: {chunks[0].shape}")

        # Analyze each chunk
        print(f"\n6. Chunk Analysis:")
        chunk_features = []

        for i, (chunk, (start_px, end_px, left_pad_px, content_end_px)) in enumerate(zip(chunks, chunk_positions)):
            print(f"\n   Chunk {i}:")
            print(f"   ├─ Pixel range: {start_px} to {end_px}")
            print(f"   ├─ Left padding: {left_pad_px}px")
            print(f"   ├─ Content ends at: {content_end_px}px in chunk")
            print(f"   ├─ Chunk shape: {chunk.shape}")

            # Process through CvT
            chunk_with_batch = chunk.unsqueeze(0)
            features = model.forward_features(chunk_with_batch)
            chunk_features.append(features)

            # Calculate spatial dimensions
            C, H, W = chunk.shape
            H_prime = (H - patch_size) // patch_stride + 1
            W_prime = (W - patch_size) // patch_stride + 1

            print(
                f"   ├─ Spatial patches: {H_prime} x {W_prime} = {H_prime * W_prime}")
            print(f"   ├─ After height pooling: {W_prime} time steps")
            print(f"   └─ Features shape: {features.shape}")

        # Analyze merging process
        print(f"\n7. Merging Analysis:")

        ignored_size = (chunker.chunk_width - chunker.stride) / 2
        ignore_patches = max(1, ignored_size // patch_stride)

        print(
            f"   Ignored region: {ignored_size}px = {ignore_patches} patches")

        total_valid_patches = 0

        for i, (features, (start_px, end_px, left_pad_px, content_end_px)) in enumerate(zip(chunk_features, chunk_positions)):
            total_patches = features.size(0)
            print(f"\n   Chunk {i} merging:")
            print(f"   ├─ Total patches: {total_patches}")

            # Calculate valid region (same logic as in model)
            if len(chunk_positions) == 1:
                # Single chunk
                start_idx = 0
                end_idx = total_patches

                if left_pad_px > 0:
                    padding_patches = left_pad_px // patch_stride
                    padding_patches = min(padding_patches, total_patches - 1)
                    start_idx = padding_patches

                actual_content_width_px = end_px - start_px
                if actual_content_width_px < chunker.chunk_width - left_pad_px:
                    content_ratio = (
                        left_pad_px + actual_content_width_px) / chunker.chunk_width
                    actual_content_patches = max(
                        1, int(content_ratio * total_patches))
                    end_idx = min(actual_content_patches, total_patches)

                print(f"   ├─ Single chunk mode")
                print(f"   ├─ Valid range: [{start_idx}:{end_idx}]")
                print(f"   ├─ Left padding removed: {start_idx}")
                print(
                    f"   └─ Right padding removed: {total_patches - end_idx}")

            else:
                if i == 0:  # First chunk
                    start_idx = 0
                    if left_pad_px > 0:
                        padding_patches = left_pad_px // patch_stride
                        padding_patches = min(
                            padding_patches, total_patches - 1)
                        start_idx = padding_patches

                    end_idx = max(
                        start_idx + 1, total_patches - ignore_patches)

                    print(f"   ├─ First chunk mode")
                    print(f"   ├─ Valid range: [{start_idx}:{end_idx}]")
                    print(f"   ├─ Left padding removed: {start_idx}")
                    print(
                        f"   └─ Right overlap ignored: {total_patches - end_idx}")

                elif i == len(chunk_positions) - 1:  # Last chunk
                    chunk_actual_width_px = end_px - start_px

                    if chunk_actual_width_px < chunker.chunk_width:
                        content_ratio = chunk_actual_width_px / chunker.chunk_width
                        chunk_actual_patches = max(
                            1, int(content_ratio * total_patches))
                        start_idx = min(
                            ignore_patches, chunk_actual_patches - 1)
                        end_idx = chunk_actual_patches
                    else:
                        start_idx = min(ignore_patches, total_patches - 1)
                        end_idx = total_patches

                    print(f"   ├─ Last chunk mode")
                    print(f"   ├─ Valid range: [{start_idx}:{end_idx}]")
                    print(f"   ├─ Left overlap ignored: {start_idx}")
                    print(
                        f"   └─ Right padding removed: {total_patches - end_idx}")

                else:  # Middle chunk
                    start_idx = min(ignore_patches, total_patches // 2)
                    end_idx = max(
                        start_idx + 1, total_patches - ignore_patches)

                    print(f"   ├─ Middle chunk mode")
                    print(f"   ├─ Valid range: [{start_idx}:{end_idx}]")
                    print(f"   ├─ Left overlap ignored: {start_idx}")
                    print(
                        f"   └─ Right overlap ignored: {total_patches - end_idx}")

            valid_patches = end_idx - start_idx
            total_valid_patches += valid_patches
            print(f"   └─ Valid patches: {valid_patches}")

        # Test actual merging
        merged_features = model._merge_chunk_features(
            chunk_features, chunk_positions)
        print(f"\n8. Final Results:")
        print(f"   ├─ Expected valid patches: {total_valid_patches}")
        print(f"   ├─ Actual merged shape: {merged_features.shape}")
        print(
            f"   ├─ Feature dimension: {merged_features.shape[1] if len(merged_features.shape) > 1 else 'N/A'}")
        print(
            f"   └─ Sequence length: {merged_features.shape[0] if len(merged_features.shape) > 0 else 0}")

        # Test full model forward pass
        image_batch = image_tensor.unsqueeze(0)
        with torch.no_grad():
            logits, lengths = model(image_batch)

        print(f"\n9. Full Model Output:")
        print(f"   ├─ Input batch: {image_batch.shape}")
        print(f"   ├─ Output logits: {logits.shape} (T_max x Batch x Vocab)")
        print(f"   ├─ Sequence length: {lengths}")
        print(f"   └─ Ready for CTC loss: ✓")


def test_edge_cases():
    """Test edge cases"""
    print(f"\n{'='*60}")
    print("TESTING EDGE CASES")
    print(f"{'='*60}")

    model = HTRModel(vocab_size=100)

    # Very small image
    small_image = create_simple_test_image(width=100, height=40)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    print("\n1. Very small image (100px wide):")
    image_tensor = transform(small_image)
    chunks, positions = model.chunker.create_chunks(image_tensor)
    print(f"   ├─ Chunks created: {len(chunks)}")
    print(f"   ├─ Positions: {positions}")

    # Process through model
    image_batch = image_tensor.unsqueeze(0)
    with torch.no_grad():
        logits, lengths = model(image_batch)
    print(f"   └─ Output shape: {logits.shape}, Length: {lengths}")

    # Very large image
    large_image = create_simple_test_image(width=2000, height=40)
    print(f"\n2. Very large image (2000px wide):")
    image_tensor = transform(large_image)
    chunks, positions = model.chunker.create_chunks(image_tensor)
    print(f"   ├─ Chunks created: {len(chunks)}")
    print(f"   ├─ Each chunk shape: {chunks[0].shape}")

    # Process through model
    image_batch = image_tensor.unsqueeze(0)
    with torch.no_grad():
        logits, lengths = model(image_batch)
    print(f"   └─ Output shape: {logits.shape}, Length: {lengths}")


if __name__ == "__main__":
    analyze_chunking_process()
    test_edge_cases()
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
