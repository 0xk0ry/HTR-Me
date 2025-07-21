"""
Comprehensive test script to visualize the HTR model's chunking and feature extraction process.
This script shows:
1. How images are converted to chunks
2. Feature extraction and pooling
3. Shape transformations at each step
4. Valid vs ignored patch indices for each chunk
5. Merging process visualization
"""

from model.HTR_ME import HTRModel, ImageChunker, CvT
from torchvision import transforms
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch.nn.functional as F
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_image(width=800, height=40, text="Sample handwritten text for testing"):
    """Create a simple test image with text"""
    # Create white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except:
        font = ImageFont.load_default()

    # Draw text
    draw.text((10, height//4), text, fill='black', font=font)

    # Add some noise/variation to make it more realistic
    pixels = np.array(img)
    noise = np.random.normal(0, 5, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(pixels)


def visualize_chunking_process():
    """Main function to demonstrate the chunking and feature extraction process"""

    # Initialize model components
    vocab_size = 100
    model = HTRModel(vocab_size=vocab_size)
    chunker = ImageChunker(target_height=40, chunk_width=320, stride=240)

    print("=== HTR Model Chunking and Feature Extraction Visualization ===\n")

    # Create test image
    test_image = create_test_image(width=800, height=40)
    print(f"1. Original Image Size: {test_image.size} (W x H)")

    # Preprocess image
    processed_image = chunker.preprocess_image(test_image)
    print(f"2. Preprocessed Image Size: {processed_image.size} (W x H)")

    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    image_tensor = transform(processed_image)
    print(f"3. Image Tensor Shape: {image_tensor.shape} (C x H x W)")

    # Create chunks
    chunks, chunk_positions = chunker.create_chunks(image_tensor)
    print(f"\n4. Chunking Results:")
    print(f"   Number of chunks: {len(chunks)}")
    print(f"   Each chunk shape: {chunks[0].shape} (C x H x W)")
    print(f"   Chunk positions: {chunk_positions}")

    # Analyze chunk parameters
    print(f"\n5. Chunking Parameters:")
    print(f"   Chunk width: {chunker.chunk_width}px")
    print(f"   Stride: {chunker.stride}px")
    print(f"   Padding: {chunker.padding}px")
    print(f"   Overlap size: {chunker.chunk_width - chunker.stride}px")

    # Process each chunk through CvT
    print(f"\n6. Feature Extraction Process:")
    chunk_features = []

    for i, chunk in enumerate(chunks):
        print(f"\n   Chunk {i}:")
        print(f"   - Input shape: {chunk.shape}")

        # Add batch dimension
        chunk_with_batch = chunk.unsqueeze(0)
        print(f"   - With batch dim: {chunk_with_batch.shape}")

        # Extract features
        features = model.forward_features(chunk_with_batch)
        chunk_features.append(features)
        print(f"   - Output features shape: {features.shape} (W' x C)")

        # Calculate patch information
        patch_stride = model.cvt.patch_embed.stride
        patch_size = model.cvt.patch_embed.patch_size

        # Calculate H' and W' manually
        H_prime = (chunk.shape[1] - patch_size) // patch_stride + 1
        W_prime = (chunk.shape[2] - patch_size) // patch_stride + 1

        print(f"   - Patch stride: {patch_stride}, Patch size: {patch_size}")
        print(f"   - Spatial patches: H'={H_prime}, W'={W_prime}")
        print(f"   - Total patches before height pooling: {H_prime * W_prime}")
        print(f"   - After height pooling: {W_prime} time steps")

    # Analyze merging process
    print(f"\n7. Merging Process Analysis:")

    # Calculate ignore parameters
    ignored_size = (chunker.chunk_width - chunker.stride) / 2  # 80px
    patch_stride = model.cvt.patch_embed.stride
    ignore_patches = max(1, ignored_size // patch_stride)

    print(
        f"   Ignored region size: {ignored_size}px = {ignore_patches} patches")

    # Analyze each chunk's valid regions
    valid_indices_info = []

    for i, (features, (start_px, end_px, left_pad_px, content_end_in_chunk_px)) in enumerate(zip(chunk_features, chunk_positions)):
        total_patches = features.size(0)
        print(f"\n   Chunk {i} Analysis:")
        print(f"   - Total patches: {total_patches}")
        print(f"   - Pixel range: {start_px}-{end_px}px")
        print(f"   - Left padding: {left_pad_px}px")

        if len(chunk_positions) == 1:
            # Single chunk
            start_idx = 0
            end_idx = total_patches

            if left_pad_px > 0:
                padding_patches = left_pad_px // patch_stride
                padding_patches = min(padding_patches, total_patches - 1)
                start_idx = padding_patches

            # Check for right padding
            actual_content_width_px = end_px - start_px
            if actual_content_width_px < chunker.chunk_width - left_pad_px:
                content_ratio = (
                    left_pad_px + actual_content_width_px) / chunker.chunk_width
                actual_content_patches = max(
                    1, int(content_ratio * total_patches))
                end_idx = min(actual_content_patches, total_patches)

            print(f"   - Single chunk: valid patches [{start_idx}:{end_idx}]")
            print(f"   - Left padding removed: {start_idx} patches")
            print(
                f"   - Right padding removed: {total_patches - end_idx} patches")

        else:
            if i == 0:
                # First chunk
                start_idx = 0
                if left_pad_px > 0:
                    padding_patches = left_pad_px // patch_stride
                    padding_patches = min(padding_patches, total_patches - 1)
                    start_idx = padding_patches

                end_idx = max(start_idx + 1, total_patches - ignore_patches)

                print(
                    f"   - First chunk: valid patches [{start_idx}:{end_idx}]")
                print(f"   - Left padding removed: {start_idx} patches")
                print(
                    f"   - Right overlap ignored: {total_patches - end_idx} patches")

            elif i == len(chunk_positions) - 1:
                # Last chunk
                chunk_actual_width_px = end_px - start_px

                if chunk_actual_width_px < chunker.chunk_width:
                    content_ratio = chunk_actual_width_px / chunker.chunk_width
                    chunk_actual_patches = max(
                        1, int(content_ratio * total_patches))
                    start_idx = min(ignore_patches, chunk_actual_patches - 1)
                    end_idx = chunk_actual_patches
                else:
                    start_idx = min(ignore_patches, total_patches - 1)
                    end_idx = total_patches

                print(
                    f"   - Last chunk: valid patches [{start_idx}:{end_idx}]")
                print(f"   - Left overlap ignored: {start_idx} patches")
                print(
                    f"   - Right padding removed: {total_patches - end_idx} patches")

            else:
                # Middle chunk
                start_idx = min(ignore_patches, total_patches // 2)
                end_idx = max(start_idx + 1, total_patches - ignore_patches)

                print(
                    f"   - Middle chunk: valid patches [{start_idx}:{end_idx}]")
                print(f"   - Left overlap ignored: {start_idx} patches")
                print(
                    f"   - Right overlap ignored: {total_patches - end_idx} patches")

        valid_indices_info.append((start_idx, end_idx, total_patches))

    # Perform actual merging
    merged_features = model._merge_chunk_features(
        chunk_features, chunk_positions)
    print(f"\n8. Final Merged Features:")
    print(
        f"   Shape: {merged_features.shape} (Total_time_steps x Feature_dim)")

    # Calculate total valid patches
    total_valid = sum(end - start for start, end, _ in valid_indices_info)
    print(f"   Total valid patches: {total_valid}")
    print(f"   Feature dimension: {merged_features.shape[1]}")

    # Test with model forward pass
    print(f"\n9. Full Model Forward Pass:")
    image_batch = image_tensor.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        logits, lengths = model(image_batch)

    print(f"   Input batch shape: {image_batch.shape}")
    print(f"   Output logits shape: {logits.shape} (T_max x Batch x Vocab)")
    print(f"   Sequence lengths: {lengths}")

    return {
        'original_image': test_image,
        'chunks': chunks,
        'chunk_positions': chunk_positions,
        'chunk_features': chunk_features,
        'valid_indices': valid_indices_info,
        'merged_features': merged_features,
        'final_logits': logits,
        'sequence_lengths': lengths
    }


def create_visualization_plot(results):
    """Create a comprehensive visualization plot"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('HTR Model: Image to Features Visualization', fontsize=16)

    # 1. Original image
    axes[0, 0].imshow(results['original_image'])
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 2. Chunks visualization
    axes[0, 1].set_title('Chunk Boundaries')
    img_width = results['original_image'].size[0]
    axes[0, 1].set_xlim(0, img_width)
    axes[0, 1].set_ylim(0, 40)

    # Draw chunk boundaries
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (start_px, end_px, left_pad_px, content_end_px) in enumerate(results['chunk_positions']):
        color = colors[i % len(colors)]
        rect = patches.Rectangle((start_px, 0), end_px - start_px, 40,
                                 linewidth=2, edgecolor=color, facecolor='none',
                                 label=f'Chunk {i}')
        axes[0, 1].add_patch(rect)
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Pixel Position')

    # 3. Feature shapes
    axes[1, 0].bar(range(len(results['chunk_features'])),
                   [f.shape[0] for f in results['chunk_features']])
    axes[1, 0].set_title('Features per Chunk (before merging)')
    axes[1, 0].set_xlabel('Chunk Index')
    axes[1, 0].set_ylabel('Number of Time Steps')

    # 4. Valid vs ignored patches
    valid_counts = [end - start for start, end, _ in results['valid_indices']]
    total_counts = [total for _, _, total in results['valid_indices']]
    ignored_counts = [total - valid for valid,
                      total in zip(valid_counts, total_counts)]

    x = range(len(valid_counts))
    axes[1, 1].bar(x, valid_counts, label='Valid Patches', alpha=0.7)
    axes[1, 1].bar(x, ignored_counts, bottom=valid_counts,
                   label='Ignored Patches', alpha=0.7)
    axes[1, 1].set_title('Valid vs Ignored Patches per Chunk')
    axes[1, 1].set_xlabel('Chunk Index')
    axes[1, 1].set_ylabel('Number of Patches')
    axes[1, 1].legend()

    # 5. Final merged features heatmap (sample) - FIX: Add .detach()
    if results['merged_features'].shape[0] > 0:
        # Show first 50 features and first 50 time steps for visualization
        sample_features = results['merged_features'][:min(50, results['merged_features'].shape[0]),
                                                     :min(50, results['merged_features'].shape[1])]
        # FIXED: Detach tensor before converting to numpy for matplotlib
        sample_features_np = sample_features.detach().numpy()
        im = axes[2, 0].imshow(sample_features_np.T,
                               aspect='auto', cmap='viridis')
        axes[2, 0].set_title('Merged Features (Sample: 50x50)')
        axes[2, 0].set_xlabel('Time Steps')
        axes[2, 0].set_ylabel('Feature Dimensions')
        plt.colorbar(im, ax=axes[2, 0])

    # 6. Final logits shape
    logits_shape = results['final_logits'].shape
    axes[2, 1].text(0.1, 0.7, f'Final Output Shape:',
                    fontsize=12, weight='bold')
    axes[2, 1].text(0.1, 0.6, f'Logits: {logits_shape}', fontsize=10)
    axes[2, 1].text(0.1, 0.5, f'Format: (T_max, Batch, Vocab)', fontsize=10)
    axes[2, 1].text(
        0.1, 0.4, f'Sequence Length: {results["sequence_lengths"]}', fontsize=10)
    axes[2, 1].text(0.1, 0.3, f'Ready for CTC Loss',
                    fontsize=10, style='italic')
    axes[2, 1].set_xlim(0, 1)
    axes[2, 1].set_ylim(0, 1)
    axes[2, 1].axis('off')
    axes[2, 1].set_title('Output Summary')

    plt.tight_layout()
    return fig


def run_comprehensive_test():
    """Run the comprehensive test and save results"""
    print("Starting comprehensive HTR chunking test...\n")

    # Run the main visualization
    results = visualize_chunking_process()

    # Create and save plot
    fig = create_visualization_plot(results)

    # Save plot
    output_path = os.path.join(os.path.dirname(
        __file__), 'chunk_visualization.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # Save detailed results
    results_path = os.path.join(os.path.dirname(
        __file__), 'chunk_analysis_results.txt')
    with open(results_path, 'w') as f:
        f.write("HTR Model Chunking Analysis Results\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Chunk Positions: {results['chunk_positions']}\n")
        f.write(f"Valid Indices: {results['valid_indices']}\n")
        f.write(f"Merged Features Shape: {results['merged_features'].shape}\n")
        f.write(f"Final Logits Shape: {results['final_logits'].shape}\n")
        f.write(f"Sequence Lengths: {results['sequence_lengths']}\n")

        f.write("\nDetailed Chunk Analysis:\n")
        for i, (features, (start_px, end_px, left_pad_px, content_end_px)) in enumerate(zip(results['chunk_features'], results['chunk_positions'])):
            f.write(f"\nChunk {i}:\n")
            f.write(f"  Pixel range: {start_px}-{end_px}\n")
            f.write(f"  Left padding: {left_pad_px}px\n")
            f.write(f"  Features shape: {features.shape}\n")
            f.write(f"  Valid indices: {results['valid_indices'][i]}\n")

    print(f"Detailed results saved to: {results_path}")

    plt.show()

    return results


if __name__ == "__main__":
    results = run_comprehensive_test()
