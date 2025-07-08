"""
Utility script to create a sample dataset for HTR training
This script demonstrates how to prepare data in the correct format
"""

import json
import os
import random
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_sample_annotations():
    """Create sample text annotations for demonstration"""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming the world",
        "Handwritten text recognition using deep learning",
        "Computer vision and natural language processing",
        "PyTorch implementation of Convolutional Vision Transformer",
        "CTC loss function for sequence prediction tasks",
        "Beam search decoding with language models",
        "Image preprocessing and data augmentation techniques",
        "Training neural networks on GPU accelerators",
        "Evaluation metrics for text recognition systems",
        "Optical character recognition in historical documents",
        "Real-time inference on mobile devices",
        "Transfer learning from pretrained models",
        "Attention mechanisms in transformer architectures",
        "End-to-end differentiable text recognition pipeline"
    ]

    annotations = []
    for i, text in enumerate(sample_texts):
        annotations.append({
            "image": f"images/sample_{i:03d}.jpg",
            "text": text
        })

    return annotations


def generate_synthetic_image(text, output_path, width=800, height=40):
    """Generate a synthetic handwritten-style image with text"""
    # Create image with white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Try to use a font that looks handwritten
    try:
        # You can download handwritten fonts and place them in a fonts directory
        font = ImageFont.truetype("arial.ttf", size=24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()

    # Calculate text position
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Adjust image width to fit text
    if text_width > width - 20:
        width = text_width + 40
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # Add some randomness to make it look more natural
    x += random.randint(-5, 5)
    y += random.randint(-3, 3)

    # Draw text
    color = (random.randint(0, 50), random.randint(
        0, 50), random.randint(0, 50))
    draw.text((x, y), text, fill=color, font=font)

    # Add some noise to make it look more realistic
    img_array = np.array(img)
    noise = np.random.normal(0, 5, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)

    # Save image
    img.save(output_path, 'JPEG', quality=90)


def create_sample_dataset(output_dir, num_samples=15):
    """Create a complete sample dataset"""
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"

    # Create directories
    output_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    # Create annotations
    annotations = create_sample_annotations()[:num_samples]

    # Generate synthetic images
    print(f"Generating {num_samples} sample images...")
    for i, annotation in enumerate(annotations):
        image_path = output_dir / annotation["image"]
        text = annotation["text"]

        print(f"Creating image {i+1}/{num_samples}: {text[:50]}...")
        generate_synthetic_image(text, image_path)

    # Save annotations
    with open(output_dir / "annotations.json", 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # Create vocabulary
    all_chars = set()
    for annotation in annotations:
        all_chars.update(annotation["text"])

    vocab = ['<blank>', '<unk>'] + sorted(list(all_chars))

    with open(output_dir / "vocab.json", 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    print(f"\\nDataset created successfully in {output_dir}")
    print(f"Images: {len(annotations)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Characters: {''.join(sorted(all_chars))}")


def convert_existing_dataset(input_dir, output_dir):
    """Convert an existing dataset to the required format"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    # Look for paired image/text files
    annotations = []
    all_chars = set()

    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    for ext in image_extensions:
        for img_file in input_dir.glob(f"*{ext}"):
            # Look for corresponding text file
            txt_file = img_file.with_suffix('.txt')
            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()

                # Copy image to new location
                new_img_path = output_dir / "images" / img_file.name
                new_img_path.parent.mkdir(exist_ok=True)

                # Copy image
                img = Image.open(img_file)
                img.save(new_img_path)

                annotations.append({
                    "image": f"images/{img_file.name}",
                    "text": text
                })

                all_chars.update(text)

    if not annotations:
        print("No paired image/text files found!")
        return

    # Save annotations
    with open(output_dir / "annotations.json", 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # Create vocabulary
    vocab = ['<blank>', '<unk>'] + sorted(list(all_chars))

    with open(output_dir / "vocab.json", 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    print(f"Dataset converted successfully!")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(annotations)}")
    print(f"Vocabulary size: {len(vocab)}")


def main():
    parser = argparse.ArgumentParser(
        description='Create or convert HTR dataset')
    parser.add_argument('--mode', choices=['create', 'convert'], required=True,
                        help='Create sample dataset or convert existing')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for dataset')
    parser.add_argument('--input_dir', type=str,
                        help='Input directory (for convert mode)')
    parser.add_argument('--num_samples', type=int, default=15,
                        help='Number of samples to create (for create mode)')

    args = parser.parse_args()

    if args.mode == 'create':
        create_sample_dataset(args.output_dir, args.num_samples)
    elif args.mode == 'convert':
        if not args.input_dir:
            print("--input_dir is required for convert mode")
            return
        convert_existing_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
