#!/usr/bin/env python3
"""
HTR Model with CvT Backbone - Usage Examples
============================================

This script demonstrates how to use the complete HTR system with:
- CvT (Convolutional Vision Transformer) backbone
- Chunking strategy for long text lines
- CTC loss for training
- CTC + Language Model decoding for inference

Author: HTR Team
Date: July 2025
"""

import os
import sys
import argparse
from pathlib import Path

def print_banner():
    print("=" * 60)
    print("HTR Model with CvT Backbone")
    print("Convolutional Vision Transformer for Handwritten Text Recognition")
    print("=" * 60)
    print()

def show_usage():
    """Show usage examples"""
    print("📖 USAGE EXAMPLES")
    print("-" * 40)
    
    print("\\n1️⃣  CREATE SAMPLE DATASET:")
    print("   python create_dataset.py --mode create --output_dir ./sample_data --num_samples 20")
    
    print("\\n2️⃣  CONVERT EXISTING DATASET:")
    print("   python create_dataset.py --mode convert --input_dir /path/to/data --output_dir ./formatted_data")
    
    print("\\n3️⃣  TRAIN THE MODEL:")
    print("   python train.py \\\\")
    print("     --data_dir ./sample_data \\\\")
    print("     --val_data_dir ./validation_data \\\\")
    print("     --output_dir ./checkpoints \\\\")
    print("     --epochs 50 \\\\")
    print("     --batch_size 8 \\\\")
    print("     --lr 1e-4")
    
    print("\\n3️⃣b TRAIN WITH SAM OPTIMIZER:")
    print("   python train.py \\\\")
    print("     --data_dir ./sample_data \\\\")
    print("     --output_dir ./checkpoints \\\\")
    print("     --epochs 50 \\\\")
    print("     --batch_size 8 \\\\")
    print("     --lr 1e-4 \\\\")
    print("     --use_sam \\\\")
    print("     --sam_rho 0.05 \\\\")
    print("     --base_optimizer adamw")
    
    print("\\n4️⃣  SINGLE IMAGE INFERENCE:")
    print("   python inference.py \\\\")
    print("     --checkpoint ./checkpoints/best_model.pth \\\\")
    print("     --image /path/to/handwritten_line.jpg")
    
    print("\\n5️⃣  BATCH INFERENCE:")
    print("   python inference.py \\\\")
    print("     --checkpoint ./checkpoints/best_model.pth \\\\")
    print("     --image_dir /path/to/images/ \\\\")
    print("     --output results.json")
    
    print("\\n6️⃣  WITH LANGUAGE MODEL:")
    print("   python inference.py \\\\")
    print("     --checkpoint ./checkpoints/best_model.pth \\\\")
    print("     --image_dir /path/to/images/ \\\\")
    print("     --lm_path /path/to/language_model.arpa \\\\")
    print("     --beam_width 100")

def show_features():
    """Show key features"""
    print("\\n🔥 KEY FEATURES")
    print("-" * 40)
    
    features = [
        "CvT (Convolutional Vision Transformer) backbone",
        "Chunking strategy for variable-length text lines",
        "Bidirectional padding to reduce boundary effects",
        "CTC loss for sequence-to-sequence training",
        "Beam search decoding with optional language models",
        "KenLM integration for character-level n-gram LM",
        "Memory-efficient processing of long text lines",
        "GPU acceleration with mixed precision support",
        "Modular design for easy customization",
        "Comprehensive training and inference scripts"
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"   {i:2d}. {feature}")

def show_architecture():
    """Show model architecture details"""
    print("\\n🏗️  MODEL ARCHITECTURE")
    print("-" * 40)
    
    print("Input Processing:")
    print("   • Resize to 40px height, preserve aspect ratio")
    print("   • Chunk into 256px segments with 192px stride")
    print("   • Add 32px bidirectional padding per chunk")
    
    print("\\nCvT Backbone:")
    print("   • Patch embedding: 7x7 conv, stride 4")
    print("   • 10 transformer blocks with conv attention")
    print("   • 384-dim embeddings, 6 attention heads")
    print("   • Convolutional positional encoding")
    
    print("\\nOutput Processing:")
    print("   • Remove padding from chunk features")
    print("   • Concatenate valid tokens from all chunks")
    print("   • MLP head: 384 → 192 → vocab_size")
    print("   • CTC loss for training")

def show_requirements():
    """Show installation requirements"""
    print("\\n📦 INSTALLATION")
    print("-" * 40)
    
    print("Core Requirements:")
    print("   pip install torch torchvision pillow numpy")
    
    print("\\nOptional (for enhanced features):")
    print("   pip install opencv-python kenlm pyctcdecode")
    
    print("\\nOr install all at once:")
    print("   pip install -r requirements.txt")

def run_quick_demo():
    """Run a quick demonstration"""
    print("\\n🚀 QUICK DEMO")
    print("-" * 40)
    
    try:
        import torch
        from model.HTR_ME import HTRModel, CTCDecoder, create_model_example
        
        print("Creating model...")
        model, decoder, vocab = create_model_example()
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created: {param_count:,} parameters")
        print(f"✓ Vocabulary: {len(vocab)} characters")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        dummy_input = torch.randn(1, 3, 40, 300).to(device)
        with torch.no_grad():
            logits, lengths = model(dummy_input)
        
        print(f"✓ Forward pass: {dummy_input.shape} → {logits.shape}")
        print(f"✓ Using device: {device}")
        
        # Test decoding
        pred_logits = logits[:lengths[0], 0, :]
        result = decoder.greedy_decode(pred_logits)
        text = ''.join([vocab[i] for i in result if i < len(vocab)])
        
        print(f"✓ Sample prediction: '{text[:30]}{'...' if len(text) > 30 else ''}'")
        print("\\n🎉 Demo completed successfully!")
        
    except ImportError as e:
        print(f"❌ Demo failed - missing dependency: {e}")
        print("Please install required packages first.")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="HTR Model with CvT Backbone - Usage Guide",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--demo', action='store_true', help='Run quick demo')
    parser.add_argument('--test', action='store_true', help='Run comprehensive tests')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.demo:
        run_quick_demo()
    elif args.test:
        print("Running comprehensive tests...")
        os.system("python test_model.py")
    else:
        show_features()
        show_architecture()
        show_requirements()
        show_usage()
        
        print("\\n💡 TIP: Run with --demo for a quick demonstration")
        print("       Run with --test for comprehensive testing")

if __name__ == "__main__":
    main()
