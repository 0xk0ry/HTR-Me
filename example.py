"""
Example workflow demonstrating HTR model training and inference
This script shows how to use all components together
"""

import torch
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')


def example_workflow():
    """Complete example workflow"""

    print("=== HTR Model Example Workflow ===\\n")

    # 1. Check dependencies
    print("1. Checking dependencies...")
    try:
        import torch
        import torchvision
        import numpy as np
        from PIL import Image
        print("   ✓ Core dependencies available")
    except ImportError as e:
        print(f"   ✗ Missing dependency: {e}")
        print("   Please install requirements: pip install -r requirements.txt")
        return

    # Optional dependencies
    optional_deps = []
    try:
        import cv2
        optional_deps.append("opencv-python")
    except ImportError:
        pass

    try:
        import kenlm
        import pyctcdecode
        optional_deps.append("kenlm + pyctcdecode")
    except ImportError:
        pass

    if optional_deps:
        print(f"   ✓ Optional dependencies: {', '.join(optional_deps)}")
    else:
        print("   ⚠ No optional dependencies (opencv, kenlm)")

    # 2. Create sample dataset
    print("\\n2. Creating sample dataset...")
    dataset_dir = Path("./sample_data")

    try:
        from create_dataset import create_sample_dataset
        create_sample_dataset(dataset_dir, num_samples=10)
        print("   ✓ Sample dataset created")
    except Exception as e:
        print(f"   ✗ Failed to create dataset: {e}")
        return

    # 3. Initialize model
    print("\\n3. Initializing model...")
    try:
        from model.HTR_ME import HTRModel, CTCDecoder, create_model_example

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {device}")

        model, decoder, vocab = create_model_example()
        model.to(device)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Model initialized with {param_count:,} parameters")
        print(f"   ✓ Vocabulary size: {len(vocab)}")

    except Exception as e:
        print(f"   ✗ Failed to initialize model: {e}")
        return

    # 4. Test forward pass
    print("\\n4. Testing forward pass...")
    try:
        # Create dummy input
        batch_size = 2
        dummy_images = torch.randn(batch_size, 3, 40, 256).to(device)

        with torch.no_grad():
            logits, lengths = model(dummy_images)

        print(f"   ✓ Forward pass successful")
        print(f"   Input shape: {dummy_images.shape}")
        print(f"   Output shape: {logits.shape}")
        print(f"   Sequence lengths: {lengths.tolist()}")

    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        return

    # 5. Test image preprocessing and chunking
    print("\\n5. Testing image preprocessing...")
    try:
        # Load a sample image
        sample_images = list(dataset_dir.glob("images/*.jpg"))
        if sample_images:
            sample_image_path = sample_images[0]

            # Test image chunking
            from PIL import Image
            image = Image.open(sample_image_path)
            print(f"   Original image size: {image.size}")

            # Preprocess
            preprocessed = model.chunker.preprocess_image(image)
            print(f"   Preprocessed size: {preprocessed.size}")

            # Create chunks
            chunks, positions = model.chunker.create_chunks(preprocessed)
            print(f"   ✓ Created {len(chunks)} chunks")
            print(f"   Chunk shapes: {[chunk.shape for chunk in chunks]}")

        else:
            print("   ⚠ No sample images found")

    except Exception as e:
        print(f"   ✗ Image preprocessing failed: {e}")

    # 6. Test inference on sample image
    print("\\n6. Testing inference...")
    try:
        if sample_images:
            sample_image_path = sample_images[0]

            # Load and preprocess image
            image = Image.open(sample_image_path).convert('RGB')
            preprocessed_image = model.chunker.preprocess_image(image)

            # Convert to tensor
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])

            image_tensor = transform(
                preprocessed_image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, lengths = model(image_tensor)

                # Get logits for the first sample
                pred_logits = logits[:lengths[0], 0, :]

                # Greedy decoding
                greedy_result = decoder.greedy_decode(pred_logits)
                greedy_text = ''.join([vocab[i]
                                      for i in greedy_result if i < len(vocab)])

                print(f"   ✓ Inference successful")
                print(f"   Sample image: {sample_image_path.name}")
                print(f"   Predicted text: '{greedy_text}'")
                print(f"   Prediction length: {len(greedy_result)} tokens")

        else:
            print("   ⚠ No sample images available for inference")

    except Exception as e:
        print(f"   ✗ Inference failed: {e}")

    # 7. Show training command
    print("\\n7. Training instructions...")
    print("   To train the model, run:")
    print(f"   python train.py \\\\")
    print(f"     --data_dir {dataset_dir} \\\\")
    print(f"     --output_dir ./checkpoints \\\\")
    print(f"     --epochs 10 \\\\")
    print(f"     --batch_size 4 \\\\")
    print(f"     --lr 1e-4")

    # 8. Show inference command
    print("\\n8. Inference instructions...")
    print("   After training, run inference with:")
    print("   python inference.py \\\\")
    print("     --checkpoint ./checkpoints/best_model.pth \\\\")
    print(f"     --image_dir {dataset_dir}/images \\\\")
    print("     --output results.json")

    # 9. Performance tips
    print("\\n9. Performance optimization tips...")
    print("   • Use GPU for faster training: CUDA device detected" if torch.cuda.is_available(
    ) else "   • Install CUDA for GPU acceleration")
    print("   • Install KenLM for better decoding: pip install kenlm pyctcdecode")
    print("   • Use mixed precision training: --mixed_precision")
    print("   • Adjust chunk_width/stride for memory optimization")

    print("\\n=== Workflow completed successfully! ===")


def test_individual_components():
    """Test individual components separately"""

    print("\\n=== Testing Individual Components ===\\n")

    # Test CvT backbone
    print("1. Testing CvT backbone...")
    try:
        from model.HTR_ME import CvT

        cvt = CvT(
            img_size=256,
            in_chans=3,
            embed_dims=[64, 192, 384],
            num_heads=[1, 3, 6],
            depths=[1, 2, 10]
        )

        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            features, (H, W) = cvt.forward_features(x)

        print(f"   ✓ CvT output shape: {features.shape}, spatial: {H}x{W}")

    except Exception as e:
        print(f"   ✗ CvT test failed: {e}")

    # Test Image Chunker
    print("\\n2. Testing Image Chunker...")
    try:
        from model.HTR_ME import ImageChunker
        from PIL import Image
        import numpy as np

        chunker = ImageChunker(
            target_height=40, chunk_width=256, stride=192, padding=32)

        # Create test image
        test_image = Image.new('RGB', (800, 60), color='white')
        preprocessed = chunker.preprocess_image(test_image)
        chunks, positions = chunker.create_chunks(preprocessed)

        print(f"   ✓ Input size: {test_image.size}")
        print(f"   ✓ Preprocessed size: {preprocessed.size}")
        print(f"   ✓ Number of chunks: {len(chunks)}")
        print(f"   ✓ Chunk positions: {positions}")

    except Exception as e:
        print(f"   ✗ Image Chunker test failed: {e}")

    # Test CTC Decoder
    print("\\n3. Testing CTC Decoder...")
    try:
        from model.HTR_ME import CTCDecoder

        vocab = ['<blank>'] + list('abcdefghijklmnopqrstuvwxyz ')
        decoder = CTCDecoder(vocab)

        # Create dummy logits
        seq_len, vocab_size = 50, len(vocab)
        logits = torch.randn(seq_len, vocab_size)

        # Test greedy decoding
        result = decoder.greedy_decode(logits)
        text = ''.join([vocab[i] for i in result if i < len(vocab)])

        print(f"   ✓ Greedy decoding successful")
        print(f"   ✓ Decoded text: '{text}'")
        print(f"   ✓ Text length: {len(text)}")

        # Test beam search (without LM)
        beam_result = decoder.beam_search_decode(logits, beam_width=10)
        print(f"   ✓ Beam search result: '{beam_result}'")

    except Exception as e:
        print(f"   ✗ CTC Decoder test failed: {e}")


if __name__ == "__main__":
    print("Starting HTR Model Example and Testing...\\n")

    try:
        example_workflow()
        test_individual_components()

    except KeyboardInterrupt:
        print("\\n\\nExample interrupted by user")
    except Exception as e:
        print(f"\\n\\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

    print("\\n=== Example completed ===")
