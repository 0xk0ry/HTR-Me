"""
Simple test to verify the HTR model implementation
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))


def test_model():
    """Test the HTR model components"""

    print("Testing HTR Model Implementation")
    print("=" * 40)

    # Import the model
    try:
        from model.HTR_ME import HTRModel, CTCDecoder, ImageChunker, CvT
        print("✓ Successfully imported HTR model components")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    # Test device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # Create vocabulary
    vocab = ['<blank>', '<unk>'] + \
        list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!? ')
    print(f"✓ Created vocabulary with {len(vocab)} characters")

    # Test CvT backbone
    print("\\nTesting CvT backbone...")
    try:
        cvt = CvT(
            img_size=320,  # 256 + 2*32 padding
            in_chans=3,
            embed_dims=[64, 192, 384],
            num_heads=[1, 3, 6],
            depths=[1, 2, 10]
        )

        # Test forward pass
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            features, (H, W) = cvt.forward_features(x)

        print(f"✓ CvT forward pass successful")
        print(f"  Input: {x.shape}")
        print(f"  Output: {features.shape}, spatial: {H}x{W}")

    except Exception as e:
        print(f"✗ CvT test failed: {e}")
        return False

    # Test Image Chunker
    print("\\nTesting Image Chunker...")
    try:
        chunker = ImageChunker(
            target_height=40, chunk_width=256, stride=192, padding=32)

        # Create test image
        test_image = Image.new('RGB', (600, 50), color='white')
        preprocessed = chunker.preprocess_image(test_image)
        chunks, positions = chunker.create_chunks(preprocessed)

        print(f"✓ Image chunking successful")
        print(f"  Original: {test_image.size}")
        print(f"  Preprocessed: {preprocessed.size}")
        print(f"  Chunks: {len(chunks)}")
        print(f"  Chunk shapes: {[tuple(chunk.shape) for chunk in chunks]}")

    except Exception as e:
        print(f"✗ Image chunker test failed: {e}")
        return False

    # Test full HTR model
    print("\\nTesting full HTR model...")
    try:
        model = HTRModel(
            vocab_size=len(vocab),
            max_length=256,
            target_height=40,
            chunk_width=256,
            stride=192,
            padding=32
        )

        model.to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created with {param_count:,} parameters")

        # Test forward pass
        batch_size = 2
        dummy_images = torch.randn(batch_size, 3, 40, 400).to(device)

        with torch.no_grad():
            logits, lengths = model(dummy_images)

        print(f"✓ Model forward pass successful")
        print(f"  Input: {dummy_images.shape}")
        print(f"  Output: {logits.shape}")
        print(f"  Lengths: {lengths.tolist()}")

    except Exception as e:
        print(f"✗ HTR model test failed: {e}")
        return False

    # Test CTC Decoder
    print("\\nTesting CTC Decoder...")
    try:
        decoder = CTCDecoder(vocab)

        # Create dummy logits
        seq_len = 50
        dummy_logits = torch.randn(seq_len, len(vocab))

        # Test greedy decoding
        result = decoder.greedy_decode(dummy_logits)
        text = ''.join([vocab[i] for i in result if i < len(vocab)])

        print(f"✓ CTC decoding successful")
        print(f"  Logits shape: {dummy_logits.shape}")
        print(f"  Decoded tokens: {len(result)}")
        print(
            f"  Decoded text: '{text[:50]}{'...' if len(text) > 50 else ''}'")

        # Test beam search
        beam_result = decoder.beam_search_decode(dummy_logits, beam_width=10)
        print(
            f"  Beam search result: '{beam_result[:50]}{'...' if len(beam_result) > 50 else ''}'")

    except Exception as e:
        print(f"✗ CTC decoder test failed: {e}")
        return False

    # Test training step
    print("\\nTesting training step...")
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Create dummy batch
        images = torch.randn(2, 3, 40, 300).to(device)
        targets = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]).to(device)
        target_lengths = torch.tensor([5, 5]).to(device)

        optimizer.zero_grad()
        logits, loss = model(images, targets, target_lengths)
        loss.backward()
        optimizer.step()

        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradients computed and applied")

    except Exception as e:
        print(f"✗ Training step failed: {e}")
        return False

    print("\\n" + "=" * 40)
    print("✓ All tests passed successfully!")
    print("The HTR model implementation is working correctly.")

    return True


if __name__ == "__main__":
    success = test_model()
    if success:
        print("\\n🎉 Ready to train your HTR model!")
        print("\\nNext steps:")
        print("1. Prepare your dataset using create_dataset.py")
        print("2. Train the model using train.py")
        print("3. Run inference using inference.py")
    else:
        print("\\n❌ Tests failed. Please check the implementation.")

    sys.exit(0 if success else 1)
