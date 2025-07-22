"""
Quick test script for 3-stage model functionality
"""
import sys
import os
sys.path.append('.')

try:
    from model.HTR_ME_3Stage import HTRModel, CTCDecoder
    print("✅ 3-stage model import successful!")
    
    # Test model creation
    print("Creating 3-stage model...")
    model = HTRModel(
        vocab_size=95,  # Default vocab size
        max_length=256,
        target_height=40,
        chunk_width=320,
        stride=240,
        embed_dims=[64, 192, 384],
        num_heads=[1, 3, 6],
        depths=[1, 2, 10]
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created successfully!")
    print(f"   Parameters: {param_count:,}")
    print(f"   Model type: {type(model).__name__}")
    
    # Test decoder
    print("Testing decoder...")
    default_vocab = ['<blank>'] + list(' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
    decoder = CTCDecoder(default_vocab)
    print(f"✅ Decoder created successfully!")
    print(f"   Vocabulary size: {len(default_vocab)}")
    
    print("\n🎉 All tests passed! 3-stage model is ready for training.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
