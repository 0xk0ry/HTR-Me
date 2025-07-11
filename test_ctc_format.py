#!/usr/bin/env python3
"""
Test script to verify the corrected HTR implementation
"""

import torch
from model.HTR_ME import HTRModel, CTCDecoder

def test_ctc_format():
    """Test that the output format is correct for CTC"""
    print("🧪 Testing CTC Output Format")
    print("=" * 50)
    
    # Create model
    vocab = ['<blank>'] + list('abcdefghijklmnopqrstuvwxyz')
    model = HTRModel(vocab_size=len(vocab), chunk_width=320, stride=240, padding=40)
    
    # Test with dummy data
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 40, 400)
    
    # Forward pass
    logits, lengths = model(dummy_images)
    
    print(f"✅ Input shape: {dummy_images.shape}")
    print(f"✅ Output logits shape: {logits.shape}")
    print(f"✅ Expected format: [T_max, batch_size, vocab_size]")
    print(f"✅ Sequence lengths: {lengths.tolist()}")
    print(f"✅ Vocab size: {len(vocab)}")
    print(f"✅ Blank token at index: 0")
    
    # Verify format
    T_max, B, V = logits.shape
    assert B == batch_size, f"Batch size mismatch: {B} != {batch_size}"
    assert V == len(vocab), f"Vocab size mismatch: {V} != {len(vocab)}"
    assert len(lengths) == batch_size, f"Lengths size mismatch: {len(lengths)} != {batch_size}"
    
    print("\n🎉 All format checks passed!")
    print(f"   Time steps per sample: {lengths.tolist()}")
    print(f"   CTC-ready format: ✅")
    
    return True

def test_feature_extraction():
    """Test that features are properly extracted as time sequences"""
    print("\n🧪 Testing Feature Extraction")
    print("=" * 50)
    
    # Create model
    vocab = ['<blank>'] + list('abc')
    model = HTRModel(vocab_size=len(vocab))
    
    # Test single chunk feature extraction
    dummy_chunk = torch.randn(1, 3, 40, 320)  # [B, C, H, W]
    features = model.forward_features(dummy_chunk)  # Should return [W', C]
    
    print(f"✅ Input chunk shape: {dummy_chunk.shape}")
    print(f"✅ Output features shape: {features.shape}")
    print(f"✅ Time dimension (W'): {features.shape[0]}")
    print(f"✅ Feature dimension (C): {features.shape[1]}")
    
    # Verify feature dimensions
    assert len(features.shape) == 2, f"Features should be 2D, got {features.shape}"
    assert features.shape[1] == model.feature_dim, f"Feature dim mismatch: {features.shape[1]} != {model.feature_dim}"
    
    print("🎉 Feature extraction test passed!")
    return True

if __name__ == "__main__":
    print("🚀 HTR Model CTC Alignment Test")
    print("=" * 60)
    
    try:
        test_feature_extraction()
        test_ctc_format()
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Your HTR model is now properly aligned for CTC training!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Please check the implementation.")
