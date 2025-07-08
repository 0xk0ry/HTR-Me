"""
Test SAM optimizer integration with HTR model
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_sam_integration():
    """Test SAM optimizer with HTR model"""
    
    print("Testing SAM Optimizer Integration")
    print("=" * 40)
    
    try:
        from utils.sam import SAM
        from model.HTR_ME import HTRModel, train_epoch
        import torch.optim as optim
        
        print("✓ Imports successful")
        
        # Create a small model for testing
        vocab_size = 10
        model = HTRModel(
            vocab_size=vocab_size,
            max_length=32,
            target_height=40,
            chunk_width=128,
            stride=96,
            padding=16,
            embed_dims=[32, 64, 128],
            num_heads=[1, 2, 4],
            depths=[1, 1, 2]
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test standard optimizer
        optimizer_std = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        print("✓ Standard AdamW optimizer created")
        
        # Test SAM optimizer
        optimizer_sam = SAM(
            model.parameters(), 
            optim.AdamW, 
            lr=1e-4, 
            weight_decay=0.01,
            rho=0.05,
            adaptive=False
        )
        print("✓ SAM optimizer created")
        
        # Create dummy data
        batch_size = 2
        images = torch.randn(batch_size, 3, 40, 200).to(device)
        targets = torch.randint(1, vocab_size, (batch_size, 10)).to(device)
        target_lengths = torch.tensor([10, 10]).to(device)
        
        dummy_batch = (images, targets, target_lengths)
        print("✓ Dummy data created")
        
        # Test standard training step
        def dummy_dataloader():
            yield dummy_batch
        
        print("\\nTesting standard training...")
        model.train()
        loss_std = train_epoch(model, dummy_dataloader(), optimizer_std, device, 
                              vocab=list(range(vocab_size)), use_sam=False)
        print(f"✓ Standard training step: loss = {loss_std:.4f}")
        
        print("\\nTesting SAM training...")
        model.train()
        loss_sam = train_epoch(model, dummy_dataloader(), optimizer_sam, device, 
                              vocab=list(range(vocab_size)), use_sam=True)
        print(f"✓ SAM training step: loss = {loss_sam:.4f}")
        
        print("\\n" + "=" * 40)
        print("✓ SAM integration test passed!")
        print("\\nSAM optimizer is ready to use with:")
        print("  python train.py --use_sam --sam_rho 0.05 --base_optimizer adamw")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sam_integration()
    sys.exit(0 if success else 1)
