
"""
Quick Performance Fixes
Apply these to your current model for immediate improvement
"""

import torch
import torch.nn as nn

class EnhancedHTRModel(nn.Module):
    def __init__(self, base_model, vocab_size):
        super().__init__()
        self.base_model = base_model
        
        # Enhanced classifier with dropout
        self.enhanced_classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(128, vocab_size)
        )
        
    def forward(self, x):
        features = self.base_model.cvt(x)
        # Use enhanced classifier
        logits = self.enhanced_classifier(features)
        return logits

def apply_quick_fixes(model_path, save_path):
    """Apply quick fixes to existing model"""
    checkpoint = torch.load(model_path)
    
    # Load existing model
    model = HTRModel(...)  # Your existing config
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create enhanced version
    enhanced_model = EnhancedHTRModel(model, len(checkpoint['vocab']))
    
    # Save enhanced model
    torch.save({
        'model_state_dict': enhanced_model.state_dict(),
        'vocab': checkpoint['vocab'],
        'enhanced': True
    }, save_path)
    
    print(f"Enhanced model saved to {save_path}")

# Usage:
# apply_quick_fixes('checkpoints_iam/best_model.pth', 'checkpoints_iam/enhanced_model.pth')
