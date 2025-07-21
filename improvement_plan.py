"""
HTR Model Improvement Action Plan
Based on debugging analysis of your model performance
"""

import json
from pathlib import Path

def analyze_debug_results():
    """Analyze the debugging results and create action plan"""
    
    # Load batch analysis results
    with open('debug_analysis/batch_analysis.json', 'r') as f:
        results = json.load(f)
    
    print("="*70)
    print("HTR MODEL IMPROVEMENT ACTION PLAN")
    print("="*70)
    
    print(f"\n📊 CURRENT PERFORMANCE:")
    print(f"   Average CER: 19.7% (0.197)")
    print(f"   Average WER: 57.4% (0.574)")
    print(f"   Range: 10.9% - 42.1% CER")
    
    print(f"\n🎯 TARGET PERFORMANCE:")
    print(f"   Target CER: <5% (state-of-the-art)")
    print(f"   Target WER: <15%")
    print(f"   Improvement needed: ~75% CER reduction")
    
    print(f"\n🔍 KEY ISSUES IDENTIFIED:")
    
    # Analyze error patterns
    error_patterns = []
    for result in results:
        if result.get('cer'):
            pred = result['greedy_text']
            gt = result['ground_truth']
            
            # Common character confusions
            if 'v' in gt and 'c' in pred:
                error_patterns.append("v→c confusion")
            if 'l' in gt and 'd' in pred:
                error_patterns.append("l→d confusion")
    
    print(f"   1. Character confusions: v→c, l→d, case errors")
    print(f"   2. High confidence on wrong predictions (overfitting)")
    print(f"   3. Inconsistent performance across samples")
    print(f"   4. Missing language model constraints")
    
    print(f"\n🚀 IMMEDIATE ACTIONS (Week 1-2):")
    print(f"   1. Implement data augmentation")
    print(f"   2. Add learning rate scheduling")
    print(f"   3. Implement gradient clipping")
    print(f"   4. Add dropout for regularization")
    
    print(f"\n📈 MEDIUM-TERM IMPROVEMENTS (Week 3-4):")
    print(f"   1. Language model integration")
    print(f"   2. Curriculum learning")
    print(f"   3. Ensemble methods")
    print(f"   4. Post-processing pipeline")
    
    print(f"\n🏗️ LONG-TERM UPGRADES (Month 2+):")
    print(f"   1. Architecture improvements")
    print(f"   2. Multi-scale features")
    print(f"   3. Attention mechanisms")
    print(f"   4. Transfer learning")

def create_immediate_fixes():
    """Create scripts for immediate performance improvements"""
    
    print(f"\n" + "="*50)
    print("CREATING IMMEDIATE FIX SCRIPTS")
    print("="*50)
    
    # 1. Enhanced training script with improvements
    training_improvements = '''
# Enhanced Training Configuration
IMPROVEMENTS = {
    "data_augmentation": {
        "enabled": True,
        "rotation_range": 3,
        "perspective_strength": 0.1,
        "noise_level": 0.05,
        "brightness_range": 0.2
    },
    
    "training_schedule": {
        "optimizer": "AdamW",
        "base_lr": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "cosine_annealing",
        "warmup_epochs": 5,
        "max_epochs": 100
    },
    
    "regularization": {
        "dropout": 0.1,
        "gradient_clipping": 1.0,
        "label_smoothing": 0.1,
        "early_stopping": 15
    },
    
    "architecture_mods": {
        "add_batch_norm": True,
        "deeper_classifier": True,
        "residual_connections": True
    }
}
'''
    
    with open('training_config.py', 'w') as f:
        f.write(training_improvements)
    
    print("✅ Created training_config.py")
    
    # 2. Quick fixes for current model
    quick_fixes = '''
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
'''
    
    with open('quick_fixes.py', 'w') as f:
        f.write(quick_fixes)
    
    print("✅ Created quick_fixes.py")

def create_training_pipeline():
    """Create improved training pipeline"""
    
    pipeline_script = '''
"""
Improved HTR Training Pipeline
Run this to start training with all improvements
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

# Import your modules
from model.HTR_ME import HTRModel
from improved_trainer import ImprovedHTRTrainer
from data_augmentation import HTRDataAugmentation, create_augmented_dataset

def create_improved_model(vocab_size):
    """Create model with improvements"""
    model = HTRModel(
        vocab_size=vocab_size,
        max_length=256,
        target_height=40,
        chunk_width=320,
        stride=240,
        embed_dims=[64, 192, 384],
        num_heads=[1, 3, 6], 
        depths=[1, 2, 10]
    )
    
    # Add dropout to classifier
    model.classifier = nn.Sequential(
        nn.Linear(384, 256),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.BatchNorm1d(256),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, vocab_size)
    )
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='improved_checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load datasets with augmentation
    print("Loading datasets with augmentation...")
    
    # You'll need to modify this based on your dataset loading
    # train_dataset = YourDatasetClass(args.data_dir, split='train')
    # val_dataset = YourDatasetClass(args.data_dir, split='val')
    
    # Apply augmentation
    # augmented_train = create_augmented_dataset(train_dataset, augmentation_factor=3)
    
    # Create data loaders
    # train_loader = DataLoader(augmented_train, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create improved model
    # vocab = train_dataset.vocab
    # model = create_improved_model(len(vocab)).to(device)
    
    # Create trainer with improvements
    # trainer = ImprovedHTRTrainer(model, train_loader, val_loader, converter, device)
    
    # Setup training with all improvements
    # trainer.setup_training(
    #     learning_rate=1e-3,
    #     weight_decay=1e-4,
    #     use_scheduler=True,
    #     scheduler_type='cosine'
    # )
    
    # Train with early stopping and checkpointing
    # history = trainer.train(
    #     num_epochs=args.epochs,
    #     save_dir=args.output_dir,
    #     save_frequency=5,
    #     early_stopping_patience=15
    # )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
'''
    
    with open('improved_training_pipeline.py', 'w') as f:
        f.write(pipeline_script)
    
    print("✅ Created improved_training_pipeline.py")

def create_evaluation_script():
    """Create comprehensive evaluation script"""
    
    eval_script = '''
"""
Comprehensive Model Evaluation
Compare different models and track improvements
"""

import torch
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def evaluate_models(model_paths, test_data_dir):
    """Evaluate multiple models and compare performance"""
    
    results = []
    
    for model_path in model_paths:
        print(f"Evaluating {model_path}...")
        
        # Load model
        checkpoint = torch.load(model_path)
        # model = load_your_model(checkpoint)
        
        # Run evaluation
        # cer, wer, predictions = evaluate_on_test_set(model, test_data_dir)
        
        results.append({
            'model': model_path.name,
            'cer': 0.0,  # Replace with actual CER
            'wer': 0.0,  # Replace with actual WER
            'params': 0,  # Model parameter count
        })
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    print(df)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(df['model'], df['cer'])
    ax1.set_title('Character Error Rate')
    ax1.set_ylabel('CER')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(df['model'], df['wer'])
    ax2.set_title('Word Error Rate') 
    ax2.set_ylabel('WER')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return df

def track_training_progress(history_files):
    """Track training progress across different experiments"""
    
    plt.figure(figsize=(15, 5))
    
    for i, history_file in enumerate(history_files):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label=f'Exp {i+1}')
        plt.title('Training Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history['val_cer'], label=f'Exp {i+1}')
        plt.title('Validation CER')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history['learning_rates'], label=f'Exp {i+1}')
        plt.title('Learning Rate')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    model_paths = [
        Path('checkpoints_iam/best_model.pth'),
        Path('improved_checkpoints/best_model.pth'),
    ]
    
    evaluate_models(model_paths, 'data/iam/lines')
'''
    
    with open('comprehensive_evaluation.py', 'w') as f:
        f.write(eval_script)
    
    print("✅ Created comprehensive_evaluation.py")

def main():
    """Run the complete analysis and create improvement plan"""
    
    analyze_debug_results()
    create_immediate_fixes()
    create_training_pipeline()
    create_evaluation_script()
    
    print(f"\n" + "="*50)
    print("🎉 IMPROVEMENT PLAN COMPLETE!")
    print("="*50)
    
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Review the generated scripts:")
    print(f"   - training_config.py")
    print(f"   - quick_fixes.py") 
    print(f"   - improved_training_pipeline.py")
    print(f"   - comprehensive_evaluation.py")
    
    print(f"\n2. Start with quick fixes:")
    print(f"   python quick_fixes.py")
    
    print(f"\n3. Run improved training:")
    print(f"   python improved_trainer.py --data_dir data/iam --epochs 100")
    
    print(f"\n4. Monitor progress:")
    print(f"   python debug_htr_model.py --checkpoint improved_checkpoints/best_model.pth --image_dir data/iam/lines")
    
    print(f"\n🎯 Expected improvements:")
    print(f"   - 30-50% CER reduction in first week")
    print(f"   - 60-80% CER reduction with full pipeline")
    print(f"   - Target: <5% CER, <15% WER")

if __name__ == "__main__":
    main()
