
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
