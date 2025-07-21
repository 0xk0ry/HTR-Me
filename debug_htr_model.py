"""
Comprehensive HTR Model Debugging and Analysis Tool
This script provides various debugging utilities to analyze model performance
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, some visualizations will be simplified")
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import sys
sys.path.append('.')

from model.HTR_ME import HTRModel, CTCDecoder
import editdistance

class HTRModelDebugger:
    def __init__(self, model_path, vocab_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model and vocab
        self.model, self.vocab = self.load_model(model_path)
        self.decoder = CTCDecoder(self.vocab)
        
        # Create reverse vocab for analysis
        self.idx_to_char = {i: char for i, char in enumerate(self.vocab)}
        self.char_to_idx = {char: i for i, char in enumerate(self.vocab)}
        
        print(f"Model loaded with vocabulary size: {len(self.vocab)}")
        print(f"Vocabulary: {''.join(self.vocab[:50])}{'...' if len(self.vocab) > 50 else ''}")

    def load_model(self, checkpoint_path):
        """Load model and vocabulary from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        vocab = checkpoint['vocab']
        
        model = HTRModel(
            vocab_size=len(vocab),
            max_length=256,
            target_height=40,
            chunk_width=320,
            stride=240,
            embed_dims=[64, 192, 384],
            num_heads=[1, 3, 6],
            depths=[1, 2, 10]
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, vocab

    def analyze_model_architecture(self):
        """Analyze and print model architecture details"""
        print("\n" + "="*50)
        print("MODEL ARCHITECTURE ANALYSIS")
        print("="*50)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Analyze each component
        print("\nComponent breakdown:")
        for name, module in self.model.named_children():
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {params:,} parameters")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024
        }

    def analyze_predictions_detailed(self, image_path, save_dir="debug_output"):
        """Detailed analysis of a single prediction"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"\n" + "="*50)
        print(f"DETAILED PREDICTION ANALYSIS: {Path(image_path).name}")
        print("="*50)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        print(f"Original image size: {image.size}")
        
        # Preprocess image
        preprocessed_image = self.model.chunker.preprocess_image(image)
        print(f"Preprocessed image size: {preprocessed_image.size}")
        
        # Convert to tensor
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(preprocessed_image).unsqueeze(0).to(self.device)
        print(f"Input tensor shape: {image_tensor.shape}")
        
        # Get model outputs with intermediate activations
        with torch.no_grad():
            # Forward pass with detailed outputs
            logits, lengths = self.model(image_tensor)
            seq_len = lengths[0].item()
            
            print(f"Output sequence length: {seq_len}")
            print(f"Logits shape: {logits.shape}")
            
            # Get logits for analysis
            pred_logits = logits[:seq_len, 0, :]  # [seq_len, vocab_size]
            
            # Confidence analysis
            probs = torch.softmax(pred_logits, dim=-1)
            max_probs, predicted_indices = torch.max(probs, dim=-1)
            
            print(f"Average confidence: {max_probs.mean().item():.3f}")
            print(f"Min confidence: {max_probs.min().item():.3f}")
            print(f"Max confidence: {max_probs.max().item():.3f}")
            
            # Analyze predictions
            greedy_result = self.decoder.greedy_decode(pred_logits)
            greedy_text = ''.join([self.vocab[i] for i in greedy_result if i < len(self.vocab)])
            
            beam_result = self.decoder.beam_search_decode(pred_logits, beam_width=100)
            
            print(f"\nGreedy prediction: '{greedy_text}'")
            print(f"Beam search prediction: '{beam_result}'")
            
            # Load ground truth if available
            gt_path = Path(image_path).with_suffix('.txt')
            ground_truth = None
            if gt_path.exists():
                with open(gt_path, 'r', encoding='utf-8') as f:
                    ground_truth = f.read().strip()
                print(f"Ground truth: '{ground_truth}'")
                
                # Calculate metrics
                cer = editdistance.eval(greedy_text, ground_truth) / len(ground_truth)
                wer_pred = greedy_text.split()
                wer_gt = ground_truth.split()
                wer = editdistance.eval(wer_pred, wer_gt) / len(wer_gt)
                
                print(f"CER: {cer:.3f}")
                print(f"WER: {wer:.3f}")
            
            # Create visualizations
            self._create_prediction_visualizations(
                image, preprocessed_image, pred_logits, probs, 
                predicted_indices, max_probs, greedy_text, 
                ground_truth, save_dir, Path(image_path).stem
            )
            
            return {
                'greedy_text': greedy_text,
                'beam_text': beam_result,
                'ground_truth': ground_truth,
                'avg_confidence': max_probs.mean().item(),
                'sequence_length': seq_len,
                'cer': cer if ground_truth else None,
                'wer': wer if ground_truth else None
            }

    def _create_prediction_visualizations(self, original_image, preprocessed_image, 
                                        logits, probs, predicted_indices, confidences,
                                        predicted_text, ground_truth, save_dir, image_name):
        """Create various visualizations for debugging"""
        
        # 1. Image preprocessing comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(preprocessed_image)
        axes[1].set_title("Preprocessed Image")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f"{image_name}_preprocessing.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Confidence plot
        plt.figure(figsize=(15, 6))
        positions = range(len(confidences))
        plt.plot(positions, confidences.cpu().numpy(), 'b-', alpha=0.7)
        plt.fill_between(positions, confidences.cpu().numpy(), alpha=0.3)
        plt.xlabel('Sequence Position')
        plt.ylabel('Confidence')
        plt.title(f'Prediction Confidence Over Sequence\nAvg: {confidences.mean():.3f}')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / f"{image_name}_confidence.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Top predictions heatmap
        top_k = 10
        seq_len = min(20, logits.shape[0])  # Show first 20 positions
        top_probs, top_indices = torch.topk(probs[:seq_len], top_k, dim=-1)
        
        # Create heatmap data
        heatmap_data = top_probs.cpu().numpy()
        labels = []
        for i in range(seq_len):
            row_labels = []
            for j in range(top_k):
                char_idx = top_indices[i, j].item()
                char = self.vocab[char_idx] if char_idx < len(self.vocab) else '?'
                char = char if char.isprintable() and char != ' ' else f'[{ord(char)}]'
                row_labels.append(char)
            labels.append(row_labels)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        if HAS_SEABORN:
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', ax=ax, 
                       xticklabels=[f'Top {i+1}' for i in range(top_k)],
                       yticklabels=[f'Pos {i}' for i in range(seq_len)])
        else:
            # Simple heatmap without seaborn
            im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
            ax.set_xticks(range(top_k))
            ax.set_xticklabels([f'Top {i+1}' for i in range(top_k)])
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels([f'Pos {i}' for i in range(seq_len)])
            plt.colorbar(im, ax=ax)
        
        # Add character annotations
        for i in range(seq_len):
            for j in range(top_k):
                text = ax.text(j+0.5, i+0.7, labels[i][j], 
                             ha="center", va="center", fontsize=8, color='red')
        
        plt.title('Top-K Predictions per Position')
        plt.tight_layout()
        plt.savefig(save_dir / f"{image_name}_top_predictions.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Character frequency analysis
        char_counts = {}
        for char in predicted_text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        if char_counts:
            plt.figure(figsize=(12, 6))
            chars = list(char_counts.keys())
            counts = list(char_counts.values())
            
            # Handle special characters for display
            display_chars = []
            for char in chars:
                if char.isprintable() and char != ' ':
                    display_chars.append(char)
                else:
                    display_chars.append(f'[{ord(char)}]')
            
            plt.bar(display_chars, counts)
            plt.xlabel('Characters')
            plt.ylabel('Frequency')
            plt.title('Character Frequency in Prediction')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_dir / f"{image_name}_char_frequency.png", dpi=150, bbox_inches='tight')
            plt.close()

    def analyze_vocabulary_coverage(self, dataset_path=None):
        """Analyze vocabulary coverage and character distribution"""
        print(f"\n" + "="*50)
        print("VOCABULARY ANALYSIS")
        print("="*50)
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Characters: {''.join(self.vocab)}")
        
        # Character type analysis
        char_types = {
            'letters': 0,
            'digits': 0, 
            'punctuation': 0,
            'spaces': 0,
            'special': 0
        }
        
        for char in self.vocab:
            if char.isalpha():
                char_types['letters'] += 1
            elif char.isdigit():
                char_types['digits'] += 1
            elif char in '.,!?;:':
                char_types['punctuation'] += 1
            elif char.isspace():
                char_types['spaces'] += 1
            else:
                char_types['special'] += 1
        
        print("\nCharacter type distribution:")
        for char_type, count in char_types.items():
            print(f"  {char_type}: {count}")
        
        return char_types

    def analyze_model_weights(self):
        """Analyze model weight distributions"""
        print(f"\n" + "="*50)
        print("MODEL WEIGHTS ANALYSIS")
        print("="*50)
        
        weight_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weights = param.data.cpu().numpy().flatten()
                stats = {
                    'mean': np.mean(weights),
                    'std': np.std(weights),
                    'min': np.min(weights),
                    'max': np.max(weights),
                    'zeros': np.sum(weights == 0),
                    'total': len(weights)
                }
                weight_stats[name] = stats
                
                print(f"\n{name}:")
                print(f"  Shape: {param.shape}")
                print(f"  Mean: {stats['mean']:.6f}")
                print(f"  Std: {stats['std']:.6f}")
                print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
                print(f"  Zero weights: {stats['zeros']}/{stats['total']} ({100*stats['zeros']/stats['total']:.1f}%)")
        
        return weight_stats

    def batch_analysis(self, image_dir, num_samples=10, save_dir="debug_output"):
        """Analyze multiple samples to find patterns"""
        print(f"\n" + "="*50)
        print(f"BATCH ANALYSIS ({num_samples} samples)")
        print("="*50)
        
        image_dir = Path(image_dir)
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Find image files
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
        
        # Sample files for analysis
        import random
        if len(image_files) > num_samples:
            image_files = random.sample(image_files, num_samples)
        
        results = []
        total_cer = 0
        total_wer = 0
        valid_samples = 0
        
        for image_file in image_files:
            try:
                result = self.analyze_predictions_detailed(image_file, save_dir)
                results.append({
                    'file': image_file.name,
                    **result
                })
                
                if result['cer'] is not None:
                    total_cer += result['cer']
                    total_wer += result['wer']
                    valid_samples += 1
                    
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        # Summary statistics
        if valid_samples > 0:
            avg_cer = total_cer / valid_samples
            avg_wer = total_wer / valid_samples
            
            print(f"\nBatch Summary:")
            print(f"Valid samples: {valid_samples}/{len(image_files)}")
            print(f"Average CER: {avg_cer:.3f}")
            print(f"Average WER: {avg_wer:.3f}")
            
            # Find best and worst performing samples
            valid_results = [r for r in results if r['cer'] is not None]
            if valid_results:
                best_sample = min(valid_results, key=lambda x: x['cer'])
                worst_sample = max(valid_results, key=lambda x: x['cer'])
                
                print(f"\nBest sample: {best_sample['file']} (CER: {best_sample['cer']:.3f})")
                print(f"Worst sample: {worst_sample['file']} (CER: {worst_sample['cer']:.3f})")
        
        # Save detailed results
        with open(save_dir / "batch_analysis.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

    def suggest_improvements(self, analysis_results=None):
        """Suggest specific improvements based on analysis"""
        print(f"\n" + "="*50)
        print("IMPROVEMENT SUGGESTIONS")
        print("="*50)
        
        suggestions = []
        
        # Architecture suggestions
        print("\n1. ARCHITECTURE IMPROVEMENTS:")
        suggestions.extend([
            "- Consider increasing model capacity (more layers/channels) if underfitting",
            "- Add dropout layers if overfitting",
            "- Experiment with different backbone architectures (ResNet, EfficientNet)",
            "- Try attention mechanisms for better sequence modeling"
        ])
        
        # Training suggestions  
        print("\n2. TRAINING IMPROVEMENTS:")
        suggestions.extend([
            "- Use learning rate scheduling (cosine annealing, step decay)",
            "- Implement gradient clipping to prevent exploding gradients",
            "- Add data augmentation (rotation, perspective, noise)",
            "- Use label smoothing for CTC loss",
            "- Implement curriculum learning (easy to hard samples)"
        ])
        
        # Data suggestions
        print("\n3. DATA IMPROVEMENTS:")
        suggestions.extend([
            "- Increase dataset size with data augmentation",
            "- Balance character distribution in training data",
            "- Add synthetic data generation",
            "- Improve image preprocessing (denoising, normalization)",
            "- Ensure proper train/val/test splits"
        ])
        
        # Decoding suggestions
        print("\n4. DECODING IMPROVEMENTS:")
        suggestions.extend([
            "- Use language model for beam search",
            "- Implement CTC prefix beam search",
            "- Add word-level constraints",
            "- Post-process with spell correction"
        ])
        
        for suggestion in suggestions:
            print(suggestion)
        
        return suggestions


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='HTR Model Debugging Tool')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, 
                       help='Single image for detailed analysis')
    parser.add_argument('--image_dir', type=str,
                       help='Directory for batch analysis')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples for batch analysis')
    parser.add_argument('--output_dir', type=str, default='debug_output',
                       help='Output directory for debug files')
    
    args = parser.parse_args()
    
    # Initialize debugger
    debugger = HTRModelDebugger(args.checkpoint)
    
    # Run analyses
    debugger.analyze_model_architecture()
    debugger.analyze_vocabulary_coverage()
    debugger.analyze_model_weights()
    
    if args.image:
        debugger.analyze_predictions_detailed(args.image, args.output_dir)
    
    if args.image_dir:
        results = debugger.batch_analysis(args.image_dir, args.num_samples, args.output_dir)
    
    debugger.suggest_improvements()


if __name__ == "__main__":
    main()
