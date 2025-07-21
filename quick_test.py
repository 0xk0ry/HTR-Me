
"""
Quick Test Script for HTR Improvements
Run this to test your model with and without language model correction
"""

import torch
from pathlib import Path
from simple_language_model import SimpleLanguageModel
import sys
sys.path.append('.')

from model.HTR_ME import HTRModel, CTCDecoder
from PIL import Image
import torchvision.transforms as transforms
import editdistance

def load_model(checkpoint_path, device):
    """Load HTR model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    model.to(device)
    model.eval()
    
    return model, vocab

def predict_image(model, decoder, image_path, device):
    """Predict text from image"""
    image = Image.open(image_path).convert('RGB')
    preprocessed_image = model.chunker.preprocess_image(image)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(preprocessed_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, lengths = model(image_tensor)
        pred_logits = logits[:lengths[0], 0, :]
        
        # Greedy decoding
        greedy_result = decoder.greedy_decode(pred_logits)
        greedy_text = ''.join([decoder.vocab[i] for i in greedy_result if i < len(decoder.vocab)])
        
        return greedy_text

def test_with_language_model():
    """Test model with and without language model correction"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = 'checkpoints_iam/best_model.pth'
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return
    
    model, vocab = load_model(model_path, device)
    decoder = CTCDecoder(vocab)
    
    # Load language model
    lm = SimpleLanguageModel()
    lm_path = 'simple_language_model.json'
    if Path(lm_path).exists():
        lm.load_model(lm_path)
    else:
        print("Language model not found. Run: python simple_language_model.py")
        return
    
    # Test on sample images
    test_dir = Path('data/iam/lines')
    image_files = list(test_dir.glob('test_*.png'))[:5]  # Test first 5 images
    
    if not image_files:
        print("No test images found in data/iam/lines")
        return
    
    print("="*70)
    print("TESTING HTR MODEL WITH LANGUAGE MODEL CORRECTION")
    print("="*70)
    
    total_improvement = 0
    valid_tests = 0
    
    for image_file in image_files:
        # Get prediction
        prediction = predict_image(model, decoder, image_file, device)
        
        # Apply language model correction
        corrected = lm.correct_text(prediction)
        
        # Load ground truth
        gt_file = image_file.with_suffix('.txt')
        if gt_file.exists():
            with open(gt_file, 'r', encoding='utf-8') as f:
                ground_truth = f.read().strip()
            
            # Calculate CER
            original_cer = editdistance.eval(prediction, ground_truth) / len(ground_truth)
            corrected_cer = editdistance.eval(corrected, ground_truth) / len(ground_truth)
            improvement = (original_cer - corrected_cer) / original_cer * 100 if original_cer > 0 else 0
            
            print(f"\nImage: {image_file.name}")
            print(f"Original:  '{prediction}'")
            print(f"Corrected: '{corrected}'")
            print(f"GT:        '{ground_truth}'")
            print(f"CER: {original_cer:.3f} → {corrected_cer:.3f} ({improvement:+.1f}%)")
            
            total_improvement += improvement
            valid_tests += 1
        else:
            print(f"\nImage: {image_file.name}")
            print(f"Original:  '{prediction}'")
            print(f"Corrected: '{corrected}'")
            print("No ground truth available")
    
    if valid_tests > 0:
        avg_improvement = total_improvement / valid_tests
        print(f"\n{'='*70}")
        print(f"SUMMARY: Average CER improvement: {avg_improvement:+.1f}%")
        print(f"{'='*70}")

if __name__ == "__main__":
    test_with_language_model()
