"""
Inference script for 3-Stage HTR model with CvT backbone
"""

import torch
import json
import argparse
from pathlib import Path
from PIL import Image
import sys
import re
sys.path.append('.')

try:
    from model.HTR_ME_3Stage import HTRModel, CTCDecoder
except ImportError:
    print("Please make sure the model directory is in your Python path")
    print("You can run: export PYTHONPATH=$PYTHONPATH:.")
    sys.exit(1)

try:
    import editdistance
except ImportError:
    print("editdistance package not found. Installing...")
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "editdistance"])
    import editdistance


def format_string_for_wer(str_input):
    """Format string for WER calculation by adding spaces around punctuation"""
    str_input = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str_input)
    str_input = re.sub('([ \n])+', " ", str_input).strip()
    return str_input


def calculate_cer(predicted, ground_truth):
    """Calculate Character Error Rate (CER) using edit distance"""
    if len(ground_truth) == 0:
        return 0.0 if len(predicted) == 0 else 1.0
    
    # Calculate edit distance between characters
    distance = editdistance.eval(predicted, ground_truth)
    return distance / len(ground_truth)


def calculate_wer(predicted, ground_truth):
    """Calculate Word Error Rate (WER) using edit distance"""
    # Format strings for WER calculation (add spaces around punctuation)
    pred_formatted = format_string_for_wer(predicted)
    gt_formatted = format_string_for_wer(ground_truth)
    
    # Split into words
    pred_words = pred_formatted.split()
    gt_words = gt_formatted.split()
    
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    
    # Calculate edit distance between word lists
    distance = editdistance.eval(pred_words, gt_words)
    return distance / len(gt_words)


def calculate_metrics_batch(predictions_greedy, predictions_beam, ground_truths):
    """
    Calculate CER and WER using the same method as HTR_VT validation
    Accumulates total edit distances and lengths, then calculates final metrics
    """
    # Initialize accumulators
    total_cer_distance_greedy = 0
    total_cer_distance_beam = 0
    total_wer_distance_greedy = 0
    total_wer_distance_beam = 0
    total_char_length = 0
    total_word_length = 0
    
    for pred_greedy, pred_beam, gt in zip(predictions_greedy, predictions_beam, ground_truths):
        # CER calculation
        cer_dist_greedy = editdistance.eval(pred_greedy, gt)
        cer_dist_beam = editdistance.eval(pred_beam, gt)
        total_cer_distance_greedy += cer_dist_greedy
        total_cer_distance_beam += cer_dist_beam
        total_char_length += len(gt)
        
        # WER calculation
        pred_greedy_formatted = format_string_for_wer(pred_greedy)
        pred_beam_formatted = format_string_for_wer(pred_beam)
        gt_formatted = format_string_for_wer(gt)
        
        pred_greedy_words = pred_greedy_formatted.split()
        pred_beam_words = pred_beam_formatted.split()
        gt_words = gt_formatted.split()
        
        wer_dist_greedy = editdistance.eval(pred_greedy_words, gt_words)
        wer_dist_beam = editdistance.eval(pred_beam_words, gt_words)
        total_wer_distance_greedy += wer_dist_greedy
        total_wer_distance_beam += wer_dist_beam
        total_word_length += len(gt_words)
    
    # Calculate final metrics
    if total_char_length > 0:
        cer_greedy = total_cer_distance_greedy / total_char_length
        cer_beam = total_cer_distance_beam / total_char_length
    else:
        cer_greedy = cer_beam = 0.0
    
    if total_word_length > 0:
        wer_greedy = total_wer_distance_greedy / total_word_length
        wer_beam = total_wer_distance_beam / total_word_length
    else:
        wer_greedy = wer_beam = 0.0
    
    return cer_greedy, cer_beam, wer_greedy, wer_beam


def load_ground_truth(image_path):
    """Load ground truth text for an image"""
    image_path = Path(image_path)
    # Replace image extension with .txt
    gt_path = image_path.with_suffix('.txt')

    if gt_path.exists():
        with open(gt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None


def load_model_and_vocab(checkpoint_path, device):
    """Load trained 3-stage model and vocabulary"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint['vocab']
    
    # Get model configuration from checkpoint
    args = checkpoint.get('args', {})
    embed_dims = args.get('embed_dims', [64, 192, 384])
    num_heads = args.get('num_heads', [1, 3, 6])
    depths = args.get('depths', [1, 2, 10])
    target_height = args.get('target_height', 40)
    chunk_width = args.get('chunk_width', 320)
    stride = args.get('stride', 240)

    print(f"Loading 3-stage model with configuration:")
    print(f"  - Embed dims: {embed_dims}")
    print(f"  - Num heads: {num_heads}")
    print(f"  - Depths: {depths}")
    print(f"  - Target height: {target_height}")
    print(f"  - Chunk width: {chunk_width}")
    print(f"  - Stride: {stride}")

    # Create model with same configuration
    model = HTRModel(
        vocab_size=len(vocab),
        max_length=256,
        target_height=target_height,
        chunk_width=chunk_width,
        stride=stride,
        embed_dims=embed_dims,
        num_heads=num_heads,
        depths=depths
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, vocab


def predict_single_image(model, decoder, image_path, device):
    """Predict text from a single image using 3-stage model"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')

    # Use the model's chunker for preprocessing
    preprocessed_image = model.chunker.preprocess_image(image)

    # Convert to tensor
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    image_tensor = transform(preprocessed_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, lengths = model(image_tensor)

        # Get logits for the first (and only) sample
        pred_logits = logits[:lengths[0], 0, :]  # [seq_len, vocab_size]

        # Greedy decoding
        greedy_result = decoder.greedy_decode(pred_logits)
        greedy_text = ''.join([decoder.vocab[i]
                              for i in greedy_result if i < len(decoder.vocab)])

        # Beam search decoding
        beam_result = decoder.beam_search_decode(pred_logits, beam_width=100)

        return {
            'greedy': greedy_text,
            'beam_search': beam_result,
            'confidence': torch.softmax(pred_logits, dim=-1).max(dim=-1)[0].mean().item()
        }


def batch_inference(model, decoder, image_dir, output_file, device):
    """Run inference on all images in a directory"""
    image_dir = Path(image_dir)
    results = []
    
    # Collect predictions and ground truths for batch metric calculation
    predictions_greedy = []
    predictions_beam = []
    ground_truths = []

    # Supported image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))

    print(f"Found {len(image_files)} images to process")

    for image_file in image_files:
        print(f"Processing: {image_file.name}")

        try:
            prediction = predict_single_image(
                model, decoder, image_file, device)

            # Load ground truth
            ground_truth = load_ground_truth(image_file)

            result = {
                'image_path': str(image_file),
                'image_name': image_file.name,
                'greedy_prediction': prediction['greedy'],
                'beam_search_prediction': prediction['beam_search'],
                'confidence': prediction['confidence'],
                'ground_truth': ground_truth,
                'model_type': '3-stage'
            }
            
            # Calculate individual metrics for display
            if ground_truth is not None:
                cer_greedy = calculate_cer(prediction['greedy'], ground_truth)
                cer_beam = calculate_cer(prediction['beam_search'], ground_truth)
                wer_greedy = calculate_wer(prediction['greedy'], ground_truth)
                wer_beam = calculate_wer(prediction['beam_search'], ground_truth)
                
                result.update({
                    'cer_greedy': cer_greedy,
                    'cer_beam_search': cer_beam,
                    'wer_greedy': wer_greedy,
                    'wer_beam_search': wer_beam
                })
                
                # Collect for batch calculation
                predictions_greedy.append(prediction['greedy'])
                predictions_beam.append(prediction['beam_search'])
                ground_truths.append(ground_truth)

                print(f"  Greedy: {prediction['greedy']}")
                print(f"  Beam Search: {prediction['beam_search']}")
                print(f"  Ground Truth: {ground_truth}")
                print(f"  CER (Greedy/Beam): {cer_greedy:.3f} / {cer_beam:.3f}")
                print(f"  WER (Greedy/Beam): {wer_greedy:.3f} / {wer_beam:.3f}")
                print(f"  Confidence: {prediction['confidence']:.3f}")
            else:
                print(f"  Greedy: {prediction['greedy']}")
                print(f"  Beam Search: {prediction['beam_search']}")
                print(f"  Confidence: {prediction['confidence']:.3f}")
                print(f"  Warning: No ground truth found for {image_file.name}")

            results.append(result)

        except Exception as e:
            print(f"  Error processing {image_file.name}: {e}")
            results.append({
                'image_path': str(image_file),
                'image_name': image_file.name,
                'error': str(e),
                'model_type': '3-stage'
            })

    # Calculate batch metrics using HTR_VT method
    if len(ground_truths) > 0:
        batch_cer_greedy, batch_cer_beam, batch_wer_greedy, batch_wer_beam = calculate_metrics_batch(
            predictions_greedy, predictions_beam, ground_truths
        )
        
        avg_metrics = {
            'batch_cer_greedy': batch_cer_greedy,
            'batch_cer_beam_search': batch_cer_beam,
            'batch_wer_greedy': batch_wer_greedy,
            'batch_wer_beam_search': batch_wer_beam,
            'samples_with_ground_truth': len(ground_truths),
            'total_samples': len(image_files),
            'calculation_method': 'HTR_VT_style_batch_calculation',
            'model_type': '3-stage'
        }

        # Add summary to results
        final_results = {
            'summary': avg_metrics,
            'detailed_results': results
        }
    else:
        final_results = {
            'summary': {
                'note': 'No ground truth files found for metric calculation',
                'model_type': '3-stage'
            },
            'detailed_results': results
        }

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\\nResults saved to: {output_file}")

    # Print summary
    if len(ground_truths) > 0:
        print(f"\\n3-Stage Model Batch Metrics (based on {len(ground_truths)} samples):")
        print(f"CER - Greedy: {batch_cer_greedy:.3f}")
        print(f"CER - Beam Search: {batch_cer_beam:.3f}")
        print(f"WER - Greedy: {batch_wer_greedy:.3f}")
        print(f"WER - Beam Search: {batch_wer_beam:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='3-Stage HTR Model Inference')
    parser.add_argument('--checkpoint', type=str,
                        required=True, help='Path to 3-stage model checkpoint')
    parser.add_argument('--image', type=str,
                        help='Path to single image for inference')
    parser.add_argument('--image_dir', type=str,
                        help='Directory containing images for batch inference')
    parser.add_argument(
        '--output', type=str, default='inference_results_3stage.json', help='Output file for results')
    parser.add_argument('--lm_path', type=str,
                        help='Path to KenLM language model')
    parser.add_argument('--beam_width', type=int, default=100,
                        help='Beam width for beam search')

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        print("Please provide either --image or --image_dir")
        return

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and vocabulary
    print("Loading 3-stage model...")
    model, vocab = load_model_and_vocab(args.checkpoint, device)
    print(f"3-stage model loaded with vocabulary size: {len(vocab)}")

    # Create decoder
    decoder = CTCDecoder(vocab, lm_path=args.lm_path)
    if decoder.use_lm:
        print("Language model loaded successfully")
    else:
        print("Using decoder without language model")

    if args.image:
        # Single image inference
        print(f"\\nProcessing single image with 3-stage model: {args.image}")
        prediction = predict_single_image(model, decoder, args.image, device)

        # Try to load ground truth
        ground_truth = load_ground_truth(args.image)

        print("\\n3-Stage Model Results:")
        print(f"Greedy Decoding: {prediction['greedy']}")
        print(f"Beam Search: {prediction['beam_search']}")
        print(f"Confidence: {prediction['confidence']:.3f}")

        if ground_truth is not None:
            cer_greedy = calculate_cer(prediction['greedy'], ground_truth)
            cer_beam = calculate_cer(prediction['beam_search'], ground_truth)
            wer_greedy = calculate_wer(prediction['greedy'], ground_truth)
            wer_beam = calculate_wer(prediction['beam_search'], ground_truth)

            print(f"Ground Truth: {ground_truth}")
            print(f"CER (Greedy): {cer_greedy:.3f}")
            print(f"CER (Beam Search): {cer_beam:.3f}")
            print(f"WER (Greedy): {wer_greedy:.3f}")
            print(f"WER (Beam Search): {wer_beam:.3f}")

        # Save single result
        result = {
            'image_path': args.image,
            'greedy_prediction': prediction['greedy'],
            'beam_search_prediction': prediction['beam_search'],
            'confidence': prediction['confidence'],
            'model_type': '3-stage'
        }

        if ground_truth is not None:
            result.update({
                'ground_truth': ground_truth,
                'cer_greedy': cer_greedy,
                'cer_beam_search': cer_beam,
                'wer_greedy': wer_greedy,
                'wer_beam_search': wer_beam
            })

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    elif args.image_dir:
        # Batch inference
        print(f"\\nProcessing images in directory with 3-stage model: {args.image_dir}")
        results = batch_inference(
            model, decoder, args.image_dir, args.output, device)

        # Print summary
        successful = sum(1 for r in results if 'error' not in r)
        print(f"\\n3-Stage Model Summary:")
        print(f"Total images: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")


if __name__ == "__main__":
    main()
