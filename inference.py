"""
Inference script for HTR model with CvT backbone
"""

import torch
import json
import argparse
from pathlib import Path
from PIL import Image
import sys
sys.path.append('.')

try:
    from model.HTR_ME import HTRModel, CTCDecoder, inference_example
except ImportError:
    print("Please make sure the model directory is in your Python path")
    print("You can run: export PYTHONPATH=$PYTHONPATH:.")
    sys.exit(1)


def load_model_and_vocab(checkpoint_path, device):
    """Load trained model and vocabulary"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint['vocab']

    # Create model with same configuration
    model = HTRModel(
        vocab_size=len(vocab),
        max_length=256,
        target_height=40,
        chunk_width=256,
        stride=192,
        padding=32,
        embed_dims=[64, 192, 384],
        num_heads=[1, 3, 6],
        depths=[1, 2, 10]
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, vocab


def predict_single_image(model, decoder, image_path, device):
    """Predict text from a single image"""
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

    # Supported image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))

    print(f"Found {len(image_files)} images to process")

    for image_file in image_files:
        print(f"Processing: {image_file.name}")

        try:
            prediction = predict_single_image(
                model, decoder, image_file, device)

            result = {
                'image_path': str(image_file),
                'image_name': image_file.name,
                'greedy_prediction': prediction['greedy'],
                'beam_search_prediction': prediction['beam_search'],
                'confidence': prediction['confidence']
            }

            results.append(result)

            print(f"  Greedy: {prediction['greedy']}")
            print(f"  Beam Search: {prediction['beam_search']}")
            print(f"  Confidence: {prediction['confidence']:.3f}")

        except Exception as e:
            print(f"  Error processing {image_file.name}: {e}")
            results.append({
                'image_path': str(image_file),
                'image_name': image_file.name,
                'error': str(e)
            })

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\\nResults saved to: {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description='HTR Model Inference')
    parser.add_argument('--checkpoint', type=str,
                        required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str,
                        help='Path to single image for inference')
    parser.add_argument('--image_dir', type=str,
                        help='Directory containing images for batch inference')
    parser.add_argument(
        '--output', type=str, default='inference_results.json', help='Output file for results')
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
    print("Loading model...")
    model, vocab = load_model_and_vocab(args.checkpoint, device)
    print(f"Model loaded with vocabulary size: {len(vocab)}")

    # Create decoder
    decoder = CTCDecoder(vocab, lm_path=args.lm_path)
    if decoder.use_lm:
        print("Language model loaded successfully")
    else:
        print("Using decoder without language model")

    if args.image:
        # Single image inference
        print(f"\\nProcessing single image: {args.image}")
        prediction = predict_single_image(model, decoder, args.image, device)

        print("\\nResults:")
        print(f"Greedy Decoding: {prediction['greedy']}")
        print(f"Beam Search: {prediction['beam_search']}")
        print(f"Confidence: {prediction['confidence']:.3f}")

        # Save single result
        result = {
            'image_path': args.image,
            'greedy_prediction': prediction['greedy'],
            'beam_search_prediction': prediction['beam_search'],
            'confidence': prediction['confidence']
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    elif args.image_dir:
        # Batch inference
        print(f"\\nProcessing images in directory: {args.image_dir}")
        results = batch_inference(
            model, decoder, args.image_dir, args.output, device)

        # Print summary
        successful = sum(1 for r in results if 'error' not in r)
        print(f"\\nSummary:")
        print(f"Total images: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")


if __name__ == "__main__":
    main()
