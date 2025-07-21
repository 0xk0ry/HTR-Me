
"""
Integration with HTR Inference
Add this to your inference.py script
"""

from simple_language_model import SimpleLanguageModel

# Load language model
lm = SimpleLanguageModel()
lm.load_model("simple_language_model.json")

def predict_with_correction(model, decoder, image_path, device):
    """Predict with language model correction"""
    
    # Original prediction
    prediction = predict_single_image(model, decoder, image_path, device)
    
    # Apply language model correction
    corrected_greedy = lm.correct_text(prediction['greedy'])
    corrected_beam = lm.correct_text(prediction['beam_search'])
    
    return {
        'greedy': prediction['greedy'],
        'greedy_corrected': corrected_greedy,
        'beam_search': prediction['beam_search'],
        'beam_search_corrected': corrected_beam,
        'confidence': prediction['confidence']
    }

# Usage in batch inference:
# for image_file in image_files:
#     result = predict_with_correction(model, decoder, image_file, device)
#     print(f"Original: {result['greedy']}")
#     print(f"Corrected: {result['greedy_corrected']}")
