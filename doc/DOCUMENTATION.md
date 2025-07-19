# HTR Model with CvT Backbone - Complete Implementation

## Overview

This is a complete implementation of a Handwritten Text Recognition (HTR) model using **CvT (Convolutional Vision Transformer)** as the backbone, with an advanced chunking strategy inspired by "Rethinking Text Line Recognition Models". The model processes single-line handwritten images and outputs character predictions using CTC Loss.

## Architecture

### 🧠 Core Components

1. **CvT Backbone**: Convolutional Vision Transformer for feature extraction
2. **Chunking Strategy**: Overlapping segments with bidirectional padding
3. **CTC Loss**: Connectionist Temporal Classification for sequence alignment
4. **Language Model Decoding**: Optional KenLM integration for improved accuracy

### 📐 Model Specifications

- **Input**: RGB images, 40px height, variable width
- **Chunking**: 256px width, 192px stride, 32px padding
- **CvT Config**: 384-dim embeddings, 6 heads, 10 blocks
- **Output**: Character probabilities per time step
- **Parameters**: ~18.4M (depending on vocabulary size)

## Files Structure

```
HTR-Me/
├── model/
│   ├── __init__.py           # Package initialization
│   └── HTR_ME.py            # Main model implementation
├── train.py                 # Training script
├── inference.py             # Inference script
├── create_dataset.py        # Dataset creation utilities
├── test_model.py           # Model testing
├── demo.py                 # Usage demonstration
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Key Features

### ✨ Advanced Chunking Strategy

- **Aspect Ratio Preservation**: Images resized to 40px height while maintaining proportions
- **Overlapping Chunks**: 256px chunks with 192px stride for context preservation
- **Bidirectional Padding**: 32px padding on each side to reduce boundary artifacts
- **Independent Processing**: Each chunk processed separately through CvT
- **Smart Merging**: Padding removal and concatenation of valid tokens

### 🔄 CvT (Convolutional Vision Transformer)

- **Patch Embedding**: 7x7 convolution with stride 4
- **Convolutional Attention**: 3x3 convolutions for positional encoding
- **Multi-Head Attention**: 6 heads for capturing diverse patterns
- **Deep Architecture**: 10 transformer blocks for rich representations
- **Layer Normalization**: Stable training with proper normalization

### 🎯 Training & Loss

- **CTC Loss**: Handles variable-length sequences without explicit alignment
- **AdamW Optimizer**: Weight decay for regularization (default)
- **SAM Optimizer**: Optional Sharpness-Aware Minimization for better generalization
- **Cosine Annealing**: Learning rate scheduling for optimal convergence
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: Optional for faster training (requires AMP)

### 🔍 Inference & Decoding

- **Greedy Decoding**: Fast baseline decoding
- **Beam Search**: Improved accuracy with beam width control
- **Language Model**: Optional KenLM integration for character-level n-grams
- **Confidence Scores**: Model uncertainty estimation

## Quick Start

### 1. Installation

```bash
# Core dependencies
pip install torch torchvision pillow numpy

# Optional enhanced features
pip install opencv-python kenlm pyctcdecode

# Or install everything
pip install -r requirements.txt
```

### 2. Test Installation

```bash
python test_model.py
```

### 3. Create Sample Dataset

```bash
python create_dataset.py --mode create --output_dir ./sample_data --num_samples 20
```

### 4. Train Model

```bash
python train.py \\
  --data_dir ./sample_data \\
  --output_dir ./checkpoints \\
  --epochs 50 \\
  --batch_size 8 \\
  --lr 1e-4

# With SAM optimizer
python train.py \\
  --data_dir ./sample_data \\
  --output_dir ./checkpoints \\
  --epochs 50 \\
  --batch_size 8 \\
  --lr 1e-4 \\
  --use_sam \\
  --sam_rho 0.05 \\
  --base_optimizer adamw
```

### 5. Run Inference

```bash
# Single image
python inference.py \\
  --checkpoint ./checkpoints/best_model.pth \\
  --image /path/to/image.jpg

# Batch processing
python inference.py \\
  --checkpoint ./checkpoints/best_model.pth \\
  --image_dir /path/to/images/ \\
  --output results.json
```

## Data Format

### Dataset Structure
```
data/
├── images/
│   ├── line_001.jpg
│   ├── line_002.jpg
│   └── ...
├── annotations.json
└── vocab.json
```

### Annotation Format
```json
[
  {
    "image": "images/line_001.jpg",
    "text": "This is handwritten text"
  },
  {
    "image": "images/line_002.jpg",
    "text": "Another line of text"
  }
]
```

## Configuration Options

### Model Configuration
```python
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
```

### Training Configuration
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

### Inference Configuration
```python
decoder = CTCDecoder(
    vocab=vocab,
    lm_path="/path/to/language_model.arpa",  # Optional
    alpha=0.5,    # Language model weight
    beta=1.0      # Word insertion bonus
)
```

## Performance Optimization

### Memory Optimization
- **Chunking**: Reduces memory usage for long text lines
- **Gradient Checkpointing**: Trade compute for memory
- **Batch Size**: Adjust based on available GPU memory

### Speed Optimization
- **GPU Processing**: CUDA acceleration for training and inference
- **Parallel Chunk Processing**: Independent chunk computation
- **Mixed Precision**: FP16 training for faster computation

### Accuracy Optimization
- **Data Augmentation**: Rotation, scaling, noise for robustness
- **Language Models**: Character-level n-grams for context
- **Ensemble Methods**: Multiple model predictions
- **Fine-tuning**: Domain-specific adaptation

## Advanced Usage

### Custom Vocabulary
```python
def create_custom_vocab():
    # Add your specific characters
    chars = "your_custom_characters"
    vocab = ['<blank>', '<unk>'] + list(chars)
    return vocab
```

### Custom Data Loader
```python
class CustomHTRDataset(Dataset):
    def __init__(self, data_dir, vocab, transform=None):
        # Your custom implementation
        pass
```

### Transfer Learning
```python
# Load pretrained weights
checkpoint = torch.load("pretrained_model.pth")
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Freeze backbone, train only classifier
for param in model.cvt.parameters():
    param.requires_grad = False
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable mixed precision

2. **Poor Recognition Accuracy**
   - Increase training epochs
   - Add data augmentation
   - Use language model decoding
   - Check data quality

3. **Slow Training**
   - Use GPU acceleration
   - Enable mixed precision
   - Optimize data loading
   - Reduce model size

4. **Import Errors**
   - Install missing dependencies
   - Check Python path
   - Verify package versions

### Performance Monitoring
```python
# Monitor GPU usage
nvidia-smi

# Profile training
python -m torch.profiler train.py

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{htr_cvt_2025,
  title={HTR Model with CvT Backbone: Convolutional Vision Transformer for Handwritten Text Recognition},
  author={HTR Team},
  year={2025},
  note={Implementation based on CvT architecture and chunking strategies}
}
```

## License

This implementation is provided for research and educational purposes. Please respect the licenses of underlying dependencies (PyTorch, etc.).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For questions and issues:
1. Check the troubleshooting section
2. Run the test suite: `python test_model.py`
3. Review the demo: `python demo.py --demo`
4. Create an issue with detailed error logs

---

**Happy Text Recognition! 🎉**
