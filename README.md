# HTR Model Configuration

## Model Architecture
- **Backbone**: CvT (Convolutional Vision Transformer)
- **Input Processing**: Chunking with overlapping segments
- **Output**: Character-level predictions with CTC loss
- **Decoding**: CTC decoding with optional KenLM language model

## Key Features

### 1. Chunking Strategy
Based on "Rethinking Text Line Recognition Models":
- Resize images to 40px height while preserving aspect ratio
- Divide horizontally into overlapping chunks (256px wide, 192px stride)
- Add bidirectional padding (32px) to reduce boundary effects
- Process chunks independently through CvT
- Remove padding and concatenate valid tokens

### 2. CvT Architecture
- **Stage 1**: 64-dim embeddings, 1 head, 1 block
- **Stage 2**: 192-dim embeddings, 3 heads, 2 blocks  
- **Stage 3**: 384-dim embeddings, 6 heads, 10 blocks
- Convolutional attention with 3x3 kernels
- Patch sizes: [7, 3, 3], Strides: [4, 2, 2]

### 3. Training Configuration
- **Loss**: CTC Loss (blank token = 0)
- **Optimizer**: AdamW with weight decay 0.01
- **Learning Rate**: 1e-4 with cosine annealing
- **Batch Size**: 8 (adjustable)
- **Gradient Clipping**: Max norm 1.0

### 4. Inference Options
- **Greedy Decoding**: Simple CTC decoding
- **Beam Search**: With optional KenLM language model
- **Beam Width**: 100 (configurable)

## Usage

### Training
```bash
python train.py \\
  --data_dir /path/to/training/data \\
  --val_data_dir /path/to/validation/data \\
  --output_dir ./checkpoints \\
  --epochs 100 \\
  --batch_size 8 \\
  --lr 1e-4 \\
  --lm_path /path/to/language_model.arpa
```

### Inference
```bash
# Single image
python inference.py \\
  --checkpoint ./checkpoints/best_model.pth \\
  --image /path/to/image.jpg \\
  --lm_path /path/to/language_model.arpa

# Batch processing
python inference.py \\
  --checkpoint ./checkpoints/best_model.pth \\
  --image_dir /path/to/images/ \\
  --output results.json \\
  --lm_path /path/to/language_model.arpa
```

## Data Format

### Training Data Structure
```
data/
├── images/
│   ├── line_001.jpg
│   ├── line_002.jpg
│   └── ...
├── annotations.json  # OR individual .txt files
└── vocab.json
```

### Annotations Format
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

### Alternative: Paired Files
- `line_001.jpg` + `line_001.txt`
- `line_002.jpg` + `line_002.txt`

## Model Parameters
- Total parameters: ~15M (depends on configuration)
- Input: RGB images, variable width, 40px height
- Output: Character probabilities per time step
- Vocabulary: Configurable (default: ASCII + common symbols)

## Dependencies
See `requirements.txt` for complete list:
- torch >= 1.9.0
- torchvision >= 0.10.0
- pillow >= 8.0.0
- numpy >= 1.21.0
- opencv-python >= 4.5.0 (optional)
- kenlm >= 0.0.0 (optional, for LM)
- pyctcdecode >= 0.4.0 (optional, for LM decoding)

## Performance Considerations
- **Memory**: Chunking reduces memory usage for long text lines
- **Speed**: Parallel chunk processing on GPU
- **Accuracy**: Bidirectional padding reduces boundary artifacts
- **Generalization**: CvT backbone provides strong feature extraction

## Extending the Model
- **Custom Vocabularies**: Modify `create_vocab()` in training script
- **Different Languages**: Provide appropriate character sets and LM
- **Multi-line**: Extend chunking to handle paragraph-level inputs
- **Different Resolutions**: Adjust `target_height` and chunk parameters
