# HTR Model Image Chunking Visualization Tools

This collection of Python scripts helps you understand and visualize how your HTR (Handwritten Text Recognition) model processes images through chunking.

## 📁 Generated Files

After running the visualization scripts, you'll find these output files in the `prototype/` folder:

### Core Visualization Files
- `test_6_original.png` - Your original input image
- `test_6_preprocessed.png` - Image resized to 40px height (model input)
- `test_6_with_boundaries.png` - Shows colored chunk boundaries on the image
- `test_6_chunk_01.png`, `test_6_chunk_02.png`, etc. - Individual chunks with padding

### Analysis Files
- `test_6_chunking_summary.png` - Complete overview with all chunks displayed
- `test_6_chunking_visualization.png` - Detailed visualization with annotations
- `test_6_parameter_comparison.png` - Comparison of different chunking parameters

## 🔧 Available Scripts

### 1. `visualize_chunking.py` - Basic Visualization
**Purpose**: Shows how your image gets chunked with the default HTR model parameters.

```bash
python visualize_chunking.py
```

**What it shows**:
- Original vs preprocessed image
- Individual chunk boundaries
- Each chunk saved as separate image
- Summary statistics

### 2. `simple_chunk_visualizer.py` - Robust Visualizer
**Purpose**: More robust version that handles edge cases better.

```bash
python simple_chunk_visualizer.py
```

**Features**:
- Error handling for various image formats
- Clean output with detailed statistics
- Overlap analysis between chunks

### 3. `chunk_parameter_explorer.py` - Parameter Analysis
**Purpose**: Compare different chunking strategies and their effects.

```bash
# Compare different parameter sets
python chunk_parameter_explorer.py --compare

# Test custom parameters
python chunk_parameter_explorer.py --chunk-width 320 --stride 256 --padding 16
```

**What it analyzes**:
- Memory usage comparison
- Coverage efficiency
- Processing intensity
- Overlap patterns

## 📊 Understanding the Output

### Chunking Process
Your HTR model processes images in these steps:

1. **Preprocessing**: Resize to 40px height while preserving aspect ratio
2. **Chunking**: Split into 256px wide segments with 192px stride
3. **Padding**: Add 32px padding on each side to reduce boundary effects
4. **Processing**: Each chunk goes through the CvT (Convolutional Vision Transformer)
5. **Merging**: Remove padding and concatenate results

### Key Metrics

- **Coverage Efficiency**: How much of the image is processed vs redundant processing
- **Overlap**: How much chunks overlap (default: 64px = 25% of chunk width)
- **Memory Usage**: Total memory required for all chunks
- **Chunks/100px**: Processing intensity metric

### Default HTR Parameters
- **Target Height**: 40px
- **Chunk Width**: 256px  
- **Stride**: 192px
- **Padding**: 32px

This means:
- Each chunk overlaps by 64px (256 - 192)
- 25% overlap provides context for better recognition
- Padding reduces artifacts at chunk boundaries

## 🎯 Example Analysis Results

For the test image `test_6.png` (1710×124 → 551×40):
- **Chunks Created**: 3
- **Memory Usage**: ~0.4MB
- **Coverage Efficiency**: 81.1%
- **Overlap**: 64px between adjacent chunks

### Chunk Breakdown:
1. **Chunk 1**: Position 0-256 (no left padding)
2. **Chunk 2**: Position 192-448 (32px left padding)  
3. **Chunk 3**: Position 384-551 (32px left padding)

## 🔍 How to Use These Tools

1. **Place your test images** in the `prototype/` folder
2. **Run any of the visualization scripts**
3. **Check the generated images** to see how chunking works
4. **Experiment with parameters** using the parameter explorer

## 💡 Tips for Optimization

- **More Overlap**: Better accuracy but higher memory usage
- **Less Overlap**: Faster processing but potential accuracy loss  
- **Larger Chunks**: Less processing overhead, higher memory per chunk
- **Smaller Chunks**: More processing overhead, lower memory per chunk
- **No Padding**: Minimal memory but potential boundary artifacts

## 🚀 Quick Start

```bash
# Basic visualization of your image chunking
python simple_chunk_visualizer.py

# Compare different chunking strategies  
python chunk_parameter_explorer.py --compare

# Test custom parameters
python chunk_parameter_explorer.py --chunk-width 320 --stride 256
```

The scripts will automatically find images in your `prototype/` folder and generate comprehensive visualizations showing exactly how your HTR model processes the images!
