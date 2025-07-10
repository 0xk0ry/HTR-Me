# ✅ Corrected HTR Image Chunking Visualization

## 🔧 Fixed Implementation

Your HTR model chunking visualization now correctly shows padding only where it's actually applied:

### 📊 Chunking Logic
- **Chunk Width**: 320px
- **Stride**: 240px  
- **Overlap**: 80px (320 - 240)
- **Padding**: 40px grey background

### 🎯 Padding Rules (Updated)

1. **First Chunk**: 
   - ✅ Gets LEFT padding of 40px grey background
   - 📏 Content covers 0→280px (280px content + 40px padding = 320px total)
   - 🎨 Shows gray background with "LEFT PAD (40px)" label

2. **Middle Chunks**: 
   - ✅ 320px wide with 80px overlap from previous chunk
   - 📏 Example: 200→520px content
   - 🎨 Shows overlap regions with semi-transparent overlay

3. **Last Chunk**: 
   - ✅ Gets RIGHT padding if needed to reach 320px width
   - 🎨 Shows gray background with "RIGHT PAD (Xpx)" label

### 📁 Generated Files

For your `test_6.png` example:
- `test_6_chunk_01.png` - First chunk: 40px left padding + 0→280px content
- `test_6_chunk_02.png` - Second chunk: 200→520px content (80px overlap)
- `test_6_chunk_03.png` - Third chunk: 440→760px content (80px overlap)
- Last chunk gets right padding if needed to reach 320px width

### 🔍 Visual Indicators

- **Gray Background**: Shows actual padding areas (40px left, variable right)
- **Semi-transparent Overlay**: Shows 80px overlap regions between chunks
- **White Text**: Labels showing padding and overlap sizes
- **Chunk Titles**: Display position, size, and overlap information

### 📐 Example Chunking Positions

For an image that's 760px wide:
1. **Chunk 1**: 40px padding + 0→280px content = 320px total
2. **Chunk 2**: 200→520px content = 320px total (80px overlap: 200→280)
3. **Chunk 3**: 440→760px content = 320px total (80px overlap: 440→520)

This implementation now correctly reflects how your HTR model actually processes images, making it clear where padding is applied and why!

## 🚀 Usage

```bash
python simple_chunk_visualizer.py
```

The script will automatically find images in the `prototype/` folder and generate comprehensive visualizations with correct padding indicators.
