# 🎯 New HTR Image Chunking Visualizer

## 🔧 Clean Implementation

This new visualizer correctly shows all aspects of the HTR model's chunking strategy with proper color coding.

## 📊 Chunking Parameters
- **Chunk Width**: 320px
- **Stride**: 240px  
- **Overlap**: 80px (320 - 240)
- **Left Padding**: 40px gray background for first chunk

## 🎨 Color Coding

### 🔴 RED Overlays = Actual Padding
- **Left padding**: 40px gray background added to first chunk
- **Right padding**: Added to last chunk if needed to reach 320px width
- This is actual padding that gets added to the image data

### ⚪ GRAY Overlays = Ignored During Merging
- **Left ignored region**: 80px overlap from previous chunk (ignored during feature merging)
- **Right ignored region**: 80px overlap with next chunk (ignored during feature merging)
- These regions contain real image content but are discarded when merging features

### 🔵 LIGHT BLUE Overlays = Overlap Regions (Informational)
- Shows the actual overlap areas between consecutive chunks
- Provides visual confirmation of the 80px overlap strategy
- Helps understand which parts of the image appear in multiple chunks

### ⚫ No Overlay = Useful Content
- Clean regions with no overlay represent the actual useful content
- This is the content that gets processed and used in the final merged features

## 📐 Example Chunking

For a 959px wide image (test_24.png):
1. **Chunk 1**: 🔴40px red padding + 0→280px content + 🔵80px light blue overlap = 320px
2. **Chunk 2**: ⚪80px gray ignored + 200→520px content + 🔵80px light blue overlap = 320px  
3. **Chunk 3**: ⚪80px gray ignored + 440→760px content + 🔵80px light blue overlap = 320px
4. **Chunk 4**: ⚪80px gray ignored + 680→959px content + 🔴red padding = 320px

## 📁 Generated Files

### Individual Chunks
- `*_new_chunk_01.png` - First chunk with red left padding and gray right ignored region
- `*_new_chunk_02.png` - Middle chunk with gray ignored regions on both sides
- `*_new_chunk_03.png` - Middle chunk with gray ignored regions on both sides
- `*_new_chunk_04.png` - Last chunk with gray left ignored region and red right padding

### Comprehensive Visualization
- `*_new_chunking_visualization.png` - Complete visualization showing:
  - Original image
  - Preprocessed image with chunk boundaries
  - Individual chunks with color-coded overlays
  - Legend explaining the color coding

## 🔍 Key Insights

1. **Overlap Strategy**: Each chunk overlaps 80px with neighbors for context
2. **Merging Strategy**: During feature merging, overlapped regions are ignored to prevent duplication
3. **Padding Strategy**: Only first and last chunks get actual padding (gray background)
4. **Efficiency**: ~75% of content is useful, 25% is overlapped for context

## 🚀 Usage

```bash
python new_chunk_visualizer.py
```

The script automatically finds images in the `prototype/` folder and generates comprehensive visualizations with correct color coding that matches how the HTR model actually processes the chunks.

## ✅ Fixes Applied

1. **Fixed last chunk overlay**: Now correctly sized based on actual right padding needed
2. **Added gray overlays**: Shows ignored regions on left/right sides during feature merging  
3. **Proper color coding**: Red for padding, gray for ignored regions, clear for useful content
4. **Accurate sizing**: All overlays are properly proportioned to the actual image regions
