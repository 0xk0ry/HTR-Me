## HTR Model Chunking Process Analysis

This document explains how your HTR model processes images through chunking, feature extraction, and merging.

### Key Configuration
- **Chunk dimensions**: 320px width × 40px height
- **Stride**: 240px (chunks overlap by 80px)
- **Padding**: 40px grey padding added to single chunks
- **Patch size**: 7×7 with stride 4
- **Final embedding**: 384 dimensions

### Process Flow

#### 1. Image → Chunks
```
Original Image (W × 40px) → Multiple 320×40px chunks
```

**Examples:**
- 200px → 1 chunk (with padding)
- 400px → 2 chunks (overlapping)
- 800px → 4 chunks
- 1000px → 4 chunks

#### 2. Chunks → CvT Features
Each 320×40px chunk becomes:
```
[3, 40, 320] → CvT → [1, H'×W', 384] → Height pooling → [W', 384]
```

Where:
- **H'** = (40-7)//4 + 1 = **9 patches** (height)
- **W'** = (320-7)//4 + 1 = **79 patches** (width)
- After height pooling: **79 time steps** × **384 features**

#### 3. Feature Merging Logic

The model removes padded and overlapping regions:

**Single Chunk:**
- Remove left padding: ~10 patches (40px ÷ 4)
- Remove right padding if image is smaller than chunk
- Result: Valid content only

**Multiple Chunks:**
- **First chunk**: Remove left padding + right overlap (40px each)
- **Middle chunks**: Remove left + right overlaps (40px each side)  
- **Last chunk**: Remove left overlap + right padding

**Ignored overlap**: 40px = 10 patches on each overlapping side

### Example Results

| Image Width | Chunks | Total Patches | Valid Patches | Final Sequence Length |
|-------------|--------|---------------|---------------|----------------------|
| 200px       | 1      | 79            | 49            | 49                   |
| 400px       | 2      | 158           | 98            | 98                   |
| 600px       | 3      | 237           | 147           | 147                  |
| 800px       | 4      | 316           | 196           | 196                  |
| 1000px      | 4      | 316           | 246           | 246                  |

### Patch Index Calculations

For each chunk with 79 total patches:

**Single Chunk (e.g., 200px image):**
```
├─ Left padding removed: patches [0:10]    (40px ÷ 4 = 10)
├─ Valid content: patches [10:59]          (49 patches)
└─ Right padding removed: patches [59:79]  (20 patches)
```

**First Chunk (multi-chunk case):**
```
├─ Left padding removed: patches [0:10]    (40px padding)
├─ Valid content: patches [10:69]          (59 patches)
└─ Right overlap ignored: patches [69:79]  (40px overlap)
```

**Middle Chunks:**
```
├─ Left overlap ignored: patches [0:10]    (40px overlap)
├─ Valid content: patches [10:69]          (59 patches)  
└─ Right overlap ignored: patches [69:79]  (40px overlap)
```

**Last Chunk:**
```
├─ Left overlap ignored: patches [0:10]    (varies by content)
├─ Valid content: patches [10:X]           (varies by image size)
└─ Right padding removed: patches [X:79]   (if image < chunk width)
```

### Shape Transformations

```
Input Image: [Batch, 3, 40, W]
     ↓ Chunking
Chunks: [N_chunks, 3, 40, 320]
     ↓ CvT + Height Pooling  
Chunk Features: [N_chunks, 79, 384]
     ↓ Merging (remove padding/overlaps)
Merged Features: [Total_valid_patches, 384]
     ↓ Classifier
Final Logits: [Total_valid_patches, 1, Vocab_size]
     ↓ CTC Format
CTC Input: [Total_valid_patches, Batch, Vocab_size]
```

### Key Insights

1. **Height pooling** is effective - reduces 9×79=711 patches to 79 time steps per chunk
2. **Overlap handling** prevents information loss at chunk boundaries
3. **Padding removal** ensures only real content is processed
4. **Sequence length** scales roughly linearly with image width
5. **Fixed chunk width** (320px) provides consistent processing

### Performance Implications

- **Memory**: Fixed chunk size means consistent memory usage regardless of image width
- **Computation**: Linear scaling with number of chunks needed
- **Accuracy**: Overlap regions get processed twice, potentially improving boundary accuracy
