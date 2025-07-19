# Corrected HTR Image Chunking and Merging Logic

## Overview

This document describes the corrected chunking and merging strategy implemented in the HTR model, where overlapping chunks are processed with specific ignored regions during feature merging.

## Chunking Parameters

- **Chunk width**: 320px
- **Stride**: 240px  
- **Overlap**: 80px (320 - 240)
- **Padding**: 40px (left padding for first chunk)
- **Ignored during merging**: 40px from each side of overlap regions

## Chunking Strategy

### First Chunk
- **Content**: 0 → 280px (280px actual content)
- **Left padding**: 40px gray padding added
- **Total chunk size**: 320px (40px padding + 280px content)
- **Ignored during merging**: 40px from right side (within overlap region)

### Middle Chunks
- **Content**: Starts at (previous_start + stride) with 80px overlap
- **Total chunk size**: 320px (full content width)
- **Ignored during merging**: 40px from left side + 40px from right side

### Last Chunk
- **Content**: Remaining content from image
- **Right padding**: Added if needed to reach 320px
- **Ignored during merging**: 40px from left side (within overlap region)

## Merging Logic (Fixed)

The corrected `_merge_chunk_features` method now implements the proper ignored region logic:

### Single Chunk
- Remove only left padding (40px)
- Use all remaining content

### First Chunk (Multi-chunk scenario)
- Remove left padding (40px) 
- Remove 40px ignored region from right side
- **Usable content**: padding_removed_content - 40px_right

### Middle Chunks
- Remove 40px ignored region from left side
- Remove 40px ignored region from right side  
- **Usable content**: chunk_content - 40px_left - 40px_right

### Last Chunk
- Remove 40px ignored region from left side
- Keep all content to the right
- **Usable content**: chunk_content - 40px_left

## Visualization Color Coding

The `new_chunk_visualizer.py` shows:

- 🔴 **RED overlays**: Actual padding (gray background added to chunks)
- 🔵 **LIGHT BLUE overlays**: Overlap regions (80px on applicable sides)  
- ⚪ **GRAY overlays**: Ignored regions during merging (40px from each side of overlap)
- ⚫ **No overlay**: Useful content that gets processed and merged

## Example for 3-Chunk Image

```
Chunk 1: [PAD_40px][content_240px][ignored_40px]
Chunk 2: [ignored_40px][content_240px][ignored_40px]  
Chunk 3: [ignored_40px][content_remaining]

Merged: [content_240px][content_240px][content_remaining]
```

## Content Efficiency

With the corrected logic:
- **Previous logic**: Removed entire 80px overlap from both sides
- **New logic**: Removes only 40px from each side of overlap  
- **Improvement**: Higher content utilization and better feature continuity

## Key Benefits

1. **Better content utilization**: Only 40px ignored instead of 80px
2. **Smoother transitions**: Maintains some overlap content for better feature continuity
3. **Accurate visualization**: Visual representation matches actual merging behavior
4. **Optimal balance**: Reduces redundancy while preserving useful overlap information

## Files Updated

- `model/HTR_ME.py`: Updated `_merge_chunk_features()` method
- `new_chunk_visualizer.py`: Corrected overlay positioning and efficiency calculation
- Test suite passes with new merging logic

This corrected implementation now properly matches the intended chunking strategy with 40px ignored regions from each side of the 80px overlap areas.
