"""
Simple ASCII visualization of the chunking process.
This creates text-based diagrams to understand the chunking visually.
"""

def visualize_chunking_ascii(image_width, chunk_width=320, stride=240, padding=40):
    """Create ASCII visualization of chunking process"""
    
    print(f"\n=== CHUNKING VISUALIZATION FOR {image_width}px IMAGE ===")
    print(f"Chunk width: {chunk_width}px, Stride: {stride}px, Padding: {padding}px")
    
    # Calculate chunks
    if image_width <= chunk_width - padding:
        # Single chunk
        chunks = [{'start': 0, 'end': image_width, 'left_pad': padding, 'type': 'single'}]
        print(f"\nSingle chunk case (image fits in one chunk)")
    else:
        # Multiple chunks
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < image_width:
            if chunk_idx == 0:
                # First chunk with padding
                end = min(start + chunk_width - padding, image_width)
                chunks.append({'start': start, 'end': end, 'left_pad': padding, 'type': 'first'})
                start += stride
            else:
                # Subsequent chunks
                end = min(start + chunk_width, image_width)
                chunk_type = 'last' if end >= image_width else 'middle'
                chunks.append({'start': start, 'end': end, 'left_pad': 0, 'type': chunk_type})
                start += stride
            chunk_idx += 1
    
    print(f"Number of chunks: {len(chunks)}")
    
    # Visualize image and chunks
    scale = max(1, image_width // 80)  # Scale for display
    image_display_width = image_width // scale
    
    print(f"\nImage visualization (scale 1:{scale}):")
    print("Image:  " + "=" * image_display_width)
    print("Pixels: 0" + " " * (image_display_width - len(str(image_width))) + str(image_width))
    
    # Show each chunk
    for i, chunk in enumerate(chunks):
        chunk_start_display = chunk['start'] // scale
        chunk_end_display = chunk['end'] // scale
        chunk_len_display = chunk_end_display - chunk_start_display
        
        # Create chunk visualization
        chunk_line = " " * chunk_start_display
        
        if chunk['type'] == 'single':
            chunk_line += f"[P{'C' * max(1, chunk_len_display-2)}P]"
        elif chunk['type'] == 'first':
            chunk_line += f"[P{'C' * max(1, chunk_len_display-3)}>>]"
        elif chunk['type'] == 'middle':
            chunk_line += f"[<<{'C' * max(1, chunk_len_display-4)}>>]"
        else:  # last
            chunk_line += f"[<<{'C' * max(1, chunk_len_display-3)}P]"
        
        print(f"Chunk{i}: {chunk_line}")
        print(f"        {chunk['start']}-{chunk['end']}px, pad:{chunk['left_pad']}px")
    
    print("\nLegend:")
    print("  C = Content pixels")
    print("  P = Padding pixels")  
    print("  << = Left overlap region")
    print("  >> = Right overlap region")
    print("  [] = Chunk boundaries")
    
    # Show patch-level analysis
    print(f"\n=== PATCH-LEVEL ANALYSIS ===")
    patch_stride = 4
    total_valid_patches = 0
    
    for i, chunk in enumerate(chunks):
        chunk_width_actual = 320  # All chunks are padded to 320px
        patches_per_chunk = (chunk_width_actual - 7) // patch_stride + 1  # 79 patches
        
        print(f"\nChunk {i} ({chunk['type']}):")
        print(f"  Chunk spans: {chunk['start']}-{chunk['end']}px")
        print(f"  Left padding: {chunk['left_pad']}px")
        print(f"  Total patches in chunk: {patches_per_chunk}")
        
        # Calculate valid patches (same logic as model)
        if len(chunks) == 1:
            # Single chunk
            start_idx = chunk['left_pad'] // patch_stride  # Remove left padding
            
            # Calculate content ratio for right padding
            actual_content_width = chunk['end'] - chunk['start']
            if actual_content_width < chunk_width_actual - chunk['left_pad']:
                content_ratio = (chunk['left_pad'] + actual_content_width) / chunk_width_actual
                actual_content_patches = max(1, int(content_ratio * patches_per_chunk))
                end_idx = min(actual_content_patches, patches_per_chunk)
            else:
                end_idx = patches_per_chunk
                
            valid_patches = end_idx - start_idx
            print(f"  Valid patch range: [{start_idx}:{end_idx}] = {valid_patches} patches")
            print(f"  Removed - Left padding: {start_idx}, Right padding: {patches_per_chunk - end_idx}")
            
        else:
            ignore_patches = max(1, (chunk_width - stride) // 2 // patch_stride)  # 10 patches
            
            if chunk['type'] == 'first':
                start_idx = chunk['left_pad'] // patch_stride
                end_idx = max(start_idx + 1, patches_per_chunk - ignore_patches)
                valid_patches = end_idx - start_idx
                print(f"  Valid patch range: [{start_idx}:{end_idx}] = {valid_patches} patches")
                print(f"  Removed - Left padding: {start_idx}, Right overlap: {patches_per_chunk - end_idx}")
                
            elif chunk['type'] == 'last':
                chunk_actual_width = chunk['end'] - chunk['start']
                if chunk_actual_width < chunk_width_actual:
                    content_ratio = chunk_actual_width / chunk_width_actual
                    chunk_actual_patches = max(1, int(content_ratio * patches_per_chunk))
                    start_idx = min(ignore_patches, chunk_actual_patches - 1)
                    end_idx = chunk_actual_patches
                else:
                    start_idx = min(ignore_patches, patches_per_chunk - 1)
                    end_idx = patches_per_chunk
                    
                valid_patches = end_idx - start_idx
                print(f"  Valid patch range: [{start_idx}:{end_idx}] = {valid_patches} patches")
                print(f"  Removed - Left overlap: {start_idx}, Right padding: {patches_per_chunk - end_idx}")
                
            else:  # middle
                start_idx = min(ignore_patches, patches_per_chunk // 2)
                end_idx = max(start_idx + 1, patches_per_chunk - ignore_patches)
                valid_patches = end_idx - start_idx
                print(f"  Valid patch range: [{start_idx}:{end_idx}] = {valid_patches} patches")
                print(f"  Removed - Left overlap: {start_idx}, Right overlap: {patches_per_chunk - end_idx}")
        
        total_valid_patches += valid_patches
        
        # Visualize patches
        patch_viz = ""
        for p in range(patches_per_chunk):
            if len(chunks) == 1:
                if start_idx <= p < end_idx:
                    patch_viz += "V"  # Valid
                else:
                    patch_viz += "X"  # Ignored/padding
            else:
                if chunk['type'] == 'first':
                    if p < start_idx:
                        patch_viz += "P"  # Padding
                    elif start_idx <= p < end_idx:
                        patch_viz += "V"  # Valid
                    else:
                        patch_viz += "O"  # Overlap
                elif chunk['type'] == 'last':
                    if p < start_idx:
                        patch_viz += "O"  # Overlap
                    elif start_idx <= p < end_idx:
                        patch_viz += "V"  # Valid
                    else:
                        patch_viz += "P"  # Padding
                else:  # middle
                    if p < start_idx:
                        patch_viz += "O"  # Left overlap
                    elif start_idx <= p < end_idx:
                        patch_viz += "V"  # Valid
                    else:
                        patch_viz += "O"  # Right overlap
        
        # Show patch visualization in groups of 10
        print(f"  Patch map: ", end="")
        for j in range(0, len(patch_viz), 10):
            print(patch_viz[j:j+10], end=" ")
        print()
        print(f"             P=Padding, V=Valid, O=Overlap, X=Ignored")
    
    print(f"\n=== FINAL RESULT ===")
    print(f"Total valid patches: {total_valid_patches}")
    print(f"Feature dimensions: {total_valid_patches} × 384")
    print(f"Ready for CTC: [{total_valid_patches}, 1, vocab_size]")

def main():
    """Test different image widths"""
    test_widths = [200, 400, 600, 800, 1000]
    
    for width in test_widths:
        visualize_chunking_ascii(width)
        print("\n" + "="*80)

if __name__ == "__main__":
    main()
