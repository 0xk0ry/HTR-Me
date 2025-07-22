"""
CvT Pipeline Visualizer
Visualizes how image chunks are processed through the CvT pipeline:
1. Chunk → Patches (PatchEmbed)
2. Patches → Stage 1 Features
3. Stage 1 → Stage 2 Features  
4. Stage 2 → Final Features
5. 2D Features → 1D Time Sequence
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from pathlib import Path
from PIL import Image
import seaborn as sns

from model.HTR_ME_3Stage import HTRModel, ImageChunker, DEFAULT_VOCAB
from chunk_visualizer_with_patch import tensor_to_pil


class CvTPipelineVisualizer:
    """Visualizes the CvT processing pipeline for HTR chunks"""

    def __init__(self, model_path=None):
        """Initialize the visualizer with a trained model or create a new one"""
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Create or load model
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
        else:
            self.model = self._create_model()

        self.model.eval()

        # Get model parameters for visualization
        # Handle both 3-stage and single-stage CvT models
        if hasattr(self.model.cvt, 'patch_embed'):
            # Single-stage model
            self.patch_size = self.model.cvt.patch_embed.patch_size
            self.stride = self.model.cvt.patch_embed.stride
            self.embed_dim = self.model.cvt.patch_embed.proj.out_channels
        else:
            # 3-stage model - use first stage parameters
            self.patch_size = self.model.cvt.stages[0].patch_embed.patch_size
            self.stride = self.model.cvt.stages[0].patch_embed.stride
            self.embed_dim = self.model.cvt.stages[0].patch_embed.proj.out_channels

    def _create_model(self, vocab=DEFAULT_VOCAB):
        """Create a new HTR model"""

        # Create model with same configuration
        model = HTRModel(
            vocab_size=len(vocab),
            max_length=256,
            target_height=40,
            chunk_width=320,
            stride=240,
            embed_dims=[64, 192, 384],
            num_heads=[1, 3, 6],
            depths=[1, 2, 10]
        )

        # Move model to device
        model.to(self.device)
        return model

    def _load_model(self, model_path):
        """Load a trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        vocab = checkpoint['vocab']

        model = self._create_model(vocab)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def extract_cvt_features(self, chunk_tensor):
        """Extract features at each stage of CvT processing"""
        with torch.no_grad():
            # Ensure chunk has correct format [B, C, H, W]
            if len(chunk_tensor.shape) == 3:
                # If [C, H, W], add batch dimension
                chunk_tensor = chunk_tensor.unsqueeze(0)  # [1, C, H, W]
            elif len(chunk_tensor.shape) == 2:
                # If [H, W], add both channel and batch dimensions
                chunk_tensor = chunk_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            chunk_tensor = chunk_tensor.to(self.device)
            
            # Check if model has 3-stage CvT
            if hasattr(self.model, 'extract_all_features'):
                # Use 3-stage model's feature extraction
                return self.model.extract_all_features(chunk_tensor)
            else:
                # Use single-stage model's feature extraction (original code)
                return self._extract_single_stage_features(chunk_tensor)

    def _extract_single_stage_features(self, chunk_tensor):
        """Extract features from single-stage CvT"""
        features_dict = {}

        # Stage 1: Patch Embedding
        x, (H, W) = self.model.cvt.patch_embed(chunk_tensor)
        features_dict['patch_embed'] = {
            'features': x.clone(),  # [1, H*W, embed_dim]
            'spatial_dims': (H, W),
            'description': f'Patch Embedding: {chunk_tensor.shape[2]}×{chunk_tensor.shape[3]} → {H}×{W} patches'
        }

        # Stage 2: Progressive through CvT blocks, capturing internal stages
        num_blocks = len(self.model.cvt.blocks)
        block_features = []
        block_internal_features = []

        for i, block in enumerate(self.model.cvt.blocks):
            # Internal stages: after norm1, after attn, after norm2, after mlp
            x_norm1 = block.norm1(x)
            attn_out = block.attn(x_norm1, H, W)
            x_attn = x + attn_out
            x_norm2 = block.norm2(x_attn)
            mlp_out = block.mlp(x_norm2)
            x_mlp = x_attn + mlp_out

            # Save features after each sub-stage
            block_internal_features.append({
                'block_idx': i,
                'after_norm1': x_norm1.clone(),
                'after_attn': x_attn.clone(),
                'after_norm2': x_norm2.clone(),
                'after_mlp': x_mlp.clone(),
                'spatial_dims': (H, W)
            })

            x = x_mlp
            if i % 2 == 0 or i == num_blocks - 1:  # Sample every 2nd block + last
                block_features.append({
                    'features': x.clone(),
                    'spatial_dims': (H, W),
                    'description': f'After Block {i+1}/{num_blocks}'
                })

        features_dict['blocks'] = block_features
        features_dict['block_internal'] = block_internal_features

        # Stage 3: Final normalization
        x = self.model.cvt.norm(x)
        features_dict['final_norm'] = {
            'features': x.clone(),
            'spatial_dims': (H, W),
            'description': f'Final Normalized Features: {H}×{W}×{self.embed_dim}'
        }

        # Stage 4: 2D → 1D conversion (height pooling)
        x_2d = x.reshape(1, H, W, -1)  # [1, H, W, C]
        x_1d = x_2d.mean(dim=1)  # [1, W, C] - average across height
        x_1d = x_1d.squeeze(0)  # [W, C]

        features_dict['time_sequence'] = {
            'features': x_1d.clone(),
            'spatial_dims': (W,),
            'description': f'Time Sequence: {W} time steps × {self.embed_dim} features'
        }

        return features_dict

    def visualize_patch_embedding(self, chunk_tensor, output_dir, base_name):
        """Visualize how the chunk is divided into patches"""
        chunk_pil = tensor_to_pil(chunk_tensor.squeeze(
            0) if len(chunk_tensor.shape) == 4 else chunk_tensor)

        fig, axes = plt.subplots(2, 1, figsize=(16, 8))

        # Original chunk
        axes[0].imshow(chunk_pil)
        axes[0].set_title(f'Input Chunk: {chunk_pil.size[0]}×{chunk_pil.size[1]} pixels',
                          fontweight='bold', fontsize=14)
        axes[0].axis('off')

        # Chunk with patch grid overlay
        axes[1].imshow(chunk_pil)

        # Calculate patch grid
        H, W = chunk_pil.size[1], chunk_pil.size[0]
        patch_h = (H - self.patch_size) // self.stride + 1
        patch_w = (W - self.patch_size) // self.stride + 1

        # Draw patch boundaries
        colors = plt.cm.Set3(np.linspace(0, 1, min(patch_h * patch_w, 12)))
        patch_idx = 0

        for i in range(patch_h):
            for j in range(patch_w):
                y = i * self.stride
                x = j * self.stride

                # Draw patch rectangle
                color = colors[patch_idx % len(colors)]
                rect = patches.Rectangle((x, y), self.patch_size, self.patch_size,
                                         linewidth=2, edgecolor=color, facecolor='none')
                axes[1].add_patch(rect)

                # Add patch number
                axes[1].text(x + self.patch_size//2, y + self.patch_size//2,
                             f'P{patch_idx+1}', ha='center', va='center',
                             fontsize=8, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

                patch_idx += 1

        axes[1].set_title(f'Patch Grid: {self.patch_size}×{self.patch_size} patches, stride={self.stride} → {patch_h}×{patch_w} = {patch_h*patch_w} patches',
                          fontweight='bold', fontsize=14)
        axes[1].axis('off')

        plt.tight_layout()

        # Save
        patch_viz_path = os.path.join(
            output_dir, f"{base_name}_patch_embedding.png")
        plt.savefig(patch_viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        return patch_viz_path, (patch_h, patch_w)

    def visualize_feature_maps(self, features_dict, patch_dims, output_dir, base_name):
        """Visualize feature maps at different stages"""
        H, W = patch_dims

        # Select key stages to visualize based on model type
        if 'stage_features' in features_dict:
            # 3-stage model
            stages_to_viz = [
                ('stage_0', 'Stage 1'),
                ('stage_1', 'Stage 2'),
                ('stage_2', 'Stage 3'),
                ('time_sequence', '1D Time Sequence')
            ]
        else:
            # Single-stage model (backward compatibility)
            stages_to_viz = [
                ('patch_embed', 'Patch Embedding'),
                ('blocks', 'CvT Blocks'),
                ('final_norm', 'Final Features'),
                ('time_sequence', '1D Time Sequence')
            ]

        # Create figure with custom layout for proper aspect ratios
        fig = plt.figure(figsize=(20, 5 * len(stages_to_viz)))

        # Define custom grid for better control
        gs = fig.add_gridspec(len(stages_to_viz), 3,
                              # Make first two columns wider for 2D plots
                              width_ratios=[8, 8, 3],
                              hspace=0.4, wspace=0.3)

        for stage_idx, (stage_key, stage_name) in enumerate(stages_to_viz):
            if 'stage_features' in features_dict and stage_key.startswith('stage_'):
                # 3-stage model: get specific stage features
                stage_num = int(stage_key.split('_')[1])
                stage_data = features_dict['stage_features'][stage_num]
            elif stage_key == 'blocks':
                # Use the last block features
                stage_data = features_dict[stage_key][-1]
            else:
                stage_data = features_dict[stage_key]

            features = stage_data['features']  # [1, N, C] or [N, C]
            if 'description' in stage_data:
                description = stage_data['description']
            else:
                spatial_dims = stage_data['spatial_dims']
                if stage_key == 'time_sequence':
                    W_stage = spatial_dims[0] if len(spatial_dims) == 1 else spatial_dims[1]
                    description = f'Time Sequence: {features.shape[0]} steps × {features.shape[1]} features'
                else:
                    H_stage, W_stage = spatial_dims
                    description = f'{stage_name}: {H_stage}×{W_stage} patches'

            if len(features.shape) == 3:
                features = features.squeeze(0)  # Remove batch dim: [N, C]

            # Create subplots for this stage
            ax1 = fig.add_subplot(gs[stage_idx, 0])
            ax2 = fig.add_subplot(gs[stage_idx, 1])
            ax3 = fig.add_subplot(gs[stage_idx, 2])

            # Visualization 1: Feature magnitude heatmap
            if stage_key == 'time_sequence':
                # For 1D sequence: [W, C]
                feat_magnitudes = torch.norm(
                    features, dim=1).cpu().numpy()  # [W]
                ax1.plot(feat_magnitudes, linewidth=2)
                ax1.set_title(f'{stage_name}: Feature Magnitudes')
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('Feature Magnitude')
                ax1.grid(True, alpha=0.3)
            else:
                # For 2D features: [H*W, C] → reshape to [H, W]
                feat_magnitudes = torch.norm(
                    features, dim=1).cpu().numpy()  # [H*W]
                # Use the correct spatial dimensions for this stage
                if 'stage_features' in features_dict and stage_key.startswith('stage_'):
                    H_viz, W_viz = stage_data['spatial_dims']
                else:
                    H_viz, W_viz = H, W
                feat_2d = feat_magnitudes.reshape(H_viz, W_viz)

                # Use equal aspect ratio for proper 1:8 visualization
                im1 = ax1.imshow(feat_2d, cmap='viridis', aspect='equal')
                ax1.set_title(f'{stage_name}: Feature Magnitudes')
                # Removed colorbar for a cleaner look

            # Visualization 2: First few feature channels
            if stage_key == 'time_sequence':
                # Show first 8 channels as lines
                for i in range(min(8, features.shape[1])):
                    ax2.plot(features[:, i].cpu().numpy(),
                             label=f'Ch {i+1}', alpha=0.7)
                ax2.set_title(f'{stage_name}: First 8 Channels')
                ax2.set_xlabel('Time Step')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.grid(True, alpha=0.3)
            else:
                # Show average of first 16 channels as 2D
                n_channels = min(16, features.shape[1])
                feat_avg = features[:, :n_channels].mean(
                    dim=1).cpu().numpy()  # [H*W]
                # Use the correct spatial dimensions for this stage
                if 'stage_features' in features_dict and stage_key.startswith('stage_'):
                    H_viz, W_viz = stage_data['spatial_dims']
                else:
                    H_viz, W_viz = H, W
                feat_2d = feat_avg.reshape(H_viz, W_viz)

                # Use equal aspect ratio for proper visualization
                im2 = ax2.imshow(feat_2d, cmap='plasma', aspect='equal')
                ax2.set_title(
                    f'{stage_name}: Avg of First {n_channels} Channels')
                # Removed colorbar for a cleaner look

            # Visualization 3: Feature statistics
            feat_mean = features.mean().item()
            feat_std = features.std().item()
            feat_min = features.min().item()
            feat_max = features.max().item()

            stats_text = f"""
            Shape: {list(features.shape)}
            Mean: {feat_mean:.4f}
            Std: {feat_std:.4f}
            Min: {feat_min:.4f}
            Max: {feat_max:.4f}
            
            {description}
            """

            ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            ax3.set_title(f'{stage_name}: Statistics')
            ax3.axis('off')

        plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)

        # Save
        features_viz_path = os.path.join(
            output_dir, f"{base_name}_cvt_features.png")
        plt.savefig(features_viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        return features_viz_path

    def create_pipeline_summary(self, chunk_tensor, features_dict, output_dir, base_name):
        """Create a summary visualization of the pipeline (vertical, after each stage)"""
        # Check if we have 3-stage features or single-stage features
        if 'stage_features' in features_dict:
            # 3-stage model
            stages = [('Input Chunk', None, None)]
            for i, stage_feat in enumerate(features_dict['stage_features']):
                feat = stage_feat['features']
                # Handle 4D features [B, C, H, W] -> convert to [H*W, C] for visualization
                if len(feat.shape) == 4:
                    B, C, H, W = feat.shape
                    feat = feat.squeeze(0).permute(1, 2, 0).reshape(H*W, C)  # [B, C, H, W] -> [H*W, C]
                elif len(feat.shape) == 3:
                    feat = feat.squeeze(0)  # Remove batch dim
                
                stages.append((f'After Stage {i+1}', feat, stage_feat['spatial_dims']))
            
            # Handle final features
            final_feat = features_dict['final_features']['features']
            if len(final_feat.shape) == 4:
                B, C, H, W = final_feat.shape
                final_feat = final_feat.squeeze(0).permute(1, 2, 0).reshape(H*W, C)
            elif len(final_feat.shape) == 3:
                final_feat = final_feat.squeeze(0)
            
            final_spatial_dims = features_dict['final_features']['spatial_dims'] if 'final_features' in features_dict else features_dict['stage_features'][-1]['spatial_dims']
            stages.append(('Final 2D Features', final_feat, final_spatial_dims))
            stages.append(('Time Sequence', features_dict['time_sequence']['features'], features_dict['time_sequence']['spatial_dims']))
        else:
            # Single-stage model (backward compatibility)
            stages = [
                ('Input Chunk', None, None),
                ('Patch Embedding', features_dict['patch_embed']['features'].squeeze(0), features_dict['patch_embed']['spatial_dims']),
                ('Final 2D Features', features_dict['final_norm']['features'].squeeze(0), features_dict['final_norm']['spatial_dims']),
                ('Time Sequence', features_dict['time_sequence']['features'], features_dict['time_sequence']['spatial_dims'])
            ]

        n_stages = len(stages)
        fig, axes = plt.subplots(n_stages, 1, figsize=(10, 2.5*n_stages))
        if n_stages == 1:
            axes = [axes]

        for i, (title, feat, spatial_dims) in enumerate(stages):
            ax = axes[i]
            if title == 'Input Chunk':
                chunk_pil = tensor_to_pil(chunk_tensor.squeeze(0) if len(chunk_tensor.shape) == 4 else chunk_tensor)
                ax.imshow(chunk_pil, aspect='equal')
                ax.set_title(f'1. Input Chunk\n320×40 pixels', fontweight='bold', fontsize=12)
                ax.axis('off')
            elif title == 'Time Sequence':
                time_features = feat  # [W, C]
                time_magnitudes = torch.norm(time_features, dim=1).cpu().numpy()
                time_len = len(time_magnitudes)
                square_size = int(np.ceil(np.sqrt(time_len)))
                time_square = np.zeros((square_size, square_size), dtype=np.float32)
                time_square.flat[:time_len] = time_magnitudes
                ax.imshow(time_square, cmap='Reds', aspect='equal', interpolation='nearest')
                ax.set_title(f'{i+1}. {title}', fontweight='bold', fontsize=12)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # 2D features: [H*W, C]
                H, W = spatial_dims
                feat_magnitudes = torch.norm(feat, dim=1).cpu().numpy()
                feat_2d = feat_magnitudes.reshape(H, W)
                ax.imshow(feat_2d, cmap='viridis', aspect='equal')
                ax.set_title(f'{i+1}. {title}\n{H}×{W} patches', fontweight='bold', fontsize=12)
                ax.axis('off')

        plt.tight_layout()
        pipeline_viz_path = os.path.join(output_dir, f"{base_name}_cvt_pipeline_summary.png")
        plt.savefig(pipeline_viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        return pipeline_viz_path

    def visualize_per_stage_blocks(self, features_dict, output_dir, base_name):
        """Create a visualization for each stage showing output after each block"""
        if 'per_stage_block_features' in features_dict:
            # 3-stage model
            for stage_data in features_dict['per_stage_block_features']:
                stage_idx = stage_data['stage_idx']
                block_feats = stage_data['block_features']
                H, W = stage_data['spatial_dims']
                
                n_blocks = len(block_feats)
                if n_blocks == 0:
                    continue
                    
                fig, axes = plt.subplots(n_blocks, 1, figsize=(10, 2.5*n_blocks))
                if n_blocks == 1:
                    axes = [axes]
                    
                for i, feat in enumerate(block_feats):
                    ax = axes[i] if n_blocks > 1 else axes[0]
                    feat_squeezed = feat.squeeze(0) if feat.dim() == 3 else feat
                    feat_magnitudes = torch.norm(feat_squeezed, dim=1).cpu().numpy().reshape(H, W)
                    ax.imshow(feat_magnitudes, cmap='viridis', aspect='equal')
                    ax.set_title(f'Stage {stage_idx+1} - After Block {i+1}\n{H}×{W} patches', fontweight='bold', fontsize=12)
                    ax.axis('off')
                    
                plt.tight_layout()
                stage_viz_path = os.path.join(output_dir, f"{base_name}_stage{stage_idx+1}_blocks.png")
                plt.savefig(stage_viz_path, dpi=150, bbox_inches='tight')
                plt.close()
        else:
            # Single-stage model (backward compatibility) 
            if 'block_internal' in features_dict:
                block_feats = [block['after_mlp'] for block in features_dict['block_internal']]
                n_blocks = len(block_feats)
                if n_blocks == 0:
                    return
                    
                H, W = features_dict['patch_embed']['spatial_dims']
                fig, axes = plt.subplots(n_blocks, 1, figsize=(10, 2.5*n_blocks))
                if n_blocks == 1:
                    axes = [axes]
                    
                for i, feat in enumerate(block_feats):
                    ax = axes[i] if n_blocks > 1 else axes[0]
                    feat_squeezed = feat.squeeze(0) if feat.dim() == 3 else feat
                    feat_magnitudes = torch.norm(feat_squeezed, dim=1).cpu().numpy().reshape(H, W)
                    ax.imshow(feat_magnitudes, cmap='viridis', aspect='equal')
                    ax.set_title(f'Single Stage - After Block {i+1}\n{H}×{W} patches', fontweight='bold', fontsize=12)
                    ax.axis('off')
                    
                plt.tight_layout()
                stage_viz_path = os.path.join(output_dir, f"{base_name}_single_stage_blocks.png")
                plt.savefig(stage_viz_path, dpi=150, bbox_inches='tight')
                plt.close()

    def visualize_chunk_processing(self, image_path, chunk_idx=0, output_dir="cvt_visualization"):
        """Main function to visualize CvT processing for a specific chunk"""
        print(f"🎯 Visualizing CvT processing for: {Path(image_path).name}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process image through chunker
        chunker = ImageChunker(target_height=40, chunk_width=320, stride=240)

        try:
            # Load and preprocess image
            original_image = Image.open(image_path).convert('RGB')
            preprocessed_image = chunker.preprocess_image(original_image)
            chunks, chunk_positions = chunker.create_chunks(preprocessed_image)

            print(f"   📊 Created {len(chunks)} chunks")

            # Select chunk to visualize
            if chunk_idx >= len(chunks):
                chunk_idx = 0
                print(
                    f"   ⚠️  Requested chunk {chunk_idx} not available, using chunk 0")

            selected_chunk = chunks[chunk_idx]
            base_name = f"{Path(image_path).stem}_chunk_{chunk_idx+1}"

            print(f"   🔍 Processing chunk {chunk_idx+1}/{len(chunks)}")

            # Extract features at each CvT stage
            features_dict = self.extract_cvt_features(selected_chunk)

            # Create visualizations
            print("   🎨 Creating patch embedding visualization...")
            patch_viz_path, patch_dims = self.visualize_patch_embedding(
                selected_chunk, output_dir, base_name)

            print("   🎨 Creating feature maps visualization...")
            features_viz_path = self.visualize_feature_maps(
                features_dict, patch_dims, output_dir, base_name)

            print("   🎨 Creating pipeline summary...")
            pipeline_viz_path = self.create_pipeline_summary(
                selected_chunk, features_dict, output_dir, base_name)

            print("   🎨 Creating per-stage block visualizations...")
            self.visualize_per_stage_blocks(features_dict, output_dir, base_name)

            # Print summary
            print(f"\n   📋 CvT Processing Summary:")
            print(f"      • Input chunk: {selected_chunk.shape}")
            print(
                f"      • Patch size: {self.patch_size}×{self.patch_size}, stride: {self.stride}")
            print(
                f"      • Patches: {patch_dims[0]}×{patch_dims[1]} = {patch_dims[0]*patch_dims[1]}")
            print(f"      • Embedding dim: {self.embed_dim}")
            
            # Handle different model types for block count
            if hasattr(self.model.cvt, 'blocks'):
                # Single-stage model
                print(f"      • CvT blocks: {len(self.model.cvt.blocks)}")
            elif hasattr(self.model.cvt, 'stages'):
                # 3-stage model
                total_blocks = sum(len(stage.blocks) for stage in self.model.cvt.stages)
                stage_blocks = [len(stage.blocks) for stage in self.model.cvt.stages]
                print(f"      • CvT stages: {len(self.model.cvt.stages)} ({stage_blocks[0]}, {stage_blocks[1]}, {stage_blocks[2]} blocks)")
                print(f"      • Total blocks: {total_blocks}")
            
            print(
                f"      • Final time sequence: {features_dict['time_sequence']['features'].shape[0]} steps")

            print(f"\n   ✅ Visualizations saved:")
            print(f"      • Patch embedding: {Path(patch_viz_path).name}")
            print(f"      • Feature maps: {Path(features_viz_path).name}")
            print(f"      • Pipeline summary: {Path(pipeline_viz_path).name}")

            return {
                'patch_viz': patch_viz_path,
                'features_viz': features_viz_path,
                'pipeline_viz': pipeline_viz_path,
                'features_dict': features_dict
            }

        except Exception as e:
            print(f"   ❌ Error processing: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function to run CvT visualization"""
    print("🔬 CvT Pipeline Visualizer")
    print("=" * 50)
    print("This tool visualizes how image chunks are processed through the CvT:")
    print("  📐 Patch embedding (image → patches)")
    print("  🧠 CvT transformer blocks (convolutional attention)")
    print("  📊 Feature maps at each stage")
    print("  ⏱️  2D → 1D time sequence conversion")
    print("=" * 50)

    # Look for images
    image_dir = "inference_data"
    if not os.path.exists(image_dir):
        print(f"❌ Directory not found: {image_dir}")
        return

    # Find images
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = []

    for file in os.listdir(image_dir):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(os.path.join(image_dir, file))

    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return

    print(f"📁 Found {len(image_files)} image(s) to process:\n")

    # Initialize visualizer
    visualizer = CvTPipelineVisualizer()

    # Process each image
    for image_path in image_files[:2]:  # Limit to first 2 images for demo
        result = visualizer.visualize_chunk_processing(image_path)
        print()

    print("✅ CvT visualization complete!")
    print(f"🗂️  All output files saved in: cvt_visualization/")


if __name__ == "__main__":
    main()
