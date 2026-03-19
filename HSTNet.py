"""
HSTNet - Hyperspectral Segmentation Transformer Network

A novel architecture for pixel-level hyperspectral image segmentation.

Architecture Flow:
==================

Input: (B, C, H, W) - e.g., (16, 100, 64, 64)
  ↓
Stage 1: Spectral Feature Extraction
  3D Conv(1→8, kernel=3×1×1) + BN + ReLU
  Output: (B, 8*C, H, W) - e.g., (16, 800, 64, 64)
  ↓
Stage 2: Spatial Feature Extraction
  Option A: U-Net (multi-scale, recommended)
  Option B: Conv2D (faster, simpler)
  Output: (B, dim, H, W) - e.g., (16, 96, 64, 64)
  ↓
Stage 3: Hierarchical Patch-based Transformer (OUR INNOVATION)
  - Divide (64×64) into patches (8×8 each) = 64 patches
  - Each patch: 64 pixels (8×8)
  
  Stage 3a: Local Attention (within patches)
    - Process each patch through transformer independently
    - Add positional embedding to each patch
    - Transformer enhances features within each patch
    - Output: (B, 64, 64, dim) - 64 patches with local features
  
  Stage 3b: Global Attention (between patches) - OPTIONAL
    - Pool each patch to get patch-level token: (B, 64, dim)
    - Apply transformer between 64 patch tokens
    - Broadcast global context back to all pixels in each patch
    - Combine: local_features + global_features
  
  - Reconstruct back to (B, dim, H, W)
  ↓
Stage 4: Segmentation Head
  Conv2D layers for pixel-level classification
  Output: (B, num_classes, H, W) - e.g., (16, 13, 64, 64)

Memory Comparison (64×64 image, dim=96):
=========================================
SSFTT Tokenization:
  - Compress 4096 pixels → 6 tokens
  - Tokenization: 6 × 4096 × 96 = 2.36M parameters
  - Attention: 6² = 36 interactions
  - Problem: Compression loses spatial detail

HSTNet Patch-based (original):
  - Divide 4096 pixels → 64 patches of 64 pixels each
  - No compression parameters needed
  - Attention per patch: 64² = 4096 interactions (within patch)
  - Process 64 patches independently (parallelizable)
  - Problem: No cross-patch communication

HSTNet Hierarchical (NEW!):
  - Local attention: 64 patches × 64² = 262,144 interactions
  - Global attention: 64² = 4,096 interactions between patches
  - Benefit: Preserves spatial detail + captures global context
  - Memory efficient: O(P²) + O(N²) where P=64, N=64

Authors: Tianhan Peng (2025-2026)
Based on: SSFTT (Sun et al.), U-Net (Ronneberger et al.)
"""

import torch
import torch.nn as nn
from einops import rearrange


# ============================================================================
# Building Blocks (Similar to SSFTT but with better documentation)
# ============================================================================

class Residual(nn.Module):
    """Residual connection wrapper"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    """Layer normalization wrapper"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLPBlock(nn.Module):
    """MLP block with GELU activation (same as SSFTT's MLP_Block)"""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention (same as SSFTT's Attention)"""
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = torch.nn.functional.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    """Transformer block (same as SSFTT's Transformer)"""
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, MultiHeadAttention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLPBlock(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x


class TransformerBlockWithAttention(nn.Module):
    """
    Transformer block with optional attention enhancement (CBAM/PSA)
    
    Used for Stage 4 ablation study to test attention mechanisms.
    """
    def __init__(self, dim, depth, heads, mlp_dim, dropout, attention_module=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            layer_modules = [
                Residual(LayerNormalize(dim, MultiHeadAttention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLPBlock(dim, mlp_dim, dropout=dropout)))
            ]
            
            # Add attention module if specified
            if attention_module == 'cbam':
                from modules.attention import CBAM
                layer_modules.append(CBAM(dim))
            elif attention_module == 'psa':
                from modules.attention import PyramidSqueezeAttention
                layer_modules.append(PyramidSqueezeAttention(dim))
            
            self.layers.append(nn.ModuleList(layer_modules))
    
    def forward(self, x, mask=None):
        for layer_modules in self.layers:
            x = layer_modules[0](x, mask=mask)  # Attention
            x = layer_modules[1](x)  # MLP
            if len(layer_modules) > 2:  # Attention enhancement
                x = layer_modules[2](x)
        return x


# ============================================================================
# U-Net Component (Integrated for multi-scale features)
# ============================================================================

class LightweightUNet(nn.Module):
    """
    Lightweight U-Net for multi-scale spatial feature extraction
    
    INNOVATION: Replaces SSFTT's simple Conv2D with U-Net
    - Encoder-decoder architecture with skip connections
    - Captures features at multiple scales
    - Preserves fine spatial details through skip connections
    
    Architecture:
      Input: (B, in_channels, H, W)
        ↓ Encoder
      (B, 128, H, W)
        ↓ Pool
      (B, 128, H/2, W/2)
        ↓ Bottleneck
      (B, 256, H/2, W/2)
        ↓ Upsample
      (B, 128, H, W)
        ↓ Skip connection + Decoder
      (B, 128, H, W)
        ↓ Output conv
      (B, out_channels, H, W)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Encoder (single level to avoid size mismatch with small patches)
        self.enc1 = self._conv_block(in_channels, 128)
        self.pool1 = nn.MaxPool2d(2)  # Downsample by 2
        
        # Bottleneck (deeper feature extraction)
        self.bottleneck = self._conv_block(128, 256)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(256, 128)  # 256 due to skip connection
        
        # Output layer
        self.out_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        """Double convolution block with Dropout for regularization: Conv-BN-ReLU-Dropout-Conv-BN-ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),  # Dropout rate 0.15 to prevent overfitting
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # (B, 128, H, W)
        x = self.pool1(enc1)  # (B, 128, H/2, W/2)
        
        # Bottleneck
        x = self.bottleneck(x)  # (B, 256, H/2, W/2)
        
        # Decoder with skip connection
        x = self.up1(x)  # (B, 128, H, W)
        
        # Match spatial dimensions if needed
        if x.shape[2:] != enc1.shape[2:]:
            x = torch.nn.functional.interpolate(x, size=enc1.shape[2:], mode='nearest')
        
        x = torch.cat([x, enc1], dim=1)  # Skip connection: (B, 256, H, W)
        x = self.dec1(x)  # (B, 128, H, W)
        
        x = self.out_conv(x)  # (B, out_channels, H, W)
        return x


# ============================================================================
# Novel Components (Our Innovations)
# ============================================================================

class SpectralFeatureExtractor(nn.Module):
    """
    3D CNN for spectral feature extraction
    
    Same as SSFTT but encapsulated and with dynamic band support.
    Extracts correlations across spectral bands.
    """
    def __init__(self, in_channels=1, out_channels=8, spectral_bands=224):
        super().__init__()
        self.spectral_bands = spectral_bands
        self.out_channels = out_channels
        
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        """(B, C, H, W) → (B, out_channels*C, H, W)"""
        x = rearrange(x, 'b c h w -> b 1 c h w')
        x = self.conv3d(x)
        x = rearrange(x, 'b c d h w -> b (c d) h w')
        return x


class SpatialFeatureExtractor(nn.Module):
    """
    INNOVATION 1: U-Net for multi-scale spatial features
    
    SSFTT uses simple Conv2D, we use U-Net for better feature extraction.
    """
    def __init__(self, in_channels, out_channels, use_unet=False):
        super().__init__()
        self.use_unet = use_unet
        
        if use_unet:
            # INNOVATION: Multi-scale features with skip connections
            self.extractor = LightweightUNet(in_channels=in_channels, out_channels=out_channels)
        else:
            # Fallback: Simple Conv2D (like SSFTT)
            self.extractor = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
    
    def forward(self, x):
        """(B, in_channels, H, W) → (B, out_channels, H, W)"""
        return self.extractor(x)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sinusoidal positional embeddings (like in MAE/ViT)
    
    Args:
        embed_dim: embedding dimension (must be even)
        grid_size: int or tuple (height, width) of the grid
        cls_token: if True, add a class token position
    
    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size
    
    grid_h_coords = torch.arange(grid_h, dtype=torch.float32)
    grid_w_coords = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_h_coords, grid_w_coords, indexing='ij')  # (H, W)
    grid = torch.stack(grid, dim=0)  # (2, H, W)
    
    grid = grid.reshape(2, -1)  # (2, H*W)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)  # (H*W, D)
    
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M, 2) for 2D or (M,) for 1D
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    
    if pos.dim() == 2:  # 2D positions
        # pos: (2, M) -> flatten to (M, 2)
        pos = pos.transpose(0, 1)  # (M, 2)
        out_h = torch.einsum('m,d->md', pos[:, 0], omega)  # (M, D/2)
        out_w = torch.einsum('m,d->md', pos[:, 1], omega)  # (M, D/2)
        
        emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)  # (M, D)
        emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)  # (M, D)
        
        # Concatenate h and w embeddings (both include sin and cos)
        emb = torch.cat([emb_h, emb_w], dim=1)  # (M, 2*D)
        # Take first embed_dim dimensions
        emb = emb[:, :embed_dim]
    else:  # 1D positions
        out = torch.einsum('m,d->md', pos, omega)  # (M, D/2)
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (M, D)
    
    return emb


class PatchBasedTransformer(nn.Module):
    """
    INNOVATION 2: Patch-based transformer instead of tokenization
    
    KEY DIFFERENCE FROM SSFTT:
    - SSFTT: Learns token_wA and token_wV to compress H×W → L tokens
      Problem: Compression loses spatial structure, memory intensive
    
    - HSTNet: Divides into fixed P×P patches, processes independently
      Benefit: Preserves spatial structure, memory efficient, parallelizable
    """
    def __init__(self, dim, depth, heads, mlp_dim, dropout, patch_size=8, use_cross_patch=True, attention_module=None):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.use_cross_patch = use_cross_patch
        self.attention_module = attention_module
        
        # Learned positional embeddings for standard patch size (P×P)
        # These will be interpolated for different sizes
        self.pos_embedding = nn.Parameter(torch.randn(1, patch_size * patch_size, dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        # Local transformer (within patches)
        # Use TransformerBlockWithAttention if attention_module is specified
        if attention_module is not None:
            self.local_transformer = TransformerBlockWithAttention(
                dim, depth, heads, mlp_dim, dropout, attention_module=attention_module
            )
        else:
            self.local_transformer = TransformerBlock(dim, depth, heads, mlp_dim, dropout)
        
        # Global transformer (between patches) - NEW!
        if use_cross_patch:
            # Global transformer doesn't need attention enhancement (only local does)
            self.global_transformer = TransformerBlock(dim, depth=1, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
            # Learned patch-level positional embeddings (will be interpolated)
            # Initialize for a reasonable default size (e.g., 8×8 = 64 patches for 64×64 image)
            self.patch_pos_embedding = nn.Parameter(torch.randn(1, 64, dim) * 0.02)

    
    def forward(self, x):
        """
        (B, C, H, W) → (B, C, H, W)
        
        Process: 
        1. Divide into patches
        2. Local attention within each patch
        3. Global attention between patches (if enabled)
        4. Reconstruct
        """
        batch_size, C, H, W = x.shape
        P = self.patch_size
        
        # Pad to multiple of patch_size
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            _, _, H_pad, W_pad = x.shape
        else:
            H_pad, W_pad = H, W
        
        # Divide into patches
        num_patches_h = H_pad // P
        num_patches_w = W_pad // P
        num_patches = num_patches_h * num_patches_w
        
        # Rearrange: (B, C, H, W) → (B, num_patches, P*P, C)
        x = rearrange(x, 'b c (nh p1) (nw p2) -> b (nh nw) (p1 p2) c',
                     nh=num_patches_h, nw=num_patches_w, p1=P, p2=P)
        
        # Use learned positional embeddings (always P×P, no need to regenerate)


        
        # Stage 1: Local attention within patches
        # (B, num_patches, P*P, C) → (B*num_patches, P*P, C)
        x_local = x.reshape(batch_size * num_patches, P * P, C)
        x_local = x_local + self.pos_embedding
        x_local = self.dropout(x_local)
        x_local = self.local_transformer(x_local)
        
        # Reshape back: (B*num_patches, P*P, C) → (B, num_patches, P*P, C)
        x_local = x_local.reshape(batch_size, num_patches, P * P, C)
        
        # Stage 2: Global attention between patches (if enabled)
        if self.use_cross_patch:
            # Pool each patch to get patch-level tokens
            # (B, num_patches, P*P, C) → (B, num_patches, C)
            patch_tokens = x_local.mean(dim=2)
            
            # Interpolate patch positional embeddings if needed
            if num_patches != self.patch_pos_embedding.shape[1]:
                # Reshape to 2D grid for interpolation
                old_size = int(self.patch_pos_embedding.shape[1] ** 0.5)
                patch_pos_emb_2d = self.patch_pos_embedding.reshape(1, old_size, old_size, C).permute(0, 3, 1, 2)  # (1, C, H, W)
                
                # Interpolate to new size
                patch_pos_emb_2d = torch.nn.functional.interpolate(
                    patch_pos_emb_2d, 
                    size=(num_patches_h, num_patches_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Reshape back to sequence
                patch_pos_embedding = patch_pos_emb_2d.permute(0, 2, 3, 1).reshape(1, num_patches, C)  # (1, num_patches, C)
            else:
                patch_pos_embedding = self.patch_pos_embedding

            
            # Add positional embedding and apply global attention
            patch_tokens = patch_tokens + patch_pos_embedding
            patch_tokens_global = self.global_transformer(patch_tokens)

            
            # Broadcast global info back to pixels
            # (B, num_patches, C) → (B, num_patches, P*P, C)
            x_global = patch_tokens_global.unsqueeze(2).expand(-1, -1, P * P, -1)
            
            # Combine local + global features
            x = x_local + x_global
        else:
            x = x_local
        
        # Reconstruct: (B, num_patches, P*P, C) → (B, C, H_pad, W_pad)
        x = rearrange(x, 'b (nh nw) (p1 p2) c -> b c (nh p1) (nw p2)',
                     nh=num_patches_h, nw=num_patches_w, p1=P, p2=P)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        
        return x


class SegmentationHead(nn.Module):
    """
    INNOVATION 3: Dense prediction head for pixel-level segmentation
    
    SSFTT outputs (B, num_classes) - one label per tile
    HSTNet outputs (B, num_classes, H, W) - one label per pixel
    """
    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        """(B, in_channels, H, W) → (B, num_classes, H, W)"""
        return self.decoder(x)


NUM_CLASS = 13


# ============================================================================
# Main Model
# ============================================================================

class HSTNet(nn.Module):
    """
    HSTNet - Hyperspectral Segmentation Transformer Network
    
    Our novel architecture combining:
    1. 3D CNN for spectral features (same as SSFTT)
    2. U-Net for multi-scale spatial features (INNOVATION)
    3. Hierarchical patch-based transformer for global context (INNOVATION)
       - Local attention within patches
       - Global attention between patches (optional)
    4. Segmentation head for dense prediction (INNOVATION)
    
    Args:
        in_channels: Number of input channels (default: 1)
        num_classes: Number of output classes
        num_tokens: Unused (kept for compatibility)
        dim: Feature dimension for transformer
        depth: Number of transformer layers
        heads: Number of attention heads
        mlp_dim: Hidden dimension for MLP blocks
        dropout: Dropout rate
        emb_dropout: Embedding dropout rate
        patch_size: Size of patches for transformer (default: 8)
        use_unet: Whether to use U-Net (True) or Conv2D (False)
        spectral_bands: Number of spectral bands in input
        use_cross_patch: Whether to use cross-patch global attention (default: False)
    """
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=6, dim=96, 
                 depth=2, heads=6, mlp_dim=384, dropout=0.15, emb_dropout=0.15, 
                 patch_size=8, use_unet=False, spectral_bands=224, use_cross_patch=False,
                 conv3d_kernel=(3, 1, 1), conv3d_out_channels=8, seg_head_type='deep', use_learned_tokens=False):
        super().__init__()
        
        self.patch_size = patch_size
        self.use_unet = use_unet
        self.spectral_bands = spectral_bands
        self.use_cross_patch = use_cross_patch
        self.use_learned_tokens = use_learned_tokens
        self.num_tokens = num_tokens
        self.conv3d_out_channels = conv3d_out_channels
        
        # Stage 1: Spectral feature extraction
        # Ablation result: (3,1,1) spectral-only kernel is best
        padding = tuple((k - 1) // 2 for k in conv3d_kernel)
        self.spectral_extractor = nn.Sequential(
            nn.Conv3d(in_channels, conv3d_out_channels, kernel_size=conv3d_kernel, padding=padding),
            nn.BatchNorm3d(conv3d_out_channels),
            nn.ReLU(),
        )
        
        # Stage 2: Spatial feature extraction
        # Ablation result: Conv2D is best (use_unet=False)
        conv3d_out_channels_total = conv3d_out_channels * spectral_bands
        self.spatial_extractor = SpatialFeatureExtractor(
            in_channels=conv3d_out_channels_total,
            out_channels=dim,
            use_unet=use_unet
        )
        
        # Stage 3: Global context modeling
        # Two options: patch-based (default) or learned tokenization (SSFTT-style)
        if use_learned_tokens:
            # Simplified learned tokenization
            # token_wA: (1, L, dim) - learnable projection to tokens
            self.token_wA = nn.Parameter(torch.empty(1, num_tokens, dim), requires_grad=True)
            nn.init.xavier_normal_(self.token_wA)
            
            # Standard transformer for tokens
            self.context_modeling = TransformerBlock(dim, depth, heads, mlp_dim, dropout)
        else:
            # Patch-based transformer (default, ablation winner)
            self.context_modeling = PatchBasedTransformer(
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                patch_size=patch_size,
                use_cross_patch=use_cross_patch
            )
        
        # Stage 4: Attention - Ablation result: MHSA only is best (no extra module)
        
        # Stage 5: Segmentation head - configurable type
        # Ablation result: deep head (E5-b) is best
        if seg_head_type == 'deep':
            self.seg_head = nn.Sequential(
                nn.Conv2d(dim, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, kernel_size=1)
            )
        elif seg_head_type == 'dilated':
            self.seg_head = nn.Sequential(
                nn.Conv2d(dim, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, kernel_size=1)
            )
        else:  # shallow (ablation E5-a)
            self.seg_head = SegmentationHead(in_channels=dim, num_classes=num_classes, hidden_channels=128)

    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W) - Hyperspectral image
            mask: Optional mask (unused, kept for compatibility)
            
        Returns:
            Output tensor (B, num_classes, H, W) - Pixel-level predictions
        """
        x = x.to(torch.float)
        
        # Stage 1: Extract spectral features
        x = rearrange(x, 'b c h w -> b 1 c h w')
        x = self.spectral_extractor(x)   # (B, 8, C, H, W)
        x = rearrange(x, 'b c d h w -> b (c d) h w')  # (B, 8*C, H, W)
        
        # Stage 2: Extract spatial features
        x = self.spatial_extractor(x)  # (B, dim, H, W)
        
        # Stage 3: Context modeling
        if self.use_learned_tokens:
            # Simplified learned tokenization for pixel-level segmentation
            B, C, H, W = x.shape
            
            # Flatten spatial: (B, C, H, W) -> (B, H*W, C)
            x_flat = rearrange(x, 'b c h w -> b (h w) c')
            
            # Simple tokenization: learnable projection
            # (B, H*W, C) @ (C, L) -> (B, H*W, L)
            tokens = torch.einsum('bpc,cl->bpl', x_flat, self.token_wA.squeeze(0).T)
            tokens = tokens.softmax(dim=1)  # Attention weights
            
            # Aggregate: (B, L, H*W) @ (B, H*W, C) -> (B, L, C)
            tokens = torch.einsum('blp,bpc->blc', tokens.transpose(1, 2), x_flat)
            
            # Transformer on tokens
            tokens = self.context_modeling(tokens)  # (B, L, C)
            
            # Reconstruct: broadcast to all pixels
            tokens_mean = tokens.mean(dim=1, keepdim=True)  # (B, 1, C)
            x = tokens_mean.expand(B, H * W, C)  # (B, H*W, C)
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)  # (B, C, H, W)
        else:
            # Patch-based transformer (default)
            x = self.context_modeling(x)  # (B, dim, H, W)
        
        # Stage 4: Dense prediction
        output = self.seg_head(x)  # (B, num_classes, H, W)
        
        return output


# Alias for backward compatibility
HSTNet_Simple = HSTNet


if __name__ == '__main__':
    print("=" * 80)
    print("HSTNet - Hyperspectral Segmentation Transformer Network")
    print("=" * 80)
    
    # Test parameters
    batch_size = 4
    spectral_bands = 100  # After band selection
    height, width = 64, 64
    num_classes = 13
    
    # Test input
    input_tensor = torch.randn(batch_size, spectral_bands, height, width)
    print(f"\nInput shape: {input_tensor.shape}")
    
    # Test HSTNet with Conv2D
    print("\n" + "=" * 80)
    print("1. HSTNet with Conv2D (Faster)")
    print("=" * 80)
    model1 = HSTNet(
        in_channels=1,
        num_classes=num_classes,
        dim=96,
        depth=2,
        heads=6,
        mlp_dim=384,
        dropout=0.15,
        emb_dropout=0.15,
        patch_size=8,
        use_unet=False,
        spectral_bands=spectral_bands,
        use_cross_patch=True
    )
    model1.eval()
    
    print(f"Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    with torch.no_grad():
        output1 = model1(input_tensor)
    print(f"Output shape: {output1.shape}")
    print(f"Expected: ({batch_size}, {num_classes}, {height}, {width})")
    print(f"✓ Success!" if output1.shape == (batch_size, num_classes, height, width) else "✗ Failed!")
    
    # Test HSTNet with U-Net
    print("\n" + "=" * 80)
    print("2. HSTNet with U-Net (Better Performance)")
    print("=" * 80)
    model2 = HSTNet(
        in_channels=1,
        num_classes=num_classes,
        dim=96,
        depth=2,
        heads=6,
        mlp_dim=384,
        dropout=0.15,
        emb_dropout=0.15,
        patch_size=8,
        use_unet=True,
        spectral_bands=spectral_bands,
        use_cross_patch=False
    )
    model2.eval()
    
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    with torch.no_grad():
        output2 = model2(input_tensor)
    print(f"Output shape: {output2.shape}")
    print(f"Expected: ({batch_size}, {num_classes}, {height}, {width})")
    print(f"✓ Success!" if output2.shape == (batch_size, num_classes, height, width) else "✗ Failed!")
    
    # Test HSTNet with Cross-Patch Attention (NEW!)
    print("\n" + "=" * 80)
    print("3. HSTNet with Cross-Patch Global Attention (NEW!)")
    print("=" * 80)
    model3 = HSTNet(
        in_channels=1,
        num_classes=num_classes,
        dim=96,
        depth=2,
        heads=6,
        mlp_dim=384,
        dropout=0.15,
        emb_dropout=0.15,
        patch_size=8,
        use_unet=True,
        spectral_bands=spectral_bands,
        use_cross_patch=True  # Enable cross-patch attention
    )
    model3.eval()
    
    print(f"Parameters: {sum(p.numel() for p in model3.parameters()):,}")
    
    with torch.no_grad():
        output3 = model3(input_tensor)
    print(f"Output shape: {output3.shape}")
    print(f"Expected: ({batch_size}, {num_classes}, {height}, {width})")
    print(f"✓ Success!" if output3.shape == (batch_size, num_classes, height, width) else "✗ Failed!")
    
    print("\n" + "=" * 80)
    print("Key Innovations vs SSFTT:")
    print("=" * 80)
    print("1. Patch-based transformer (vs learned tokenization)")
    print("   - Memory: O(P²) vs O(L×H×W)")
    print("   - Preserves spatial structure")
    print("\n2. Hierarchical attention (NEW!)")
    print("   - Local: Within 8×8 patches (64 pixels)")
    print("   - Global: Between 64 patch tokens")
    print("   - Best of both worlds: detail + context")
    print("\n3. U-Net integration (vs simple Conv2D)")
    print("   - Multi-scale features")
    print("   - Skip connections")
    print("\n4. Pixel-level segmentation (vs tile-level)")
    print("   - Dense prediction: (B,C,H,W) vs (B,C)")
    print("   - Handles mixed materials")
    print("\n5. Dynamic band support")
    print(f"   - Works with {spectral_bands} bands (or any count)")
    print("   - No hardcoded dimensions")
    print("\n" + "=" * 80)
    print("Memory Comparison (64×64 image):")
    print("=" * 80)
    print("SSFTT Tokenization:")
    print("  - 4096 pixels → 6 tokens (99.85% compression)")
    print("  - Attention: 6² = 36 interactions")
    print("  - Problem: Loses spatial detail")
    print("\nHSTNet Patch-based (original):")
    print("  - 4096 pixels → 64 patches of 64 pixels")
    print("  - Attention per patch: 64² = 4096 interactions")
    print("  - Problem: No cross-patch communication")
    print("\nHSTNet Hierarchical (NEW!):")
    print("  - Local: 64 patches × 64² = 262,144 interactions")
    print("  - Global: 64² = 4,096 interactions between patches")
    print("  - Benefit: Detail + context, memory efficient")
