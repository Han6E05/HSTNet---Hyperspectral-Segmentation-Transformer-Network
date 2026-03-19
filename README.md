# HSTNet — Hyperspectral Segmentation Transformer

A novel Transformer architecture for pixel-level hyperspectral image segmentation. Combines 3D CNN spectral compression, U-Net spatial features, and hierarchical patch-based attention that preserves full spatial resolution without tokenization. Achieves **96.66% cosine similarity** on 13-class textile material segmentation, outperforming SSFTT-LSTM (94.59%) and SSFTT (92.85%).

---

## Architecture

```
Input (B, 224, H, W)  — hyperspectral tile, 224 spectral bands
    |
    v
Stage 1 — Spectral Feature Extraction
    3D Conv (3x1x1) + BN + ReLU
    224 bands -> 8 feature channels
    Captures correlations between adjacent wavelengths
    Output: (B, 8x224, H, W) after reshape
    |
    v
Stage 2 — Spatial Feature Extraction
    Conv2D (default) or LightweightUNet encoder-decoder
    Projects to dim=96 feature channels
    U-Net variant adds multi-scale skip connections
    Output: (B, 96, H, W)
    |
    v
Stage 3 — Hierarchical Patch-based Transformer
    Divide HxW into non-overlapping 8x8 patches
    Local self-attention within each patch (preserves all spatial detail)
    Optional global cross-patch attention between patch tokens
    Sinusoidal positional embeddings
    O(P^2) + O(N^2) complexity  (P=64 pixels/patch, N=64 patches for 64x64 input)
    Output: (B, 96, H, W)  — full resolution maintained
    |
    v
Stage 4 — Deep Segmentation Head
    Conv2d(96->128) -> BN -> ReLU  x3
    Conv2d(128->num_classes, kernel=1)
    Output: (B, num_classes, H, W)
```

**Why patch-based instead of tokenization?**
Tokenization-based methods (e.g. SSFTT) compress 4096 pixels into a handful of tokens for efficiency, but this loses fine-grained spatial detail required for dense pixel-level prediction. HSTNet divides the image into patches *without compression*, so every pixel retains its own feature vector throughout the network.

---

## Loss Function

HSTNet uses **Masked KL Divergence Loss** (`MaskedKLDivergenceLoss` in `elements/calc_loss.py`).

**Why KL Divergence instead of Cross-Entropy?**

The ground truth labels are soft probability distributions — each pixel holds a 13-dimensional vector of material composition percentages (e.g. `[0.6, 0.4, 0, ...]` for 60% cotton + 40% polyester). Cross-entropy assumes hard one-hot labels and penalizes any deviation from a single class, which is fundamentally wrong for this task. KL Divergence directly measures the distance between two probability distributions, making it the natural choice for soft label segmentation.

```python
# Pixel-level: excludes background pixels (index 0) from loss
criterion = MaskedKLDivergenceLoss(ignore_indices=[0])

# Forward: outputs (B, C, H, W), targets (B, C, H, W)
loss = criterion(model_output, soft_label_target)
```

The `ignore_indices=[0]` mask excludes background pixels (class 0) from loss computation entirely, following standard semantic segmentation practice.

---

## Training

### Dataset

- **Input**: 64x64 hyperspectral tiles, 224 spectral bands (900-1700nm, Specim FX17 camera)
- **Labels**: Per-pixel soft probability vectors (13 material classes)
- **Preprocessing**: Per-tile min-max normalization; spatial augmentations only (flips, rotations) — no color jitter, which would corrupt spectral signatures
- **Split**: Pre-split `train_set.npz` / `test_set.npz`; training set further split 85/15 train/val with stratification by composition label

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Max epochs | 100 |
| LR scheduler | ReduceLROnPlateau (patience=20, factor=0.5) |
| Early stopping | patience=15, min_delta=1e-4 |
| Mixed precision | AMP (float16 forward, float32 gradients) |


### Training Script

```bash
# Main training pipeline
python train_Tianhan.py

## Evaluation

Metrics are computed on the held-out test set using the best checkpoint (highest validation cosine similarity):

- **Cosine Similarity** (primary): Angular similarity between predicted and ground truth composition vectors, averaged over spatial dimensions per tile.
- **MAE / MSE**: Mean absolute / squared error on composition percentages.

```python
# Cosine similarity computation (per sample)
pred_comp = softmax(output).mean(dim=(2, 3))  # (B, num_classes)
true_comp = target.mean(dim=(2, 3))           # (B, num_classes)
cos_sim = cosine_similarity(pred_comp, true_comp)
```

---

## Usage

```python
import torch
from HSTNet import HSTNet

model = HSTNet(
    in_channels=1,
    num_classes=13,
    spectral_bands=224,
    dim=96,                # feature dimension
    depth=2,               # transformer layers per patch
    heads=6,               # attention heads
    mlp_dim=384,
    dropout=0.15,
    patch_size=8,          # local patch size (8x8)
    use_unet=False,        # True: U-Net spatial extractor
    use_cross_patch=False, # True: global cross-patch attention
    seg_head_type='deep'   # 'deep' | 'shallow' | 'dilated'
)

# Input:  (B, spectral_bands, H, W)
# Output: (B, num_classes, H, W)  — pixel-level soft predictions
x = torch.randn(2, 224, 64, 64)
out = model(x)  # (2, 13, 64, 64)

# Apply softmax to get probability distributions
probs = torch.softmax(out, dim=1)
```

---

## Requirements

```
torch >= 2.0.0
einops
numpy
scikit-learn
```

---

## Results

| Model | Test Cosine Similarity |
|---|---|
| **HSTNet** | **96.66%** |
| SSFTT-LSTM | 94.59% |
| SSFTT-UNet | 93.49% |
| SSFTT | 92.85% |
| 3D-CNN | 92.09% |
| CNN-LSTM | 91.38% |
| 2D-CNN | 84.96% |
| 1D-CNN | 84.91% |
| U-Net | 81.06% |

---

## Ablation Study Summary

Five architectural components were systematically explored across 15 experiments (E1-E5):

| Stage | Component | Winner | Key Finding |
|---|---|---|---|
| E1 | 3D Conv kernel | (3x1x1) | Separating spectral/spatial processing outperforms isotropic (3x3x3) |
| E2 | Spatial extractor | U-Net | Multi-scale skip connections improve over plain Conv2D |
| E3 | Attention design | Patch-based | Tokenization loses spatial detail; patch-based preserves full resolution |
| E4 | Attention module | Pure MHSA | CBAM/PSA add complexity without benefit |
| E5 | Segmentation head | Deep (3 layers) | More capacity improves final prediction refinement |

---

## License

Free to use
