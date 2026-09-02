# IFBONet-V3: Two-Stage Camouflaged Object Detection

A two-stage encoder–decoder network for **Camouflaged Object Detection (COD)** on the **COD10K-v3** benchmark, built on a shared **Swin-Base** backbone with twin decoder heads for coarse and refined segmentation.

Pretrained checkpoints are included via **Git LFS** (~1 GB each).

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Architecture Pipeline](#architecture-pipeline-detailed)
- [Best Checkpoint Metrics](#best-checkpoint-metrics-cod10k-v3-test-set)
- [Requirements](#requirements)
- [Setup & Cloning](#setup--cloning)
- [Running the Code](#running-the-code)
- [Reproducibility](#reproducibility)
- [Inspecting Checkpoint Metrics Without GPU](#inspecting-checkpoint-metrics-without-gpu)
- [Evaluation & Grad-CAM Visualisation](#evaluation--grad-cam-visualisation)
- [Checkpoint Contents](#checkpoint-contents)
- [License](#license)

---

## Repository Structure

```
cod_models/
├── models/
│   └── IFBONet_twin_refined.py        # Full model definition + dataset + training
│                                       #   + evaluation + Grad-CAM (Jupyter-compatible)
├── IFBO_NET_V3_checkpoints/
│   ├── model_best_ephi.pkl            # Best E-measure checkpoint   (~1.05 GB, Git LFS)
│   ├── model_best_fbw.pkl             # Best Weighted F-measure     (~1.05 GB, Git LFS)
│   └── model_best_salpha.pkl          # Best S-measure checkpoint   (~1.05 GB, Git LFS)
├── _extract_ckpt_meta.py              # Utility: inspect checkpoint metrics without GPU/torch
├── .gitattributes                     # Git LFS tracking rule: *.pkl
├── .gitignore
├── LICENSE                            # Apache 2.0
└── README.md
```

---

## Architecture Pipeline (Detailed)

### Overview: Two-Stage Detect-and-Refine

IFBONetV3 runs the input through a shared encoder **twice**. Stage 1 produces a coarse camouflage map. That map is used to soft-mask the original input (suppressing background), and the masked image is re-encoded for a refined Stage 2 prediction.

```
Input x ∈ (B, 3, 224, 224)
│
╔════════════════════ STAGE 1 (Coarse) ════════════════════════╗
║                                                               ║
║   SwinEncoder(x) ──► feats1 = [f1, f2, f3, f4]               ║
║                                                               ║
║   DecoderHead_1(feats1) ──► M1 (mask logit), E1 (edge logit) ║
║                              both (B, 1, 224, 224)            ║
╚══════════════════════════════════════════════════════════════╝
│
│  Soft Masking (background suppression):
│    x_masked = x × (0.9 × M1.detach() + 0.1)
│
╔════════════════════ STAGE 2 (Refined) ═══════════════════════╗
║                                                               ║
║   SwinEncoder(x_masked) ──► feats2 = [f1', f2', f3', f4']    ║
║   (same encoder weights, different input)                     ║
║                                                               ║
║   DecoderHead_2(feats2) ──► M2 (mask logit), E2 (edge logit) ║
║   (separate decoder weights from Head 1)                      ║
╚══════════════════════════════════════════════════════════════╝
│
▼
Output: M1, E1, M2, E2  (all raw logits — no sigmoid)
Final prediction at inference: σ(M2)
```

**Key design decisions:**
- **Shared encoder, twin decoders**: One `SwinEncoder` (weight-shared across both passes), two independent `DecoderHead` modules (separate weights for coarse vs refined).
- **`M1.detach()`**: Prevents Stage 2 gradients from flowing back through Stage 1's decoder, so each decoder learns its own role.
- **`mask_blend = 0.9`**: Background pixels are dampened to ~10% intensity (not zeroed), preserving some contextual information for Stage 2.

---

### 1. SwinEncoder — Shared Backbone

Uses `swin_base_patch4_window7_224` from the `timm` library (ImageNet-1K pretrained).

```
Input: (B, 3, 224, 224)
  │
  ▼ patch_embed
  │  4×4 non-overlapping patches → linear projection
  │  Output: (B, 56, 56, 128)
  │
  ├─ Swin Stage 1 ──► f1: (B, 128, 56, 56)    [1/4 resolution]
  │   2 Swin Transformer blocks, window size 7×7, shifted windows
  │
  ├─ Swin Stage 2 ──► f2: (B, 256, 28, 28)    [1/8 resolution]
  │   Patch merging (2× spatial downsample, 2× channel expand)
  │   + 2 Swin Transformer blocks
  │
  ├─ Swin Stage 3 ──► f3: (B, 512, 14, 14)    [1/16 resolution]
  │   Patch merging + 18 Swin Transformer blocks
  │
  └─ Swin Stage 4 ──► f4: (B, 1024, 7, 7)     [1/32 resolution]
      Patch merging + 2 Swin Transformer blocks
```

- Outputs are permuted from `(B,H,W,C)` → `(B,C,H,W)` for CNN compatibility.
- Compatible with both timm ≥0.6 (4D output) and older timm (3D `(B,L,C)` output).

---

### 2. DecoderHead — Multi-Scale Decode, Fuse, and Refine

Each decoder contains: **FOM ×4 → FID ×3 (bottom-up) → FHIM → BRM → mask_head + edge_head**.

```
Encoder features:   f1 (56²)    f2 (28²)    f3 (14²)    f4 (7²)
                     │            │            │            │
                     ▼            ▼            ▼            ▼
              ┌──────────────────────────────────────────────────┐
              │              FOM × 4  (per-scale)                │
              │  1×1 Conv (C_enc→64) → BN → LeakyReLU(0.2)      │
              │  → Dropout2d(0.1) → CBAM(64)                    │
              │                                                  │
              │  c0: (B,64,56,56)   from f1 (128-ch)            │
              │  c1: (B,64,28,28)   from f2 (256-ch)            │
              │  c2: (B,64,14,14)   from f3 (512-ch)            │
              │  c3: (B,64, 7, 7)   from f4 (1024-ch)           │
              └───┬──────┬──────────┬───────────┬───────────────┘
                  │      │          │           │
              ┌───┼──────┼──────────▼───────────▼───────────────┐
              │   │      │   FID ×3  (bottom-up pairwise fusion) │
              │   │      │                                       │
              │   │      │   f32 = FID(c2, c3)  → (B,64,14,14)  │
              │   │      │   Fuse deepest two scales             │
              │   │      │                                       │
              │   │      └─► f21 = FID(c1, f32) → (B,64,28,28)  │
              │   │          Fuse scale 2 with fused(3,4)        │
              │   │                                              │
              │   └────────► f10 = FID(c0, f21) → (B,64,56,56)  │
              │              Fuse scale 1 with fused(2,3,4)      │
              └───┬──────────┬──────────┬───────────────────────┘
                  │          │          │
              ┌───▼──────────▼──────────▼───────────────────────┐
              │         FHIM  (all-scale global merge)           │
              │                                                  │
              │  Inputs: [c0, f21, f32, c3]  (4 feature maps)   │
              │  All bilinear-upsampled to 56×56                 │
              │  Concat → (B, 64×4=256, 56, 56)                 │
              │  Conv 3×3 (256→64) → BN → LeakyReLU             │
              │  Conv 3×3 (64→64)  → BN → LeakyReLU             │
              │  CBAM(64)                                        │
              │  Output: (B, 64, 56, 56)                         │
              └──────────────────┬──────────────────────────────┘
                                 │
              ┌──────────────────▼──────────────────────────────┐
              │          BRM  (Boundary Refinement)              │
              │                                                  │
              │  Conv 3×3 → BN → LeakyReLU(0.2)                 │
              │  Dilated Conv 3×3 (dilation=2) → BN → LeakyReLU │
              │                                                  │
              │  Morphological edge extraction:                  │
              │    dilated = MaxPool2d(x, k=3, s=1, p=1)         │
              │    eroded  = −MaxPool2d(−x, k=3, s=1, p=1)       │
              │    output  = x + (dilated − eroded)              │
              │                    ↑ residual boundary signal     │
              │  Output: (B, 64, 56, 56)                         │
              └────────┬────────────────────┬───────────────────┘
                       │                    │
                       ▼                    ▼
              ┌──────────────┐     ┌──────────────┐
              │  mask_head   │     │  edge_head   │
              │  Conv 3×3    │     │  Conv 3×3    │
              │  → BN → ReLU │     │  → BN → ReLU │
              │  Conv 1×1    │     │  Conv 1×1    │
              │  (64→32→1)   │     │  (64→32→1)   │
              └──────┬───────┘     └──────┬───────┘
                     │                    │
                     ▼                    ▼
              Bilinear upsample      Bilinear upsample
              to (224, 224)          to (224, 224)
                     │                    │
              mask_logit (B,1,224,224)  edge_logit (B,1,224,224)
```

---

### 3. Module Details

#### FOM — Feature Optimisation Module

Applied independently to each of the 4 encoder scales.

```
(B, C_enc, H, W)  →  Conv2d 1×1 (C_enc→64) → BN → LeakyReLU(0.2) → Dropout2d(0.1)
                  →  CBAM(64)
                  →  (B, 64, H, W)
```

Reduces heterogeneous encoder channels (128/256/512/1024) to a uniform 64-ch representation with attention-based feature selection and regularisation dropout.

#### FID — Feature Integration & Differentiation

Fuses two adjacent scales in a bottom-up manner.

```
hi_res (B,64,H_hi,W_hi)  ──► branch_hi: Conv3×3 → BN → LeakyReLU ──┐
                                                                      ├► Concat (B,128,H_hi,W_hi)
lo_res (B,64,H_lo,W_lo)  ──► branch_lo: Conv3×3 → BN → LeakyReLU   │     │
                              + bilinear upsample to (H_hi,W_hi) ───┘     ▼
                                                                    Fusion: Conv 1×1 (128→64)
                                                                    → BN → LeakyReLU → CBAM(64)
                                                                    → (B,64,H_hi,W_hi)
```

The lo-res (deeper, more semantic) features are upsampled to match the hi-res (shallower, finer spatial detail) resolution. Both branches apply independent 3×3 convolutions before fusion.

#### FHIM — Feature Hierarchical Integration Module

Merges all 4 scales simultaneously into a single unified feature map.

```
[c0 (56²), f21 (28²), f32 (14²), c3 (7²)]
  → Bilinear upsample all to 56×56
  → Concat → (B, 256, 56, 56)
  → Conv 3×3 (256→64) → BN → LeakyReLU
  → Conv 3×3 (64→64)  → BN → LeakyReLU
  → CBAM(64)
  → (B, 64, 56, 56)
```

Note: receives both raw FOM features (c0, c3) and FID-fused features (f21, f32), mixing bottom-up aggregated features with original per-scale features.

#### BRM — Boundary Refinement Module

Sharpens object boundaries using a learnable morphological operation.

```
x → Conv 3×3 → BN → LeakyReLU → Dilated Conv 3×3 (d=2) → BN → LeakyReLU
  │
  ├─ dilated = MaxPool2d(x, k=3, s=1, p=1)      ← morphological dilation
  ├─ eroded  = −MaxPool2d(−x, k=3, s=1, p=1)    ← morphological erosion
  ├─ boundary = dilated − eroded                  ← edge signal
  │
  └─ output = x + boundary                        ← residual addition
```

The dilation–erosion difference highlights edge regions. Adding it as a residual enhances boundary features without losing interior information. The dilated convolution (dilation=2) provides wider receptive field before the morphological step.

#### CBAM — Convolutional Block Attention Module

Used inside every FOM, FID, FHIM, and BRM module.

```
Input: (B, C, H, W)
  │
  ├─ Channel Attention:
  │    AvgPool → (B,C,1,1) ──┐
  │                          ├─ Shared FC: Conv1×1(C→C/8) → ReLU → Conv1×1(C/8→C)
  │    MaxPool → (B,C,1,1) ──┘
  │    Sum → Sigmoid → element-wise scale input
  │
  └─ Spatial Attention:
       Mean across channels → (B,1,H,W) ──┐
                                           ├─ Concat → Conv2d 7×7 → Sigmoid
       Max across channels  → (B,1,H,W) ──┘
       Element-wise scale input
  │
Output: (B, C, H, W)
```

Reduction ratio = 8, spatial kernel = 7×7.

---

### 4. Soft Masking — Stage 1 to Stage 2 Bridge

```python
x_masked = x * (self.mask_blend * M1.detach() + (1.0 - self.mask_blend))
# mask_blend = 0.9
```

| Pixel type | M1 logit | Effective multiplier | Effect |
|------------|----------|---------------------|--------|
| Strong foreground | High positive | ≈ 0.9×1.0 + 0.1 = **~1.0** | Preserved |
| Strong background | High negative | ≈ 0.9×0.0 + 0.1 = **~0.1** | Suppressed to 10% |
| Ambiguous | Near zero | ≈ 0.9×0.5 + 0.1 = **~0.55** | Partially retained |

- `M1.detach()` breaks the gradient path: Stage 2's loss only updates Decoder2 and the encoder (via Stage 2's forward pass), not Decoder1.
- Background is dampened, not zeroed — some context survives for Stage 2's encoder.

---

### 5. Loss Function — Boundary-Aware Weighted BCE + Dice

```
structure_loss(pred_logit, GT):

  Pixel weights:  w = 1 + 5 × |AvgPool₃₁(GT) − GT|
                  → Interior/exterior pixels: weight = 1
                  → Boundary pixels: weight up to 6
                  → Forces the model to focus on boundary accuracy

  BCE = BCEWithLogits(pred_logit, GT, weight=w)         ← numerically stable, AMP-safe

  Dice:  pred = σ(pred_logit)
         inter = Σ(pred × GT × w)     over spatial dims
         union = Σ((pred + GT) × w)
         dice_loss = 1 − (2·inter + 1) / (union + 1)    ← smoothed

  Return: BCE + mean(Dice)
```

**Per-stage loss:**
```
L_stage = structure_loss(mask_logit, GT) + 0.4 × structure_loss(edge_logit, GT)
```

**Total training loss:**
```
L_total = 0.4 × L_stage1 + 1.0 × L_stage2
```

Stage 2 (refined) is weighted 2.5× more than Stage 1 (coarse), since the refined output is the final prediction.

---

### 6. Weight Initialisation

Applied once at model construction (`IFBONetV3._init_weights`):

- **Conv2d**: Kaiming normal (`fan_out`, `relu` nonlinearity). Biases initialised to zero.
- **BatchNorm2d**: Weight = 1, Bias = 0.
- **Swin backbone**: Retains its ImageNet-pretrained weights (not overwritten because `_init_weights` runs on all `self.modules()` but Swin's internal layers are `nn.Linear` / `LayerNorm`, not `Conv2d` / `BatchNorm2d`).

---

### 7. Evaluation Metrics

All metrics are computed per-batch and averaged over the full dataset.

| Metric | Notation | How it's computed in the code | Range |
|--------|----------|-------------------------------|-------|
| **S-measure** | S_α | Foreground accuracy + background accuracy, weighted by α=0.5. Prediction binarised at 0.5. | [0, 1] ↑ |
| **E-measure** | E_φ | Enhanced alignment measure: adaptive threshold at `min(2×mean(pred), 1.0)`, then alignment matrix `((2(p−p̄)(g−ḡ) / ((p−p̄)²+(g−ḡ)²+ε) + 1)² / 4)` averaged. | [0, 1] ↑ |
| **Weighted F-measure** | F_β^w | Boundary-distance-weighted precision and recall (β²=1.0). Uses distance transform, Gaussian blur (σ=3.5, k=7), and exponential boundary weighting. | [0, 1] ↑ |
| **MAE** | MAE | `mean(\|σ(logit) − GT\|)` — absolute error on sigmoid probabilities. | [0, 1] ↓ |
| **Accuracy** | Acc | Pixel-wise `mean(round(σ(logit)) == GT)` at threshold 0.5. | [0, 1] ↑ |

---

### 8. Data Pipeline

**Dataset**: [COD10K-v3](https://github.com/DengPingFan/SINet-V2#dataset)
- Train split and test split are physically separate directories (`Train/` and `Test/`).
- Image–mask pairs loaded from `Image/` (`.jpg`) and `GT_Object/` (`.png`).
- No validation split — test set used for per-epoch evaluation (standard COD benchmark practice).

**Training augmentations** (jointly applied to image + mask with the same random params):

| Augmentation | Probability / Range | Applied to |
|-------------|-------------------|-----------|
| Random horizontal flip | p = 0.5 | Image + Mask |
| Random vertical flip | p = 0.7 | Image + Mask |
| Random rotation | ±20° | Image + Mask |
| Random resized crop | scale 0.65–1.0, ratio 0.75–1.33, resized to 224×224 | Image + Mask (nearest interp for mask) |
| ColorJitter | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05 | Image only |
| ImageNet normalisation | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | Image only |
| Binarise mask | threshold = 0.5 | Mask only |

**Test preprocessing** (deterministic, no randomness):

| Step | Applied to |
|------|-----------|
| Bilinear resize to 224×224 | Image |
| Nearest-neighbour resize to 224×224 | Mask |
| ImageNet normalisation | Image only |
| Binarise mask at 0.5 | Mask only |

**DataLoader settings**: batch_size=8, num_workers=4, pin_memory=True. Train loader shuffled, test loader sequential.

---

### 9. Training Configuration

| Parameter | Value | Code reference |
|-----------|-------|---------------|
| Input size | 224 × 224 | `IMG_SIZE = 224` |
| Batch size | 8 | `BATCH_SIZE = 8` |
| Optimizer | AdamW, two param groups | `torch.optim.AdamW([...])` |
| Backbone LR | 1e-5 | 10× smaller than decoders |
| Decoder LR | 1e-4 | — |
| Weight decay | 1e-4 | Both groups |
| Scheduler | CosineAnnealingWarmRestarts | `T_0=50, T_mult=2, eta_min=1e-7` |
| Warmup | 5 epochs, linear ramp | `WARMUP_EPOCHS = 5` |
| Total epochs | 1000 | `NUM_EPOCHS = 1000` |
| Mixed precision | AMP via `torch.cuda.amp.GradScaler` | Enabled when CUDA available |
| Gradient clipping | `clip_grad_norm_`, max_norm=2.0 | After `scaler.unscale_` |
| Mask blend | 0.9 | `IFBONetV3(mask_blend=0.9)` |
| Feature channels | 64 | `FEAT_CH = 64` |
| CBAM reduction | 8 | Default in `ChannelAttention` |
| CBAM spatial kernel | 7×7 | Default in `SpatialAttention` |

**Per-epoch workflow:**
1. `train_one_epoch()` — full pass over train_loader with `model.train()`, gradient updates, AMP
2. `evaluate()` — full pass over test_loader with `model.eval()`, `torch.no_grad()`
3. Check if any test metric improved → save corresponding `model_best_*.pkl`
4. Compute composite score → save `model_best_composite.pkl` if improved
5. Every 10 epochs → save `model_latest.pkl` for safe resume

---

### 10. Checkpoint Saving Strategy

Five checkpoint types are managed:

| Checkpoint file | Saved when | Contents |
|----------------|-----------|----------|
| `model_best_fbw.pkl` | Test F_β^w improves | epoch, model/optimizer/scheduler/scaler state dicts, best_scores, test_metrics, train_metrics, metric="fbw" |
| `model_best_salpha.pkl` | Test S_α improves | Same structure, metric="salpha" |
| `model_best_ephi.pkl` | Test E_φ improves | Same structure, metric="ephi" |
| `model_best_composite.pkl` | Composite score improves | Same + composite_score, weights dict |
| `model_latest.pkl` | Every 10 epochs | epoch, model/optimizer/scheduler/scaler state dicts, history |

**Composite score formula:**
```
composite = 0.35 × F_β^w + 0.35 × S_α + 0.20 × E_φ − 0.10 × MAE
```

**Resume logic**: At startup the code loads `RESUME_FROM` (default `model_best_fbw.pkl`) and restores all state dicts + epoch counter + best_scores. Change `RESUME_FROM` to any checkpoint file to resume from a different point.

---

### 11. TTA (Test-Time Augmentation)

Optional multi-scale + horizontal-flip inference, disabled by default (`USE_TTA_EVAL = False`).

```
Scales: [0.75, 1.0, 1.25]
For each scale:
  1. Resize input to (H×s, W×s)
  2. Forward pass → M2
  3. Upsample M2 back to (H, W)
  4. Flip input horizontally → forward pass → flip prediction back
  5. Upsample to (H, W)
Average all 6 predictions (3 scales × 2 orientations)
```

---

## Best Checkpoint Metrics (COD10K-v3 Test Set)

| Checkpoint | Saved at Epoch | S_α ↑ | E_φ ↑ | F_β^w ↑ | MAE ↓ | Accuracy |
|:-----------|:--------------:|:-----:|:-----:|:-------:|:-----:|:--------:|
| `model_best_salpha.pkl` | 185 | **0.9406** | 0.9250 | 0.7980 | 0.0275 | 0.9754 |
| `model_best_fbw.pkl` | 299 | 0.9334 | 0.9402 | **0.8156** | 0.0232 | 0.9783 |
| `model_best_ephi.pkl` | 314 | 0.9314 | **0.9469** | 0.8138 | 0.0229 | 0.9786 |

> **Note for papers:** Each row is a *different checkpoint saved at a different epoch*. For paper benchmark tables, report all metrics from a **single** checkpoint (e.g. choose `model_best_fbw.pkl` and report all its metrics together). Do not mix-and-match metrics from different checkpoints.

---

## Requirements

| Package | Min version | Purpose |
|---------|-------------|---------|
| Python | ≥ 3.8 | — |
| PyTorch | ≥ 1.12 | Model, training, AMP |
| torchvision | ≥ 0.13 | Transforms, functional |
| timm | ≥ 0.6 | Swin-Base backbone |
| scipy | — | Distance transform for F_β^w metric |
| opencv-python | — | Otsu thresholding in Grad-CAM visualisation |
| matplotlib | — | Plotting |
| tqdm | — | Progress bars |
| Pillow | — | Image I/O |
| Git LFS | — | Download large checkpoint files |

```bash
pip install torch torchvision timm scipy opencv-python matplotlib tqdm pillow
```

---

## Setup & Cloning

### Step 1: Install Git LFS

Git LFS is required to download the checkpoint files (~1 GB each).

```bash
# Install Git LFS (run once per machine)
git lfs install
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/Sarthak6o1/cod_models.git
cd cod_models
```

Git LFS will automatically download the three `.pkl` checkpoint files during cloning.

**Verify the checkpoints are fully downloaded (not LFS pointers):**

```bash
ls -lh IFBO_NET_V3_checkpoints/
# Each .pkl file should be ~1.0 GB
# If they are ~130 bytes, they are LFS pointers — run:
git lfs pull
```

### Step 3: Install Python Dependencies

```bash
pip install torch torchvision timm scipy opencv-python matplotlib tqdm pillow
```

### Step 4: Download COD10K-v3 Dataset

Download from [COD10K-v3 (SINet-V2 repo)](https://github.com/DengPingFan/SINet-V2#dataset) and organise as:

```
COD10K-v3/
├── Train/
│   ├── Image/           # RGB images (.jpg)
│   ├── GT_Object/       # Binary ground truth masks (.png)
│   ├── GT_Edge/         # Edge maps
│   └── GT_Instance/     # Instance annotations
└── Test/
    ├── Image/           # RGB images (.jpg)
    ├── GT_Object/       # Binary ground truth masks (.png)
    ├── GT_Edge/
    └── GT_Instance/
```

Only `Image/` and `GT_Object/` are used by the model.

### Step 5: Update the Dataset Path

In `models/IFBONet_twin_refined.py`, update **line 35**:

```python
# Change this to your local path:
dataset_path = "/path/to/your/COD10K-v3"
```

Also update **line 3** for your GPU index (or remove the line to use the default GPU):

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # your GPU index
```

---

## Running the Code

The file `models/IFBONet_twin_refined.py` is a Jupyter-compatible script. Each `# %%` marker defines a cell boundary.

### Option A: Jupyter Notebook / VS Code / Colab

1. Open `models/IFBONet_twin_refined.py` in **Jupyter Lab**, **VS Code** (with Python/Jupyter extension), or upload to **Google Colab**.
2. Run cells sequentially from top to bottom.
3. Cells are organised as:
   - **Cells 1–3**: Imports, seed, dataset setup
   - **Cells 4–8**: Model definition (SwinEncoder, CBAM, FOM, FID, FHIM, BRM, DecoderHead, IFBONetV3)
   - **Cells 9–10**: Loss functions and metrics
   - **Cell 11**: TTA prediction utility
   - **Cells 12–13**: Checkpoint/prediction dirs, model + optimizer + scheduler setup
   - **Cells 14–15**: `train_one_epoch()` and `evaluate()` functions
   - **Cells 16–17**: Resume from checkpoint + training loop (1000 epochs)
   - **Cell 18**: Print checkpoint metrics
   - **Cell 19**: Grad-CAM visualisation on test set

### Option B: Command Line

```bash
python models/IFBONet_twin_refined.py
```

This runs all cells sequentially (training + evaluation + visualisation).

---

## Reproducibility

### Fixed Random Seeds

All sources of randomness are seeded at startup:

```python
SEED = 42
random.seed(SEED)          # Python random (augmentations)
np.random.seed(SEED)       # NumPy (F_β^w metric computation)
torch.manual_seed(SEED)    # PyTorch CPU
torch.cuda.manual_seed_all(SEED)  # PyTorch all GPUs
```

### No Data Leakage

- Train and test data come from **physically separate directories** (`Train/` vs `Test/`).
- **No dataset statistics** (mean/std) are computed from data — fixed ImageNet constants are used.
- Evaluation runs under `model.eval()` + `torch.no_grad()`: no gradients computed, no parameters updated, no dropout randomness on test data.
- Augmentations are only applied when `is_train=True`.

### Deterministic Reproduction

To reproduce training from scratch:
1. Clone the repo, install dependencies, download COD10K-v3.
2. Delete or rename the `IFBO_NET_V3_checkpoints/` directory (so the resume logic starts from scratch).
3. Run the training script.

To reproduce evaluation using provided checkpoints:
1. Clone the repo with `git lfs pull`.
2. Run only the evaluation / Grad-CAM cells (skip the training loop cell).

For bit-exact reproducibility across identical GPU hardware, optionally add:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Inspecting Checkpoint Metrics Without GPU

The utility script `_extract_ckpt_meta.py` extracts metadata from checkpoint files without loading model weights or requiring PyTorch/CUDA:

```bash
python _extract_ckpt_meta.py
```

**Output for each checkpoint:**
- File path and size
- Saved epoch
- `best_scores` dict (best F_β^w, S_α, E_φ seen up to that point)
- `test_metrics` dict (loss, loss1, loss2, mae, salpha, ephi, fbw, acc)
- `train_metrics` dict (same keys)
- Which metric triggered the save

This uses a custom `SkipUnpickler` that stubs out PyTorch tensor classes, so it works on any machine with just Python 3.

---

## Evaluation & Grad-CAM Visualisation

The final cell of the script performs Grad-CAM analysis on the test set:

1. **Loads** `model_best_fbw.pkl` into a fresh `IFBONetV3` instance.
2. **Registers hooks** on `decoder2.brm.conv` (the BRM convolution block in the refinement decoder).
3. **For each test image:**
   - Forward pass through the full two-stage model (`return_coarse=True`) to get M2.
   - Backward pass on `mean(M2)` to compute gradients at the hooked layer.
   - Computes Grad-CAM: channel-wise gradient mean → weighted sum of activations → ReLU → normalise → bilinear upsample to 224×224.
   - Applies **Otsu thresholding** on `σ(M2)` for automatic binarisation.
4. **Displays 5-panel figure** per test image:

```
┌────────────┬────────────┬──────────────────┬────────────────────┬────────────────┐
│   Input    │  Ground    │  Predicted Mask  │  Otsu Binarised    │  Grad-CAM      │
│   Image    │  Truth     │  (jet colormap)  │  Predicted Mask    │  Overlay       │
│            │  Mask      │  σ(M2)           │  cv2.THRESH_OTSU   │  (α=0.4)       │
└────────────┴────────────┴──────────────────┴────────────────────┴────────────────┘
```

---

## Checkpoint Contents

Each `.pkl` checkpoint is a `torch.save()` dict with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `epoch` | int | Epoch number when saved (1-indexed) |
| `model_state_dict` | OrderedDict | All model weights (609 parameters) |
| `optimizer_state_dict` | dict | AdamW state (2 param groups) |
| `scheduler_state_dict` | dict | CosineAnnealingWarmRestarts state (11 entries) |
| `scaler_state_dict` | dict | GradScaler state (5 entries) |
| `best_scores` | dict | `{"fbw": float, "salpha": float, "ephi": float}` — best values seen so far |
| `test_metrics` | dict | `{loss, loss1, loss2, mae, salpha, ephi, fbw, acc}` at saved epoch |
| `train_metrics` | dict | Same keys as test_metrics |
| `metric` | str | Which metric triggered save (`"fbw"`, `"salpha"`, or `"ephi"`) |
| `composite_score` | float | *(only in model_best_composite.pkl)* |
| `weights` | dict | *(only in model_best_composite.pkl)* composite weight dict |
| `history` | dict | *(only in model_latest.pkl)* `{"train": [...], "test": [...]}` per-epoch metrics |

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
