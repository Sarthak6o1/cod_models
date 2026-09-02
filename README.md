# IFBONet-V3: Two-Stage Camouflaged Object Detection

A two-stage encoder-decoder network for Camouflaged Object Detection (COD) on the COD10K-v3 benchmark, built on a shared Swin-Base backbone with twin decoder heads for coarse localization and boundary-refined segmentation.

This repository contains the complete source code, neural network architecture, loss functions, training pipeline, and evaluation/visualization scripts.

---

## Table of Contents

- [Quick Start: Setup & Execution](#quick-start-setup--execution)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Dataset Preparation](#3-dataset-preparation)
  - [4. Running the Code](#4-running-the-code)
- [Repository Structure](#repository-structure)
- [Architecture Pipeline](#architecture-pipeline)
  - [High-Level Two-Stage Flow](#high-level-two-stage-flow)
  - [1. SwinEncoder (Shared Backbone)](#1-swinencoder-shared-backbone)
  - [2. DecoderHead Architecture](#2-decoderhead-architecture)
  - [3. FOM (Feature Optimisation Module)](#3-fom-feature-optimisation-module)
  - [4. FID (Feature Integration & Differentiation)](#4-fid-feature-integration--differentiation)
  - [5. FHIM (Feature Hierarchical Integration Module)](#5-fhim-feature-hierarchical-integration-module)
  - [6. BRM (Boundary Refinement Module)](#6-brm-boundary-refinement-module)
  - [7. CBAM (Convolutional Block Attention Module)](#7-cbam-convolutional-block-attention-module)
  - [8. Soft Masking (Stage 1 to Stage 2 Bridge)](#8-soft-masking-stage-1-to-stage-2-bridge)
  - [9. Loss Function (Boundary-Aware Weighted BCE + Dice)](#9-loss-function-boundary-aware-weighted-bce--dice)
  - [10. Evaluation Metrics](#10-evaluation-metrics)
  - [11. Data Pipeline & Augmentation](#11-data-pipeline--augmentation)
  - [12. Training Configuration & Hyperparameters](#12-training-configuration--hyperparameters)
  - [13. Test-Time Augmentation (TTA)](#13-test-time-augmentation-tta)
- [Benchmark Results (COD10K-v3 Test Set)](#benchmark-results-cod10k-v3-test-set)
- [Evaluation & Grad-CAM Visualisation](#evaluation--grad-cam-visualisation)
- [License](#license)

---

## Quick Start: Setup & Execution

### 1. Clone Repository

The repository is lightweight and clones instantly:

```bash
git clone https://github.com/Sarthak6o1/cod_models.git
cd cod_models
```

### 2. Install Dependencies

```bash
pip install torch torchvision timm scipy opencv-python matplotlib tqdm pillow
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 1.12 (CUDA support recommended)
- `timm` >= 0.6 (`swin_base_patch4_window7_224` backbone)
- `torchvision`, `scipy`, `opencv-python`, `matplotlib`, `tqdm`, `Pillow`

### 3. Dataset Preparation

Download the standard COD10K-v3 dataset and structure it as follows:

```
COD10K-v3/
├── Train/
│   ├── Image/           # RGB images (.jpg)
│   ├── GT_Object/       # Ground truth binary masks (.png)
│   ├── GT_Edge/         # Edge annotations
│   └── GT_Instance/     # Instance annotations
└── Test/
    ├── Image/           # RGB images (.jpg)
    ├── GT_Object/       # Ground truth binary masks (.png)
    ├── GT_Edge/
    └── GT_Instance/
```

### 4. Running the Code

Open `models/IFBONet_twin_refined.py` and update line 35 with your dataset directory:

```python
dataset_path = "/path/to/your/COD10K-v3"
```

#### Option A: Jupyter / Interactive Notebook
The script is formatted with `# %%` cell markers, making it directly runnable in VS Code, Jupyter Lab, or Google Colab cell by cell.

#### Option B: Command Line
```bash
python models/IFBONet_twin_refined.py
```

---

## Repository Structure

```
cod_models/
├── models/
│   └── IFBONet_twin_refined.py   # Full model architecture, dataset, training,
│                                  # evaluation, and Grad-CAM visualization
├── _extract_ckpt_meta.py         # Utility to inspect checkpoint metadata
├── .gitignore                    # Ignores large checkpoint files and caches
├── LICENSE                       # Apache 2.0
└── README.md
```

---

## Architecture Pipeline

### High-Level Two-Stage Flow

IFBONet-V3 executes two passes over each image using a shared Swin-Base encoder:
- **Stage 1**: Detects coarse candidate regions ($M_1$) and edges ($E_1$).
- **Soft Masking**: Suppresses background context while retaining foreground information: $x_{\text{masked}} = x \cdot (0.9 \cdot M_1 + 0.1)$.
- **Stage 2**: Refines boundaries and internal features to generate the final mask ($M_2$) and edge prediction ($E_2$).

```
Input x in (B, 3, 224, 224)
|
+-------------------- STAGE 1 (Coarse Detection) --------------------+
|                                                                     |
|   +-------------------------------------+                           |
|   |         Shared SwinEncoder          |                           |
|   |  patch_embed -> 4 Swin Transformer  |                           |
|   |  stages with window attention       |                           |
|   +------------------+------------------+                           |
|                      |                                              |
|    feats1 = [f1, f2, f3, f4]                                        |
|    f1: (B, 128, 56, 56)                                             |
|    f2: (B, 256, 28, 28)                                             |
|    f3: (B, 512, 14, 14)                                             |
|    f4: (B, 1024, 7,  7)                                             |
|                      |                                              |
|   +------------------v------------------+                           |
|   |        DecoderHead 1 (Coarse)       |                           |
|   |  FOM -> FID -> FHIM -> BRM ->       |                           |
|   |  mask_head -> M1 (B, 1, 224, 224)   |                           |
|   |  edge_head -> E1 (B, 1, 224, 224)   |                           |
|   +------------------+------------------+                           |
|                      |                                              |
+----------------------v---- SOFT MASKING ----------------------------+
|                                                                     |
|   x_masked = x * (0.9 * M1.detach() + 0.1)                          |
|   - Suppresses background, amplifies coarse foreground              |
|   - M1.detach() isolates Stage 2 gradients from Stage 1             |
|   - Output: (B, 3, 224, 224)                                        |
|                                                                     |
+-------------------- STAGE 2 (Refinement) ---------------------------+
|                                                                     |
|   +-------------------------------------+                           |
|   |      SAME Shared SwinEncoder        |                           |
|   |      (weight-shared, 2nd pass)      |                           |
|   +------------------+------------------+                           |
|                      |                                              |
|    feats2 = [f1', f2', f3', f4']                                    |
|                      |                                              |
|   +------------------v------------------+                           |
|   |     DecoderHead 2 (Refinement)      |                           |
|   |     (separate weights from Head 1)  |                           |
|   |  FOM -> FID -> FHIM -> BRM ->       |                           |
|   |  mask_head -> M2 (B, 1, 224, 224)   |                           |
|   |  edge_head -> E2 (B, 1, 224, 224)   |                           |
|   +-------------------------------------+                           |
|                                                                     |
+---------------------------------------------------------------------+

Output: M1, E1, M2, E2 (raw logits)
Final inference prediction = sigmoid(M2)
```

---

### 1. SwinEncoder (Shared Backbone)

```
Input: (B, 3, 224, 224)
  |
  v patch_embed (4x4 patches -> linear projection)
  |
  (B, 56, 56, 128)
  |
  +-- Swin Stage 1 ---> f1: (B, 128, 56, 56)   [1/4 resolution]
  |   [2 Swin Transformer blocks, window size 7x7]
  |
  +-- Swin Stage 2 ---> f2: (B, 256, 28, 28)   [1/8 resolution]
  |   [patch merging 2x downsample + 2 Swin blocks]
  |
  +-- Swin Stage 3 ---> f3: (B, 512, 14, 14)   [1/16 resolution]
  |   [patch merging + 18 Swin blocks]
  |
  +-- Swin Stage 4 ---> f4: (B, 1024, 7, 7)    [1/32 resolution]
      [patch merging + 2 Swin blocks]
```

- Pretrained on ImageNet-1K using `swin_base_patch4_window7_224`.
- Shared weights between Stage 1 and Stage 2 passes.

---

### 2. DecoderHead Architecture

Each decoder head consists of: **FOM x 4 -> FID x 3 -> FHIM -> BRM -> mask_head + edge_head**.

```
Encoder features:   f1 (56^2)    f2 (28^2)    f3 (14^2)    f4 (7^2)
                     |            |            |            |
                     v            v            v            v
              +--------------------------------------------------+
              |              FOM x 4 (per-scale)                 |
              |  1x1 Conv (C_enc->64) -> BN -> LeakyReLU(0.2)   |
              |  -> Dropout2d(0.1) -> CBAM(64)                   |
              |                                                  |
              |  c0: (B, 64, 56, 56)  from f1 (128-ch)           |
              |  c1: (B, 64, 28, 28)  from f2 (256-ch)           |
              |  c2: (B, 64, 14, 14)  from f3 (512-ch)           |
              |  c3: (B, 64,  7,  7)  from f4 (1024-ch)          |
              +---+------+----------+-----------+----------------+
                  |      |          |           |
              +---+------+----------v-----------v----------------+
              |   |      |   FID x 3 (bottom-up pairwise fusion) |
              |   |      |                                       |
              |   |      |   f32 = FID(c2, c3)  -> (B, 64, 14, 14)|
              |   |      |   Fuse deepest two scales             |
              |   |      |                                       |
              |   |      +--> f21 = FID(c1, f32) -> (B, 64, 28, 28)|
              |   |          Fuse scale 2 with fused(3,4)        |
              |   |                                              |
              |   +---------> f10 = FID(c0, f21) -> (B, 64, 56, 56)|
              |              Fuse scale 1 with fused(2,3,4)      |
              +---+----------+----------+------------------------+
                  |          |          |
              +---v----------v----------v------------------------+
              |         FHIM (all-scale global merge)            |
              |                                                  |
              |  Inputs: [c0, f21, f32, c3]                      |
              |  All bilinear-upsampled to 56x56                 |
              |  Concat -> (B, 256, 56, 56)                      |
              |  Conv 3x3 (256->64) -> BN -> LeakyReLU          |
              |  Conv 3x3 (64->64)  -> BN -> LeakyReLU          |
              |  CBAM(64)                                        |
              |  Output: (B, 64, 56, 56)                         |
              +------------------+-------------------------------+
                                 |
              +------------------v-------------------------------+
              |          BRM (Boundary Refinement)               |
              |                                                  |
              |  Conv 3x3 -> BN -> LeakyReLU(0.2)                |
              |  Dilated Conv 3x3 (dilation=2) -> BN -> LeakyReLU|
              |                                                  |
              |  Morphological edge extraction:                  |
              |    dilated = MaxPool2d(x, k=3, s=1, p=1)         |
              |    eroded  = -MaxPool2d(-x, k=3, s=1, p=1)       |
              |    output  = x + (dilated - eroded)              |
              |    Output: (B, 64, 56, 56)                       |
              +--------+--------------------+--------------------+
                       |                    |
                       v                    v
              +--------------+     +--------------+
              |  mask_head   |     |  edge_head   |
              |  Conv 3x3    |     |  Conv 3x3    |
              |  -> BN->ReLU |     |  -> BN->ReLU |
              |  Conv 1x1    |     |  Conv 1x1    |
              |  (64->32->1) |     |  (64->32->1) |
              +------+-------+     +------+-------+
                     |                    |
                     v                    v
              Bilinear upsample    Bilinear upsample
              to (224, 224)        to (224, 224)
                     |                    |
              mask_logit           edge_logit
              (B, 1, 224, 224)     (B, 1, 224, 224)
```

---

### 3. FOM (Feature Optimisation Module)

```
Input: (B, C_enc, H, W)
  |
  +-- Conv2d 1x1 (C_enc -> 64)
  +-- BatchNorm2d
  +-- LeakyReLU(0.2)
  +-- Dropout2d(0.1)
  +-- CBAM(64)
  |
Output: (B, 64, H, W)
```

Reduces heterogeneous backbone channel dimensions to uniform 64 channels with attention selection.

---

### 4. FID (Feature Integration & Differentiation)

```
Inputs: hi_res (B, 64, H_hi, W_hi), lo_res (B, 64, H_lo, W_lo)
  |
  +-- branch_hi: Conv 3x3 -> BN -> LeakyReLU on hi_res
  +-- branch_lo: Conv 3x3 -> BN -> LeakyReLU on lo_res + Bilinear upsample to (H_hi, W_hi)
  |
  +-- Concat -> (B, 128, H_hi, W_hi)
  +-- Fusion Conv 1x1 (128 -> 64) -> BN -> LeakyReLU
  +-- CBAM(64)
  |
Output: (B, 64, H_hi, W_hi)
```

Integrates high-level semantic abstractions with shallow spatial resolution progressively.

---

### 5. FHIM (Feature Hierarchical Integration Module)

```
Inputs: [c0 (56x56), f21 (28x28), f32 (14x14), c3 (7x7)]
  |
  +-- Bilinear upsample all to 56x56
  +-- Concat -> (B, 256, 56, 56)
  +-- Conv 3x3 (256 -> 64) -> BN -> LeakyReLU
  +-- Conv 3x3 (64 -> 64)  -> BN -> LeakyReLU
  +-- CBAM(64)
  |
Output: (B, 64, 56, 56)
```

Produces a unified representation containing multi-scale context simultaneously.

---

### 6. BRM (Boundary Refinement Module)

```
Input: (B, 64, 56, 56)
  |
  +-- Conv 3x3 (64 -> 64) -> BN -> LeakyReLU
  +-- Dilated Conv 3x3 (dilation=2) -> BN -> LeakyReLU
  |
  +-- Morphological edge extraction:
  |     dilated = MaxPool2d(x, kernel=3, stride=1, pad=1)
  |     eroded  = -MaxPool2d(-x, kernel=3, stride=1, pad=1)
  |     boundary = dilated - eroded
  |
  +-- output = x + boundary (residual connection)
  |
Output: (B, 64, 56, 56)
```

Applies learnable morphological dilation and erosion to isolate and amplify boundary transition zones.

---

### 7. CBAM (Convolutional Block Attention Module)

```
Input: (B, C, H, W)
  |
  +-- Channel Attention:
  |     AvgPool(B, C, 1, 1) --+
  |                           +-- Shared FC: Conv1x1(C->C/8) -> ReLU -> Conv1x1(C/8->C)
  |     MaxPool(B, C, 1, 1) --+
  |     Sum -> Sigmoid -> Channel-wise multiplication
  |
  +-- Spatial Attention:
        Mean across channels (B, 1, H, W) --+
                                             +-- Concat -> Conv2d 7x7 -> Sigmoid
        Max across channels  (B, 1, H, W) --+
        Spatial element-wise multiplication
  |
Output: (B, C, H, W)
```

---

### 8. Soft Masking (Stage 1 to Stage 2 Bridge)

```python
x_masked = x * (0.9 * M1.detach() + 0.1)
```

- Foreground regions receive full weight (~1.0x).
- Background noise is dampened to 10% (0.1x) without complete zeroing, maintaining context.
- `M1.detach()` breaks computational graph to decouple Stage 1 and Stage 2 gradient backpropagation.

---

### 9. Loss Function (Boundary-Aware Weighted BCE + Dice)

```
structure_loss(pred_logit, GT):
  w = 1 + 5 * |AvgPool_31x31(GT) - GT|
  BCE = BCEWithLogits(pred_logit, GT, weight=w)
  Dice = 1 - (2 * sum(sigmoid(pred_logit) * GT * w) + 1) / (sum((sigmoid(pred_logit) + GT) * w) + 1)
  return BCE + mean(Dice)

L_stage = structure_loss(mask_logit, GT) + 0.4 * structure_loss(edge_logit, GT)
L_total = 0.4 * L_stage1 + 1.0 * L_stage2
```

---

### 10. Evaluation Metrics

- **Structure-measure ($S_\alpha$)**: Evaluates object-aware and region-aware structural similarity ($\alpha = 0.5$).
- **Enhanced-alignment measure ($E_\phi$)**: Evaluates pixel-level errors and image-level statistics simultaneously.
- **Weighted F-measure ($F_\beta^w$)**: Boundary-distance weighted precision and recall ($\beta^2 = 1.0$).
- **Mean Absolute Error ($\text{MAE}$)**: Average per-pixel absolute difference between continuous prediction and ground truth.
- **Accuracy**: Binary pixel accuracy at threshold 0.5.

---

### 11. Data Pipeline & Augmentation

**Training Augmentations (applied jointly to image and mask):**
- Random horizontal flip ($p = 0.5$)
- Random vertical flip ($p = 0.3$)
- Random rotation ($\pm 20^\circ$)
- Random resized crop (scale 0.65 to 1.0, aspect ratio 0.75 to 1.33) -> $224 \times 224$
- Photometric ColorJitter on image (brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
- ImageNet normalization: Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]

**Testing Preprocessing (deterministic):**
- Bilinear resize to $224 \times 224$ (Image)
- Nearest-neighbor resize to $224 \times 224$ (Mask)
- ImageNet normalization

---

### 12. Training Configuration & Hyperparameters

| Parameter | Value | Details |
|:---|:---|:---|
| Input Size | 224 x 224 | Resized input dimensions |
| Batch Size | 8 | Multi-worker pinned memory |
| Optimizer | AdamW | Weight decay $1 \times 10^{-4}$ |
| Backbone Learning Rate | $1 \times 10^{-5}$ | 10x smaller for pretrained Swin backbone |
| Decoder Learning Rate | $1 \times 10^{-4}$ | Higher for task-specific heads |
| LR Scheduler | CosineAnnealingWarmRestarts | $T_0 = 50, T_{\text{mult}} = 2, \eta_{\text{min}} = 10^{-7}$ |
| Warmup | 5 epochs | Linear warmup ramp |
| Total Epochs | 1000 | Deep convergence |
| Mixed Precision | PyTorch AMP | GradScaler enabled |
| Gradient Clipping | max_norm = 2.0 | Gradient stabilization |
| Seed | 42 | Full deterministic reproducibility |

---

### 13. Test-Time Augmentation (TTA)

The code contains built-in multi-scale and horizontal-flip test-time augmentation (`tta_predict`):
- Multi-scale evaluations at scales: $[0.75, 1.0, 1.25]$
- Horizontal flip on each scale
- Averages all 6 prediction outputs for boundary smoothing.

---

## Benchmark Results (COD10K-v3 Test Set)

Quantitative evaluation on the official COD10K-v3 test benchmark (2,026 test images):

| Model Checkpoint Criterion | Best Epoch | S_alpha | E_phi | F_beta_w | MAE | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Best E-measure (`model_best_ephi`) | 314 | 0.9314 | 0.9469 | 0.8138 | 0.0229 | 0.9786 |
| Best Weighted F-measure (`model_best_fbw`) | 299 | 0.9334 | 0.9402 | 0.8156 | 0.0232 | 0.9783 |
| Best S-measure (`model_best_salpha`) | 185 | 0.9406 | 0.9250 | 0.7980 | 0.0275 | 0.9754 |

---

## Evaluation & Grad-CAM Visualisation

The final section of `models/IFBONet_twin_refined.py` hooks into `decoder2.brm.conv` and generates 5-panel visualizations on test samples:
1. **Input Image** (RGB)
2. **Ground Truth Mask** (Binary)
3. **Predicted Mask** (Jet Colormap continuous probabilities)
4. **Otsu Binarized Mask** (Automatic optimal thresholding via OpenCV)
5. **Grad-CAM Overlay** (Spatial heatmap attention focus on camouflaged object)

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
