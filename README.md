# IFBONet-V3: Two-Stage Camouflaged Object Detection

Official PyTorch implementation of **IFBONet-V3** (Iterative Feature Boundary Optimization Network), a two-stage encoder–decoder architecture for **Camouflaged Object Detection (COD)** built on a shared **Swin-Base** backbone with dual decoder heads for coarse localization and boundary-refined segmentation.

> 📦 **Note**: This repository contains the complete source code, model architecture, loss functions, training loop, and evaluation/visualization pipeline. Large model weight files (`.pkl`) are excluded to keep the repository lightweight (< 100 KB) and instantly cloneable without Git LFS requirements.

---

## ⚡ Quick Start: Cloning & Setup

### 1. Clone the Repository
The repository is lightweight and clones instantly in seconds:

```bash
git clone https://github.com/Sarthak6o1/cod_models.git
cd cod_models
```

### 2. Install Requirements

```bash
pip install torch torchvision timm scipy opencv-python matplotlib tqdm pillow
```

**Environment Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 1.12 (CUDA recommended)
- `timm` ≥ 0.6 (`swin_base_patch4_window7_224` backbone)
- `torchvision`, `scipy`, `opencv-python`, `matplotlib`, `tqdm`, `Pillow`

### 3. Dataset Setup (COD10K-v3)

Download the standard [COD10K-v3 Benchmark Dataset](https://github.com/DengPingFan/SINet-V2#dataset) and organize directory structure as:

```
COD10K-v3/
├── Train/
│   ├── Image/           # Training RGB images (.jpg)
│   ├── GT_Object/       # Binary ground truth masks (.png)
│   ├── GT_Edge/         # Edge annotations
│   └── GT_Instance/     # Instance maps
└── Test/
    ├── Image/           # Testing RGB images (.jpg)
    ├── GT_Object/       # Binary ground truth masks (.png)
    ├── GT_Edge/
    └── GT_Instance/
```

### 4. Configure Path & Run

Open [`models/IFBONet_twin_refined.py`](models/IFBONet_twin_refined.py) and update the local dataset path at line 35:

```python
dataset_path = "/path/to/your/COD10K-v3"
```

#### Option A: Run as Interactive Notebook (Recommended)
[`models/IFBONet_twin_refined.py`](models/IFBONet_twin_refined.py) is structured with `# %%` cell blocks, allowing you to run it directly inside **VS Code (Jupyter extension)**, **Jupyter Lab**, or **Google Colab**.

#### Option B: Run via Command Line
```bash
python models/IFBONet_twin_refined.py
```

---

## 📂 Repository Structure

```
cod_models/
├── models/
│   └── IFBONet_twin_refined.py   # Complete model architecture, dataset loader,
│                                  # training loop, validation, and Grad-CAM visualization
├── _extract_ckpt_meta.py         # Standalone utility to inspect checkpoint metadata
├── .gitignore                    # Ignores large checkpoint files (.pkl) and caches
├── LICENSE                       # Apache 2.0
└── README.md
```

---

## 🏗️ Architecture Pipeline (Detailed)

### High-Level Two-Stage Workflow

IFBONet-V3 processes each image through a weight-shared Swin-Base backbone in two distinct passes:
1. **Stage 1 (Coarse Discovery)**: Finds candidate camouflaged regions and outputs coarse mask $M_1$ and edge map $E_1$.
2. **Soft-Masking Bridge**: Suppresses background noise while preserving context: $x_{\text{masked}} = x \cdot (0.9 \cdot M_1 + 0.1)$.
3. **Stage 2 (Boundary Refinement)**: Re-encodes the focused feature map to produce the refined mask $M_2$ and high-fidelity edge map $E_2$.

```
Input Image x ∈ (B, 3, 224, 224)
│
╔════════════════════ STAGE 1 (Coarse Detection) ═══════════════════╗
║                                                                   ║
║   Shared SwinEncoder(x) ──► feats1 = [f1, f2, f3, f4]            ║
║     f1: (B, 128, 56, 56)                                          ║
║     f2: (B, 256, 28, 28)                                          ║
║     f3: (B, 512, 14, 14)                                          ║
║     f4: (B, 1024, 7,  7)                                          ║
║                                                                   ║
║   DecoderHead_1(feats1) ──► M1 (coarse mask), E1 (coarse edge)   ║
║     both (B, 1, 224, 224) raw logits                              ║
╚═══════════════════════════════════════════════════════════════════╝
│
│  Soft Masking (Context-Preserving Foreground Focus):
│    x_masked = x × (0.9 × M1.detach() + 0.1)
│
╔════════════════════ STAGE 2 (Refinement Stage) ═══════════════════╗
║                                                                   ║
║   Shared SwinEncoder(x_masked) ──► feats2 = [f1', f2', f3', f4']  ║
║   (re-uses same encoder backbone)                                 ║
║                                                                   ║
║   DecoderHead_2(feats2) ──► M2 (refined mask), E2 (refined edge) ║
║   (dedicated refinement decoder weights)                          ║
╚═══════════════════════════════════════════════════════════════════╝
│
▼
Final Prediction at Inference: σ(M2)
```

---

### Detailed Module Breakdown

#### 1. Shared SwinEncoder Backbone
- Base model: `swin_base_patch4_window7_224` (pretrained on ImageNet-1K).
- Extracts 4 hierarchical multi-scale feature maps:
  - $f_1 \in \mathbb{R}^{B \times 128 \times 56 \times 56}$ ($1/4$ resolution)
  - $f_2 \in \mathbb{R}^{B \times 256 \times 28 \times 28}$ ($1/8$ resolution)
  - $f_3 \in \mathbb{R}^{B \times 512 \times 14 \times 14}$ ($1/16$ resolution)
  - $f_4 \in \mathbb{R}^{B \times 1024 \times 7 \times 7}$ ($1/32$ resolution)

#### 2. DecoderHead Architecture
Each decoder head processes multi-scale features through the following sequential stages:
- **FOM (Feature Optimisation Module)**: Projects each encoder scale to a uniform 64 channels using $1 \times 1$ convolution, BatchNorm, LeakyReLU(0.2), Dropout(0.1), and CBAM attention.
- **FID (Feature Integration & Differentiation)**: Performs bottom-up pairwise fusion across adjacent scales:
  - $f_{32} = \text{FID}(c_2, c_3)$
  - $f_{21} = \text{FID}(c_1, f_{32})$
  - $f_{10} = \text{FID}(c_0, f_{21})$
- **FHIM (Feature Hierarchical Integration Module)**: Bilinearly upsamples all scales $[c_0, f_{21}, f_{32}, c_3]$ to $56 \times 56$, concatenates them to 256 channels, and fuses them via stacked $3 \times 3$ convolutions and CBAM.
- **BRM (Boundary Refinement Module)**: Applies dilated convolutions ($d=2$) and morphological edge residual extraction:
  $$\text{boundary} = \text{MaxPool}_{3\times3}(x) - (-\text{MaxPool}_{3\times3}(-x))$$
  $$\text{Output} = x + \text{boundary}$$
- **Prediction Heads**: Separate convolutional heads for mask logits ($M$) and edge logits ($E$), upsampled to $(224 \times 224)$.

#### 3. CBAM Attention Mechanism
Integrated inside FOM, FID, and FHIM modules:
- **Channel Attention**: Joint Average-Pooling and Max-Pooling $\to$ Shared MLP with reduction ratio 8 $\to$ Sigmoid channel scaling.
- **Spatial Attention**: Channel mean and max projections $\to 7 \times 7$ Convolution $\to$ Sigmoid spatial weight map.

---

## 🎯 Loss Function & Optimization

### Boundary-Aware Weighted Structure Loss

The network is trained with a composite boundary-weighted BCE and Dice loss:

$$\mathcal{L}_{\text{structure}}(P, Y) = \text{BCE}_{\text{weighted}}(P, Y) + \text{Dice}_{\text{weighted}}(\sigma(P), Y)$$

Where pixel weights $w$ emphasize object boundaries:
$$w = 1 + 5 \cdot \left| \text{AvgPool}_{31\times31}(Y) - Y \right|$$

### Joint Multi-Task Loss

$$\mathcal{L}_{\text{stage}} = \mathcal{L}_{\text{structure}}(M, Y) + 0.4 \cdot \mathcal{L}_{\text{structure}}(E, Y)$$

$$\mathcal{L}_{\text{total}} = 0.4 \cdot \mathcal{L}_{\text{stage1}} + 1.0 \cdot \mathcal{L}_{\text{stage2}}$$

---

## ⚙️ Training Configuration

| Hyperparameter | Value | Description |
|:---|:---|:---|
| **Input Resolution** | $224 \times 224$ | Resized RGB inputs |
| **Batch Size** | 8 | Effective batch size |
| **Optimizer** | AdamW | Weight decay $10^{-4}$ |
| **Backbone LR** | $1 \times 10^{-5}$ | 10× lower for pretrained weights |
| **Decoder LR** | $1 \times 10^{-4}$ | Higher LR for randomly initialized heads |
| **LR Scheduler** | CosineAnnealingWarmRestarts | $T_0=50, T_{\text{mult}}=2, \eta_{\text{min}}=10^{-7}$ |
| **Warmup** | 5 epochs | Linear warmup ramp |
| **Total Epochs** | 1000 | Full convergence training |
| **Precision** | PyTorch AMP (fp16) | Automatic mixed precision with `GradScaler` |
| **Gradient Clipping** | $\text{max\_norm} = 2.0$ | Stabilizes transformer gradients |

---

## 📊 Evaluation Metrics & Benchmark Performance

Evaluated on the **COD10K-v3 Test Set** (2,026 test images):

| Metric | Target Metric Optimized | Best Epoch | $S_\alpha$ ↑ | $E_\phi$ ↑ | $F_\beta^w$ ↑ | MAE ↓ | Accuracy ↑ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$E_\phi$ Best Model** | `ephi` | **314** | 0.9314 | **0.9469** | 0.8138 | 0.0229 | 0.9786 |
| **$F_\beta^w$ Best Model** | `fbw` | **299** | 0.9334 | 0.9402 | **0.8156** | 0.0232 | 0.9783 |
| **$S_\alpha$ Best Model** | `salpha` | **185** | **0.9406** | 0.9250 | 0.7980 | 0.0275 | 0.9754 |

- **$S_\alpha$ (Structure-measure)**: Structural similarity to ground truth ($\alpha = 0.5$).
- **$E_\phi$ (Enhanced-alignment measure)**: Pixel-level matching with image-level statistics.
- **$F_\beta^w$ (Weighted F-measure)**: Boundary-distance weighted precision & recall ($\beta^2 = 1.0$).
- **$\text{MAE}$ (Mean Absolute Error)**: Average absolute pixel discrepancy.

---

## 🔍 Qualitative Grad-CAM Visualizations

The script includes built-in Grad-CAM visual interpretability hooks attached to `decoder2.brm.conv`, producing a 5-panel diagnostic figure per test sample:
1. **Input Image** (RGB)
2. **Ground Truth Mask** (Binary)
3. **Predicted Mask** (Jet Colormap continuous probabilities)
4. **Otsu Binarized Prediction** (Thresholded segmentation)
5. **Grad-CAM Heatmap Overlay** (Attention focus on camouflaged target)

---

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).
