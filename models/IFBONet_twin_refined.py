# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import timm
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
from scipy.ndimage import distance_transform_edt as bwdist, convolve

print(torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
torch.cuda.empty_cache()

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# %%
dataset_path = "/home/23ucs697/ObjectDetection/COD10K-v3"
train_path   = os.path.join(dataset_path, "Train")
test_path    = os.path.join(dataset_path, "Test")

def get_subfolders(base_path):
    return {
        'image':       os.path.join(base_path, 'Image'),
        'gt_object':   os.path.join(base_path, 'GT_Object'),
        'gt_edge':     os.path.join(base_path, 'GT_Edge'),
        'gt_instance': os.path.join(base_path, 'GT_Instance'),
    }

train_folders = get_subfolders(train_path)
test_folders  = get_subfolders(test_path)

def count_files(folder_dict):
    for key, folder in folder_dict.items():
        if os.path.exists(folder):
            print(f"  ✅ {key}: {len(os.listdir(folder))} files")
        else:
            print(f"  Missing: {folder}")

print("Train set:"); count_files(train_folders)
print("Test  set:"); count_files(test_folders)

# %%
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class COD10KDataset(Dataset):
    """
    Returns (image, mask) pairs.
    • Training  : joint random flips / rotation / colour jitter / crop
    • Validation: only resize + normalise
    """
    def __init__(self, folders, is_train=False, img_size=224):
        self.image_paths = sorted(os.listdir(folders['image']))
        self.image_dir   = folders['image']
        self.mask_dir    = folders['gt_object']
        self.is_train    = is_train
        self.img_size    = img_size

        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.image_paths)

    def _joint_transform(self, image, mask):
        """Apply the SAME random spatial transform to image and mask."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.3:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # Random rotation ±20°
        angle = random.uniform(-20, 20)
        image = TF.rotate(image, angle)
        mask  = TF.rotate(mask,  angle)

        # Random resized crop
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.65, 1.0), ratio=(0.75, 1.33))
        image = TF.resized_crop(image, i, j, h, w, (self.img_size, self.img_size))
        mask  = TF.resized_crop(mask,  i, j, h, w, (self.img_size, self.img_size),
                                interpolation=TF.InterpolationMode.NEAREST)
        return image, mask

    def __getitem__(self, idx):
        name     = self.image_paths[idx]
        img_path = os.path.join(self.image_dir, name)
        msk_path = os.path.join(self.mask_dir,  name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(msk_path).convert('L')

        if self.is_train:
            image, mask = self._joint_transform(image, mask)
            # Colour jitter on image only
            image = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)(image)
        else:
            image = TF.resize(image, (self.img_size, self.img_size))
            mask  = TF.resize(mask,  (self.img_size, self.img_size),
                              interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = self.normalize(image)
        mask  = TF.to_tensor(mask)           # [1, H, W] in [0, 1]
        mask  = (mask > 0.5).float()         # binarise
        return image, mask

# %%
IMG_SIZE   = 224
BATCH_SIZE = 8          # swin_base is heavier; reduce if OOM
NUM_WORKERS = 4

train_dataset = COD10KDataset(train_folders, is_train=True,  img_size=IMG_SIZE)
test_dataset  = COD10KDataset(test_folders,  is_train=False, img_size=IMG_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
print(f"Test : {len(test_dataset)}  samples, {len(test_loader)}  batches")

# %%
# ── 4. Swin-Base Encoder ─────────────────────────────────────
class SwinEncoder(nn.Module):
    """
    Swin-Base backbone.
    Returns 4 feature maps with channels [128, 256, 512, 1024].
    """
    def __init__(self, model_name='swin_base_patch4_window7_224'):
        super().__init__()
        base = timm.create_model(model_name, pretrained=True)
        self.patch_embed  = base.patch_embed
        self.layers       = base.layers
        self.out_channels = [128, 256, 512, 1024]

    def forward(self, x):
        features = []
        x = self.patch_embed(x)   # (B, H, W, C) or (B, L, C) depending on timm version

        for layer in self.layers:
            x = layer(x)

            # Handle both (B, H, W, C) [4D] and (B, L, C) [3D] outputs
            if x.dim() == 4:
                # timm >= 0.6 typically outputs (B, H, W, C)
                B, H, W, C = x.shape
                feat = x.permute(0, 3, 1, 2).contiguous()   # → (B, C, H, W)
            elif x.dim() == 3:
                # older timm outputs (B, L, C)
                B, L, C = x.shape
                H = W = int(L ** 0.5)
                feat = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
            else:
                raise ValueError(f"Unexpected encoder output shape: {x.shape}")

            features.append(feat)

        return features   # [(B,128,56,56), (B,256,28,28), (B,512,14,14), (B,1024,7,7)]

# %%
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))

# %%
class FOM(nn.Module):
    """Feature Optimisation Module — channel reduction to `out_channels` + CBAM."""
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
        )
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        return self.cbam(self.conv(x))


class FID(nn.Module):
    """Feature Integration & Differentiation — fuses two adjacent scales."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch_hi = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))
        self.branch_lo = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))
        self.cbam = CBAM(out_channels)

    def forward(self, hi_res, lo_res):
        """hi_res is the shallower (larger) feature, lo_res is the deeper (smaller)."""
        x_hi = self.branch_hi(hi_res)
        x_lo = self.branch_lo(lo_res)
        x_lo = F.interpolate(x_lo, size=x_hi.shape[2:], mode='bilinear', align_corners=False)
        return self.cbam(self.fusion(torch.cat([x_hi, x_lo], dim=1)))


class FHIM(nn.Module):
    """Feature Hierarchical Integration Module — merges N feature maps."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True),
        )
        self.cbam = CBAM(out_channels)

    def forward(self, *features):
        target = features[0].shape[2:]
        ups = [F.interpolate(f, size=target, mode='bilinear', align_corners=False)
               if f.shape[2:] != target else f for f in features]
        return self.cbam(self.conv(torch.cat(ups, dim=1)))


class BRM(nn.Module):
    """Boundary Refinement Module — morphological edge enhancement."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # Dilated conv for wider receptive field
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x       = self.conv(x)
        dilated = F.max_pool2d(x,  kernel_size=3, stride=1, padding=1)
        eroded  = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        return x + (dilated - eroded)   # residual boundary signal

# %%
# ── 7. Decoder Head ──────────────────────────────────────────
FEAT_CH = 64

class DecoderHead(nn.Module):
    def __init__(self, enc_channels=(128, 256, 512, 1024), feat_ch=FEAT_CH):
        super().__init__()
        self.fom = nn.ModuleList([FOM(c, feat_ch) for c in enc_channels])

        self.fid_32 = FID(feat_ch, feat_ch)
        self.fid_21 = FID(feat_ch, feat_ch)
        self.fid_10 = FID(feat_ch, feat_ch)

        self.fhim    = FHIM(feat_ch * 4, feat_ch)
        self.brm     = BRM(feat_ch)

        self.mask_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch // 2, 1, 1),
        )
        self.edge_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch // 2, 1, 1),
        )

    def forward(self, feats, out_size=(224, 224)):
        c = [self.fom[i](feats[i]) for i in range(4)]

        f32 = self.fid_32(c[2], c[3])
        f21 = self.fid_21(c[1], f32)
        f10 = self.fid_10(c[0], f21)

        f_fhim = self.fhim(c[0], f21, f32, c[3])
        f_brm  = self.brm(f_fhim)

        # NO sigmoid here — raw logits, sigmoid applied only at inference
        mask_logit = F.interpolate(
            self.mask_head(f_brm), size=out_size, mode='bilinear', align_corners=False)
        edge_logit = F.interpolate(
            self.edge_head(f_brm), size=out_size, mode='bilinear', align_corners=False)

        return mask_logit, edge_logit   # raw logits

# %%
class IFBONetV3(nn.Module):
    """
    Two-stage camouflage object detection network.
    Stage 1 : coarse segmentation → M1, E1
    Masking  : x_masked = x * (blend * M1 + (1-blend))
    Stage 2  : refinement on focused input → M2, E2
    """
    def __init__(self, mask_blend: float = 0.9,
                 enc_model: str = 'swin_base_patch4_window7_224'):
        super().__init__()
        self.encoder    = SwinEncoder(enc_model)
        enc_channels    = self.encoder.out_channels          # [128, 256, 512, 1024]
        self.decoder1   = DecoderHead(enc_channels, FEAT_CH)
        self.decoder2   = DecoderHead(enc_channels, FEAT_CH)
        self.mask_blend = mask_blend

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_coarse: bool = True):
        # Stage 1
        feats1 = self.encoder(x)
        M1, E1 = self.decoder1(feats1)

        # Soft masking — suppress background
        x_masked = x * (self.mask_blend * M1.detach() + (1.0 - self.mask_blend))

        # Stage 2
        feats2 = self.encoder(x_masked)
        M2, E2 = self.decoder2(feats2)

        if return_coarse:
            return M1, E1, M2, E2
        return M2, E2

# %%
# ── 9. Loss Functions ────────────────────────────────────────
def structure_loss(pred_logit, mask):
    """
    Weighted BCEWithLogits + Dice loss.
    pred_logit: raw logits (no sigmoid applied)
    mask      : binary float in [0, 1]
    """
    # Boundary-aware pixel weights
    weight = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    # BCEWithLogits is AMP-safe
    bce = F.binary_cross_entropy_with_logits(pred_logit, mask,
                                              weight=weight, reduction='mean')

    # Dice on sigmoid of logits
    pred = torch.sigmoid(pred_logit)
    inter = (pred * mask * weight).sum(dim=(1, 2, 3))
    union = ((pred + mask) * weight).sum(dim=(1, 2, 3))
    dice  = 1 - (2 * inter + 1) / (union + 1)

    return bce + dice.mean()


def _single_loss(pred_logit, edge_logit, mask):
    loss_mask = structure_loss(pred_logit, mask)
    loss_edge = structure_loss(edge_logit, mask)
    return loss_mask + 0.4 * loss_edge


def loss_fn_v2(M1, E1, M2, E2, mask, w1: float = 0.4, w2: float = 1.0):
    loss1 = _single_loss(M1, E1, mask)
    loss2 = _single_loss(M2, E2, mask)
    total = w1 * loss1 + w2 * loss2
    return total, loss1.item(), loss2.item()

# %%
# ── compute_fbw helper (required by compute_fbw_batch) ──────

def _gauss2D(shape=(7, 7), sigma=5):
    m, n = [(ss - 1) / 2 for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    h /= h.sum()
    return h


def compute_fbw(pred: np.ndarray, gt: np.ndarray,
                beta2: float = 1.0, sigma: int = 7, eps: float = 1e-8) -> float:
    pred = np.clip(pred.astype(np.float64), 0, 1)
    gt   = gt.astype(bool)
    if gt.sum() == 0:
        return 1.0 - pred.mean()
    E  = np.abs(pred - gt.astype(np.float64))
    Dst, Idxt = bwdist(~gt, return_indices=True)
    Et = E.copy()
    bg = np.where(~gt)
    Et[bg] = E[Idxt[0][bg], Idxt[1][bg]]
    K  = _gauss2D((sigma, sigma), sigma / 2)
    EA = convolve(Et, weights=K, mode="reflect")
    MIN_E_EA = np.where(gt & (EA < E), EA, E)
    B  = np.ones_like(gt, dtype=np.float64)
    B[~gt] = 2 - np.exp(np.log(0.5) / 5 * Dst[~gt])
    Ew  = MIN_E_EA * B
    TPw = gt.sum() - Ew[gt].sum()
    FPw = Ew[~gt].sum()
    R   = 1 - Ew[gt].mean() if gt.sum() > 0 else 0
    P   = TPw / (TPw + FPw + eps)
    return (1 + beta2) * R * P / (R + beta2 * P + eps)

# %%
# ── 10. Metrics ──────────────────────────────────────────────
def to_prob(logit):
    """Convert raw logit to probability in [0,1]."""
    return torch.sigmoid(logit)

def compute_mae(pred_logit, gt):
    return torch.abs(to_prob(pred_logit) - gt).mean().item()

def compute_smeasure(pred_logit, mask, alpha=0.5):
    pred = (to_prob(pred_logit) > 0.5).float()
    mask = (mask > 0.5).float()
    y    = mask.mean()
    if y == 0:   return 1 - pred.mean().item()
    if y == 1:   return pred.mean().item()
    fg = (pred * mask).sum() / (mask.sum() + 1e-6)
    bg = ((1 - pred) * (1 - mask)).sum() / ((1 - mask).sum() + 1e-6)
    return (alpha * fg + (1 - alpha) * bg).item()

def compute_ephi(pred_logit, mask, eps=1e-8):
    pred = to_prob(pred_logit).detach().cpu().float().clamp(0, 1)
    mask = (mask.detach().cpu().float() > 0.5).float()
    th   = min(2 * pred.mean().item(), 1.0)
    pb   = (pred >= th).float()
    fg   = mask.sum().item()
    size = mask.numel()
    if fg == 0:    return 1.0 - pb.mean().item()
    if fg == size: return pb.mean().item()
    pm, gm = pb.mean(), mask.mean()
    align  = 2 * (pb - pm) * (mask - gm) / (((pb - pm)**2 + (mask - gm)**2) + eps)
    return (((align + 1)**2) / 4).mean().item()

def compute_fbw_batch(pred_logit: torch.Tensor, masks: torch.Tensor) -> float:
    preds = to_prob(pred_logit).detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    return float(np.mean([
        compute_fbw(preds[i].squeeze(), masks[i].squeeze())
        for i in range(len(preds))
    ]))

def compute_accuracy(pred_logit, mask):
    pred = (to_prob(pred_logit) > 0.5).float()
    return (pred == (mask > 0.5).float()).float().mean().item()

# %%
def tta_predict(model, img, scales=(0.75, 1.0, 1.25), device='cuda'):
    """Average predictions over multiple scales + horizontal flip."""
    preds = []
    h, w  = img.shape[2:]
    for s in scales:
        hs, ws = int(h * s), int(w * s)
        x = F.interpolate(img, (hs, ws), mode='bilinear', align_corners=False)
        M2, _ = model(x, return_coarse=False)
        preds.append(F.interpolate(M2, (h, w), mode='bilinear', align_corners=False))
        # Horizontal flip
        xf    = torch.flip(x, dims=[3])
        M2f, _ = model(xf, return_coarse=False)
        M2f   = torch.flip(M2f, dims=[3])
        preds.append(F.interpolate(M2f, (h, w), mode='bilinear', align_corners=False))
    return torch.stack(preds, dim=0).mean(dim=0)

# %%
CHECKPOINT_DIR = "IFBO_NET_V3_checkpoints"
PREDICTION_DIR = "IFBO_NET_V3_predictions"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)
print("Checkpoints:", os.path.abspath(CHECKPOINT_DIR))
print("Predictions:", os.path.abspath(PREDICTION_DIR))


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = IFBONetV3(mask_blend=0.9).to(device)

# Differential LRs: backbone gets 10× smaller LR than heads
backbone_params = list(model.encoder.parameters())
decoder_params  = list(model.decoder1.parameters()) + list(model.decoder2.parameters())

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5,  'weight_decay': 1e-4},
    {'params': decoder_params,  'lr': 1e-4,  'weight_decay': 1e-4},
])

# Cosine annealing with warm restarts — LR never dies
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=2, eta_min=1e-7)

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

print(f"Backbone params : {sum(p.numel() for p in backbone_params) / 1e6:.1f}M")
print(f"Decoder params  : {sum(p.numel() for p in decoder_params)  / 1e6:.1f}M")
print(f"Total params    : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")


# %%
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total = dict(loss=0, loss1=0, loss2=0,
                 mae=0, salpha=0, ephi=0, fbw=0, acc=0)

    for img, mask in tqdm(loader, desc="[Train]", leave=False):
        img, mask = img.to(device), mask.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            M1, E1, M2, E2 = model(img, return_coarse=True)
            loss, l1, l2   = loss_fn_v2(M1, E1, M2, E2, mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            total["loss"]   += loss.item()
            total["loss1"]  += l1
            total["loss2"]  += l2
            total["mae"]    += compute_mae(M2, mask)
            total["salpha"] += compute_smeasure(M2, mask)
            total["ephi"]   += compute_ephi(M2, mask)
            total["fbw"]    += compute_fbw_batch(M2, mask)
            total["acc"]    += compute_accuracy(M2, mask)

    n = len(loader)
    return {k: v / n for k, v in total.items()}

# %%
def evaluate(model, loader, epoch, save_preds=False, use_tta=False):
    model.eval()
    total = dict(loss=0, loss1=0, loss2=0,
                 mae=0, salpha=0, ephi=0, fbw=0, acc=0)

    with torch.no_grad():
        for idx, (img, mask) in enumerate(tqdm(loader, desc="[Test ]", leave=False)):
            img, mask = img.to(device), mask.to(device)

            if use_tta:
                M1, E1 = model(img, return_coarse=False)   # dummy — TTA ignores coarse
                M1, E1, M2, E2 = model(img, return_coarse=True)
                M2 = tta_predict(model, img)
                E2 = E1   # edge head not TTA'd
            else:
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    M1, E1, M2, E2 = model(img, return_coarse=True)

            loss, l1, l2 = loss_fn_v2(M1, E1, M2, E2, mask)

            total["loss"]   += loss.item()
            total["loss1"]  += l1
            total["loss2"]  += l2
            total["mae"]    += compute_mae(M2, mask)
            total["salpha"] += compute_smeasure(M2, mask)
            total["ephi"]   += compute_ephi(M2, mask)
            total["fbw"]    += compute_fbw_batch(M2, mask)
            total["acc"]    += compute_accuracy(M2, mask)

            if save_preds:
                pred_np = (torch.sigmoid(M2) > 0.5).float().cpu().numpy()   # sigmoid here
                gt_np   = mask.cpu().numpy()
                for b in range(pred_np.shape[0]):
                    plt.imsave(os.path.join(PREDICTION_DIR,
                        f"ep{epoch}_b{idx}_{b}_pred.png"), pred_np[b, 0], cmap="gray")
                    plt.imsave(os.path.join(PREDICTION_DIR,
                        f"ep{epoch}_b{idx}_{b}_gt.png"),   gt_np[b, 0],   cmap="gray")

    n = len(loader)
    return {k: v / n for k, v in total.items()}

# %%
# ── Resume from best FbW checkpoint ─────────────────────────

RESUME_FROM = "model_best_fbw.pkl"   # change to any: model_best_salpha.pkl
                                      #                 model_best_ephi.pkl
                                      #                 model_best_composite.pkl
                                      #                 model_latest.pkl

resume_path = os.path.join(CHECKPOINT_DIR, RESUME_FROM)

history              = {"train": [], "test": []}
best_scores          = {"fbw": float('-inf'), "salpha": float('-inf'), "ephi": float('-inf')}
best_composite_score = float('-inf')
start_epoch          = 0

if os.path.exists(resume_path):
    print(f"Loading checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch          = ckpt["epoch"]
    best_scores          = ckpt.get("best_scores",  best_scores)
    best_composite_score = ckpt.get("composite_score", float('-inf'))

    if "history" in ckpt:
        history = ckpt["history"]

    print(f"Resuming from epoch {start_epoch}")
    print(f"   Best FbW   : {best_scores['fbw']:.4f}")
    print(f"   Best S_alpha : {best_scores['salpha']:.4f}")
    print(f"   Best E_phi : {best_scores['ephi']:.4f}")
    print(f"   Composite  : {best_composite_score:.4f}")
else:
    print(f"Not found: {resume_path} — starting from scratch.")

# ── Training loop (resume-aware) ────────────────────────────

WARMUP_EPOCHS = 5
NUM_EPOCHS    = 1000
USE_TTA_EVAL  = False

def warmup_lr(epoch, warmup_epochs, base_lr=1e-4):
    return base_lr * min(1.0, (epoch + 1) / warmup_epochs)

for epoch in range(start_epoch, NUM_EPOCHS):

    # Warmup only applies if still in warmup window
    if epoch < WARMUP_EPOCHS:
        for g in optimizer.param_groups:
            g['lr'] = warmup_lr(epoch, WARMUP_EPOCHS, g.get('initial_lr', g['lr']))

    train_m = train_one_epoch(model, train_loader, optimizer, scaler)
    test_m  = evaluate(model, test_loader, epoch,
                       save_preds=False, use_tta=USE_TTA_EVAL)

    # Step scheduler (after warmup)
    if epoch >= WARMUP_EPOCHS:
        scheduler.step(epoch - WARMUP_EPOCHS)

    history["train"].append(train_m)
    history["test"].append(test_m)

    lr_now = optimizer.param_groups[1]['lr']
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}  |  LR={lr_now:.2e}")
    print(f"  Train  Loss {train_m['loss']:.4f}"
          f"  (L1={train_m['loss1']:.4f} L2={train_m['loss2']:.4f})"
          f"  MAE {train_m['mae']:.4f}"
          f"  S_alpha {train_m['salpha']:.4f}"
          f"  E_phi {train_m['ephi']:.4f}"
          f"  FbW {train_m['fbw']:.4f}"
          f"  Acc {train_m['acc']:.4f}")
    print(f"  Test   Loss {test_m['loss']:.4f}"
          f"  (L1={test_m['loss1']:.4f} L2={test_m['loss2']:.4f})"
          f"  MAE {test_m['mae']:.4f}"
          f"  S_alpha {test_m['salpha']:.4f}"
          f"  E_phi {test_m['ephi']:.4f}"
          f"  FbW {test_m['fbw']:.4f}"
          f"  Acc {test_m['acc']:.4f}")

    # Save best per-metric checkpoints
    for metric in ["fbw", "salpha", "ephi"]:
        if test_m[metric] > best_scores[metric]:
            best_scores[metric] = test_m[metric]
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "best_scores":          best_scores,
                "test_metrics":         test_m,
                "train_metrics":        train_m,
                "metric":               metric,
            }, os.path.join(CHECKPOINT_DIR, f"model_best_{metric}.pkl"))
            print(f"  Saved best {metric.upper()}: {best_scores[metric]:.4f}")

    # Save best composite checkpoint
    weights   = {"fbw": 0.35, "salpha": 0.35, "ephi": 0.20, "mae": 0.10}
    composite = (weights["fbw"]    * test_m["fbw"]
               + weights["salpha"] * test_m["salpha"]
               + weights["ephi"]   * test_m["ephi"]
               - weights["mae"]    * test_m["mae"])

    if composite > best_composite_score:
        best_composite_score = composite
        torch.save({
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "best_scores":          best_scores,
            "test_metrics":         test_m,
            "train_metrics":        train_m,
            "composite_score":      best_composite_score,
            "weights":              weights,
        }, os.path.join(CHECKPOINT_DIR, "model_best_composite.pkl"))
        print(f"  Saved best COMPOSITE: {best_composite_score:.4f}")

    # Save latest checkpoint every 10 epochs (safe resume)
    if (epoch + 1) % 10 == 0:
        torch.save({
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "history":              history,
        }, os.path.join(CHECKPOINT_DIR, "model_latest.pkl"))

# %%
def show_checkpoint_metrics(filename, ckpt_dir=CHECKPOINT_DIR):
    path = os.path.join(ckpt_dir, filename)
    if not os.path.exists(path):
        print(f"Not found: {path}"); return
    ckpt = torch.load(path, map_location="cpu")
    print(f"\n=== {filename} ===")
    print(f"  Epoch          : {ckpt.get('epoch')}")
    print(f"  Metric saved   : {ckpt.get('metric', 'composite')}")
    print(f"  Composite score: {ckpt.get('composite_score')}")
    print("\n  Test metrics:")
    pprint(ckpt.get("test_metrics"))
    print("\n  Train metrics:")
    pprint(ckpt.get("train_metrics"))

show_checkpoint_metrics("model_best_fbw.pkl")
show_checkpoint_metrics("model_best_salpha.pkl")
show_checkpoint_metrics("model_best_ephi.pkl")
show_checkpoint_metrics("model_best_composite.pkl")

# %%
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_metric = "fbw"
ckpt_path = f"IFBO_NET_V3_checkpoints/model_best_{best_metric}.pkl"

model_test = IFBONetV3().to(device)
checkpoint = torch.load(ckpt_path, map_location=device)
model_test.load_state_dict(checkpoint["model_state_dict"])
model_test.eval()

activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

target_layer = model_test.decoder2.brm.conv
fh = target_layer.register_forward_hook(forward_hook)
bh = target_layer.register_backward_hook(backward_hook)

with torch.enable_grad():
    for idx, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        masks = masks.to(device)

        for b in range(images.size(0)):
            activations.clear()
            gradients.clear()

            img = images[b].unsqueeze(0).requires_grad_(True)
            mask = masks[b].unsqueeze(0)

            _, _, preds, _ = model_test(img, return_coarse=True)  # ✅ correct output
            score = preds.mean()

            model_test.zero_grad()
            score.backward()

            grad = gradients[0].cpu().detach().numpy()[0]
            act = activations[0].cpu().detach().numpy()[0]

            weights = np.mean(grad, axis=(1, 2))
            cam = np.zeros(act.shape[1:], dtype=np.float32)

            for i, w in enumerate(weights):
                cam += w * act[i]

            cam = np.maximum(cam, 0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            cam_tensor = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
            cam_up = F.interpolate(
                cam_tensor,
                size=(img.size(2), img.size(3)),
                mode='bilinear',
                align_corners=False
            ).squeeze().cpu().numpy()

            # ✅ logits → sigmoid (same as your pipeline)
            pred_mask = torch.sigmoid(preds[0, 0]).cpu().detach().numpy()

            pred_mask_8bit = (pred_mask * 255).astype(np.uint8)

            otsu_thresh_val, mask_otsu = cv2.threshold(
                pred_mask_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            mask_np = mask[0, 0].cpu().detach().numpy()
            img_np = img[0].detach().cpu().permute(1, 2, 0).numpy()

            print(f"Batch {idx} Img {b}: Otsu Threshold Value = {otsu_thresh_val}")

            plt.figure(figsize=(20, 5))

            plt.subplot(1, 5, 1)
            plt.imshow(img_np)
            plt.title("Input Image")
            plt.axis("off")

            plt.subplot(1, 5, 2)
            plt.imshow(mask_np, cmap="gray")
            plt.title("Ground Truth Mask")
            plt.axis("off")

            plt.subplot(1, 5, 3)
            plt.imshow(pred_mask, cmap="jet")
            plt.title(f"Predicted Mask (Jet) [{best_metric}]")
            plt.axis("off")

            plt.subplot(1, 5, 4)
            plt.imshow(mask_otsu, cmap="gray")
            plt.title("Predicted Mask (Otsu Binarized)")
            plt.axis("off")

            plt.subplot(1, 5, 5)
            plt.imshow(img_np)
            plt.imshow(cam_up, cmap="jet", alpha=0.4)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title("Grad-CAM Overlay")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

fh.remove()
bh.remove()
