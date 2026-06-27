# %% [markdown]
# # Phase 2: Loss Function Comparison with Fixed ResNet18 Backbone
#
# **Objective**: Compare 6 loss functions under identical conditions using a fixed
# pretrained ResNet18 (iResNet18) backbone from InsightFace.
#
# **Design**: Transfer fine-tuning — MS1MV3-pretrained R18 → CASIA-WebFace dataset.
# All losses share the same pretrained initialization and dataset, ensuring a fair comparison.
#
# **Losses**: ArcFace, CosFace, CurricularFace, ElasticFace, AdaFace, MagFace
#
# **Runtime**: Google Colab T4 GPU, report mode (5 epochs per loss).

# %% [markdown]
# ---
# ## Cell 0: Install Dependencies

# %%
# === Install dependencies ===
!pip install -q easydict ptflops gdown opencv-python scikit-learn
!pip install -q mxnet==1.9.1

# Patch np.bool for mxnet compatibility
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'float'):
    np.float = np.float_
if not hasattr(np, 'object'):
    np.object = np.object_

try:
    import mxnet
    print(f"mxnet OK: {mxnet.__version__}")
except ImportError:
    print("WARNING: mxnet not installed — .bin eval files cannot be loaded.")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# %% [markdown]
# ---
# ## Cell 1: Mount Google Drive + Global Config

# %%
import os
from google.colab import drive
drive.mount('/content/drive')

# ============================================================
#  GLOBAL CONFIG — Edit here before running
# ============================================================

RUN_MODE = "report"          # "report" = 5 epochs, full pipeline
BACKBONE = "r18"             # iResNet18
BATCH_SIZE = 64              # OOM? → reduce SAMPLE_RATE first, then BATCH_SIZE
PHASE2_EPOCHS = 5
SAMPLE_RATE = 1.0            # OOM fallback: 0.5
FP16 = True
NUM_WORKERS = 2
FORCE_RETRAIN = False        # True = retrain even if model.pt exists
SAVE_EVERY_EPOCH = True
BACKUP_EVERY_EPOCH = True

LOSS_LIST = [
    "arcface",
    "cosface",
    "curricularface",
    "elasticface",
    "adaface",
    "magface",
]

# === Paths ===
DRIVE_ROOT = "/content/drive/MyDrive"
PROJECT_DIR = f"{DRIVE_ROOT}/phase2_resnet18_loss_comparison"
PRETRAINED_CKPT = f"{PROJECT_DIR}/pretrained/backbone.pth"
DATASET_DIR = "/content/faces_webface_112x112"
WORK_DIR = "/content/insightface/recognition/arcface_torch"
OUTPUT_ROOT = "/content/phase2_outputs"
BACKUP_ROOT = PROJECT_DIR

# Dataset info (CASIA-WebFace)
NUM_CLASSES = 10572
NUM_IMAGE = 490623

# Optimizer
LR = 0.01         # Lower LR for fine-tuning (not training from scratch)
WEIGHT_DECAY = 5e-4
WARMUP_EPOCH = 1
OPTIMIZER = "sgd"

# Create Drive directory structure
for subdir in ["pretrained", "models", "checkpoints", "train_logs",
               "eval_logs", "degraded_eval_logs", "benchmark_logs",
               "configs", "final_backup"]:
    os.makedirs(f"{BACKUP_ROOT}/{subdir}", exist_ok=True)
for loss_name in LOSS_LIST:
    os.makedirs(f"{BACKUP_ROOT}/models/{loss_name}", exist_ok=True)
    os.makedirs(f"{BACKUP_ROOT}/checkpoints/{loss_name}", exist_ok=True)

print("=" * 60)
print("  Phase 2: Loss Function Comparison")
print("=" * 60)
print(f"  RUN_MODE       = {RUN_MODE}")
print(f"  BACKBONE       = {BACKBONE}")
print(f"  BATCH_SIZE     = {BATCH_SIZE}")
print(f"  PHASE2_EPOCHS  = {PHASE2_EPOCHS}")
print(f"  SAMPLE_RATE    = {SAMPLE_RATE}")
print(f"  FP16           = {FP16}")
print(f"  FORCE_RETRAIN  = {FORCE_RETRAIN}")
print(f"  LR             = {LR}")
print(f"  LOSS_LIST      = {LOSS_LIST}")
print(f"  PRETRAINED     = {PRETRAINED_CKPT}")
print(f"  DATASET        = {DATASET_DIR}")
print(f"  BACKUP         = {BACKUP_ROOT}")
print("=" * 60)

# %% [markdown]
# ---
# ## Cell 2: Check GPU

# %%
!nvidia-smi

import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU NOT AVAILABLE! Go to Runtime → Change runtime type → T4 GPU")

gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f"\nGPU: {gpu_name}")
print(f"VRAM: {vram_gb:.1f} GB")
print(f"CUDA version: {torch.version.cuda}")

# %% [markdown]
# ---
# ## Cell 3: Clone / Setup InsightFace

# %%
import os

if not os.path.exists('/content/insightface'):
    !git clone --depth 1 https://github.com/deepinsight/insightface.git /content/insightface
    print("[OK] Cloned insightface")
else:
    print("[OK] insightface already exists")

WORK_DIR = '/content/insightface/recognition/arcface_torch'
os.chdir(WORK_DIR)
print(f"Working directory: {os.getcwd()}")

# Verify r18 backbone
from backbones import get_model
import torch

test_model = get_model("r18", dropout=0, fp16=False, num_features=512)
with torch.no_grad():
    dummy = torch.randn(2, 3, 112, 112)
    out = test_model(dummy)
params_m = sum(p.numel() for p in test_model.parameters()) / 1e6
print(f"[OK] r18 backbone: output={list(out.shape)}, params={params_m:.2f}M")
del test_model, out, dummy

# === Create degradation module (3 core types only) ===
# The upstream InsightFace repo does NOT include this module.
# Project scope: gaussian_blur, low_resolution, low_illumination only.
os.makedirs('degradation', exist_ok=True)

with open('degradation/__init__.py', 'w') as f:
    f.write('from .transforms import DegradationTransform, SUPPORTED_DEGRADATIONS\n')

deg_code = r'''"""
Image degradation transforms for evaluation — 3 core types only.

All transforms accept/return numpy array (H, W, 3) uint8.
Severity levels: 1 (mild) to 5 (severe).
"""
import cv2
import numpy as np

SUPPORTED_DEGRADATIONS = [
    "gaussian_blur",
    "low_resolution",
    "low_illumination",
]

_GAUSSIAN_BLUR_SIGMA = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.5, 5: 5.0}
_LOW_RES_SIZE = {1: 56, 2: 42, 3: 28, 4: 20, 5: 14}
_LOW_ILLUM_GAMMA = {1: 1.3, 2: 1.6, 3: 2.0, 4: 2.7, 5: 3.5}


def apply_gaussian_blur(image, severity=1):
    sigma = _GAUSSIAN_BLUR_SIGMA.get(severity, 2.0)
    ksize = int(np.ceil(sigma * 3)) * 2 + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_low_resolution(image, severity=1):
    h, w = image.shape[:2]
    low = _LOW_RES_SIZE.get(severity, 28)
    small = cv2.resize(image, (low, low), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_low_illumination(image, severity=1):
    gamma = _LOW_ILLUM_GAMMA.get(severity, 2.0)
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


_DEGRADATION_FN = {
    "gaussian_blur": apply_gaussian_blur,
    "low_resolution": apply_low_resolution,
    "low_illumination": apply_low_illumination,
}


class DegradationTransform:
    """Configurable, reproducible image degradation."""

    def __init__(self, degradation_type, severity=1, seed=42):
        if degradation_type not in SUPPORTED_DEGRADATIONS:
            raise ValueError(
                f"Unknown: {degradation_type}. Supported: {SUPPORTED_DEGRADATIONS}")
        if not 1 <= severity <= 5:
            raise ValueError(f"Severity must be 1-5, got {severity}")
        self.degradation_type = degradation_type
        self.severity = severity
        self._fn = _DEGRADATION_FN[degradation_type]

    def apply(self, image):
        return self._fn(image, severity=self.severity)

    def __repr__(self):
        return f"DegradationTransform({self.degradation_type}, s={self.severity})"
'''

with open('degradation/transforms.py', 'w') as f:
    f.write(deg_code)
print("[OK] degradation/ module (3 core types: blur, resolution, illumination)")

# %% [markdown]
# ---
# ## Cell 4: Dataset Check

# %%
import os

DATASET_DIR_ACTUAL = DATASET_DIR

# Check dataset exists — try to download CASIA if missing
if not os.path.exists(DATASET_DIR_ACTUAL):
    print(f"Dataset not found at {DATASET_DIR_ACTUAL}")
    print("Attempting to download CASIA-WebFace (faces_webface_112x112)...")
    # Common gdown ID for CASIA-WebFace
    !pip install -q gdown
    !gdown --id 1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l -O /content/faces_webface_112x112.zip
    if os.path.exists("/content/faces_webface_112x112.zip"):
        !cd /content && unzip -q -o faces_webface_112x112.zip
        print("[OK] Dataset extracted")
    else:
        print("ERROR: Could not download dataset.")
        print("Please upload CASIA-WebFace (faces_webface_112x112/) to /content/")
        print("Must contain: train.rec, train.idx, property, lfw.bin, cfp_fp.bin, agedb_30.bin")

# Verify dataset files
required_files = ['train.rec', 'train.idx']
for f in required_files:
    path = os.path.join(DATASET_DIR_ACTUAL, f)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024**2
        print(f"  [OK] {f}: {size_mb:.1f} MB")
    else:
        print(f"  [MISSING] {f}")
        raise FileNotFoundError(f"Required file missing: {path}")

# Read num_classes from property file
prop_file = os.path.join(DATASET_DIR_ACTUAL, 'property')
if os.path.exists(prop_file):
    with open(prop_file, 'r') as f:
        content = f.read().strip()
    print(f"  [OK] property: {content}")
    # Format: "num_classes,height,width" e.g. "10572,112,112"
    parts = content.split(',')
    detected_classes = int(parts[0])
    if detected_classes != NUM_CLASSES:
        print(f"  WARNING: property says {detected_classes} classes, config says {NUM_CLASSES}")
        print(f"  Updating NUM_CLASSES to {detected_classes}")
        NUM_CLASSES = detected_classes
else:
    print(f"  [INFO] No property file, using NUM_CLASSES={NUM_CLASSES}")

# Check eval bins
eval_bins = ['lfw.bin', 'cfp_fp.bin', 'agedb_30.bin']
for b in eval_bins:
    path = os.path.join(DATASET_DIR_ACTUAL, b)
    exists = "OK" if os.path.exists(path) else "MISSING"
    print(f"  [{exists}] {b}")

print(f"\nDataset: {DATASET_DIR_ACTUAL}")
print(f"NUM_CLASSES: {NUM_CLASSES}")
print(f"NUM_IMAGE: {NUM_IMAGE}")

# %% [markdown]
# ---
# ## Cell 5: Load & Verify Pretrained ResNet18 Backbone
#
# Checkpoint from InsightFace model zoo: `ms1mv3_arcface_r18_fp16/backbone.pth`
# trained on MS1MV3 with ArcFace. We use this as the shared starting point
# for all loss functions (transfer fine-tuning).

# %%
import torch
import os
import shutil
from backbones import get_model

# Check pretrained checkpoint exists
if not os.path.exists(PRETRAINED_CKPT):
    # Try alternate locations
    alt_paths = [
        f"{DRIVE_ROOT}/ms1mv3_arcface_r18_fp16/backbone.pth",
        f"{DRIVE_ROOT}/pretrained/backbone.pth",
        f"{PROJECT_DIR}/backbone.pth",
    ]
    found = False
    for alt in alt_paths:
        if os.path.exists(alt):
            print(f"[INFO] Found checkpoint at {alt}, copying to {PRETRAINED_CKPT}")
            shutil.copy2(alt, PRETRAINED_CKPT)
            found = True
            break
    if not found:
        print("=" * 60)
        print("ERROR: Pretrained checkpoint not found!")
        print(f"Expected: {PRETRAINED_CKPT}")
        print()
        print("Please download 'ms1mv3_arcface_r18_fp16' from InsightFace OneDrive:")
        print("https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d")
        print()
        print("Upload backbone.pth to Google Drive at:")
        print(f"  {PRETRAINED_CKPT}")
        print("=" * 60)
        raise FileNotFoundError(f"Pretrained checkpoint not found: {PRETRAINED_CKPT}")

# Load and verify
print(f"Loading pretrained checkpoint: {PRETRAINED_CKPT}")
ckpt_size = os.path.getsize(PRETRAINED_CKPT) / 1024**2
print(f"  Checkpoint size: {ckpt_size:.1f} MB")

backbone = get_model("r18", dropout=0, fp16=FP16, num_features=512)
state_dict = torch.load(PRETRAINED_CKPT, map_location='cpu')

# Handle different checkpoint formats
if 'state_dict_backbone' in state_dict:
    state_dict = state_dict['state_dict_backbone']
elif 'model' in state_dict:
    state_dict = state_dict['model']

# Load with strict=False to see missing/unexpected keys
result = backbone.load_state_dict(state_dict, strict=False)
print(f"  Missing keys:    {len(result.missing_keys)}")
print(f"  Unexpected keys: {len(result.unexpected_keys)}")
if result.missing_keys:
    print(f"    Missing: {result.missing_keys[:5]}...")
if result.unexpected_keys:
    print(f"    Unexpected: {result.unexpected_keys[:5]}...")

# Must have zero missing keys for valid pretrained
if len(result.missing_keys) > 0:
    print("WARNING: Some backbone weights are missing! Training may not converge well.")

# Dummy forward pass
backbone = backbone.cuda().eval()
dummy_input = torch.randn(4, 3, 112, 112).cuda()

with torch.no_grad():
    embeddings = backbone(dummy_input)

print(f"\n  Dummy forward OK:")
print(f"    Input:  {list(dummy_input.shape)}")
print(f"    Output: {list(embeddings.shape)}")
print(f"    Output dtype: {embeddings.dtype}")

# Verify we can get raw feature norm (for AdaFace/MagFace)
# IResNet.forward() returns features AFTER BN but BEFORE L2-norm.
# PartialFC_V2_Extended computes norms from these raw features.
raw_norms = torch.norm(embeddings, dim=1)
print(f"    Raw feature norms: mean={raw_norms.mean():.2f}, "
      f"std={raw_norms.std():.2f}, "
      f"min={raw_norms.min():.2f}, max={raw_norms.max():.2f}")

# Cleanup
del backbone, dummy_input, embeddings, raw_norms
torch.cuda.empty_cache()

print("\n[OK] Pretrained R18 backbone verified successfully.")
print("     All 6 losses will start from this exact checkpoint.")

# %% [markdown]
# ---
# ## Cell 6: Implement All 6 Loss Functions
#
# Each loss follows the unified interface:
#   `forward(logits, labels, embeddings=None, norms=None)`
#
# - **ArcFace**: angular margin m2=0.5
# - **CosFace**: additive cosine margin m3=0.4
# - **CurricularFace**: curriculum learning with hard negative mining
# - **ElasticFace**: stochastic per-sample margin ~ N(m, σ)
# - **AdaFace**: quality-adaptive margin using feature norm
# - **MagFace**: magnitude-aware margin + regularization g(a_i)

# %%
import os

phase2_losses_code = r'''"""
Phase 2 Loss Functions for Fair Comparison.

All losses: forward(logits, labels, embeddings=None, norms=None) -> logits
AdaFace/MagFace require norms (raw feature norm before L2-normalization).
MagFace stores _last_mag_reg for regularization term.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. ArcFace — Standard angular margin (m=0.5)
# =====================================================================

class ArcFaceLoss(nn.Module):
    """ArcFace: Additive Angular Margin Loss (CVPR 2019).
    cos(theta + m) with m=0.5, s=64.
    """
    def __init__(self, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels, embeddings=None, norms=None):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.m
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s
        return logits


# =====================================================================
# 2. CosFace — Additive cosine margin (m=0.4)
# =====================================================================

class CosFaceLoss(nn.Module):
    """CosFace: Large Margin Cosine Loss (CVPR 2018).
    cos(theta) - m with m=0.4, s=64.
    """
    def __init__(self, s=64.0, m=0.4):
        super().__init__()
        self.s = s
        self.m = m

    def forward(self, logits, labels, embeddings=None, norms=None):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits


# =====================================================================
# 3. CurricularFace — Curriculum learning + hard negative mining
# =====================================================================

class CurricularFaceLoss(nn.Module):
    """CurricularFace: Adaptive Curriculum Learning Loss (CVPR 2020).

    Maintains running average t of positive target cosine similarity.
    For hard negatives (cos_j > cos(theta_yi + m)):
        cos_j -> cos_j * (t + cos_j)
    This adaptively emphasizes hard negatives as training progresses.

    Reference: https://arxiv.org/abs/2004.00288
    """
    def __init__(self, s=64.0, m=0.5, alpha=0.99):
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.alpha = alpha
        self.register_buffer('t', torch.zeros(1))

    def forward(self, logits, labels, embeddings=None, norms=None):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            # cos(theta_yi + m) using angle addition
            cos_theta = target_logit.clone()
            sin_theta = torch.sqrt(1.0 - cos_theta.pow(2).clamp(0, 1))
            cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

            # Safe margin (when theta > pi - m, use linear fallback)
            final_target = torch.where(
                cos_theta > self.threshold,
                cos_theta_m,
                cos_theta - self.mm
            )

            # Update t: EMA of mean positive target cosine
            self.t = self.alpha * self.t + (1.0 - self.alpha) * cos_theta.mean()

            # Hard negative modulation (vectorized)
            # For negatives where cos_j > cos(theta_yi + m): cos_j -> cos_j * (t + cos_j)
            neg_logits = logits[index].clone()

            # One-hot mask for target class
            one_hot = torch.zeros_like(neg_logits)
            one_hot.scatter_(1, labels[index].view(-1, 1), 1.0)

            # Hard negative mask: cos_j > cos(theta_yi + m) AND not target
            hard_mask = (neg_logits > cos_theta_m.unsqueeze(1)) & (one_hot == 0)

            # Modulate hard negatives
            modulated = neg_logits * (self.t + neg_logits)
            neg_logits = torch.where(hard_mask, modulated, neg_logits)

            # Set target logits
            neg_logits.scatter_(1, labels[index].view(-1, 1),
                                final_target.unsqueeze(1))
            logits[index] = neg_logits

        logits = logits * self.s
        return logits


# =====================================================================
# 4. ElasticFace — Stochastic per-sample margin
# =====================================================================

class ElasticFaceLoss(nn.Module):
    """ElasticFace: Elastic Margin Loss (CVPRW 2022).

    Samples per-sample margin m_i ~ N(m_mean, m_std) during training.
    Uses fixed m_mean at eval. This regularizes the decision boundary
    and improves generalization.

    Reference: https://arxiv.org/abs/2109.09416
    """
    def __init__(self, s=64.0, m=0.5, std=0.0125):
        super().__init__()
        self.s = s
        self.m = m
        self.std = std

    def forward(self, logits, labels, embeddings=None, norms=None):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()

            # Per-sample margin sampling
            if self.training:
                margins = torch.normal(
                    mean=self.m, std=self.std,
                    size=target_logit.shape,
                    device=target_logit.device
                )
                margins = margins.clamp(min=0.05)  # ensure positive margin
            else:
                margins = self.m

            final_target_logit = target_logit + margins
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()

        logits = logits * self.s
        return logits


# =====================================================================
# 5. AdaFace — Quality-adaptive margin using feature norm
# =====================================================================

class AdaFaceLoss(nn.Module):
    """AdaFace: Quality Adaptive Margin for Face Recognition (CVPR 2022).

    Uses feature norm (before L2-normalization) as image quality indicator.
    - g_angular = m * margin_scaler * (-1)  → added to theta via one-hot
    - g_additive = m + m * margin_scaler    → subtracted from cosine target
    - margin_scaler = clip(h * (norm - mean) / std, -1, 1)

    Reference: https://arxiv.org/abs/2204.00964
    Official:  https://github.com/mk-minchul/AdaFace
    """
    def __init__(self, s=64.0, m=0.4, h=0.333, t_alpha=0.01):
        super().__init__()
        self.s = s
        self.m = m
        self.h = h
        self.t_alpha = t_alpha
        self.eps = 1e-3

        # EMA running statistics for feature norms
        self.register_buffer('batch_mean', torch.ones(1) * 20.0)
        self.register_buffer('batch_std', torch.ones(1) * 100.0)

    def forward(self, logits, labels, embeddings=None, norms=None):
        index = torch.where(labels != -1)[0]

        if norms is None:
            # Fallback: standard ArcFace if norms not provided
            return self._arcface_fallback(logits, labels, index)

        # Safe norms (clip + detach, per official code)
        safe_norms = torch.clip(norms, min=0.001, max=100).clone().detach()

        # Update EMA of feature norm statistics
        with torch.no_grad():
            mean = safe_norms[index].mean()
            std = safe_norms[index].std()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        # margin_scaler: z-score scaled by h, clipped to [-1, 1]
        margin_scaler = (safe_norms[index].view(-1) - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = (margin_scaler * self.h).clamp(-1, 1)

        with torch.no_grad():
            # g_angular: applied to theta via one-hot mask (per paper Eq. 6)
            # g_angular = m * margin_scaler * (-1)
            # high quality (scaler>0) → g_angular<0 → theta decreases → harder margin
            # low quality (scaler<0) → g_angular>0 → theta increases → easier margin
            # NOTE: sign convention — we ADD g_angular to theta, and g_angular = -m*scaler
            m_arc = torch.zeros_like(logits[index])
            g_angular = self.m * margin_scaler * (-1)
            m_arc.scatter_(1, labels[index].view(-1, 1), g_angular.unsqueeze(1))

            theta = logits[index].acos()
            theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
            logits[index] = theta_m.cos()

            # g_additive: subtracted from cosine of target class (per paper Eq. 7)
            # g_add = m + m * margin_scaler
            m_cos = torch.zeros_like(logits[index])
            g_add = self.m + self.m * margin_scaler
            m_cos.scatter_(1, labels[index].view(-1, 1), g_add.unsqueeze(1))
            logits[index] = logits[index] - m_cos

        logits = logits * self.s
        return logits

    def _arcface_fallback(self, logits, labels, index):
        target_logit = logits[index, labels[index].view(-1)]
        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            logits[index, labels[index].view(-1)] = target_logit + self.m
            logits.cos_()
        return logits * self.s


# =====================================================================
# 6. MagFace — Magnitude-aware margin + regularization
# =====================================================================

class MagFaceLoss(nn.Module):
    """MagFace: A Universal Representation for FR & Quality (CVPR 2021).

    Larger feature magnitude (higher quality) → larger margin.
    Includes magnitude regularization: g(a_i) = a_i/u_a^2 + 1/a_i

    Total loss = classification_loss + lambda_g * mean(g(a_i))

    Reference: https://arxiv.org/abs/2103.06627
    """
    def __init__(self, s=64.0, l_a=10.0, u_a=110.0,
                 l_m=0.45, u_m=0.8, lambda_g=20.0):
        super().__init__()
        self.s = s
        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m
        self.lambda_g = lambda_g

        # Stores last regularization term (accessed in training loop)
        self._last_mag_reg = torch.tensor(0.0)

    def _calc_margin(self, norms):
        """m(a_i) = l_m + (u_m - l_m) * (a_i - l_a) / (u_a - l_a)"""
        a = torch.clamp(norms, self.l_a, self.u_a)
        margin = self.l_m + (self.u_m - self.l_m) * (a - self.l_a) / (self.u_a - self.l_a)
        return margin

    def _calc_reg(self, norms):
        """g(a_i) = a_i / u_a^2 + 1 / a_i"""
        a = torch.clamp(norms, self.l_a, self.u_a)
        return a / (self.u_a ** 2) + 1.0 / a

    def forward(self, logits, labels, embeddings=None, norms=None):
        index = torch.where(labels != -1)[0]

        if norms is None:
            raise ValueError(
                "MagFace requires feature norms (raw, before L2-norm). "
                "Use PartialFC_V2_Extended.")

        # Per-sample adaptive margin based on feature magnitude
        margins = self._calc_margin(norms[index].view(-1))
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target = target_logit + margins
            logits[index, labels[index].view(-1)] = final_target
            logits.cos_()

        logits = logits * self.s

        # Magnitude regularization (has gradients through norms)
        reg = self._calc_reg(norms[index].view(-1))
        self._last_mag_reg = self.lambda_g * reg.mean()

        return logits


# =====================================================================
# Loss factory
# =====================================================================

def get_phase2_loss(loss_name):
    """Create loss function by name. Returns (loss_fn, needs_norms)."""
    losses = {
        "arcface":        (lambda: ArcFaceLoss(s=64.0, m=0.5), False),
        "cosface":        (lambda: CosFaceLoss(s=64.0, m=0.4), False),
        "curricularface": (lambda: CurricularFaceLoss(s=64.0, m=0.5), False),
        "elasticface":    (lambda: ElasticFaceLoss(s=64.0, m=0.5, std=0.0125), False),
        "adaface":        (lambda: AdaFaceLoss(s=64.0, m=0.4, h=0.333), True),
        "magface":        (lambda: MagFaceLoss(s=64.0, l_a=10, u_a=110,
                                                l_m=0.45, u_m=0.8, lambda_g=20.0), True),
    }
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(losses.keys())}")
    factory, needs_norms = losses[loss_name]
    return factory(), needs_norms
'''

# Write phase2_losses.py
with open(os.path.join(WORK_DIR, 'phase2_losses.py'), 'w') as f:
    f.write(phase2_losses_code)
print("[OK] phase2_losses.py written")

# Also backup to Drive
import shutil
shutil.copy2(os.path.join(WORK_DIR, 'phase2_losses.py'),
             os.path.join(BACKUP_ROOT, 'configs', 'phase2_losses.py'))

# === Test each loss ===
import sys
sys.path.insert(0, WORK_DIR)
from phase2_losses import get_phase2_loss
import torch

print("\nTesting all 6 losses with dummy data:")
B, C = 8, 100  # batch, num_classes
dummy_logits_base = torch.randn(B, C).clamp(-1, 1)
dummy_labels = torch.zeros(B, 1, dtype=torch.long)
dummy_labels[:, 0] = torch.arange(B) % C
dummy_norms = torch.rand(B, 1) * 50 + 10  # norms in [10, 60]

for loss_name in LOSS_LIST:
    loss_fn, needs_norms = get_phase2_loss(loss_name)
    loss_fn.train()
    logits = dummy_logits_base.clone()
    labels = dummy_labels.clone()
    norms = dummy_norms.clone() if needs_norms else None

    try:
        out = loss_fn(logits, labels, norms=norms)
        extra = ""
        if hasattr(loss_fn, '_last_mag_reg'):
            extra = f", mag_reg={loss_fn._last_mag_reg.item():.4f}"
        if hasattr(loss_fn, 't') and isinstance(loss_fn.t, torch.Tensor):
            extra += f", t={loss_fn.t.item():.4f}"
        if hasattr(loss_fn, 'batch_mean'):
            extra += f", norm_ema={loss_fn.batch_mean.item():.2f}"
        print(f"  [OK] {loss_name:<18} output_shape={list(out.shape)}"
              f"  range=[{out.min():.1f}, {out.max():.1f}]{extra}")
    except Exception as e:
        print(f"  [FAIL] {loss_name}: {e}")

print("\n[OK] All 6 loss functions verified.")

# %% [markdown]
# ---
# ## Cell 7: Create `phase2_train.py` Training Script
#
# Single-GPU training via `torchrun --standalone --nproc_per_node=1`.
# Uses PartialFC_V2_Extended for AdaFace/MagFace (passes raw norms).
# Loads pretrained backbone, creates fresh classifier head per loss.

# %%
import os

phase2_train_code = r'''#!/usr/bin/env python3
"""
phase2_train.py — Phase 2 training: single loss with pretrained R18.

Usage:
    torchrun --standalone --nproc_per_node=1 phase2_train.py \
        --loss_name arcface \
        --pretrained /path/to/backbone.pth \
        --rec /path/to/dataset \
        --output /path/to/output \
        --num_classes 10572 --num_image 490623 \
        --epochs 5 --batch_size 64 --lr 0.01 \
        --sample_rate 1.0 --fp16
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.nn.functional import normalize, linear

from backbones import get_model
from dataset import get_dataloader
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging

# Import Phase 2 losses
from phase2_losses import get_phase2_loss


class PartialFC_V2_Extended(PartialFC_V2):
    """Extended PartialFC that passes raw norms to loss (for AdaFace/MagFace)."""

    def forward(self, local_embeddings, local_labels):
        local_labels.squeeze_()
        local_labels = local_labels.long()
        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size

        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size)).cuda()
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size).long().cuda()
            for _ in range(self.world_size)
        ]

        if self.world_size > 1:
            from partial_fc_v2 import AllGather
            _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
            distributed.all_gather(_gather_labels, local_labels)
            embeddings = torch.cat(_list_embeddings)
        else:
            embeddings = local_embeddings
            _gather_labels[0] = local_labels

        labels = torch.cat(_gather_labels)
        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local)
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive)
        else:
            weight = self.weight

        # Compute norms BEFORE normalizing (for AdaFace/MagFace)
        norms = torch.norm(embeddings, dim=1, keepdim=True)

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels,
                                      embeddings=norm_embeddings, norms=norms)
        loss = self.dist_cross_entropy(logits, labels)
        return loss, norms


def main(args):
    # Init distributed
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        distributed.init_process_group("nccl")
    except KeyError:
        rank = 0
        local_rank = 0
        world_size = 1
        distributed.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:12584",
            rank=0, world_size=1)

    setup_seed(seed=2048, cuda_deterministic=False)
    torch.cuda.set_device(local_rank)
    os.makedirs(args.output, exist_ok=True)
    init_logging(rank, args.output)

    # Data
    train_loader = get_dataloader(
        args.rec, local_rank, args.batch_size,
        False, False, 2048, args.num_workers)

    # Backbone
    backbone = get_model("r18", dropout=0.0, fp16=args.fp16,
                          num_features=512).cuda()

    # Load pretrained weights (backbone only)
    if args.pretrained and os.path.exists(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location='cpu')
        if 'state_dict_backbone' in state_dict:
            state_dict = state_dict['state_dict_backbone']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        result = backbone.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded pretrained: missing={len(result.missing_keys)}, "
                     f"unexpected={len(result.unexpected_keys)}")
    else:
        logging.warning(f"No pretrained checkpoint: {args.pretrained}")

    backbone = torch.nn.parallel.DistributedDataParallel(
        backbone, broadcast_buffers=False,
        device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.train()
    backbone._set_static_graph()

    # Loss + Head
    margin_loss, needs_norms = get_phase2_loss(args.loss_name)
    PFC_Class = PartialFC_V2_Extended if needs_norms else PartialFC_V2

    module_partial_fc = PFC_Class(
        margin_loss, 512, args.num_classes,
        args.sample_rate, False)
    module_partial_fc.train().cuda()

    # Optimizer
    opt = torch.optim.SGD(
        [{"params": backbone.parameters()},
         {"params": module_partial_fc.parameters()}],
        lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # LR scheduler
    total_batch = args.batch_size * world_size
    warmup_step = args.num_image // total_batch * args.warmup_epoch
    total_step = args.num_image // total_batch * args.epochs

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt, warmup_iters=warmup_step, total_iters=total_step)

    amp_scaler = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        logging.info(f"Resuming from {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location='cpu')
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        backbone.module.load_state_dict(ckpt["state_dict_backbone"])
        module_partial_fc.load_state_dict(ckpt["state_dict_softmax_fc"])
        opt.load_state_dict(ckpt["state_optimizer"])
        lr_scheduler.load_state_dict(ckpt["state_lr_scheduler"])
        if "state_scaler" in ckpt:
            amp_scaler.load_state_dict(ckpt["state_scaler"])
        del ckpt
        logging.info(f"Resumed: epoch={start_epoch}, step={global_step}")

    # Log config
    config_dict = vars(args)
    logging.info(f"=== Phase 2 Training: {args.loss_name} ===")
    for k, v in config_dict.items():
        logging.info(f"  {k:<25} {v}")

    # Callbacks
    val_targets = ['lfw', 'cfp_fp', 'agedb_30']
    callback_verification = CallBackVerification(
        val_targets=val_targets, rec_prefix=args.rec,
        summary_writer=None, wandb_logger=None)
    callback_logging = CallBackLogging(
        frequent=50, total_step=total_step,
        batch_size=args.batch_size, start_step=global_step, writer=None)

    loss_am = AverageMeter()

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)

        epoch_start = time.time()
        norm_sum, norm_count = 0.0, 0

        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)

            if needs_norms:
                loss, norms = module_partial_fc(local_embeddings, local_labels)
                # MagFace regularization
                if args.loss_name == "magface" and hasattr(margin_loss, '_last_mag_reg'):
                    loss = loss + margin_loss._last_mag_reg
                # Track norm stats
                with torch.no_grad():
                    norm_sum += norms.mean().item()
                    norm_count += 1
            else:
                loss = module_partial_fc(local_embeddings, local_labels)

            if args.fp16:
                amp_scaler.scale(loss).backward()
                if global_step % 1 == 0:  # gradient_acc = 1
                    amp_scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp_scaler.step(opt)
                    amp_scaler.update()
                    opt.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step()
                opt.zero_grad()

            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, args.fp16,
                                 lr_scheduler.get_last_lr()[0], amp_scaler)
                if global_step % 2000 == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        epoch_time = time.time() - epoch_start
        norm_info = ""
        if norm_count > 0:
            norm_info = f", mean_norm={norm_sum/norm_count:.2f}"
        logging.info(f"[{args.loss_name}] Epoch {epoch+1}/{args.epochs} "
                     f"done in {epoch_time:.0f}s, loss={loss_am.avg:.4f}{norm_info}")

        # Save checkpoint every epoch
        if rank == 0:
            ckpt = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
                "state_scaler": amp_scaler.state_dict(),
                "loss_name": args.loss_name,
                "config": config_dict,
            }
            ckpt_path = os.path.join(args.output, "checkpoint_gpu_0.pt")
            torch.save(ckpt, ckpt_path)

            model_path = os.path.join(args.output, "model.pt")
            torch.save(backbone.module.state_dict(), model_path)

            # Backup to Drive if path provided
            if args.backup_dir:
                import shutil
                os.makedirs(args.backup_dir, exist_ok=True)
                shutil.copy2(ckpt_path,
                             os.path.join(args.backup_dir, "checkpoint_gpu_0.pt"))
                shutil.copy2(model_path,
                             os.path.join(args.backup_dir, "model.pt"))

    # Final save
    if rank == 0:
        final_path = os.path.join(args.output, "model.pt")
        torch.save(backbone.module.state_dict(), final_path)
        logging.info(f"[{args.loss_name}] Training complete. Saved: {final_path}")

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    p = argparse.ArgumentParser()
    p.add_argument("--loss_name", type=str, required=True)
    p.add_argument("--pretrained", type=str, default="")
    p.add_argument("--rec", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--backup_dir", type=str, default="")
    p.add_argument("--resume_ckpt", type=str, default="")
    p.add_argument("--num_classes", type=int, default=10572)
    p.add_argument("--num_image", type=int, default=490623)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--sample_rate", type=float, default=1.0)
    p.add_argument("--warmup_epoch", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--fp16", action="store_true")
    main(p.parse_args())
'''

with open(os.path.join(WORK_DIR, 'phase2_train.py'), 'w') as f:
    f.write(phase2_train_code)
print("[OK] phase2_train.py written")

import shutil
shutil.copy2(os.path.join(WORK_DIR, 'phase2_train.py'),
             os.path.join(BACKUP_ROOT, 'configs', 'phase2_train.py'))

# %% [markdown]
# ---
# ## Cell 8: Train Each Loss with Resume/Skip + Backup
#
# For each loss:
# 1. Check if model.pt already exists on Drive → skip
# 2. Check if checkpoint exists → resume
# 3. Otherwise → train from pretrained R18
# 4. Backup model + checkpoint + log to Drive after training

# %%
import os
import shutil
import time

os.chdir(WORK_DIR)

print("=" * 70)
print("  Phase 2 Training: 6 Losses × Fixed R18 Backbone")
print(f"  Epochs per loss: {PHASE2_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}, Sample rate: {SAMPLE_RATE}")
print("=" * 70)

training_summary = {}

for loss_name in LOSS_LIST:
    print(f"\n{'='*60}")
    print(f"  Loss: {loss_name.upper()}")
    print(f"{'='*60}")

    output_dir = f"{OUTPUT_ROOT}/{loss_name}"
    backup_model_dir = f"{BACKUP_ROOT}/models/{loss_name}"
    backup_ckpt_dir = f"{BACKUP_ROOT}/checkpoints/{loss_name}"
    backup_log_dir = f"{BACKUP_ROOT}/train_logs"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(backup_model_dir, exist_ok=True)
    os.makedirs(backup_ckpt_dir, exist_ok=True)
    os.makedirs(backup_log_dir, exist_ok=True)

    # Check if already done
    drive_model = f"{backup_model_dir}/model.pt"
    if os.path.exists(drive_model) and not FORCE_RETRAIN:
        print(f"  [SKIP] {loss_name} — model.pt already exists on Drive")
        # Copy to local for evaluation
        local_model = f"{output_dir}/model.pt"
        if not os.path.exists(local_model):
            shutil.copy2(drive_model, local_model)
        training_summary[loss_name] = "skipped (already done)"
        continue

    # Check for resume checkpoint
    resume_ckpt = ""
    drive_ckpt = f"{backup_ckpt_dir}/checkpoint_gpu_0.pt"
    if os.path.exists(drive_ckpt) and not FORCE_RETRAIN:
        resume_ckpt = drive_ckpt
        print(f"  [RESUME] Found checkpoint on Drive")

    # Build training command
    fp16_flag = "--fp16" if FP16 else ""
    cmd = (
        f"torchrun --standalone --nproc_per_node=1 phase2_train.py "
        f"--loss_name {loss_name} "
        f"--pretrained {PRETRAINED_CKPT} "
        f"--rec {DATASET_DIR} "
        f"--output {output_dir} "
        f"--backup_dir {backup_model_dir} "
        f"--num_classes {NUM_CLASSES} "
        f"--num_image {NUM_IMAGE} "
        f"--epochs {PHASE2_EPOCHS} "
        f"--batch_size {BATCH_SIZE} "
        f"--lr {LR} "
        f"--sample_rate {SAMPLE_RATE} "
        f"--warmup_epoch {WARMUP_EPOCH} "
        f"--num_workers {NUM_WORKERS} "
        f"{fp16_flag}"
    )

    if resume_ckpt:
        cmd += f" --resume_ckpt {resume_ckpt}"

    print(f"  CMD: {cmd}")
    t0 = time.time()

    # Run training
    ret = os.system(cmd)

    elapsed = time.time() - t0
    print(f"\n  [{loss_name}] Finished in {elapsed/60:.1f} min, exit={ret}")

    # Backup results
    local_model = f"{output_dir}/model.pt"
    local_ckpt = f"{output_dir}/checkpoint_gpu_0.pt"

    if os.path.exists(local_model):
        shutil.copy2(local_model, f"{backup_model_dir}/model.pt")
        print(f"  [BACKUP] model.pt → Drive")
        training_summary[loss_name] = f"done ({elapsed/60:.1f} min)"
    else:
        print(f"  [ERROR] model.pt not found!")
        training_summary[loss_name] = f"FAILED (exit={ret})"

    if os.path.exists(local_ckpt):
        shutil.copy2(local_ckpt, f"{backup_ckpt_dir}/checkpoint_gpu_0.pt")

    # Backup training log
    log_file = f"{output_dir}/training.log"
    if os.path.exists(log_file):
        shutil.copy2(log_file, f"{backup_log_dir}/{loss_name}.log")

    torch.cuda.empty_cache()

# Print summary
print("\n" + "=" * 60)
print("  TRAINING SUMMARY")
print("=" * 60)
for loss_name, status in training_summary.items():
    print(f"  {loss_name:<18} {status}")
print("=" * 60)

# %% [markdown]
# ---
# ## Cell 9: Clean Evaluation
#
# Evaluate each trained model on LFW, CFP-FP, AgeDB-30 (clean).

# %%
import os
import json
import pickle
import numpy as np
import torch
import sklearn.preprocessing
from backbones import get_model

os.chdir(WORK_DIR)

try:
    import mxnet as mx
except ImportError:
    mx = None
    print("WARNING: mxnet not available, skipping .bin evaluation")

from eval.verification import evaluate

@torch.no_grad()
def eval_model_clean(model_path, rec_prefix, targets, network="r18",
                     batch_size=64, embedding_size=512):
    """Evaluate a model on clean verification benchmarks."""
    device = 'cuda'
    backbone = get_model(network, dropout=0, fp16=False,
                          num_features=embedding_size)
    backbone.load_state_dict(torch.load(model_path, map_location='cpu'))
    backbone = backbone.to(device).eval()

    results = {}
    for target in targets:
        bin_path = os.path.join(rec_prefix, f"{target}.bin")
        if not os.path.exists(bin_path):
            print(f"  [SKIP] {target}.bin not found")
            continue

        try:
            with open(bin_path, 'rb') as f:
                bins, issame = pickle.load(f)
        except UnicodeDecodeError:
            with open(bin_path, 'rb') as f:
                bins, issame = pickle.load(f, encoding='bytes')

        num = len(issame) * 2
        embeddings_list = []
        for flip in [False, True]:
            embs = np.zeros((num, embedding_size), dtype=np.float32)
            for i in range(0, num, batch_size):
                j = min(i + batch_size, num)
                imgs = []
                for k in range(i, j):
                    img = mx.image.imdecode(bins[k]).asnumpy()
                    if flip:
                        img = img[:, ::-1, :].copy()
                    imgs.append(img)
                batch = np.stack(imgs)
                batch = torch.from_numpy(batch.transpose(0, 3, 1, 2).astype(np.float32))
                batch = ((batch / 255.0) - 0.5) / 0.5
                embs[i:j] = backbone(batch.to(device)).cpu().numpy()
            embeddings_list.append(embs)

        embs = sklearn.preprocessing.normalize(embeddings_list[0] + embeddings_list[1])
        _, _, accuracy, val, val_std, far = evaluate(embs, issame, nrof_folds=10)
        acc = np.mean(accuracy)
        results[target] = round(acc * 100, 2)
        print(f"  {target}: {acc*100:.2f}%")

    del backbone
    torch.cuda.empty_cache()
    return results

# Run clean evaluation
if mx is None:
    print("Skipping clean evaluation (mxnet not available)")
    clean_results = {}
else:
    val_targets = ['lfw', 'cfp_fp', 'agedb_30']
    clean_results = {}

    for loss_name in LOSS_LIST:
        model_path = f"{OUTPUT_ROOT}/{loss_name}/model.pt"
        if not os.path.exists(model_path):
            # Try Drive backup
            model_path = f"{BACKUP_ROOT}/models/{loss_name}/model.pt"
        if not os.path.exists(model_path):
            print(f"\n[SKIP] {loss_name} — no model.pt found")
            continue

        print(f"\n=== {loss_name.upper()} ===")
        clean_results[loss_name] = eval_model_clean(
            model_path, DATASET_DIR, val_targets)

    # Print comparison table
    print("\n" + "=" * 70)
    print("  CLEAN ACCURACY COMPARISON")
    print("=" * 70)
    header = f"  {'Loss':<18}"
    for t in val_targets:
        header += f" {t:>10}"
    print(header)
    print("  " + "-" * 48)
    for loss_name, res in clean_results.items():
        row = f"  {loss_name:<18}"
        for t in val_targets:
            row += f" {res.get(t, 'N/A'):>10}"
        print(row)
    print("=" * 70)

    # Save results
    results_path = f"{BACKUP_ROOT}/eval_logs/clean_results.json"
    with open(results_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nSaved: {results_path}")

# %% [markdown]
# ---
# ## Cell 10: Degraded Evaluation
#
# Evaluate under gaussian_blur, low_resolution, low_illumination at severity 1,3,5.
# Compute robustness drop = clean_acc - degraded_acc.

# %%
import os
import json
import numpy as np
import torch

os.chdir(WORK_DIR)

if mx is None:
    print("Skipping degraded evaluation (mxnet not available)")
    degraded_results = {}
else:
    from degradation.transforms import DegradationTransform

    degradation_types = ["gaussian_blur", "low_resolution", "low_illumination"]
    severities = [1, 3, 5]
    val_targets = ['lfw']  # Use LFW for degraded eval (faster)

    @torch.no_grad()
    def eval_model_degraded(model_path, rec_prefix, target, deg_type, severity,
                            network="r18", batch_size=64, embedding_size=512):
        device = 'cuda'
        backbone = get_model(network, dropout=0, fp16=False,
                              num_features=embedding_size)
        backbone.load_state_dict(torch.load(model_path, map_location='cpu'))
        backbone = backbone.to(device).eval()

        bin_path = os.path.join(rec_prefix, f"{target}.bin")
        with open(bin_path, 'rb') as f:
            bins, issame = pickle.load(f)

        deg = DegradationTransform(deg_type, severity=severity, seed=42)
        num = len(issame) * 2

        embeddings_list = []
        for flip in [False, True]:
            embs = np.zeros((num, embedding_size), dtype=np.float32)
            for i in range(0, num, batch_size):
                j = min(i + batch_size, num)
                imgs = []
                for k in range(i, j):
                    img = mx.image.imdecode(bins[k]).asnumpy()
                    img = deg.apply(img)  # Apply degradation
                    if flip:
                        img = img[:, ::-1, :].copy()
                    imgs.append(img)
                batch = np.stack(imgs)
                batch = torch.from_numpy(batch.transpose(0, 3, 1, 2).astype(np.float32))
                batch = ((batch / 255.0) - 0.5) / 0.5
                embs[i:j] = backbone(batch.to(device)).cpu().numpy()
            embeddings_list.append(embs)

        embs = sklearn.preprocessing.normalize(embeddings_list[0] + embeddings_list[1])
        _, _, accuracy, _, _, _ = evaluate(embs, issame, nrof_folds=10)
        del backbone
        torch.cuda.empty_cache()
        return round(np.mean(accuracy) * 100, 2)

    degraded_results = {}
    for loss_name in LOSS_LIST:
        model_path = f"{OUTPUT_ROOT}/{loss_name}/model.pt"
        if not os.path.exists(model_path):
            model_path = f"{BACKUP_ROOT}/models/{loss_name}/model.pt"
        if not os.path.exists(model_path):
            print(f"[SKIP] {loss_name} — no model")
            continue

        print(f"\n=== {loss_name.upper()} ===")
        degraded_results[loss_name] = {}

        for deg_type in degradation_types:
            for sev in severities:
                key = f"{deg_type}_s{sev}"
                print(f"  Evaluating {key}...", end=" ")
                acc = eval_model_degraded(
                    model_path, DATASET_DIR, 'lfw', deg_type, sev)
                degraded_results[loss_name][key] = acc
                print(f"{acc:.2f}%")

    # Print robustness comparison table
    print("\n" + "=" * 90)
    print("  DEGRADED ACCURACY & ROBUSTNESS DROP (LFW)")
    print("=" * 90)

    conditions = [f"{d}_s{s}" for d in degradation_types for s in severities]
    header = f"  {'Loss':<16} {'Clean':>7}"
    for c in conditions:
        header += f" {c[-8:]:>9}"
    print(header)
    print("  " + "-" * (16 + 7 + 9 * len(conditions)))

    for loss_name in LOSS_LIST:
        if loss_name not in degraded_results:
            continue
        clean_acc = clean_results.get(loss_name, {}).get('lfw', 0)
        row = f"  {loss_name:<16} {clean_acc:>7.2f}"
        for c in conditions:
            deg_acc = degraded_results[loss_name].get(c, 0)
            drop = clean_acc - deg_acc
            row += f" {deg_acc:>5.2f}({drop:+.1f})"
        print(row)
    print("=" * 90)

    # Save
    results_path = f"{BACKUP_ROOT}/degraded_eval_logs/degraded_results.json"
    with open(results_path, 'w') as f:
        json.dump(degraded_results, f, indent=2)
    print(f"\nSaved: {results_path}")

# %% [markdown]
# ---
# ## Cell 11: Benchmark Model Efficiency
#
# Since all losses use the same R18 backbone, benchmark runs once.
# Reports: Params(M), Size(MB), GPU latency batch=1 and batch=16.

# %%
import os
import json
import time
import torch
import numpy as np
from backbones import get_model

os.chdir(WORK_DIR)

print("=" * 60)
print("  Model Efficiency Benchmark (ResNet18 / iResNet18)")
print("  Note: Same backbone for all losses — benchmark once")
print("=" * 60)

model = get_model("r18", dropout=0, fp16=False, num_features=512).cuda().eval()

# Params
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  Parameters: {total_params:.2f}M")

# Model size
import tempfile
tmp = os.path.join(WORK_DIR, '_tmp_bench.pt')
torch.save(model.state_dict(), tmp)
size_mb = os.path.getsize(tmp) / 1024**2
os.remove(tmp)
print(f"  Model size: {size_mb:.1f} MB (FP32)")

# GPU latency
for bs in [1, 16]:
    dummy = torch.randn(bs, 3, 112, 112).cuda()
    # Warmup
    for _ in range(20):
        _ = model(dummy)
    torch.cuda.synchronize()
    times = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(dummy)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  GPU latency (batch={bs}): {np.mean(times):.2f} ± {np.std(times):.2f} ms")

# FLOPs
try:
    from ptflops import get_model_complexity_info
    model_cpu = get_model("r18", dropout=0, fp16=False, num_features=512).eval()
    macs, _ = get_model_complexity_info(
        model_cpu, (3, 112, 112), as_strings=False, print_per_layer_stat=False)
    print(f"  FLOPs: {macs/1e9:.3f} GFLOPs")
    del model_cpu
except ImportError:
    print("  FLOPs: N/A (install ptflops)")

# Save benchmark
benchmark = {
    "backbone": "r18 (iResNet18)",
    "params_m": round(total_params, 2),
    "size_mb": round(size_mb, 1),
    "note": "Same backbone for all 6 losses — inference identical"
}
bench_path = f"{BACKUP_ROOT}/benchmark_logs/benchmark_results.json"
with open(bench_path, 'w') as f:
    json.dump(benchmark, f, indent=2)
print(f"\nSaved: {bench_path}")

del model
torch.cuda.empty_cache()

# %% [markdown]
# ---
# ## Cell 12: Final Backup

# %%
import os
import shutil
import time

final_dir = f"{BACKUP_ROOT}/final_backup"
os.makedirs(final_dir, exist_ok=True)

print("Creating final backup...")

# Copy all results
for subdir in ["models", "train_logs", "eval_logs", "degraded_eval_logs",
               "benchmark_logs", "configs"]:
    src = f"{BACKUP_ROOT}/{subdir}"
    dst = f"{final_dir}/{subdir}"
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  [OK] {subdir}/")

# Copy notebook source
nb_src = os.path.join(WORK_DIR, 'colab_phase2_loss_comparison.py')
if os.path.exists(nb_src):
    shutil.copy2(nb_src, f"{final_dir}/colab_phase2_loss_comparison.py")

print(f"\n[OK] Final backup complete: {final_dir}")
print(f"  Total size: ", end="")
total = 0
for root, dirs, files in os.walk(final_dir):
    for f in files:
        total += os.path.getsize(os.path.join(root, f))
print(f"{total/1024**2:.1f} MB")

# %% [markdown]
# ---
# ## Cell 13: How to Run / Troubleshooting
#
# ### Run Order
# 1. Cell 0: Install deps (run once per session)
# 2. Cell 1: Mount Drive + config (edit paths first)
# 3. Cell 2: Check GPU
# 4. Cell 3: Clone InsightFace
# 5. Cell 4: Dataset check
# 6. Cell 5: Load pretrained R18
# 7. Cell 6: Create loss functions
# 8. Cell 7: Create training script
# 9. Cell 8: Train all 6 losses (main cell, ~30-60 min total)
# 10. Cell 9: Clean evaluation
# 11. Cell 10: Degraded evaluation
# 12. Cell 11: Benchmark
# 13. Cell 12: Final backup
#
# ### OOM Handling
# If you get CUDA OOM during training:
# 1. First try: reduce `SAMPLE_RATE = 0.5` (keep batch=64)
# 2. Then try: reduce `BATCH_SIZE = 32`
# 3. IMPORTANT: if you change batch/sample_rate, use same for ALL losses
#
# ### Resume After Colab Disconnect
# 1. Set `FORCE_RETRAIN = False` (default)
# 2. Re-run all cells from Cell 0
# 3. Cell 8 will auto-skip completed losses and resume from checkpoints
# 4. Losses with model.pt on Drive → skipped
# 5. Losses with checkpoint_gpu_0.pt on Drive → resumed
# 6. Losses with nothing → trained from pretrained R18
#
# ### Pretrained Checkpoint
# Download `ms1mv3_arcface_r18_fp16/backbone.pth` from:
# https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d
#
# Upload to: `{DRIVE_ROOT}/phase2_resnet18_loss_comparison/pretrained/backbone.pth`
#
# ### Dataset
# Default: CASIA-WebFace (faces_webface_112x112, 10572 classes)
# If using MS1MV3: update NUM_CLASSES=93431, NUM_IMAGE=5179510
#
# ### Key Design Decisions
# - **Transfer fine-tuning**: R18 pretrained on MS1MV3, fine-tuned on CASIA
# - **Fair comparison**: All losses start from identical pretrained backbone
# - **Classifier head**: Fresh PartialFC for each loss (random init)
# - **AdaFace/MagFace**: Use raw feature norm before L2-norm via PartialFC_V2_Extended
# - **MagFace regularization**: g(a_i) = a_i/u_a² + 1/a_i added to loss
# - **Default report mode**: 6 losses × 5 epochs × batch 64, backup every epoch

# %%
print("=" * 60)
print("  Phase 2 Notebook Ready")
print("=" * 60)
print()
print("Default config (report mode):")
print(f"  6 losses: {', '.join(LOSS_LIST)}")
print(f"  {PHASE2_EPOCHS} epochs per loss")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Sample rate: {SAMPLE_RATE}")
print(f"  FP16: {FP16}")
print(f"  Backup: every epoch to Google Drive")
print()
print("Outputs on Google Drive:")
print(f"  {BACKUP_ROOT}/")
print("    models/{loss}/model.pt")
print("    checkpoints/{loss}/checkpoint_gpu_0.pt")
print("    train_logs/{loss}.log")
print("    eval_logs/clean_results.json")
print("    degraded_eval_logs/degraded_results.json")
print("    benchmark_logs/benchmark_results.json")
print("    final_backup/")
