# %% [markdown]
# # Proposed 4.3 Core 20-Epoch 5-Eval + Synthetic Degraded Eval Kaggle Runner
#
# FIX INCLUDED:
# - patch write_manifest() mkdir lỗi run_manifest.json
# - patch perceptibility_attention.py để tránh lỗi fp16/fp32:
#   RuntimeError: Input type (c10::Half) and bias type (float) should be the same
# - patch attention_maps() đúng với bản mới return dict:
#   {"channel": A_c, "spatial": A_s, "combined": M}
# - Core runner dùng train_proposed_4_3_core_kaggle.py
# - UI centers severity = 5
# - degraded eval severity = 1,3,5

# %%
from pathlib import Path
import os
import subprocess
import sys
import json
import shutil
import zipfile

REPO_URL = "https://github.com/Tai12345-Ai/Lightweight_Face_Recognition.git"
BRANCH = "main"

CODE_ROOT = Path("/kaggle/working/Lightweight_Face_Recognition")
ARCFACE_DIR = CODE_ROOT / "insightface" / "recognition" / "arcface_torch"
INPUT_ROOT = Path("/kaggle/input")

# -----------------------------
# 1. Clone / update repo
# -----------------------------
if not CODE_ROOT.exists():
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(CODE_ROOT)], check=True)
else:
    subprocess.run(["git", "pull", "--ff-only"], cwd=CODE_ROOT, check=True)

os.chdir(ARCFACE_DIR)
sys.path.insert(0, str(ARCFACE_DIR))

# -----------------------------
# 2. Install dependencies
# -----------------------------
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "tensorboard",
    "easydict",
    "onnx",
    "opencv-python",
    "scikit-learn",
    "pandas",
    "matplotlib",
], check=True)

# -----------------------------
# 3. PATCH lỗi run_manifest.json
# PHẢI patch TRƯỚC khi import kaggle_5eval_degraded_common
# -----------------------------
common_file = ARCFACE_DIR / "experiments" / "kaggle" / "kaggle_5eval_degraded_common.py"
text = common_file.read_text(encoding="utf-8")

already_patched = "current_exp_dir.mkdir(parents=True, exist_ok=True)" in text

old_block = '''    out = Path(current_exp_dir) / "run_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
'''

new_block = '''    current_exp_dir = Path(current_exp_dir)
    current_exp_dir.mkdir(parents=True, exist_ok=True)

    out = current_exp_dir / "run_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
'''

if already_patched:
    print("Patch already exists: write_manifest() has mkdir.")
elif old_block in text:
    common_file.write_text(text.replace(old_block, new_block), encoding="utf-8")
    print("Patched write_manifest(): mkdir added.")
else:
    print("[WARN] Could not patch by exact block. Showing context around run_manifest.json:")
    idx = text.find("run_manifest.json")
    print(text[max(0, idx - 500): idx + 500])
    raise RuntimeError("Patch failed. Need manual check of write_manifest() in kaggle_5eval_degraded_common.py")

# Clear possible module cache, just in case.
for k in list(sys.modules.keys()):
    if "kaggle_5eval_degraded_common" in k:
        sys.modules.pop(k)

# -----------------------------
# 3.5 PATCH lỗi FP16/FP32 trong perceptibility_attention.py
# Compatible với bản attention_maps() mới return dict:
# {
#   "channel": A_c,
#   "spatial": A_s,
#   "combined": M,
# }
# -----------------------------
attention_file = ARCFACE_DIR / "lightweight_fr" / "proposed_4_3" / "perceptibility_attention.py"
att_text = attention_file.read_text(encoding="utf-8")

if "orig_dtype = feature_map.dtype" in att_text and "x = feature_map.float()" in att_text:
    print("Patch already exists: perceptibility_attention.py is dtype-safe.")
else:
    start = att_text.find("    def attention_maps(")
    end = att_text.find("    def apply_attention(", start)

    if start == -1 or end == -1:
        print("[WARN] Could not locate attention_maps/apply_attention block. Showing context:")
        idx = att_text.find("attention_maps")
        print(att_text[max(0, idx - 500): idx + 1200])
        raise RuntimeError("Patch failed. Need manual check of perceptibility_attention.py")

    replacement = '''    def attention_maps(self, feature_map: torch.Tensor):
        if feature_map.ndim != 4:
            raise ValueError(
                f"PerceptibilityAttentionModule expects 4D input [B,C,H,W], got {feature_map.shape}"
            )

        orig_dtype = feature_map.dtype

        # Attention layers are float32 by default. During fp16 training/eval,
        # feature_map can be float16, causing:
        # RuntimeError: Input type (c10::Half) and bias type (float) should be the same.
        # Compute attention maps in fp32, then cast maps back to original dtype.
        x = feature_map.float() if feature_map.dtype in (torch.float16, torch.bfloat16) else feature_map

        # --- Channel attention ---
        avg_c = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]
        max_c = F.adaptive_max_pool2d(x, 1)  # [B, C, 1, 1]
        A_c = torch.sigmoid(self.channel_mlp(avg_c) + self.channel_mlp(max_c))  # [B, C, 1, 1]
        F_c = A_c * x  # [B, C, H, W]

        # --- Spatial attention ---
        avg_s = F_c.mean(dim=1, keepdim=True)          # [B, 1, H, W]
        max_s = F_c.max(dim=1, keepdim=True).values    # [B, 1, H, W]
        A_s = torch.sigmoid(
            self.spatial_conv(torch.cat([avg_s, max_s], dim=1))
        )  # [B, 1, H, W]

        M = A_c * A_s  # [B, C, H, W]

        return {
            "channel": A_c.to(dtype=orig_dtype),
            "spatial": A_s.to(dtype=orig_dtype),
            "combined": M.to(dtype=orig_dtype),
        }

'''

    new_att_text = att_text[:start] + replacement + att_text[end:]
    attention_file.write_text(new_att_text, encoding="utf-8")
    print("Patched perceptibility_attention.py: dtype-safe dict attention_maps().")

# Clear possible module cache.
for k in list(sys.modules.keys()):
    if "perceptibility_attention" in k or "proposed_4_3_attention_model" in k:
        sys.modules.pop(k)

# -----------------------------
# 4. Import runner sau khi patch
# -----------------------------
import kaggle_5eval_degraded_common as common

_original_build_proposed_command = common.build_proposed_command


def _build_core_proposed_command(config, current_exp_dir, train_data_dir, eval_dir, num_classes):
    cmd = _original_build_proposed_command(config, current_exp_dir, train_data_dir, eval_dir, num_classes)
    for i, item in enumerate(cmd):
        if str(item) == "train_soft_gated_lambda_kaggle.py":
            cmd[i] = "train_proposed_4_3_core_kaggle.py"
            break
    else:
        raise RuntimeError("Could not replace train_soft_gated_lambda_kaggle.py with Core trainer wrapper.")
    return cmd


common.build_proposed_command = _build_core_proposed_command

from kaggle_5eval_degraded_common import (
    run_5eval_degraded_runner,
    resolve_train_data_dir,
    resolve_pretrained_backbone,
    DEFAULT_TRAIN_DATA_DIR,
    DEFAULT_PRETRAINED_BACKBONE,
    run,
)

# -----------------------------
# 5. Data paths
# -----------------------------
def find_train_data_dir():
    for rec in sorted(INPUT_ROOT.rglob("train.rec")):
        if (rec.parent / "train.idx").exists():
            print("Detected TRAIN_DATA_DIR:", rec.parent)
            return rec.parent
    raise FileNotFoundError("Could not find train.rec + train.idx under /kaggle/input.")


def find_eval_dir():
    required = ["lfw.bin", "cfp_fp.bin", "cplfw.bin", "agedb_30.bin", "calfw.bin"]
    for lfw in sorted(INPUT_ROOT.rglob("lfw.bin")):
        if all((lfw.parent / name).exists() for name in required):
            print("Detected EVAL_DIR:", lfw.parent)
            return lfw.parent
    raise FileNotFoundError("Could not find 5-eval .bin directory under /kaggle/input.")


def find_pretrained_backbone():
    hits = sorted(INPUT_ROOT.rglob("backbone.pth"))
    if not hits:
        raise FileNotFoundError("Could not find backbone.pth under /kaggle/input.")
    chosen = hits[0]
    print("Detected PRETRAINED_BACKBONE:", chosen)
    return chosen


TRAIN_DATA_DIR = find_train_data_dir()
EVAL_DIR = find_eval_dir()
PRETRAINED_BACKBONE = find_pretrained_backbone()

assert (TRAIN_DATA_DIR / "train.rec").exists()
assert (TRAIN_DATA_DIR / "train.idx").exists()
assert (EVAL_DIR / "lfw.bin").exists()
assert PRETRAINED_BACKBONE.exists()

# -----------------------------
# 6. Core configuration
# -----------------------------
RUNNER_FILE = "kaggle_proposed_4_3_core_5eval_degraded_runner.py"
RUNNER_KIND = "proposed4_3"
OUTPUT_SUBDIR = "proposed_4_3_core"
BACKUP_ZIP_NAME = "proposed_4_3_core_20ep_5eval_degraded_s135.zip"

# Core degraded eval must also load the attention wrapper so inference uses x'.
DEGRADED_EVAL_SCRIPT = "eval_degraded_proposed_4_3_full.py"

LOSS_NAME = "proposed_4_3_multi_ui_attention"
BACKBONE = "r18"

EPOCHS = 20
BATCH_SIZE = 128
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
WARMUP_EPOCHS = 1.0
EVAL_EVERY = 1
SAVE_EVERY_EPOCHS = 1
SAVE_EVERY_STEPS = 300
NUM_WORKERS = 2
USE_FP16 = True
MAX_TRAIN_MINUTES = 660

# Margin/loss base params.
S = 64.0
M = 0.4
H = 0.333

# Core setting: disable explicit UI extra loss.
# Giữ các UI params khác vì loss class hiện tại trong repo vẫn cần.
UI_LAMBDA = 0.0
UI_RHO = 0.20
UI_TAU_RI = 1.0
UI_TAU_EASY = 2.0
UI_D_MARGIN = 0.25
UI_ALPHA = 10.0
UI_BETA = 5.0
UI_HARD_BOOST = 0.10
UI_DANGEROUS_DOWNWEIGHT = 0.35
UI_SAMPLE_WEIGHT_MIN = 0.50

# True attention setting for Core wrapper.
ENABLE_ATTENTION = True
ATTENTION_GAMMA = 0.03
ATTENTION_REDUCTION = 16
ATTENTION_ALPHA = 0.25
CENTERED_ATTENTION = False
RI_LAMBDA = 0.05
ATTENTION_SPATIAL_LAMBDA = 1e-4
ATTENTION_CHANNEL_LAMBDA = 0.0
ATTENTION_TV_LAMBDA = 1e-4

EVAL_TARGETS = [
    "lfw",
    "cfp_fp",
    "cplfw",
    "agedb_30",
    "calfw",
]

VAL_TARGETS = EVAL_TARGETS
HQ_EVAL_TARGETS = EVAL_TARGETS

RUN_DEGRADED_EVAL = True
DEGRADED_TARGETS = EVAL_TARGETS
DEGRADED_DEGRADATIONS = [
    "gaussian_blur",
    "motion_blur",
    "low_resolution",
    "jpeg_compression",
    "low_illumination",
    "alignment_perturb",
]
DEGRADED_SEVERITIES = "1,3,5"
DEGRADED_BATCH_SIZE = 128

# Offline multi-UI centers for existing Proposed 4.3 trainer.
UI_CENTER_NUM_SAMPLES = 50000
UI_CENTER_SEVERITIES = "5"
UI_CENTER_OUTPUT = Path("/kaggle/working/ui_centers/proposed_4_3_core_multi_ui_centers_s5.pth")


# -----------------------------
# 7. Build / find multi-UI centers
# -----------------------------
def find_existing_ui_centers():
    """Prefer a user-provided UI centers file from /kaggle/input if available."""
    candidates = []

    for root in [INPUT_ROOT, Path("/kaggle/working")]:
        if not root.exists():
            continue

        for p in root.rglob("*.pth"):
            name = p.name.lower()
            full = str(p).lower()

            if ("ui" in name or "center" in name or "centers" in name) and "multi" in full:
                candidates.append(p)

    if candidates:
        candidates = sorted(
            candidates,
            key=lambda x: (0 if "s5" in str(x).lower() else 1, len(str(x))),
        )
        chosen = candidates[0]
        if "s5" not in str(chosen).lower():
            print(
                "[WARN] Existing UI centers are not marked s5. "
                "UI centers should preferably be built from severity 5 only.",
                chosen,
            )
        print("Found existing multi-UI centers:", chosen)
        return chosen

    return None


def ensure_multi_ui_centers():
    existing = find_existing_ui_centers()
    if existing is not None:
        return existing

    UI_CENTER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if UI_CENTER_OUTPUT.exists():
        print("Using existing UI centers:", UI_CENTER_OUTPUT)
        return UI_CENTER_OUTPUT

    train_dir = TRAIN_DATA_DIR
    pretrained_backbone = PRETRAINED_BACKBONE

    print("Building multi-UI centers because none were found in /kaggle/input.")
    print("TRAIN_DATA_DIR:", train_dir)
    print("PRETRAINED_BACKBONE:", pretrained_backbone)
    print("UI_CENTER_OUTPUT:", UI_CENTER_OUTPUT)

    cmd = [
        sys.executable,
        "build_multi_ui_centers.py",
        "--data-dir", str(train_dir),
        "--pretrained-backbone", str(pretrained_backbone),
        "--backbone", BACKBONE,
        "--output", str(UI_CENTER_OUTPUT),
        "--num-samples", str(UI_CENTER_NUM_SAMPLES),
        "--batch-size", str(DEGRADED_BATCH_SIZE),
        "--num-workers", str(NUM_WORKERS),
        "--degradations", ",".join(DEGRADED_DEGRADATIONS),
        "--severities", UI_CENTER_SEVERITIES,
        "--attention-alpha", str(ATTENTION_ALPHA),
        "--include-global",
        "--overwrite",
    ]

    if CENTERED_ATTENTION:
        cmd.append("--centered-attention")

    if USE_FP16:
        cmd.append("--fp16")

    run(cmd, cwd=ARCFACE_DIR)
    return UI_CENTER_OUTPUT


MULTI_UI_CENTERS = str(ensure_multi_ui_centers())
print("MULTI_UI_CENTERS:", MULTI_UI_CENTERS)

# -----------------------------
# 8. Run train + clean eval + degraded eval + backup
# CHỈ GỌI 1 LẦN
# -----------------------------
run_5eval_degraded_runner(globals())

# -----------------------------
# 9. Optional post-run report/plots
# -----------------------------
try:
    from kaggle_proposed_4_3_core_report import make_report

    make_report(
        backup_zip_name=BACKUP_ZIP_NAME,
        output_subdir=OUTPUT_SUBDIR,
        eval_targets=EVAL_TARGETS,
        degraded_targets=DEGRADED_TARGETS,
        degraded_degradations=DEGRADED_DEGRADATIONS,
    )

except Exception as exc:
    print("[WARN] Could not generate plots/report:", repr(exc))
