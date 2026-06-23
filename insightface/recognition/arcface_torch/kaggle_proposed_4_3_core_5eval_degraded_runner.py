# %% [markdown]
# # Proposed 4.3 Core 20-Epoch 5-Eval + Synthetic Degraded Eval Kaggle Runner
#
# This runner is designed to match the existing Kaggle runners in the repo.
# It uses the repo's Proposed 4.3 training engine, but configures it as a practical Core version:
#   - multi-UI centers are still built/loaded because the existing trainer requires them;
#   - UI extra loss is disabled with UI_LAMBDA = 0.0;
#   - perceptibility attention is enabled;
#   - the 5 clean evals and synthetic degraded eval are run after training.
#
# NOTE: This is a runnable Core-v0 for the current repo. A strict mathematical Core with
# explicit RI-predictor, preserve loss, identity-anchor, and Delta C/N/U diagnostics needs
# a deeper trainer patch.

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

if not CODE_ROOT.exists():
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(CODE_ROOT)], check=True)
else:
    subprocess.run(["git", "pull", "--ff-only"], cwd=CODE_ROOT, check=True)

os.chdir(ARCFACE_DIR)
sys.path.insert(0, str(ARCFACE_DIR))

# Install before optional UI-center building, because build_multi_ui_centers.py uses cv2/sklearn stack.
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "tensorboard", "easydict", "onnx", "opencv-python", "scikit-learn", "pandas", "matplotlib"
], check=True)

from kaggle_5eval_degraded_common import (
    run_5eval_degraded_runner,
    resolve_train_data_dir,
    resolve_pretrained_backbone,
    DEFAULT_TRAIN_DATA_DIR,
    DEFAULT_PRETRAINED_BACKBONE,
    run,
)

# -----------------------------
# Core-v0 configuration
# -----------------------------
RUNNER_FILE = "kaggle_proposed_4_3_core_5eval_degraded_runner.py"
RUNNER_KIND = "proposed4_3"  # use existing Proposed 4.3 command builder
OUTPUT_SUBDIR = "proposed_4_3_core"
BACKUP_ZIP_NAME = "proposed_4_3_core_20ep_5eval_degraded_s5.zip"

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
MAX_TRAIN_MINUTES = 600

# Margin/loss base params.
S = 64.0
M = 0.4
H = 0.333

# Core setting: disable explicit UI extra loss.
# Keep the other UI params because the current repo's 4.3 loss class requires them.
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

# Attention auxiliary setting from the current repo.
ENABLE_ATTENTION = True
ATTENTION_GAMMA = 0.03
ATTENTION_REDUCTION = 16

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
DEGRADED_SEVERITIES = "5"
DEGRADED_BATCH_SIZE = 128

# Offline multi-UI centers for the existing repo's 4.3 trainer.
UI_CENTER_NUM_SAMPLES = 50000
UI_CENTER_OUTPUT = Path("/kaggle/working/ui_centers/proposed_4_3_core_multi_ui_centers_s5.pth")


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
        candidates = sorted(candidates, key=lambda x: len(str(x)))
        print("Found existing multi-UI centers:", candidates[0])
        return candidates[0]
    return None


def ensure_multi_ui_centers():
    existing = find_existing_ui_centers()
    if existing is not None:
        return existing

    UI_CENTER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if UI_CENTER_OUTPUT.exists():
        return UI_CENTER_OUTPUT

    train_dir = resolve_train_data_dir(DEFAULT_TRAIN_DATA_DIR)
    pretrained_backbone = resolve_pretrained_backbone(DEFAULT_PRETRAINED_BACKBONE)
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
        "--severities", DEGRADED_SEVERITIES,
        "--include-global",
        "--overwrite",
    ]
    if USE_FP16:
        cmd.append("--fp16")
    run(cmd, cwd=ARCFACE_DIR)
    return UI_CENTER_OUTPUT


MULTI_UI_CENTERS = str(ensure_multi_ui_centers())

# Run train + clean eval + degraded eval + backup.
run_5eval_degraded_runner(globals())

# Optional post-run report/plots.
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
