# %% [markdown]
# # Proposed 4.3 (Multi-UI + Attention) 20-Epoch 5-Eval + Degraded Eval Kaggle Runner

# %%
from pathlib import Path
import os
import subprocess
import sys
import shutil

REPO_URL = "https://github.com/Tai12345-Ai/Lightweight_Face_Recognition.git"
BRANCH = "main"
CODE_ROOT = Path("/kaggle/working/Lightweight_Face_Recognition")
ARCFACE_DIR = CODE_ROOT / "insightface" / "recognition" / "arcface_torch"

if CODE_ROOT.exists():
    shutil.rmtree(CODE_ROOT)
os.chdir("/kaggle/working")
subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(CODE_ROOT)], cwd="/kaggle/working", check=True)
conflict_check = subprocess.run(
    [
        "grep",
        "-R",
        "-n",
        "-E",
        "--include=*.py",
        "^(<<<<<<<|=======|>>>>>>>)",
        str(ARCFACE_DIR),
    ],
    check=False,
    capture_output=True,
    text=True,
)

if conflict_check.stdout.strip():
    print(conflict_check.stdout)
    raise RuntimeError("Repo still contains merge conflict markers.")
else:
    print("No conflict markers found.")
os.chdir(ARCFACE_DIR)
sys.path.insert(0, str(ARCFACE_DIR))

from kaggle_5eval_degraded_common import (
    run_5eval_degraded_runner,
    resolve_train_data_dir,
    resolve_pretrained_backbone,
    EXPERIMENTS_ROOT,
    INPUT_ROOT,
    DEFAULT_TRAIN_DATA_DIR,
    DEFAULT_PRETRAINED_BACKBONE,
)

RUNNER_FILE = "kaggle_proposed_4_3_5eval_degraded_runner.py"
RUNNER_KIND = "proposed4_3"
OUTPUT_SUBDIR = "proposed4_3_multi_ui_attention"
BACKUP_ZIP_NAME = "proposed4_3_attention_20ep_5eval_degraded_s5.zip"

LOSS_NAME = "multi_ui_perceptibility_competition_quality_adaptive_soft_gated_ada_curricular"

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

S = 64.0
M = 0.4
H = 0.333

UI_LAMBDA = 0.05
UI_RHO = 0.2
UI_TAU_RI = 1.0
UI_TAU_EASY = 2.0
UI_D_MARGIN = 0.25
UI_ALPHA = 10.0
UI_BETA = 5.0
UI_HARD_BOOST = 0.1
UI_DANGEROUS_DOWNWEIGHT = 0.35
UI_SAMPLE_WEIGHT_MIN = 0.5

ENABLE_ATTENTION = True
ATTENTION_GAMMA = 0.05
ATTENTION_REDUCTION = 16

MULTI_UI_CENTERS = Path(
    "/kaggle/working/experiments/proposed4_3_multi_ui_attention/multi_ui_centers_r18_arcface_s5.pth"
)

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
DEGRADED_TARGETS = [
    "lfw",
    "cfp_fp",
    "cplfw",
    "agedb_30",
    "calfw",
]
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


# --- Build multi-UI centers if missing ---
def _build_centers_if_needed():
    """Build multi-UI centers from CASIA-WebFace + backbone before training."""
    centers_path = Path(str(MULTI_UI_CENTERS))
    if centers_path.exists():
        print(f"Multi-UI centers already exist: {centers_path}")
        return

    print("Building multi-UI centers...")
    try:
        train_data_dir = resolve_train_data_dir(DEFAULT_TRAIN_DATA_DIR)
    except FileNotFoundError:
        print("WARNING: Cannot find train data to build multi-UI centers. Skipping.")
        return

    try:
        pretrained_backbone = resolve_pretrained_backbone(DEFAULT_PRETRAINED_BACKBONE)
    except (FileNotFoundError, RuntimeError):
        print("WARNING: Cannot find pretrained backbone to build multi-UI centers. Skipping.")
        return

    centers_path.parent.mkdir(parents=True, exist_ok=True)
    build_cmd = [
        sys.executable,
        "build_multi_ui_centers.py",
        "--data-dir",
        str(train_data_dir),
        "--pretrained-backbone",
        str(pretrained_backbone),
        "--backbone",
        BACKBONE,
        "--output",
        str(centers_path),
        "--num-samples",
        "50000",
        "--batch-size",
        "128",
        "--num-workers",
        "2",
        "--degradations",
        ",".join(DEGRADED_DEGRADATIONS),
        "--severities",
        "5",
        "--include-global",
    ]
    if USE_FP16:
        build_cmd.append("--fp16")
    subprocess.run([str(c) for c in build_cmd], cwd=ARCFACE_DIR, check=True)
    print(f"Multi-UI centers saved to: {centers_path}")


_build_centers_if_needed()
run_5eval_degraded_runner(globals())
