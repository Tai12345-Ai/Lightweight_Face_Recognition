# %% [markdown]
# # ArcFace 20-Epoch 5-Eval + Degraded Eval Kaggle Runner

# %%
from pathlib import Path
import os
import subprocess
import sys

REPO_URL = "https://github.com/Tai12345-Ai/Lightweight_Face_Recognition.git"
BRANCH = "main"
CODE_ROOT = Path("/kaggle/working/Lightweight_Face_Recognition")
ARCFACE_DIR = CODE_ROOT / "insightface" / "recognition" / "arcface_torch"

if not CODE_ROOT.exists():
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(CODE_ROOT)], check=True)
else:
    subprocess.run(["git", "pull", "--ff-only"], cwd=CODE_ROOT, check=True)
os.chdir(ARCFACE_DIR)
sys.path.insert(0, str(ARCFACE_DIR))

from kaggle_5eval_degraded_common import run_5eval_degraded_runner

RUNNER_FILE = "kaggle_arcface_5eval_degraded_runner.py"
RUNNER_KIND = "phase2"
OUTPUT_SUBDIR = "phase2_loss"
BACKUP_ZIP_NAME = "phase2_arcface_20ep_5eval_degraded_s5.zip"

LOSS_NAME = "arcface"
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

EVAL_TARGETS = [
    "lfw",
    "cfp_fp",
    "cplfw",
    "agedb_30",
    "calfw",
]
VAL_TARGETS = EVAL_TARGETS
HQ_EVAL_TARGETS = [
    "lfw",
    "cfp_fp",
    "cplfw",
    "agedb_30",
    "calfw",
]

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

run_5eval_degraded_runner(globals())
