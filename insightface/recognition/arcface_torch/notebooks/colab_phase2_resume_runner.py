# %% [markdown]
# # Phase 2 Colab Resume Runner
#
# This runner uses the same `train_phase2_kaggle.py` script as Kaggle.
# It is meant for switching between Kaggle and Colab without changing the
# training protocol. Checkpoints are restored from Google Drive before training
# and backed up after each run.
#
# Default training queue:
#
# 1. arcface
# 2. adaface
# 3. curricularface
# 4. proposed
# 5. magface
# 6. elasticface
# 7. cosface

# %%
from google.colab import drive

drive.mount("/content/drive")

# %% [markdown]
# ## Configuration
#
# Edit the paths below before running. Keep `OUTPUT_DIR` and `DRIVE_BACKUP_ROOT`
# stable across Kaggle and Colab so `--resume` can find the same experiment
# folder structure.

# %%
from pathlib import Path

REPO_URL = "https://github.com/Tai12345-Ai/Lightweight_Face_Recognition.git"
BRANCH = "main"

CODE_ROOT = Path("/content/Lightweight_Face_Recognition")
ARCFACE_DIR = CODE_ROOT / "insightface" / "recognition" / "arcface_torch"

DATA_DIR = "/content/drive/MyDrive/datasets/faces_webface_112x112"
PRETRAINED_BACKBONE = "/content/drive/MyDrive/phase2_resnet18_loss/pretrained/backbone.pth"

OUTPUT_DIR = "/content/outputs"
DRIVE_BACKUP_ROOT = Path("/content/drive/MyDrive/phase2_resnet18_loss/outputs")

BACKBONE = "r18"
EPOCHS = 20
BATCH_SIZE = 128
LR = 0.01
WARMUP_EPOCHS = 1.0
EVAL_EVERY = 2
SAVE_EVERY_STEPS = 300
MAX_TRAIN_MINUTES = 600
NUM_WORKERS = 2
VAL_TARGETS = ""  # Use "lfw,cfp_fp,agedb_30" only if those .bin files exist.

LOSS_QUEUE = [
    "arcface",
    "adaface",
    "curricularface",
    "proposed",
    "magface",
    "elasticface",
    "cosface",
]

# %% [markdown]
# ## Prepare Code

# %%
import os
import shutil
import subprocess
import sys


def run(cmd, cwd=None):
    print("+", " ".join(str(item) for item in cmd))
    subprocess.check_call([str(item) for item in cmd], cwd=cwd)


if not CODE_ROOT.exists():
    run(["git", "clone", "--branch", BRANCH, REPO_URL, str(CODE_ROOT)])
else:
    run(["git", "fetch", "origin", BRANCH], cwd=CODE_ROOT)
    run(["git", "checkout", BRANCH], cwd=CODE_ROOT)
    run(["git", "pull", "--ff-only", "origin", BRANCH], cwd=CODE_ROOT)

run([sys.executable, "-m", "pip", "install", "-r", "requirement.txt"], cwd=ARCFACE_DIR)

# %% [markdown]
# ## Restore Previous Checkpoints From Drive

# %%
local_phase2 = Path(OUTPUT_DIR) / "phase2_loss"
drive_phase2 = DRIVE_BACKUP_ROOT / "phase2_loss"
local_phase2.mkdir(parents=True, exist_ok=True)

if drive_phase2.exists():
    shutil.copytree(drive_phase2, local_phase2, dirs_exist_ok=True)
    print(f"Restored checkpoints from {drive_phase2} to {local_phase2}")
else:
    print(f"No previous Drive backup found at {drive_phase2}")

assert Path(DATA_DIR).exists(), f"DATA_DIR not found: {DATA_DIR}"
assert Path(PRETRAINED_BACKBONE).exists(), f"PRETRAINED_BACKBONE not found: {PRETRAINED_BACKBONE}"

# %% [markdown]
# ## Train And Backup
#
# The runner stops the queue if the current loss is not complete. That way, if
# Colab reaches `MAX_TRAIN_MINUTES`, the next session resumes the same loss
# instead of accidentally moving to the next experiment.

# %%
import json


def experiment_dir(loss_name):
    return Path(OUTPUT_DIR) / "phase2_loss" / f"{BACKBONE}_{loss_name}"


def is_loss_complete(loss_name):
    metrics_path = experiment_dir(loss_name) / "metrics.json"
    if not metrics_path.exists():
        return False
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    finished_epochs = len(metrics.get("epochs", []))
    return finished_epochs >= EPOCHS


def backup_loss(loss_name):
    src = experiment_dir(loss_name)
    if not src.exists():
        print(f"No local experiment to backup for {loss_name}: {src}")
        return
    dst = DRIVE_BACKUP_ROOT / "phase2_loss" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"Backed up {loss_name} to {dst}")


def train_loss(loss_name):
    exp_dir = experiment_dir(loss_name)
    latest = exp_dir / "latest.pt"
    resume = latest.exists()

    cmd = [
        sys.executable,
        "train_phase2_kaggle.py",
        "--loss",
        loss_name,
        "--backbone",
        BACKBONE,
        "--data-dir",
        DATA_DIR,
        "--output-dir",
        OUTPUT_DIR,
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--lr",
        str(LR),
        "--warmup-epochs",
        str(WARMUP_EPOCHS),
        "--eval-every",
        str(EVAL_EVERY),
        "--save-every",
        "1",
        "--save-every-steps",
        str(SAVE_EVERY_STEPS),
        "--max-train-minutes",
        str(MAX_TRAIN_MINUTES),
        "--num-workers",
        str(NUM_WORKERS),
        "--val-targets",
        VAL_TARGETS,
        "--fp16",
    ]
    if resume:
        cmd.append("--resume")
    else:
        cmd.extend(["--pretrained-backbone", PRETRAINED_BACKBONE])

    run(cmd, cwd=ARCFACE_DIR)


for loss_name in LOSS_QUEUE:
    if is_loss_complete(loss_name):
        print(f"[SKIP] {loss_name} already has {EPOCHS} epochs.")
        continue

    train_loss(loss_name)
    backup_loss(loss_name)

    if not is_loss_complete(loss_name):
        print(f"[STOP] {loss_name} is not complete yet. Resume it next session.")
        break

print("Done. Current queue:", LOSS_QUEUE)
