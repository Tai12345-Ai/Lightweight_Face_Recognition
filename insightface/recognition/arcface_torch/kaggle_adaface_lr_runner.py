# %% [markdown]
# # AdaFace Split-LR Kaggle Runner
#
# Standalone Kaggle runner for comparing AdaFace under different backbone/head
# learning rates. Change `LOSS_NAME` to run another Phase 2 loss with the same
# LR sweep.

# %% [markdown]
# ## Cell 1: Clone Repo And Install Dependencies

# %%
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
import zipfile

REPO_URL = "https://github.com/Tai12345-Ai/Lightweight_Face_Recognition.git"
BRANCH = "main"

CODE_ROOT = Path("/kaggle/working/Lightweight_Face_Recognition")
ARCFACE_DIR = CODE_ROOT / "insightface" / "recognition" / "arcface_torch"

TRAIN_DATA_DIR = Path("/kaggle/input/CASIA-WebFace/casia-webface")
EVAL_DIR = Path("/kaggle/input/CASIA-WebFace/eval")
PRETRAINED_BACKBONE = Path("/kaggle/input/backbone/backbone.pth")
OUTPUT_ROOT = Path("/kaggle/working/experiments")

EVAL_TARGETS = [
    "lfw",
    "cfp_ff",
    "cfp_fp",
    "agedb_30",
    "calfw",
    "cplfw",
    "sllfw",
    "talfw",
]


def run(cmd, cwd=None, check=True):
    print("+", " ".join(str(x) for x in cmd))
    return subprocess.run([str(x) for x in cmd], cwd=cwd, check=check)


def normalize_train_data_dir(path):
    path = Path(path)
    nested_train = path / "casia-webface"
    nested_eval = path / "eval"
    if nested_train.is_dir() and nested_eval.is_dir():
        print("Detected nested CASIA-WebFace layout. Using train folder:", nested_train)
        return nested_train
    return path


def detect_num_classes(train_dir, default=10575):
    property_path = Path(train_dir) / "property"
    if property_path.exists():
        content = property_path.read_text(encoding="utf-8").strip()
        try:
            num_classes = int(content.split(",")[0].strip())
            print("Detected num_classes from property:", num_classes)
            return num_classes
        except Exception as exc:
            warnings.warn(
                f"Could not parse {property_path}: {content!r} ({exc}). "
                f"Using default NUM_CLASSES={default}.",
                RuntimeWarning,
            )
    print("Using default NUM_CLASSES:", default)
    return default


if not CODE_ROOT.exists():
    run(["git", "clone", "--branch", BRANCH, REPO_URL, str(CODE_ROOT)])
else:
    run(["git", "pull", "--ff-only"], cwd=CODE_ROOT)

os.chdir(ARCFACE_DIR)
print("Working dir:", Path.cwd())

run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "tensorboard",
    "easydict",
    "onnx",
    "opencv-python",
    "scikit-learn",
])

# %% [markdown]
# ## Cell 2: Check Inputs

# %%
if not TRAIN_DATA_DIR.exists():
    candidates = [
        p for p in Path("/kaggle/input").rglob("casia-webface")
        if p.is_dir() and ((p / "property").exists() or (p / "train.rec").exists())
    ]
    assert candidates, f"TRAIN_DATA_DIR not found: {TRAIN_DATA_DIR}"
    TRAIN_DATA_DIR = candidates[0]
TRAIN_DATA_DIR = normalize_train_data_dir(TRAIN_DATA_DIR)
NUM_CLASSES = detect_num_classes(TRAIN_DATA_DIR)

if not EVAL_DIR.exists():
    candidates = [
        p.parent for p in Path("/kaggle/input").rglob("lfw.bin")
        if (p.parent / "agedb_30.bin").exists()
    ]
    assert candidates, f"EVAL_DIR not found: {EVAL_DIR}"
    EVAL_DIR = candidates[0]

if not PRETRAINED_BACKBONE.exists():
    candidates = sorted(Path("/kaggle/input").rglob("backbone.pth"))
    assert candidates, f"PRETRAINED_BACKBONE not found: {PRETRAINED_BACKBONE}"
    PRETRAINED_BACKBONE = candidates[0]

print("TRAIN_DATA_DIR:", TRAIN_DATA_DIR)
print("EVAL_DIR:", EVAL_DIR)
print("PRETRAINED_BACKBONE:", PRETRAINED_BACKBONE)
print("NUM_CLASSES:", NUM_CLASSES)

assert TRAIN_DATA_DIR.exists()
assert EVAL_DIR.exists()
assert PRETRAINED_BACKBONE.exists()

# %% [markdown]
# ## Cell 3: Restore Previous Outputs

# %%
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PHASE2_ROOT = OUTPUT_ROOT / "phase2_loss"
restored = False

for zip_candidate in sorted(Path("/kaggle/input").rglob("phase2_adaface_lr_sweep.zip")):
    print("Found previous backup zip:", zip_candidate)
    extract_dir = Path("/kaggle/working/restore_phase2_adaface_lr")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_candidate, "r") as zf:
        zf.extractall(extract_dir)

    for folder_candidate in extract_dir.rglob("phase2_loss"):
        shutil.copytree(folder_candidate, PHASE2_ROOT, dirs_exist_ok=True)
        restored = True
        break
    if restored:
        break

if not restored:
    for folder_candidate in sorted(Path("/kaggle/input").rglob("phase2_loss")):
        print("Found previous phase2_loss folder:", folder_candidate)
        shutil.copytree(folder_candidate, PHASE2_ROOT, dirs_exist_ok=True)
        restored = True
        break

if restored:
    print("Restored previous outputs to:", PHASE2_ROOT)
else:
    print("No previous phase2 output found. Starting from pretrained backbone.")

# %% [markdown]
# ## Cell 4: Preflight

# %%
run([
    sys.executable,
    "-m",
    "py_compile",
    "losses_extended.py",
    "train_phase2_kaggle.py",
    "eval_degraded_phase2.py",
    "recordio_fallback.py",
], cwd=ARCFACE_DIR)

import torch
from backbones import get_model
from train_phase2_kaggle import build_dataset

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


class Args:
    data_dir = str(TRAIN_DATA_DIR)
    image_size = 112
    num_classes = NUM_CLASSES


dataset, inferred_num_classes = build_dataset(Args())
print("dataset size:", len(dataset))
print("num_classes:", inferred_num_classes)
assert inferred_num_classes in (10572, 10575), inferred_num_classes

backbone = get_model("r18", dropout=0.0, fp16=False, num_features=512)
print("Backbone ready:", type(backbone).__name__)

# %% [markdown]
# ## Cell 5: AdaFace LR Sweep

# %%
LOSS_NAME = "adaface"
BACKBONE = "r18"
EPOCHS = 8
BATCH_SIZE = 128
LR = 0.01
LR_SWEEP = [
    (0.001, 0.01),
    (0.0003, 0.003),
]
WARMUP_EPOCHS = 1.0
EVAL_EVERY = 1
SAVE_EVERY_STEPS = 300
SAVE_EVERY_EPOCHS = 1
MAX_TRAIN_MINUTES = 660
MIN_TRAIN_MINUTES_TO_START = 2
NUM_WORKERS = 2
USE_FP16 = True

SWEEP_START_TIME = time.time()


def remaining_train_minutes():
    if MAX_TRAIN_MINUTES <= 0:
        return MAX_TRAIN_MINUTES
    elapsed_minutes = (time.time() - SWEEP_START_TIME) / 60.0
    return max(0.0, MAX_TRAIN_MINUTES - elapsed_minutes)


def float_tag(value):
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def exp_dir(backbone_lr, head_lr):
    return (
        OUTPUT_ROOT
        / "phase2_loss"
        / (
            f"{BACKBONE}_{LOSS_NAME}"
            f"_blr_{float_tag(backbone_lr)}_hlr_{float_tag(head_lr)}"
        )
    )


def is_complete(backbone_lr, head_lr):
    metrics_path = exp_dir(backbone_lr, head_lr) / "metrics.json"
    if not metrics_path.exists():
        return False
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return len(metrics.get("epochs", [])) >= EPOCHS


print("Cell 5 NUM_CLASSES:", NUM_CLASSES)
assert NUM_CLASSES < 100000, f"Bad NUM_CLASSES={NUM_CLASSES}"
assert NUM_CLASSES in (10572, 10575), f"Unexpected NUM_CLASSES={NUM_CLASSES}"

for backbone_lr, head_lr in LR_SWEEP:
    if is_complete(backbone_lr, head_lr):
        print(f"[SKIP] {LOSS_NAME} blr={backbone_lr} hlr={head_lr} complete.")
        continue

    train_minutes_left = remaining_train_minutes()
    if MAX_TRAIN_MINUTES > 0 and train_minutes_left < MIN_TRAIN_MINUTES_TO_START:
        print(
            f"[STOP] sweep time budget exhausted "
            f"({train_minutes_left:.1f} minutes left). Resume next session."
        )
        break

    latest = exp_dir(backbone_lr, head_lr) / "latest.pt"
    cmd = [
        sys.executable,
        "train_phase2_kaggle.py",
        "--loss",
        LOSS_NAME,
        "--backbone",
        BACKBONE,
        "--data-dir",
        str(TRAIN_DATA_DIR),
        "--eval-dir",
        str(EVAL_DIR),
        "--output-dir",
        str(OUTPUT_ROOT),
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--lr",
        str(LR),
        "--backbone-lr",
        str(backbone_lr),
        "--head-lr",
        str(head_lr),
        "--warmup-epochs",
        str(WARMUP_EPOCHS),
        "--eval-every",
        str(EVAL_EVERY),
        "--save-every",
        str(SAVE_EVERY_EPOCHS),
        "--save-every-steps",
        str(SAVE_EVERY_STEPS),
        "--max-train-minutes",
        f"{train_minutes_left:.2f}",
        "--num-workers",
        str(NUM_WORKERS),
        "--num-classes",
        str(NUM_CLASSES),
        "--val-targets",
        ",".join(EVAL_TARGETS),
    ]
    if USE_FP16:
        cmd.append("--fp16")

    print(
        f"[BUDGET] loss={LOSS_NAME} backbone_lr={backbone_lr} head_lr={head_lr} "
        f"remaining_sweep_train_minutes={train_minutes_left:.1f}"
    )

    if latest.exists():
        print(f"[RESUME] {LOSS_NAME} blr={backbone_lr} hlr={head_lr} from {latest}")
        cmd.append("--resume")
    else:
        print(f"[START] {LOSS_NAME} blr={backbone_lr} hlr={head_lr} from pretrained backbone")
        cmd.extend(["--pretrained-backbone", str(PRETRAINED_BACKBONE)])

    run(cmd, cwd=ARCFACE_DIR)

    if not is_complete(backbone_lr, head_lr):
        print(
            f"[STOP] {LOSS_NAME} blr={backbone_lr} hlr={head_lr} "
            "is not complete yet. Resume next session."
        )
        break

print("Done. Loss:", LOSS_NAME)
print("LR sweep:", LR_SWEEP)

# %% [markdown]
# ## Cell 6: Progress

# %%
root = OUTPUT_ROOT / "phase2_loss"
if not root.exists():
    print("No phase2_loss folder yet:", root)
else:
    for exp in sorted(root.glob(f"{BACKBONE}_{LOSS_NAME}*")):
        metrics_path = exp / "metrics.json"
        latest = exp / "latest.pt"
        best = exp / "best.pth"

        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            epochs = len(metrics.get("epochs", []))
            best_epoch = metrics.get("best_epoch")
            best_score = metrics.get("best_score")
        else:
            epochs = 0
            best_epoch = None
            best_score = None

        print(exp.name)
        print("  epochs:", epochs)
        print("  latest:", latest.exists())
        print("  best:", best.exists())
        print("  best_epoch:", best_epoch)
        print("  best_score:", best_score)

# %% [markdown]
# ## Cell 7: Export Eval By Epoch

# %%
import pandas as pd


def complete_accuracy_mean(eval_metrics, targets):
    values = []
    for target in targets:
        item = eval_metrics.get(target)
        if item is None or "accuracy" not in item:
            return None
        values.append(float(item["accuracy"]))
    return float(sum(values) / len(values)) if values else None


rows = []
for metrics_path in sorted((OUTPUT_ROOT / "phase2_loss").glob(f"{BACKBONE}_{LOSS_NAME}*/metrics.json")):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    exp_name = metrics_path.parent.name

    for ep in metrics.get("epochs", []):
        evals = ep.get("eval", {}) or {}
        row = {
            "experiment": exp_name,
            "epoch": ep.get("epoch"),
            "loss": ep.get("loss"),
            "mean_norm": ep.get("mean_norm"),
            "lr": ep.get("lr"),
            "backbone_lr": ep.get("backbone_lr"),
            "head_lr": ep.get("head_lr"),
            "HQ_Avg": complete_accuracy_mean(
                evals, ["lfw", "cfp_ff", "cfp_fp", "agedb_30", "calfw", "cplfw"]
            ),
            "LQ_Avg": complete_accuracy_mean(evals, ["sllfw", "talfw"]),
            "All_Avg": complete_accuracy_mean(evals, EVAL_TARGETS),
        }
        for name, item in evals.items():
            row[name] = item.get("accuracy")
            row[f"{name}_std"] = item.get("std")
            row[f"{name}_xnorm"] = item.get("xnorm")
        rows.append(row)

df = pd.DataFrame(rows)
display(df)

out_csv = f"/kaggle/working/{LOSS_NAME}_lr_eval_by_epoch.csv"
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)

try:
    from IPython.display import FileLink, display

    display(FileLink(out_csv))
except Exception as exc:
    print("Could not render download link:", exc)

# %% [markdown]
# ## Cell 8: Backup

# %%
zip_base = "/kaggle/working/phase2_adaface_lr_sweep"
zip_path = Path(zip_base + ".zip")
if zip_path.exists():
    zip_path.unlink()

root = OUTPUT_ROOT / "phase2_loss"
if root.exists():
    shutil.make_archive(zip_base, "zip", str(OUTPUT_ROOT), "phase2_loss")
    print("Saved:", zip_path)
    print("Size MB:", zip_path.stat().st_size / 1024 / 1024)
    try:
        from IPython.display import FileLink, display

        print("Download:")
        display(FileLink(str(zip_path)))
    except Exception as exc:
        print("Could not render notebook download link:", exc)
        print("Download path:", zip_path)
else:
    print("No phase2 outputs yet:", root)
