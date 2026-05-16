# %% [markdown]
# # Soft-Gated Ada-CurricularFace Lambda Sweep
#
# Standalone Kaggle runner for testing fixed ``lambda_gate`` values before
# promoting the loss into the main Phase 2 loss registry.
#
# It trains only on:
# ``/kaggle/input/CASIA-WebFace/casia-webface``
#
# It evaluates only on .bin files under:
# ``/kaggle/input/CASIA-WebFace/eval``

# %% [markdown]
# ## Cell 1: Clone Repo And Install Dependencies
#
# Pull the repo into `/kaggle/working`, switch into `arcface_torch`, and install
# only the runtime packages needed by the training/eval scripts.

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
    "cfp_fp",
    "cplfw",
    "agedb_30",
    "calfw",
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
#
# Use the expected Kaggle inputs:
# `/kaggle/input/CASIA-WebFace/{casia-webface,eval}` and
# `/kaggle/input/backbone/backbone.pth`. The script refuses ambiguous `.pth`
# selection if the expected backbone path is missing.

# %%
if not TRAIN_DATA_DIR.exists():
    candidates = [p for p in Path("/kaggle/input").rglob("casia-webface") if p.is_dir()]
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

missing_eval = [name for name in EVAL_TARGETS if not (EVAL_DIR / f"{name}.bin").exists()]
assert not missing_eval, f"Missing eval bins in {EVAL_DIR}: {missing_eval}"

if not PRETRAINED_BACKBONE.exists():
    backbone_candidates = sorted(Path("/kaggle/input").rglob("backbone.pth"))
    if backbone_candidates:
        if len(backbone_candidates) > 1:
            warnings.warn(
                "Expected /kaggle/input/backbone/backbone.pth, but found multiple "
                "backbone.pth files. Using the first sorted candidate:\n"
                + "\n".join(str(p) for p in backbone_candidates),
                RuntimeWarning,
            )
        PRETRAINED_BACKBONE = backbone_candidates[0]
    else:
        pth_candidates = sorted(Path("/kaggle/input").rglob("*.pth"))
        assert pth_candidates, "No pretrained backbone .pth found under /kaggle/input"
        if len(pth_candidates) > 1:
            raise RuntimeError(
                "Expected /kaggle/input/backbone/backbone.pth, but it was not found. "
                "Multiple .pth files exist, so refusing to choose one silently. "
                "Candidates:\n" + "\n".join(str(p) for p in pth_candidates)
            )
        warnings.warn(
            "Expected /kaggle/input/backbone/backbone.pth, but it was not found. "
            f"Using the only .pth candidate: {pth_candidates[0]}",
            RuntimeWarning,
        )
        PRETRAINED_BACKBONE = pth_candidates[0]
else:
    print("Using expected backbone:", PRETRAINED_BACKBONE)

if not PRETRAINED_BACKBONE.exists():
    pth_candidates = sorted(Path("/kaggle/input").rglob("*.pth"))
    raise FileNotFoundError(
        "Pretrained backbone checkpoint not found. Expected "
        f"/kaggle/input/backbone/backbone.pth. Available .pth files:\n"
        + "\n".join(str(p) for p in pth_candidates)
    )

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("TRAIN_DATA_DIR:", TRAIN_DATA_DIR)
print("EVAL_DIR:", EVAL_DIR)
print("PRETRAINED_BACKBONE:", PRETRAINED_BACKBONE)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("NUM_CLASSES:", NUM_CLASSES)

# %% [markdown]
# ## Cell 3: Restore Previous Sweep Outputs
#
# `Save & Run All` starts from a clean `/kaggle/working`. To resume across
# Kaggle versions, add a previous version output as an input dataset. This cell
# restores either `soft_gated_lambda_sweep.zip` or a `soft_gated_lambda_sweep`
# folder from `/kaggle/input` back into `/kaggle/working/experiments`.

# %%
SWEEP_ROOT = OUTPUT_ROOT / "soft_gated_lambda_sweep"
restored = False

for zip_candidate in sorted(Path("/kaggle/input").rglob("soft_gated_lambda_sweep.zip")):
    print("Restoring previous sweep zip:", zip_candidate)
    with zipfile.ZipFile(zip_candidate, "r") as f:
        f.extractall(OUTPUT_ROOT)
    restored = True
    break

if not restored:
    for folder_candidate in sorted(Path("/kaggle/input").rglob("soft_gated_lambda_sweep")):
        if folder_candidate.is_dir():
            print("Restoring previous sweep folder:", folder_candidate)
            shutil.copytree(folder_candidate, SWEEP_ROOT, dirs_exist_ok=True)
            restored = True
            break

if restored:
    print("Restored previous sweep outputs to:", SWEEP_ROOT)
else:
    print("No previous sweep output input found. Starting from pretrained backbone.")

# %% [markdown]
# ## Cell 4: Preflight
#
# Compile the soft-gated loss, standalone train script, and reused Phase 2
# helpers before launching long Kaggle jobs.

# %%
run([
    sys.executable,
    "-m",
    "py_compile",
    "soft_gated_losses.py",
    "train_soft_gated_lambda_kaggle.py",
    "train_phase2_kaggle.py",
    "eval_degraded_phase2.py",
    "recordio_fallback.py",
], cwd=ARCFACE_DIR)

import torch
from backbones import get_model
from train_phase2_kaggle import torch_load_cpu, extract_backbone_state, build_dataset

NUM_CLASSES = 10575

print("TRAIN_DATA_DIR:", TRAIN_DATA_DIR)
print("EVAL_DIR:", EVAL_DIR)
print("PRETRAINED_BACKBONE:", PRETRAINED_BACKBONE)
print("EVAL_TARGETS:", ",".join(EVAL_TARGETS))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

assert device == "cuda", "CUDA/GPU is required"
assert NUM_CLASSES in (10572, 10575), f"Unexpected NUM_CLASSES={NUM_CLASSES}"
assert PRETRAINED_BACKBONE.exists(), f"Missing backbone: {PRETRAINED_BACKBONE}"

# %% [markdown]
# ## Cell 5: Lambda Sweep
#
# Train `soft_gated_ada_curricular` for fixed lambda values. Each lambda has
# its own output folder and resumes from `latest.pt` if present.

# %%
BACKBONE = "r18"
EPOCHS = 20
BATCH_SIZE = 128
LR = 0.01
BACKBONE_LR = 0.001
HEAD_LR = 0.01
WARMUP_EPOCHS = 1.0
EVAL_EVERY = 1
SAVE_EVERY_STEPS = 300
SAVE_EVERY_EPOCHS = 1
MAX_TRAIN_MINUTES = 480
MIN_TRAIN_MINUTES_TO_START = 2
NUM_WORKERS = 2
USE_FP16 = True

S = 64.0
M = 0.4
H = 0.333
LAMBDA_SWEEP = [0.5, 0.4, 0.2, 0.0]
SWEEP_START_TIME = time.time()


def remaining_train_minutes():
    if MAX_TRAIN_MINUTES <= 0:
        return MAX_TRAIN_MINUTES
    elapsed_minutes = (time.time() - SWEEP_START_TIME) / 60.0
    return max(0.0, MAX_TRAIN_MINUTES - elapsed_minutes)


def lambda_tag(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text.replace("-", "m").replace(".", "p")


def float_tag(value):
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def exp_dir(lambda_gate):
    return (
        OUTPUT_ROOT
        / "soft_gated_lambda_sweep"
        / (
            f"{BACKBONE}_soft_gated_ada_curricular_lambda_{lambda_tag(lambda_gate)}"
            f"_blr_{float_tag(BACKBONE_LR)}_hlr_{float_tag(HEAD_LR)}"
        )
    )


def is_complete(lambda_gate):
    metrics_path = exp_dir(lambda_gate) / "metrics.json"
    if not metrics_path.exists():
        return False
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return len(metrics.get("epochs", [])) >= EPOCHS


print("Cell 5 NUM_CLASSES:", NUM_CLASSES)
assert NUM_CLASSES < 100000, f"Bad NUM_CLASSES={NUM_CLASSES}"
assert NUM_CLASSES in (10572, 10575), f"Unexpected NUM_CLASSES={NUM_CLASSES}"

for lambda_gate in LAMBDA_SWEEP:
    if is_complete(lambda_gate):
        print(f"[SKIP] lambda_gate={lambda_gate} complete.")
        continue

    train_minutes_left = remaining_train_minutes()
    if MAX_TRAIN_MINUTES > 0 and train_minutes_left < MIN_TRAIN_MINUTES_TO_START:
        print(
            f"[STOP] sweep time budget exhausted "
            f"({train_minutes_left:.1f} minutes left). Resume next session."
        )
        break

    latest = exp_dir(lambda_gate) / "latest.pt"
    cmd = [
        sys.executable,
        "train_soft_gated_lambda_kaggle.py",
        "--loss",
        "soft_gated_ada_curricular",
        "--network",
        BACKBONE,
        "--s",
        str(S),
        "--m",
        str(M),
        "--h",
        str(H),
        "--lambda_gate",
        str(lambda_gate),
        "--train_data",
        str(TRAIN_DATA_DIR),
        "--eval_dir",
        str(EVAL_DIR),
        "--output_dir",
        str(OUTPUT_ROOT),
        "--epochs",
        str(EPOCHS),
        "--batch_size",
        str(BATCH_SIZE),
        "--lr",
        str(LR),
        "--backbone_lr",
        str(BACKBONE_LR),
        "--head_lr",
        str(HEAD_LR),
        "--warmup_epochs",
        str(WARMUP_EPOCHS),
        "--eval_every",
        str(EVAL_EVERY),
        "--save_every",
        str(SAVE_EVERY_EPOCHS),
        "--save_every_steps",
        str(SAVE_EVERY_STEPS),
        "--max_train_minutes",
        f"{train_minutes_left:.2f}",
        "--num_workers",
        str(NUM_WORKERS),
        "--num_classes",
        str(NUM_CLASSES),
        "--eval_targets",
        ",".join(EVAL_TARGETS),
    ]
    if USE_FP16:
        cmd.append("--fp16")

    print(
        f"[BUDGET] lambda_gate={lambda_gate} "
        f"remaining_sweep_train_minutes={train_minutes_left:.1f}"
    )

    if latest.exists():
        print(f"[RESUME] lambda_gate={lambda_gate} from {latest}")
        cmd.append("--resume")
    else:
        print(f"[START] lambda_gate={lambda_gate} from pretrained backbone")
        cmd.extend(["--pretrained_backbone", str(PRETRAINED_BACKBONE)])

    run(cmd, cwd=ARCFACE_DIR)

    if not is_complete(lambda_gate):
        print(f"[STOP] lambda_gate={lambda_gate} is not complete yet. Resume next session.")
        break

print("Done. Lambda sweep:", LAMBDA_SWEEP)
print("Learning rates: backbone_lr=", BACKBONE_LR, "head_lr=", HEAD_LR)

# %% [markdown]
# ## Cell 6: Progress
#
# Print epoch count, latest checkpoint, best checkpoint, and best score for
# each lambda experiment.

# %%
root = OUTPUT_ROOT / "soft_gated_lambda_sweep"
if not root.exists():
    print("No soft_gated_lambda_sweep folder yet:", root)
else:
    for exp in sorted(root.glob("r18_soft_gated_ada_curricular_lambda_*")):
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
# ## Cell 7: Backup
#
# Zip the sweep outputs so they can be downloaded or saved as a Kaggle version
# artifact. The cell also displays a direct notebook download link for the zip.

# %%
zip_base = "/kaggle/working/soft_gated_lambda_sweep"
zip_path = Path(zip_base + ".zip")
if zip_path.exists():
    zip_path.unlink()

root = OUTPUT_ROOT / "soft_gated_lambda_sweep"
if root.exists():
    shutil.make_archive(zip_base, "zip", str(OUTPUT_ROOT), "soft_gated_lambda_sweep")
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
    print("No sweep outputs yet:", root)
