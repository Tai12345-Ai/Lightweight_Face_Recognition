# %% [markdown]
# # Kaggle Phase 2 Runner
#
# Copy these cells into a Kaggle notebook. Use GPU T4/T4 x2, not TPU.
# Inputs expected:
# - CASIA-WebFace RecordIO dataset with train.rec and train.idx.
# - Eval bins with lfw.bin, cfp_fp.bin, agedb_30.bin.
# - Pretrained r18 backbone.pth.
# - Optional previous phase2_outputs.zip or outputs/phase2_loss for resume.

# %% [markdown]
# ## Cell 1: Clone Repo And Install Dependencies

# %%
from pathlib import Path
import os
import subprocess
import sys

REPO_URL = "https://github.com/Tai12345-Ai/Lightweight_Face_Recognition.git"
BRANCH = "main"

CODE_ROOT = Path("/kaggle/working/Lightweight_Face_Recognition")
ARCFACE_DIR = CODE_ROOT / "insightface" / "recognition" / "arcface_torch"


def run(cmd, cwd=None, check=True):
    print("+", " ".join(map(str, cmd)))
    return subprocess.run([str(x) for x in cmd], cwd=cwd, check=check)


if not CODE_ROOT.exists():
    run(["git", "clone", "--branch", BRANCH, REPO_URL, str(CODE_ROOT)])
else:
    run(["git", "pull", "--ff-only"], cwd=CODE_ROOT)

os.chdir(ARCFACE_DIR)
print("Working dir:", Path.cwd())

# Do not install from requirement.txt because it contains deprecated sklearn.
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

import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Has train script:", (ARCFACE_DIR / "train_phase2_kaggle.py").exists())
print("Has degraded eval:", (ARCFACE_DIR / "eval_degraded_phase2.py").exists())
print("Has recordio fallback:", (ARCFACE_DIR / "recordio_fallback.py").exists())

# %% [markdown]
# ## Cell 2: Detect Inputs And Build phase2_data

# %%
from pathlib import Path
import os
import shutil

INPUT = Path("/kaggle/input")
OUTPUT_DIR = Path("/kaggle/working/outputs")
PHASE2_DATA_DIR = Path("/kaggle/working/phase2_data")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PHASE2_DATA_DIR.mkdir(parents=True, exist_ok=True)

train_candidates = sorted({
    p.parent for p in INPUT.rglob("train.rec")
    if (p.parent / "train.idx").exists()
})

eval_candidates = sorted({
    p.parent for p in INPUT.rglob("lfw.bin")
    if (p.parent / "cfp_fp.bin").exists()
    and (p.parent / "agedb_30.bin").exists()
})

backbone_candidates = sorted(INPUT.rglob("backbone.pth"))

print("=== train candidates ===")
for p in train_candidates:
    print(p)

print("\n=== eval candidates ===")
for p in eval_candidates:
    print(p)

print("\n=== backbone candidates ===")
for p in backbone_candidates:
    print(p)

assert train_candidates, "train.rec + train.idx not found"
assert eval_candidates, "lfw.bin + cfp_fp.bin + agedb_30.bin not found"
assert backbone_candidates, "backbone.pth not found"

TRAIN_SOURCE_DIR = train_candidates[0]
EVAL_SOURCE_DIR = eval_candidates[0]
PRETRAINED_BACKBONE = backbone_candidates[0]


def link_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
        print("linked:", dst, "->", src)
    except OSError:
        shutil.copy2(src, dst)
        print("copied:", dst, "<-", src)


for name in ["train.rec", "train.idx", "property", "train.lst"]:
    src = TRAIN_SOURCE_DIR / name
    if src.exists():
        link_or_copy(src, PHASE2_DATA_DIR / name)

for name in ["lfw.bin", "cfp_fp.bin", "agedb_30.bin"]:
    src = EVAL_SOURCE_DIR / name
    if src.exists():
        link_or_copy(src, PHASE2_DATA_DIR / name)

TRAIN_DATA_DIR = PHASE2_DATA_DIR
EVAL_DATA_DIR = PHASE2_DATA_DIR

assert (TRAIN_DATA_DIR / "train.rec").exists()
assert (TRAIN_DATA_DIR / "train.idx").exists()
assert PRETRAINED_BACKBONE.exists()

has_eval_bins = all(
    (EVAL_DATA_DIR / f"{name}.bin").exists()
    for name in ["lfw", "cfp_fp", "agedb_30"]
)
VAL_TARGETS = "lfw,cfp_fp,agedb_30" if has_eval_bins else ""

print("\nSELECTED PATHS")
print("TRAIN_SOURCE_DIR:", TRAIN_SOURCE_DIR)
print("EVAL_SOURCE_DIR:", EVAL_SOURCE_DIR)
print("TRAIN_DATA_DIR:", TRAIN_DATA_DIR)
print("EVAL_DATA_DIR:", EVAL_DATA_DIR)
print("PRETRAINED_BACKBONE:", PRETRAINED_BACKBONE)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("VAL_TARGETS:", VAL_TARGETS or "(disabled)")

print("\nPHASE2_DATA_DIR contents:")
for p in sorted(PHASE2_DATA_DIR.iterdir()):
    print(" -", p.name, "->", p.resolve())

# %% [markdown]
# ## Cell 3: Restore Previous Checkpoints

# %%
import shutil
import zipfile
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working/outputs")
WORK_PHASE2 = OUTPUT_DIR / "phase2_loss"
WORK_PHASE2.mkdir(parents=True, exist_ok=True)

restored = False

for z in Path("/kaggle/input").rglob("phase2_outputs.zip"):
    print("Restoring zip:", z)
    with zipfile.ZipFile(z, "r") as f:
        f.extractall("/kaggle/working")
    restored = True
    break

if not restored:
    for p in Path("/kaggle/input").rglob("phase2_loss"):
        if p.is_dir():
            print("Restoring phase2_loss folder:", p)
            shutil.copytree(p, WORK_PHASE2, dirs_exist_ok=True)
            restored = True
            break

if restored:
    print("Restored to:", WORK_PHASE2)
else:
    print("No previous checkpoint found. This is a fresh run.")

print("\nExisting experiments:")
for p in sorted(WORK_PHASE2.glob("r18_*")):
    print(" -", p.name)

# %% [markdown]
# ## Cell 4: Preflight Check

# %%
import torch
from pathlib import Path

run([
    sys.executable,
    "-m",
    "py_compile",
    "train_phase2_kaggle.py",
    "losses_extended.py",
    "eval_degraded_phase2.py",
    "recordio_fallback.py",
], cwd=ARCFACE_DIR)

from backbones import get_model
from train_phase2_kaggle import torch_load_cpu, extract_backbone_state, build_dataset

NUM_CLASSES = 10575

print("TRAIN_DATA_DIR:", TRAIN_DATA_DIR)
print("EVAL_DATA_DIR:", EVAL_DATA_DIR)
print("PRETRAINED_BACKBONE:", PRETRAINED_BACKBONE)
print("VAL_TARGETS:", VAL_TARGETS or "(disabled)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

ckpt = torch_load_cpu(PRETRAINED_BACKBONE)
state = extract_backbone_state(ckpt)

model = get_model("r18", dropout=0, fp16=False, num_features=512)
msg = model.load_state_dict(state, strict=False)
print("missing keys:", len(msg.missing_keys))
print("unexpected keys:", len(msg.unexpected_keys))

model = model.to(device).eval()
x = torch.randn(2, 3, 112, 112, device=device)
with torch.no_grad():
    y = model(x)
print("forward ok:", y.shape)


class Args:
    data_dir = str(TRAIN_DATA_DIR)
    image_size = 112
    num_classes = NUM_CLASSES


dataset, inferred_num_classes = build_dataset(Args())
print("dataset size:", len(dataset))
print("num_classes:", inferred_num_classes)

img, label = dataset[0]
print("sample image:", img.shape, img.dtype)
print("sample label:", label)

# %% [markdown]
# ## Cell 5: Smoke Test

# %%
import shutil
from pathlib import Path

# Smoke test only checks whether code, paths, checkpoints, and losses run.
# It intentionally uses a low max_train_minutes and frequent step checkpoints.
# Do not use smoke test outputs for reporting.
RUN_SMOKE = True
USE_FP16 = True
SMOKE_NUM_WORKERS = 2
NUM_CLASSES = 10575
SMOKE_LOSSES = [
    "arcface",
    "adaface",
    "magface",
    "proposed",
]

RUN_BATCH128_SPEED_TEST = False

smoke = Path("/kaggle/working/smoke_outputs")
if smoke.exists():
    shutil.rmtree(smoke)
    print("Deleted old smoke_outputs")

if RUN_SMOKE:
    for smoke_loss in SMOKE_LOSSES:
        print(f"\n[SMOKE] Testing loss: {smoke_loss}")
        cmd = [
            sys.executable,
            "train_phase2_kaggle.py",
            "--loss",
            smoke_loss,
            "--backbone",
            "r18",
            "--pretrained-backbone",
            str(PRETRAINED_BACKBONE),
            "--data-dir",
            str(TRAIN_DATA_DIR),
            "--output-dir",
            "/kaggle/working/smoke_outputs",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--lr",
            "0.01",
            "--warmup-epochs",
            "0",
            "--eval-every",
            "2",
            "--save-every",
            "1",
            "--save-every-steps",
            "20",
            "--max-train-minutes",
            "5",
            "--num-workers",
            str(SMOKE_NUM_WORKERS),
            "--num-classes",
            str(NUM_CLASSES),
            "--val-targets",
            "",
        ]
        if USE_FP16:
            cmd.append("--fp16")
        try:
            run(cmd, cwd=ARCFACE_DIR)
        except subprocess.CalledProcessError as exc:
            print(f"[SMOKE FAILED] loss={smoke_loss}")
            raise exc
else:
    print("Smoke test skipped.")

if RUN_BATCH128_SPEED_TEST:
    batch128_smoke = Path("/kaggle/working/smoke_batch128_outputs")
    if batch128_smoke.exists():
        shutil.rmtree(batch128_smoke)
        print("Deleted old smoke_batch128_outputs")

    print("\n[SMOKE] Running ArcFace batch 128 speed/OOM test")
    cmd = [
        sys.executable,
        "train_phase2_kaggle.py",
        "--loss",
        "arcface",
        "--backbone",
        "r18",
        "--pretrained-backbone",
        str(PRETRAINED_BACKBONE),
        "--data-dir",
        str(TRAIN_DATA_DIR),
        "--output-dir",
        str(batch128_smoke),
        "--epochs",
        "1",
        "--batch-size",
        "128",
        "--lr",
        "0.01",
        "--warmup-epochs",
        "0",
        "--eval-every",
        "2",
        "--save-every",
        "1",
        "--save-every-steps",
        "100",
        "--max-train-minutes",
        "5",
        "--num-workers",
        str(SMOKE_NUM_WORKERS),
        "--num-classes",
        str(NUM_CLASSES),
        "--val-targets",
        "",
        "--fp16",
    ]
    run(cmd, cwd=ARCFACE_DIR)

# %% [markdown]
# ## Cell 6: Official Training With Resume

# %%
import json
import sys
from pathlib import Path

BACKBONE = "r18"
EPOCHS = 20
BATCH_SIZE = 128
LR = 0.01
WARMUP_EPOCHS = 1.0
EVAL_EVERY = 2
SAVE_EVERY_STEPS = 300
MAX_TRAIN_MINUTES = 600
NUM_WORKERS = 2
NUM_CLASSES = 10575
USE_FP16 = True

# Official training uses save_every_steps=300 to reduce checkpoint overhead.
# If the session stops, rerun the notebook and the script resumes from latest.pt.
LOSS_QUEUE = [
    "arcface",
    "adaface",
    "curricularface",
    "proposed",
    "magface",
    "elasticface",
    "cosface",
]


def exp_dir(loss):
    return OUTPUT_DIR / "phase2_loss" / f"{BACKBONE}_{loss}"


def is_complete(loss):
    metrics_path = exp_dir(loss) / "metrics.json"
    if not metrics_path.exists():
        return False
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return len(metrics.get("epochs", [])) >= EPOCHS


for loss in LOSS_QUEUE:
    if is_complete(loss):
        print(f"[SKIP] {loss} complete.")
        continue

    latest = exp_dir(loss) / "latest.pt"

    cmd = [
        sys.executable,
        "train_phase2_kaggle.py",
        "--loss",
        loss,
        "--backbone",
        BACKBONE,
        "--data-dir",
        str(TRAIN_DATA_DIR),
        "--output-dir",
        str(OUTPUT_DIR),
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
        "--num-classes",
        str(NUM_CLASSES),
        "--val-targets",
        VAL_TARGETS,
    ]
    if USE_FP16:
        cmd.append("--fp16")

    if latest.exists():
        print(f"[RESUME] {loss} from {latest}")
        cmd.append("--resume")
    else:
        print(f"[START] {loss} from pretrained backbone")
        cmd.extend(["--pretrained-backbone", str(PRETRAINED_BACKBONE)])

    run(cmd, cwd=ARCFACE_DIR)

    if not is_complete(loss):
        print(f"[STOP] {loss} is not complete yet. Resume next session.")
        break

print("Done. Queue:", LOSS_QUEUE)

# %% [markdown]
# ## Cell 7: Progress Check

# %%
from pathlib import Path
import json

root = OUTPUT_DIR / "phase2_loss"

if not root.exists():
    print("No phase2_loss folder yet:", root)
else:
    for exp in sorted(root.glob("r18_*")):
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
# ## Cell 8: Backup Outputs

# %%
import shutil
from pathlib import Path

zip_base = "/kaggle/working/phase2_outputs"
zip_path = Path(zip_base + ".zip")

if zip_path.exists():
    zip_path.unlink()

if OUTPUT_DIR.exists():
    shutil.make_archive(zip_base, "zip", "/kaggle/working", "outputs")
    print("Saved:", zip_path)
    print("Size MB:", zip_path.stat().st_size / 1024 / 1024)
    print("Now click Kaggle Save Version.")
else:
    print("OUTPUT_DIR does not exist yet:", OUTPUT_DIR)

# %% [markdown]
# ## Cell 9: Degraded Evaluation

# %%
RUN_DEGRADED_EVAL = False

if RUN_DEGRADED_EVAL:
    assert VAL_TARGETS, "Missing lfw.bin/cfp_fp.bin/agedb_30.bin."

    run([
        sys.executable,
        "eval_degraded_phase2.py",
        "--checkpoint-dir",
        str(OUTPUT_DIR / "phase2_loss"),
        "--backbone",
        "r18",
        "--data-dir",
        str(EVAL_DATA_DIR),
        "--output",
        str(OUTPUT_DIR / "degraded_eval"),
        "--targets",
        "lfw,cfp_fp,agedb_30",
        "--degradations",
        "clean,blur,lowres,jpeg,brightness,noise",
        "--batch-size",
        "128",
        "--fp16",
    ], cwd=ARCFACE_DIR)
else:
    print("Degraded eval skipped.")
