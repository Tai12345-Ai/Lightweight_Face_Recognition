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

# %%
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import warnings

REPO_URL = "https://github.com/Tai12345-Ai/Lightweight_Face_Recognition.git"
BRANCH = "main"

CODE_ROOT = Path("/kaggle/working/Lightweight_Face_Recognition")
ARCFACE_DIR = CODE_ROOT / "insightface" / "recognition" / "arcface_torch"

TRAIN_DATA_DIR = Path("/kaggle/input/CASIA-WebFace/casia-webface")
EVAL_DIR = Path("/kaggle/input/CASIA-WebFace/eval")
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
# ## Check Inputs

# %%
if not TRAIN_DATA_DIR.exists():
    candidates = [p for p in Path("/kaggle/input").rglob("casia-webface") if p.is_dir()]
    assert candidates, f"TRAIN_DATA_DIR not found: {TRAIN_DATA_DIR}"
    TRAIN_DATA_DIR = candidates[0]

if not EVAL_DIR.exists():
    candidates = [
        p.parent for p in Path("/kaggle/input").rglob("lfw.bin")
        if (p.parent / "agedb_30.bin").exists()
    ]
    assert candidates, f"EVAL_DIR not found: {EVAL_DIR}"
    EVAL_DIR = candidates[0]

missing_eval = [name for name in EVAL_TARGETS if not (EVAL_DIR / f"{name}.bin").exists()]
assert not missing_eval, f"Missing eval bins in {EVAL_DIR}: {missing_eval}"

backbone_candidates = sorted(Path("/kaggle/input").rglob("backbone.pth"))
if backbone_candidates:
    if len(backbone_candidates) > 1:
        warnings.warn(
            "Multiple backbone.pth files found. Using the first sorted candidate:\n"
            + "\n".join(str(p) for p in backbone_candidates),
            RuntimeWarning,
        )
    PRETRAINED_BACKBONE = backbone_candidates[0]
else:
    pth_candidates = sorted(Path("/kaggle/input").rglob("*.pth"))
    assert pth_candidates, "No pretrained backbone .pth found under /kaggle/input"
    if len(pth_candidates) > 1:
        raise RuntimeError(
            "No file named backbone.pth was found, and multiple .pth files exist. "
            "Refusing to choose one silently. Candidates:\n"
            + "\n".join(str(p) for p in pth_candidates)
        )
    warnings.warn(
        f"No backbone.pth found. Using the only .pth candidate: {pth_candidates[0]}",
        RuntimeWarning,
    )
    PRETRAINED_BACKBONE = pth_candidates[0]

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("TRAIN_DATA_DIR:", TRAIN_DATA_DIR)
print("EVAL_DIR:", EVAL_DIR)
print("PRETRAINED_BACKBONE:", PRETRAINED_BACKBONE)
print("OUTPUT_ROOT:", OUTPUT_ROOT)

# %% [markdown]
# ## Preflight

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

# %% [markdown]
# ## Lambda Sweep

# %%
BACKBONE = "r18"
EPOCHS = 20
BATCH_SIZE = 128
LR = 0.01
WARMUP_EPOCHS = 1.0
EVAL_EVERY = 2
SAVE_EVERY_STEPS = 300
MAX_TRAIN_MINUTES = 600
NUM_WORKERS = 2
USE_FP16 = True

S = 64.0
M = 0.4
H = 0.333
LAMBDA_SWEEP = [0.0, 0.2, 0.4, 0.5]


def lambda_tag(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text.replace("-", "m").replace(".", "p")


def exp_dir(lambda_gate):
    return (
        OUTPUT_ROOT
        / "soft_gated_lambda_sweep"
        / f"{BACKBONE}_soft_gated_ada_curricular_lambda_{lambda_tag(lambda_gate)}"
    )


def is_complete(lambda_gate):
    metrics_path = exp_dir(lambda_gate) / "metrics.json"
    if not metrics_path.exists():
        return False
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return len(metrics.get("epochs", [])) >= EPOCHS


for lambda_gate in LAMBDA_SWEEP:
    if is_complete(lambda_gate):
        print(f"[SKIP] lambda_gate={lambda_gate} complete.")
        continue

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
        "--warmup_epochs",
        str(WARMUP_EPOCHS),
        "--eval_every",
        str(EVAL_EVERY),
        "--save_every",
        "1",
        "--save_every_steps",
        str(SAVE_EVERY_STEPS),
        "--max_train_minutes",
        str(MAX_TRAIN_MINUTES),
        "--num_workers",
        str(NUM_WORKERS),
        "--eval_targets",
        ",".join(EVAL_TARGETS),
    ]
    if USE_FP16:
        cmd.append("--fp16")

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

# %% [markdown]
# ## Progress

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
# ## Backup

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
else:
    print("No sweep outputs yet:", root)
