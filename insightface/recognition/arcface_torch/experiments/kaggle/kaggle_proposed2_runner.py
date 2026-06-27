# %% [markdown]
# # Proposed V2 Adaptive Soft-Gated Ada-CurricularFace Sweep
#
# Standalone Kaggle runner for:
# `adaptive_soft_gated_ada_curricular_v2`
#
# It trains only on:
# `/kaggle/input/CASIA-WebFace/casia-webface`
#
# It evaluates only on .bin files under:
# `/kaggle/input/CASIA-WebFace/eval`

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
HQ_EVAL_TARGETS = ["lfw", "cfp_fp", "cplfw", "agedb_30", "calfw"]
LQ_EVAL_TARGETS = ["sllfw", "talfw"]


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
# ## Cell 3: Restore Previous Proposed V2 Outputs
#
# `Save & Run All` starts from a clean `/kaggle/working`. To resume across
# Kaggle versions, add a previous version output as an input dataset. This cell
# restores either `proposed2_sweep.zip` or a `proposed2_sweep` folder from
# `/kaggle/input` back into `/kaggle/working/experiments`.

# %%
SWEEP_ROOT = OUTPUT_ROOT / "proposed2_sweep"
restored = False

for zip_candidate in sorted(Path("/kaggle/input").rglob("proposed2_sweep.zip")):
    print("Restoring previous proposed2 sweep zip:", zip_candidate)
    with zipfile.ZipFile(zip_candidate, "r") as f:
        f.extractall(OUTPUT_ROOT)
    restored = True
    break

if not restored:
    for folder_candidate in sorted(Path("/kaggle/input").rglob("proposed2_sweep")):
        if folder_candidate.is_dir():
            print("Restoring previous proposed2 sweep folder:", folder_candidate)
            shutil.copytree(folder_candidate, SWEEP_ROOT, dirs_exist_ok=True)
            restored = True
            break

if restored:
    print("Restored previous proposed2 outputs to:", SWEEP_ROOT)
else:
    print("No previous proposed2 output input found. Starting from pretrained backbone.")

# %% [markdown]
# ## Cell 4: Preflight
#
# Compile the proposed loss, standalone train script, reused Phase 2 helpers,
# and this runner before launching long Kaggle jobs.

# %%
run([
    sys.executable,
    "-m",
    "py_compile",
    "soft_gated_losses.py",
    "train_soft_gated_lambda_kaggle.py",
    "train_phase2_kaggle.py",
    "kaggle_proposed2_runner.py",
], cwd=ARCFACE_DIR)

import torch

print("TRAIN_DATA_DIR:", TRAIN_DATA_DIR)
print("EVAL_DIR:", EVAL_DIR)
print("PRETRAINED_BACKBONE:", PRETRAINED_BACKBONE)
print("EVAL_TARGETS:", ",".join(EVAL_TARGETS))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

assert device.type == "cuda", "CUDA/GPU is required"
assert NUM_CLASSES in (10572, 10575), f"Unexpected NUM_CLASSES={NUM_CLASSES}"
assert PRETRAINED_BACKBONE.exists(), f"Missing backbone: {PRETRAINED_BACKBONE}"

# %% [markdown]
# ## Cell 5: One-Batch Debug

# %%
from backbones import get_model
from soft_gated_losses import AdaptiveSoftGatedAdaCurricularFaceV2Loss
from train_phase2_kaggle import MarginSoftmaxHead, amp_autocast, make_grad_scaler

BACKBONE = "r18"
BACKBONE_LR = 2e-4
HEAD_LR = 2e-3
BATCH_SIZE = 128
USE_FP16 = True

S = 64.0
M = 0.4
H = 0.333
LAMBDA_WARMUP_EPOCHS = 2.0

DEBUG_LAMBDA_MAX = 0.3
DEBUG_ALPHA_MAX = 0.5
DEBUG_GATE_GAMMA = 5.0
DEBUG_ALPHA_QUALITY_FLOOR = 0.5
DEBUG_BATCH_SIZE = min(8, BATCH_SIZE)

use_amp = bool(USE_FP16 and device.type == "cuda")
debug_backbone = get_model(
    BACKBONE,
    dropout=0.0,
    fp16=use_amp,
    num_features=512,
).to(device)
debug_margin_loss = AdaptiveSoftGatedAdaCurricularFaceV2Loss(
    s=S,
    m=M,
    h=H,
    lambda_max=DEBUG_LAMBDA_MAX,
    alpha_max=DEBUG_ALPHA_MAX,
    gate_gamma=DEBUG_GATE_GAMMA,
    alpha_quality_floor=DEBUG_ALPHA_QUALITY_FLOOR,
    lambda_warmup_epochs=LAMBDA_WARMUP_EPOCHS,
)
debug_margin_loss.set_epoch(1)
debug_head = MarginSoftmaxHead(
    embedding_size=512,
    num_classes=NUM_CLASSES,
    margin_loss=debug_margin_loss,
    fp16=use_amp,
).to(device)
debug_optimizer = torch.optim.SGD(
    [
        {"params": debug_backbone.parameters(), "lr": BACKBONE_LR, "name": "backbone"},
        {"params": debug_head.parameters(), "lr": HEAD_LR, "name": "head"},
    ],
    momentum=0.9,
    weight_decay=5e-4,
)
debug_scaler = make_grad_scaler(use_amp)
debug_images = torch.randn(DEBUG_BATCH_SIZE, 3, 112, 112, device=device)
debug_labels = (torch.arange(DEBUG_BATCH_SIZE, device=device) % NUM_CLASSES).long()

debug_optimizer.zero_grad(set_to_none=True)
with amp_autocast(use_amp):
    debug_embeddings = debug_backbone(debug_images)
    debug_loss, debug_logits, debug_norms = debug_head(debug_embeddings, debug_labels)

assert torch.isfinite(debug_loss).item(), "debug loss is NaN or Inf"
assert debug_logits.shape == (DEBUG_BATCH_SIZE, NUM_CLASSES), debug_logits.shape

if use_amp:
    debug_scaler.scale(debug_loss).backward()
    debug_scaler.step(debug_optimizer)
    debug_scaler.update()
else:
    debug_loss.backward()
    debug_optimizer.step()

stats = debug_margin_loss.last_stats
assert -1.0 <= stats["q_min"] <= 1.0, stats
assert -1.0 <= stats["q_max"] <= 1.0, stats
assert 0.0 <= max(stats["q_min"], 0.0) <= 1.0, stats
assert 0.0 <= max(stats["q_max"], 0.0) <= 1.0, stats
assert 0.0 <= stats["q_pos_mean"] <= 1.0, stats
assert DEBUG_ALPHA_QUALITY_FLOOR <= stats["quality_alpha_min"] <= 1.0, stats
assert DEBUG_ALPHA_QUALITY_FLOOR <= stats["quality_alpha_max"] <= 1.0, stats
assert 0.0 <= stats["lambda_i_mean"] <= DEBUG_LAMBDA_MAX + 1e-6, stats
assert 0.0 <= stats["lambda_i_max"] <= DEBUG_LAMBDA_MAX + 1e-6, stats
assert 0.0 <= stats["D_mean"] <= 1.0, stats
assert 0.0 <= stats["D_max"] <= 1.0, stats
assert 0.0 <= stats["alpha_mean"] <= DEBUG_ALPHA_MAX + 1e-6, stats
assert 0.0 <= stats["alpha_max_actual"] <= DEBUG_ALPHA_MAX + 1e-6, stats

print("debug_loss:", float(debug_loss.detach().cpu().item()))
print("debug_logits_shape:", tuple(debug_logits.shape))
print("debug_norm_mean:", float(debug_norms.detach().mean().cpu().item()))
print("last_stats:", json.dumps(stats, indent=2, sort_keys=True))

del debug_backbone, debug_head, debug_margin_loss, debug_optimizer
del debug_images, debug_labels, debug_embeddings, debug_logits, debug_norms, debug_loss
if device.type == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
# ## Cell 6: Proposed V2 Sweep

# %%
EPOCHS = 10
EVAL_EVERY = 1
SAVE_EVERY_EPOCHS = 1
SAVE_EVERY_STEPS = 300
NUM_WORKERS = 2
MAX_TRAIN_MINUTES = 480
MIN_TRAIN_MINUTES_TO_START = 2
SWEEP_START_TIME = time.time()

PROPOSED2_SWEEP = [
    {
        "lambda_max": 0.3,
        "alpha_max": 0.5,
        "gate_gamma": 5.0,
        "alpha_quality_floor": 0.5,
    },
    {
        "lambda_max": 0.3,
        "alpha_max": 1.0,
        "gate_gamma": 5.0,
        "alpha_quality_floor": 0.5,
    },
    {
        "lambda_max": 0.3,
        "alpha_max": 0.5,
        "gate_gamma": 10.0,
        "alpha_quality_floor": 0.5,
    },
]


def remaining_train_minutes():
    if MAX_TRAIN_MINUTES <= 0:
        return MAX_TRAIN_MINUTES
    elapsed_minutes = (time.time() - SWEEP_START_TIME) / 60.0
    return max(0.0, MAX_TRAIN_MINUTES - elapsed_minutes)


def float_tag(value):
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def exp_dir(config):
    return (
        OUTPUT_ROOT
        / "proposed2_sweep"
        / (
            f"{BACKBONE}_proposed2"
            f"_lmax_{float_tag(config['lambda_max'])}"
            f"_amax_{float_tag(config['alpha_max'])}"
            f"_gamma_{float_tag(config['gate_gamma'])}"
            f"_qfloor_{float_tag(config['alpha_quality_floor'])}"
            f"_blr_{float_tag(BACKBONE_LR)}"
            f"_hlr_{float_tag(HEAD_LR)}"
        )
    )


def is_complete(config):
    metrics_path = exp_dir(config) / "metrics.json"
    if not metrics_path.exists():
        return False
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return len(metrics.get("epochs", [])) >= EPOCHS


print("Cell 6 NUM_CLASSES:", NUM_CLASSES)
assert NUM_CLASSES < 100000, f"Bad NUM_CLASSES={NUM_CLASSES}"
assert NUM_CLASSES in (10572, 10575), f"Unexpected NUM_CLASSES={NUM_CLASSES}"

for config in PROPOSED2_SWEEP:
    if is_complete(config):
        print(f"[SKIP] config={config} complete.")
        continue

    train_minutes_left = remaining_train_minutes()
    if MAX_TRAIN_MINUTES > 0 and train_minutes_left < MIN_TRAIN_MINUTES_TO_START:
        print(
            f"[STOP] sweep time budget exhausted "
            f"({train_minutes_left:.1f} minutes left). Resume next session."
        )
        break

    latest = exp_dir(config) / "latest.pt"
    cmd = [
        sys.executable,
        "train_soft_gated_lambda_kaggle.py",
        "--loss",
        "adaptive_soft_gated_ada_curricular_v2",
        "--network",
        BACKBONE,
        "--s",
        str(S),
        "--m",
        str(M),
        "--h",
        str(H),
        "--lambda_max",
        str(config["lambda_max"]),
        "--alpha_max",
        str(config["alpha_max"]),
        "--gate_gamma",
        str(config["gate_gamma"]),
        "--alpha_quality_floor",
        str(config["alpha_quality_floor"]),
        "--lambda_warmup_epochs",
        str(LAMBDA_WARMUP_EPOCHS),
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
        str(HEAD_LR),
        "--backbone_lr",
        str(BACKBONE_LR),
        "--head_lr",
        str(HEAD_LR),
        "--warmup_epochs",
        "1.0",
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
        "[BUDGET] "
        f"lambda_max={config['lambda_max']} alpha_max={config['alpha_max']} "
        f"gate_gamma={config['gate_gamma']} "
        f"alpha_quality_floor={config['alpha_quality_floor']} "
        f"remaining_sweep_train_minutes={train_minutes_left:.1f}"
    )

    if latest.exists():
        print(f"[RESUME] config={config} from {latest}")
        cmd.append("--resume")
    else:
        print(f"[START] config={config} from pretrained backbone")
        cmd.extend(["--pretrained_backbone", str(PRETRAINED_BACKBONE)])

    run(cmd, cwd=ARCFACE_DIR)

    if not is_complete(config):
        print(f"[STOP] config={config} is not complete yet. Resume next session.")
        break

print("Done. Proposed2 sweep:", PROPOSED2_SWEEP)
print("Learning rates: backbone_lr=", BACKBONE_LR, "head_lr=", HEAD_LR)

# %% [markdown]
# ## Cell 7: Progress And Best Scores

# %%
def complete_accuracy_mean(eval_metrics, targets):
    values = []
    for target in targets:
        item = eval_metrics.get(target)
        if item is None or "accuracy" not in item:
            return None
        values.append(float(item["accuracy"]))
    return float(sum(values) / len(values)) if values else None


def select_epoch_score(epoch_record):
    group_eval = epoch_record.get("group_eval", {}) or {}
    evals = epoch_record.get("eval", {}) or {}
    if "HQ_Avg" in group_eval:
        return group_eval["HQ_Avg"], "HQ_Avg"
    eval7 = group_eval.get("Eval7_Avg")
    if eval7 is None:
        eval7 = complete_accuracy_mean(evals, EVAL_TARGETS)
    if eval7 is not None:
        return eval7, "Eval7_Avg"
    if evals:
        values = [float(item["accuracy"]) for item in evals.values() if "accuracy" in item]
        if values:
            return float(sum(values) / len(values)), "mean_validation_accuracy"
    return None, None


root = OUTPUT_ROOT / "proposed2_sweep"
if not root.exists():
    print("No proposed2_sweep folder yet:", root)
else:
    for exp in sorted(root.glob("r18_proposed2_*")):
        metrics_path = exp / "metrics.json"
        latest = exp / "latest.pt"
        best = exp / "best.pth"

        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            epochs = metrics.get("epochs", [])
            best_epoch = metrics.get("best_epoch")
            best_score = metrics.get("best_score")
            best_metric = metrics.get("best_metric")
            if best_score is None:
                selected = [
                    (ep.get("epoch"),) + select_epoch_score(ep)
                    for ep in epochs
                    if select_epoch_score(ep)[0] is not None
                ]
                if selected:
                    best_epoch, best_score, best_metric = max(
                        selected,
                        key=lambda item: item[1],
                    )
        else:
            epochs = []
            best_epoch = None
            best_score = None
            best_metric = None

        print(exp.name)
        print("  epochs:", len(epochs))
        print("  latest:", latest.exists())
        print("  best:", best.exists())
        print("  best_epoch:", best_epoch)
        print("  best_metric:", best_metric)
        print("  best_score:", best_score)

# %% [markdown]
# ## Cell 8: Export Eval By Epoch

# %%
import pandas as pd

rows = []
for metrics_path in sorted((OUTPUT_ROOT / "proposed2_sweep").glob("r18_proposed2_*/metrics.json")):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    exp_name = metrics_path.parent.name

    for ep in metrics.get("epochs", []):
        evals = ep.get("eval", {}) or {}
        group_eval = ep.get("group_eval", {}) or {}
        row = {
            "experiment": exp_name,
            "epoch": ep.get("epoch"),
            "loss": ep.get("loss"),
            "mean_norm": ep.get("mean_norm"),
            "lr": ep.get("lr"),
            "backbone_lr": ep.get("backbone_lr"),
            "head_lr": ep.get("head_lr"),
            "lambda_max": ep.get("lambda_max"),
            "alpha_max": ep.get("alpha_max"),
            "gate_gamma": ep.get("gate_gamma"),
            "alpha_quality_floor": ep.get("alpha_quality_floor"),
            "lambda_warmup_epochs": ep.get("lambda_warmup_epochs"),
            "HQ_Avg": group_eval.get(
                "HQ_Avg",
                complete_accuracy_mean(evals, HQ_EVAL_TARGETS),
            ),
            "LQ_Avg": group_eval.get(
                "LQ_Avg",
                complete_accuracy_mean(evals, LQ_EVAL_TARGETS),
            ),
            "Eval7_Avg": group_eval.get(
                "Eval7_Avg",
                complete_accuracy_mean(evals, EVAL_TARGETS),
            ),
        }
        for key in (
            "q_mean",
            "q_std",
            "q_min",
            "q_max",
            "q_pos_mean",
            "alpha_quality_floor",
            "quality_alpha_mean",
            "quality_alpha_min",
            "quality_alpha_max",
            "lambda_i_mean",
            "lambda_i_max",
            "u_pos_mean",
            "arc_anchor_mean",
            "tau_mean",
            "D_mean",
            "D_max",
            "alpha_mean",
            "alpha_max_actual",
            "soft_hard_ratio",
            "effective_mod_ratio",
            "curricular_t",
        ):
            row[key] = ep.get(key)
        for name, item in evals.items():
            row[name] = item.get("accuracy")
            row[f"{name}_std"] = item.get("std")
            row[f"{name}_xnorm"] = item.get("xnorm")
        rows.append(row)

df = pd.DataFrame(rows)
try:
    from IPython.display import display

    display(df)
except Exception:
    print(df)

out_csv = "/kaggle/working/proposed2_eval_by_epoch.csv"
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)

try:
    from IPython.display import FileLink, display

    display(FileLink(out_csv))
except Exception as exc:
    print("Could not render download link:", exc)

# %% [markdown]
# ## Cell 9: Backup

# %%
zip_base = "/kaggle/working/proposed2_sweep"
zip_path = Path(zip_base + ".zip")
if zip_path.exists():
    zip_path.unlink()

root = OUTPUT_ROOT / "proposed2_sweep"
if root.exists():
    shutil.make_archive(zip_base, "zip", str(OUTPUT_ROOT), "proposed2_sweep")
    csv_path = Path("/kaggle/working/proposed2_eval_by_epoch.csv")
    if csv_path.exists():
        with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, csv_path.name)
        print("Included CSV:", csv_path)
    else:
        print("CSV not found. Run Cell 8 first:", csv_path)
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
    print("No proposed2 outputs yet:", root)
