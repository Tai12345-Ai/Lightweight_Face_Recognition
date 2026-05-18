# %% [markdown]
# # Proposed 3 Competition-Aware AdaFace Runner
#
# Standalone Kaggle runner for:
# `competition_aware_adaface`
#
# It trains on:
# `/kaggle/input/CASIA-WebFace/casia-webface`
#
# It evaluates on .bin files under:
# `/kaggle/input/CASIA-WebFace/eval`

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


def float_tag(value):
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


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
        PRETRAINED_BACKBONE = pth_candidates[0]
else:
    print("Using expected backbone:", PRETRAINED_BACKBONE)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("TRAIN_DATA_DIR:", TRAIN_DATA_DIR)
print("EVAL_DIR:", EVAL_DIR)
print("PRETRAINED_BACKBONE:", PRETRAINED_BACKBONE)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("NUM_CLASSES:", NUM_CLASSES)

# %% [markdown]
# ## Restore Previous Outputs

# %%
SWEEP_ROOT = OUTPUT_ROOT / "proposed3_competition_aware_adaface"
restored = False

for zip_candidate in sorted(Path("/kaggle/input").rglob("proposed3_competition_aware_adaface.zip")):
    print("Restoring previous proposed3 zip:", zip_candidate)
    with zipfile.ZipFile(zip_candidate, "r") as f:
        f.extractall(OUTPUT_ROOT)
    restored = True
    break

if not restored:
    for folder_candidate in sorted(Path("/kaggle/input").rglob("proposed3_competition_aware_adaface")):
        if folder_candidate.is_dir():
            print("Restoring previous proposed3 folder:", folder_candidate)
            shutil.copytree(folder_candidate, SWEEP_ROOT, dirs_exist_ok=True)
            restored = True
            break

print("Restored previous proposed3 outputs to:" if restored else "No previous proposed3 output input found.", SWEEP_ROOT)

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
    "kaggle_proposed3_runner.py",
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
# ## One-Batch Debug

# %%
from backbones import get_model
from soft_gated_losses import CompetitionAwareAdaFaceLoss
from train_phase2_kaggle import MarginSoftmaxHead, amp_autocast, make_grad_scaler

BACKBONE = "r18"
BACKBONE_LR = 2e-4
HEAD_LR = 2e-3
BATCH_SIZE = 128
USE_FP16 = True

S = 64.0
M = 0.4
H = 0.333
DEBUG_BATCH_SIZE = min(8, BATCH_SIZE)

use_amp = bool(USE_FP16 and device.type == "cuda")
debug_backbone = get_model(
    BACKBONE,
    dropout=0.0,
    fp16=use_amp,
    num_features=512,
).to(device)
debug_margin_loss = CompetitionAwareAdaFaceLoss(s=S, m=M, h=H)
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
assert 0.0 <= stats["d_mean"] <= 1.0, stats
assert 0.0 <= stats["d_max"] <= 1.0, stats
assert -1.0 <= stats["q_star_min"] <= 1.0, stats
assert -1.0 <= stats["q_star_max"] <= 1.0, stats

print("debug_loss:", float(debug_loss.detach().cpu().item()))
print("debug_logits_shape:", tuple(debug_logits.shape))
print("debug_norm_mean:", float(debug_norms.detach().mean().cpu().item()))
print("last_stats:", json.dumps(stats, indent=2, sort_keys=True))

del debug_backbone, debug_head, debug_margin_loss, debug_optimizer
del debug_images, debug_labels, debug_embeddings, debug_logits, debug_norms, debug_loss
if device.type == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
# ## Train Proposed 3

# %%
LOSS_NAME = "competition_aware_adaface"
EPOCHS = 10
EVAL_EVERY = 1
SAVE_EVERY_EPOCHS = 1
SAVE_EVERY_STEPS = 300
NUM_WORKERS = 2
MAX_TRAIN_MINUTES = 480
MIN_TRAIN_MINUTES_TO_START = 2
RUN_START_TIME = time.time()


def exp_dir():
    return (
        OUTPUT_ROOT
        / "proposed3_competition_aware_adaface"
        / (
            f"{BACKBONE}_proposed3_competition_aware_adaface"
            f"_blr_{float_tag(BACKBONE_LR)}"
            f"_hlr_{float_tag(HEAD_LR)}"
        )
    )


def is_complete():
    metrics_path = exp_dir() / "metrics.json"
    if not metrics_path.exists():
        return False
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return len(metrics.get("epochs", [])) >= EPOCHS


def remaining_train_minutes():
    if MAX_TRAIN_MINUTES <= 0:
        return MAX_TRAIN_MINUTES
    elapsed_minutes = (time.time() - RUN_START_TIME) / 60.0
    return max(0.0, MAX_TRAIN_MINUTES - elapsed_minutes)


print("Cell train NUM_CLASSES:", NUM_CLASSES)
assert NUM_CLASSES < 100000, f"Bad NUM_CLASSES={NUM_CLASSES}"
assert NUM_CLASSES in (10572, 10575), f"Unexpected NUM_CLASSES={NUM_CLASSES}"

if is_complete():
    print("[SKIP] proposed3 complete:", exp_dir())
else:
    train_minutes_left = remaining_train_minutes()
    assert (
        MAX_TRAIN_MINUTES <= 0 or train_minutes_left >= MIN_TRAIN_MINUTES_TO_START
    ), f"Not enough train time left: {train_minutes_left:.1f} minutes"

    latest = exp_dir() / "latest.pt"
    cmd = [
        sys.executable,
        "train_soft_gated_lambda_kaggle.py",
        "--loss",
        LOSS_NAME,
        "--network",
        BACKBONE,
        "--s",
        str(S),
        "--m",
        str(M),
        "--h",
        str(H),
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
        f"loss={LOSS_NAME} backbone_lr={BACKBONE_LR} head_lr={HEAD_LR} "
        f"remaining_train_minutes={train_minutes_left:.1f}"
    )

    if latest.exists():
        print(f"[RESUME] proposed3 from {latest}")
        cmd.append("--resume")
    else:
        print("[START] proposed3 from pretrained backbone")
        cmd.extend(["--pretrained_backbone", str(PRETRAINED_BACKBONE)])

    run(cmd, cwd=ARCFACE_DIR)

print("Done. Proposed3 dir:", exp_dir())

# %% [markdown]
# ## Progress And Best Scores

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
    hq_avg = complete_accuracy_mean(evals, HQ_EVAL_TARGETS)
    if hq_avg is not None:
        return hq_avg, "HQ_Avg"
    eval7 = group_eval.get("Eval7_Avg")
    if eval7 is None:
        eval7 = complete_accuracy_mean(evals, EVAL_TARGETS)
    if eval7 is not None:
        return eval7, "Eval7_Avg"
    return None, None


metrics_path = exp_dir() / "metrics.json"
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
            best_epoch, best_score, best_metric = max(selected, key=lambda item: item[1])
    print(exp_dir().name)
    print("  epochs:", len(epochs))
    print("  latest:", (exp_dir() / "latest.pt").exists())
    print("  best:", (exp_dir() / "best.pth").exists())
    print("  best_epoch:", best_epoch)
    print("  best_metric:", best_metric)
    print("  best_score:", best_score)
else:
    print("No metrics yet:", metrics_path)

# %% [markdown]
# ## Export Eval By Epoch

# %%
try:
    import pandas as pd

    rows = []
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        for ep in metrics.get("epochs", []):
            evals = ep.get("eval", {}) or {}
            group_eval = ep.get("group_eval", {}) or {}
            row = {
                "experiment": exp_dir().name,
                "epoch": ep.get("epoch"),
                "loss": ep.get("loss"),
                "mean_norm": ep.get("mean_norm"),
                "lr": ep.get("lr"),
                "backbone_lr": ep.get("backbone_lr"),
                "head_lr": ep.get("head_lr"),
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
                "d_mean",
                "d_max",
                "q_star_mean",
                "q_star_std",
                "q_star_min",
                "q_star_max",
                "c_minus_mean",
                "arc_anchor_mean",
                "u_pos_star_mean",
                "competition_active_ratio",
                "high_quality_hard_ratio",
                "low_quality_hard_ratio",
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

    out_csv = "/kaggle/working/proposed3_eval_by_epoch.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
except Exception as exc:
    print("Could not export proposed3 CSV:", exc)

# %% [markdown]
# ## Backup

# %%
zip_base = "/kaggle/working/proposed3_competition_aware_adaface"
zip_path = Path(zip_base + ".zip")
if zip_path.exists():
    zip_path.unlink()

root = OUTPUT_ROOT / "proposed3_competition_aware_adaface"
if root.exists():
    shutil.make_archive(zip_base, "zip", str(OUTPUT_ROOT), "proposed3_competition_aware_adaface")
    csv_path = Path("/kaggle/working/proposed3_eval_by_epoch.csv")
    if csv_path.exists():
        with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, csv_path.name)
        print("Included CSV:", csv_path)
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
    print("No proposed3 outputs yet:", root)
