# %% [markdown]
# # Proposed 4 Competition-Adaptive Soft-Gated Ada-CurricularFace Runner
#
# Standalone Kaggle runner for:
# `competition_adaptive_soft_gated_ada_curricular`
#
# It trains on:
# `/kaggle/input/CASIA-WebFace/casia-webface`
#
# It resumes from previous outputs when a Kaggle dataset contains:
# `/kaggle/input/proposed/proposed4_competition_adaptive`

# %%
from pathlib import Path
import json
import math
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

INPUT_ROOT = Path("/kaggle/input")
TRAIN_DATA_DIR = INPUT_ROOT / "CASIA-WebFace" / "casia-webface"
EVAL_DIR = INPUT_ROOT / "CASIA-WebFace" / "eval"
PRETRAINED_BACKBONE = INPUT_ROOT / "backbone" / "backbone.pth"
OUTPUT_ROOT = Path("/kaggle/working/experiments")
SWEEP_FOLDER = "proposed4_competition_adaptive"

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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def run(cmd, cwd=None, check=True):
    print("+", " ".join(str(x) for x in cmd))
    return subprocess.run([str(x) for x in cmd], cwd=cwd, check=check)


def unique_existing(paths):
    seen = set()
    for path in paths:
        path = Path(path)
        if path in seen or not path.exists():
            continue
        seen.add(path)
        yield path


def input_dirs(max_depth=2):
    if not INPUT_ROOT.exists():
        return []

    dirs = []

    def visit(path, depth):
        if depth > max_depth or not path.is_dir():
            return
        dirs.append(path)
        if depth == max_depth or (depth > 0 and has_train_data(path)):
            return
        try:
            children = sorted(child for child in path.iterdir() if child.is_dir())
        except OSError:
            return
        for child in children:
            visit(child, depth + 1)

    visit(INPUT_ROOT, 0)
    return dirs


def has_recordio_train_data(path):
    path = Path(path)
    return (path / "train.rec").exists() and (path / "train.idx").exists()


def has_imagefolder_train_data(path):
    path = Path(path)
    if not path.is_dir() or has_recordio_train_data(path):
        return False
    try:
        class_dirs = [child for child in path.iterdir() if child.is_dir()]
    except OSError:
        return False
    if len(class_dirs) < 2:
        return False
    for class_dir in class_dirs[:20]:
        try:
            has_image = any(
                item.is_file() and item.suffix.lower() in IMAGE_EXTS
                for item in class_dir.iterdir()
            )
            if has_image:
                return True
        except OSError:
            continue
    return False


def has_train_data(path):
    return has_recordio_train_data(path) or has_imagefolder_train_data(path)


def normalize_train_data_dir(path):
    path = Path(path)
    for name in ("webface_112x112", "casia-webface"):
        nested_train = path / name
        if has_train_data(nested_train):
            print("Detected nested train layout. Using train folder:", nested_train)
            return nested_train
    return path


def resolve_train_data_dir(preferred):
    preferred_paths = [
        preferred,
        INPUT_ROOT / "CASIA-WebFace" / "casia-webface",
        INPUT_ROOT / "webface-112x112" / "webface_112x112",
        INPUT_ROOT / "WebFace 112x112" / "webface_112x112",
    ]

    candidates = []
    for path in preferred_paths:
        candidates.append(path)
        candidates.append(Path(path).parent)
    candidates.extend(input_dirs(max_depth=2))

    for candidate in unique_existing(candidates):
        normalized = normalize_train_data_dir(candidate)
        if has_train_data(normalized):
            return normalized

    raise FileNotFoundError(
        "Could not find WebFace/CASIA train data under /kaggle/input. "
        "Expected a folder containing train.rec/train.idx or ImageFolder class folders."
    )


def resolve_eval_dir(train_dir):
    candidates = [
        Path(train_dir) / "eval",
        Path(train_dir).parent / "eval",
        INPUT_ROOT / "webface-112x112" / "eval",
        INPUT_ROOT / "CASIA-WebFace" / "eval",
    ]
    candidates.extend(input_dirs(max_depth=2))

    for candidate in unique_existing(candidates):
        if any((candidate / f"{name}.bin").exists() for name in EVAL_TARGETS):
            return candidate
    return None


def input_file_candidates(filename, max_depth=2):
    for directory in input_dirs(max_depth=max_depth):
        candidate = directory / filename
        if candidate.exists():
            yield candidate


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


def complete_accuracy_mean(eval_metrics, targets):
    values = []
    for target in targets:
        item = eval_metrics.get(target)
        if item is None or "accuracy" not in item:
            return None
        values.append(float(item["accuracy"]))
    return float(sum(values) / len(values)) if values else None


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
TRAIN_DATA_DIR = resolve_train_data_dir(TRAIN_DATA_DIR)
NUM_CLASSES = detect_num_classes(TRAIN_DATA_DIR)

EVAL_DIR = resolve_eval_dir(TRAIN_DATA_DIR)
if EVAL_DIR is None:
    warnings.warn(
        "No verification .bin folder found under /kaggle/input. "
        "Training will resume without per-epoch eval.",
        RuntimeWarning,
    )
    EVAL_TARGETS = []
else:
    available_eval = [name for name in EVAL_TARGETS if (EVAL_DIR / f"{name}.bin").exists()]
    missing_eval = [name for name in EVAL_TARGETS if name not in available_eval]
    if missing_eval:
        warnings.warn(
            f"Missing eval bins in {EVAL_DIR}: {missing_eval}. "
            f"Using available targets only: {available_eval}",
            RuntimeWarning,
        )
    EVAL_TARGETS = available_eval

if not PRETRAINED_BACKBONE.exists():
    backbone_candidates = sorted(input_file_candidates("backbone.pth", max_depth=2))
    if backbone_candidates:
        if len(backbone_candidates) > 1:
            candidate_list = "\n".join(str(p) for p in backbone_candidates)
            msg = f"Multiple backbone.pth files found. Using {backbone_candidates[0]}."
            warnings.warn(msg + "\nCandidates:\n" + candidate_list, RuntimeWarning)
        PRETRAINED_BACKBONE = backbone_candidates[0]
    else:
        pth_candidates = sorted(
            candidate
            for directory in input_dirs(max_depth=2)
            for candidate in directory.glob("*.pth")
        )
        if not pth_candidates:
            raise FileNotFoundError("No pretrained backbone .pth found under /kaggle/input")
        if len(pth_candidates) > 1:
            candidate_list = "\n".join(str(p) for p in pth_candidates)
            raise RuntimeError(
                "Multiple .pth files found, refusing to choose one silently.\nCandidates:\n"
                + candidate_list
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
print("EVAL_TARGETS:", ",".join(EVAL_TARGETS) if EVAL_TARGETS else "(none)")

# %% [markdown]
# ## Restore Previous Outputs

# %%
SWEEP_ROOT = OUTPUT_ROOT / SWEEP_FOLDER
restored = False

zip_candidates = sorted(
    unique_existing(
        [INPUT_ROOT / "proposed" / f"{SWEEP_FOLDER}.zip", INPUT_ROOT / f"{SWEEP_FOLDER}.zip"]
        + [directory / f"{SWEEP_FOLDER}.zip" for directory in input_dirs(max_depth=1)]
    )
)
folder_candidates = sorted(
    unique_existing(
        [INPUT_ROOT / "proposed" / SWEEP_FOLDER, INPUT_ROOT / SWEEP_FOLDER]
        + [directory / SWEEP_FOLDER for directory in input_dirs(max_depth=1)]
    )
)

for zip_candidate in zip_candidates:
    print("Restoring previous proposed4 zip:", zip_candidate)
    with zipfile.ZipFile(zip_candidate, "r") as f:
        f.extractall(OUTPUT_ROOT)
    restored = True
    break

if not restored:
    for folder_candidate in folder_candidates:
        print("Restoring previous proposed4 folder:", folder_candidate)
        shutil.copytree(folder_candidate, SWEEP_ROOT, dirs_exist_ok=True)
        restored = True
        break

for csv_candidate in input_file_candidates("proposed4_eval_by_epoch.csv", max_depth=1):
    csv_dst = Path("/kaggle/working/proposed4_eval_by_epoch.csv")
    shutil.copy2(csv_candidate, csv_dst)
    print("Restored previous proposed4 CSV:", csv_candidate, "->", csv_dst)
    break

print("Restored previous proposed4 outputs to:" if restored else "No previous proposed4 output input found.", SWEEP_ROOT)

# %% [markdown]
# ## Preflight

# %%
compile_files = [
    "degradation/transforms.py",
    "eval_degraded_6phase2.py",
    "soft_gated_losses.py",
    "train_soft_gated_lambda_kaggle.py",
    "train_phase2_kaggle.py",
]
if (ARCFACE_DIR / "kaggle_proposed4_runner.py").exists():
    compile_files.append("kaggle_proposed4_runner.py")

run([sys.executable, "-m", "py_compile"] + compile_files, cwd=ARCFACE_DIR)

import torch

print("TRAIN_DATA_DIR:", TRAIN_DATA_DIR)
print("EVAL_DIR:", EVAL_DIR)
print("PRETRAINED_BACKBONE:", PRETRAINED_BACKBONE)
print("EVAL_TARGETS:", ",".join(EVAL_TARGETS) if EVAL_TARGETS else "(none)")

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
from soft_gated_losses import CompetitionAdaptiveSoftGatedAdaCurricularFaceLoss
from train_phase2_kaggle import MarginSoftmaxHead, amp_autocast, make_grad_scaler

BACKBONE = "r18"
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
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
debug_margin_loss = CompetitionAdaptiveSoftGatedAdaCurricularFaceLoss(s=S, m=M, h=H)
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
assert 0.0 <= stats["lambda_i_mean"] <= H + 1e-6, stats
assert 0.0 <= stats["lambda_i_max"] <= H + 1e-6, stats
assert math.isfinite(stats["tau_mean"]), stats
assert 0.0 <= stats["hard_negative_ratio"] <= 1.0, stats

print("debug_loss:", float(debug_loss.detach().cpu().item()))
print("debug_logits_shape:", tuple(debug_logits.shape))
print("debug_norm_mean:", float(debug_norms.detach().mean().cpu().item()))
print("last_stats:", json.dumps(stats, indent=2, sort_keys=True))

del debug_backbone, debug_head, debug_margin_loss, debug_optimizer
del debug_images, debug_labels, debug_embeddings, debug_logits, debug_norms, debug_loss
if device.type == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
# ## Train Proposed 4

# %%
LOSS_NAME = "competition_adaptive_soft_gated_ada_curricular"
EPOCHS = 10
EVAL_EVERY = 1
SAVE_EVERY_EPOCHS = 1
SAVE_EVERY_STEPS = 300
NUM_WORKERS = 2
MAX_TRAIN_MINUTES = 600
MIN_TRAIN_MINUTES_TO_START = 2
RUN_OPTIONAL_2X_LR = False
RUN_START_TIME = time.time()
RUN_DEGRADED_EVAL = True
DEGRADED_TARGETS = ["lfw", "cfp_fp", "cplfw", "agedb_30", "calfw"]
DEGRADED_DEGRADATIONS = [
    "gaussian_blur",
    "motion_blur",
    "low_resolution",
    "jpeg_compression",
    "low_illumination",
    "alignment_perturb",
]
DEGRADED_SEVERITIES = "3"
DEGRADED_BATCH_SIZE = 128

TRAIN_CONFIGS = [
    {"backbone_lr": 1e-4, "head_lr": 1e-3},
]
if RUN_OPTIONAL_2X_LR:
    TRAIN_CONFIGS.append({"backbone_lr": 2e-4, "head_lr": 2e-3})


def exp_dir(backbone_lr, head_lr):
    return (
        OUTPUT_ROOT
        / SWEEP_FOLDER
        / (
            f"{BACKBONE}_proposed4_comp_adaptive_soft_gated_ada_curricular"
            f"_blr_{float_tag(backbone_lr)}"
            f"_hlr_{float_tag(head_lr)}"
        )
    )


def is_complete(backbone_lr, head_lr):
    metrics_path = exp_dir(backbone_lr, head_lr) / "metrics.json"
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


def run_degraded_eval(current_exp_dir):
    if not RUN_DEGRADED_EVAL:
        return
    if EVAL_DIR is None:
        warnings.warn("No EVAL_DIR found; skipping degraded eval.", RuntimeWarning)
        return

    best_checkpoint = Path(current_exp_dir) / "best.pth"
    if not best_checkpoint.exists():
        warnings.warn(
            f"Missing best.pth for degraded eval: {best_checkpoint}. Skipping.",
            RuntimeWarning,
        )
        return

    degraded_cmd = [
        sys.executable,
        "eval_degraded_6phase2.py",
        "--backbone",
        BACKBONE,
        "--checkpoint",
        str(best_checkpoint),
        "--checkpoint-label",
        Path(current_exp_dir).name,
        "--data-dir",
        str(EVAL_DIR),
        "--output",
        str(Path(current_exp_dir) / "degraded_eval"),
        "--targets",
        ",".join(DEGRADED_TARGETS),
        "--degradations",
        ",".join(DEGRADED_DEGRADATIONS),
        "--severities",
        DEGRADED_SEVERITIES,
        "--batch-size",
        str(DEGRADED_BATCH_SIZE),
        "--device",
        device.type,
    ]
    if USE_FP16:
        degraded_cmd.append("--fp16")

    run(degraded_cmd, cwd=ARCFACE_DIR)


print("Cell train NUM_CLASSES:", NUM_CLASSES)
assert NUM_CLASSES < 100000, f"Bad NUM_CLASSES={NUM_CLASSES}"
assert NUM_CLASSES in (10572, 10575), f"Unexpected NUM_CLASSES={NUM_CLASSES}"

for cfg in TRAIN_CONFIGS:
    backbone_lr = cfg["backbone_lr"]
    head_lr = cfg["head_lr"]
    current_exp_dir = exp_dir(backbone_lr, head_lr)

    if is_complete(backbone_lr, head_lr):
        print("[SKIP] proposed4 complete:", current_exp_dir)
        run_degraded_eval(current_exp_dir)
        continue

    train_minutes_left = remaining_train_minutes()
    assert (
        MAX_TRAIN_MINUTES <= 0 or train_minutes_left >= MIN_TRAIN_MINUTES_TO_START
    ), f"Not enough train time left: {train_minutes_left:.1f} minutes"

    latest = current_exp_dir / "latest.pt"
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
        "--output_dir",
        str(OUTPUT_ROOT),
        "--epochs",
        str(EPOCHS),
        "--batch_size",
        str(BATCH_SIZE),
        "--lr",
        str(head_lr),
        "--backbone_lr",
        str(backbone_lr),
        "--head_lr",
        str(head_lr),
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
    if EVAL_DIR is not None:
        cmd.extend(["--eval_dir", str(EVAL_DIR)])
    if USE_FP16:
        cmd.append("--fp16")

    print(
        "[BUDGET] "
        f"loss={LOSS_NAME} backbone_lr={backbone_lr} head_lr={head_lr} "
        f"remaining_train_minutes={train_minutes_left:.1f}"
    )

    if latest.exists():
        print(f"[RESUME] proposed4 from {latest}")
        cmd.append("--resume")
    else:
        print("[START] proposed4 from pretrained backbone")
        cmd.extend(["--pretrained_backbone", str(PRETRAINED_BACKBONE)])

    run(cmd, cwd=ARCFACE_DIR)
    if is_complete(backbone_lr, head_lr):
        run_degraded_eval(current_exp_dir)
    else:
        print("[SKIP] degraded eval because proposed4 is not complete yet:", current_exp_dir)

print("Done. Proposed4 root:", SWEEP_ROOT)

# %% [markdown]
# ## Progress And Best Scores

# %%
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


for metrics_path in sorted(SWEEP_ROOT.glob("*/metrics.json")):
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
    print(metrics_path.parent.name)
    print("  epochs:", len(epochs))
    print("  latest:", (metrics_path.parent / "latest.pt").exists())
    print("  best:", (metrics_path.parent / "best.pth").exists())
    print("  best_epoch:", best_epoch)
    print("  best_metric:", best_metric)
    print("  best_score:", best_score)

# %% [markdown]
# ## Export Eval By Epoch

# %%
try:
    import pandas as pd

    rows = []
    for metrics_path in sorted(SWEEP_ROOT.glob("*/metrics.json")):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        for ep in metrics.get("epochs", []):
            evals = ep.get("eval", {}) or {}
            group_eval = ep.get("group_eval", {}) or {}
            row = {
                "experiment": metrics_path.parent.name,
                "epoch": ep.get("epoch"),
                "loss": ep.get("loss"),
                "mean_norm": ep.get("mean_norm"),
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
            for name in EVAL_TARGETS:
                item = evals.get(name, {}) or {}
                row[name] = item.get("accuracy")
            for key in (
                "q_mean",
                "q_std",
                "q_min",
                "q_max",
                "u_pos_mean",
                "arc_anchor_mean",
                "c_minus_mean",
                "d_mean",
                "d_max",
                "lambda_i_mean",
                "lambda_i_max",
                "tau_mean",
                "hard_negative_ratio",
                "competition_active_ratio",
                "curricular_t",
            ):
                row[key] = ep.get(key)
            rows.append(row)

    columns = [
        "experiment",
        "epoch",
        "loss",
        "mean_norm",
        "backbone_lr",
        "head_lr",
        "HQ_Avg",
        "LQ_Avg",
        "Eval7_Avg",
        "lfw",
        "cfp_fp",
        "cplfw",
        "agedb_30",
        "calfw",
        "sllfw",
        "talfw",
        "q_mean",
        "q_std",
        "q_min",
        "q_max",
        "u_pos_mean",
        "arc_anchor_mean",
        "c_minus_mean",
        "d_mean",
        "d_max",
        "lambda_i_mean",
        "lambda_i_max",
        "tau_mean",
        "hard_negative_ratio",
        "competition_active_ratio",
        "curricular_t",
    ]
    df = pd.DataFrame(rows, columns=columns)
    try:
        from IPython.display import display

        display(df)
    except Exception:
        print(df)

    out_csv = "/kaggle/working/proposed4_eval_by_epoch.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
except Exception as exc:
    print("Could not export proposed4 CSV:", exc)

# %% [markdown]
# ## Plot Eval Metrics

# %%
plot_path = Path("/kaggle/working/proposed4_eval_plot.png")
try:
    import matplotlib.pyplot as plt

    if "df" not in globals():
        raise RuntimeError("No dataframe named df. Run the export cell first.")

    plot_df = df.copy()
    plot_df["epoch"] = pd.to_numeric(plot_df["epoch"], errors="coerce")
    for metric in ("HQ_Avg", "LQ_Avg", "Eval7_Avg"):
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 5))
    for experiment, group in plot_df.groupby("experiment"):
        group = group.sort_values("epoch")
        for metric in ("HQ_Avg", "LQ_Avg", "Eval7_Avg"):
            metric_group = group.dropna(subset=["epoch", metric])
            if not metric_group.empty:
                ax.plot(
                    metric_group["epoch"],
                    metric_group[metric],
                    marker="o",
                    label=f"{experiment} {metric}",
                )

    ax.set_title("Proposed4 Eval Metrics by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    try:
        from IPython.display import display

        display(fig)
    except Exception:
        print("Plot display is unavailable; saved plot only.")
    plt.close(fig)
    print("Saved plot:", plot_path)
except Exception as exc:
    print("Could not render proposed4 plot:", exc)

# %% [markdown]
# ## Download Plot

# %%
plot_path = Path("/kaggle/working/proposed4_eval_plot.png")
if plot_path.exists():
    try:
        from IPython.display import FileLink, display

        print("Download plot:")
        display(FileLink(str(plot_path)))
    except Exception as exc:
        print("Could not render plot download link:", exc)
        print("Plot download path:", plot_path)
else:
    print("No plot found yet:", plot_path)

# %% [markdown]
# ## Backup

# %%
zip_base = f"/kaggle/working/{SWEEP_FOLDER}"
zip_path = Path(zip_base + ".zip")
if zip_path.exists():
    zip_path.unlink()

if SWEEP_ROOT.exists():
    shutil.make_archive(zip_base, "zip", str(OUTPUT_ROOT), SWEEP_FOLDER)
    csv_path = Path("/kaggle/working/proposed4_eval_by_epoch.csv")
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
    print("No proposed4 outputs yet:", SWEEP_ROOT)
