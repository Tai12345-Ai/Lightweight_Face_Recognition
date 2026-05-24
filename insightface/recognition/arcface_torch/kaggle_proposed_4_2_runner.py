# %% [markdown]
# # Proposed 4.2 UI-Aware Recognizability Runner
#
# Standalone Kaggle runner for:
# `ui_aware_competition_quality_adaptive_soft_gated_ada_curricular`
#
# It trains on CASIA-WebFace, evaluates the normal verification bins per epoch,
# then runs shared degraded evaluation on the five high-quality eval bins:
# LFW, CFP-FP, CPLFW, AgeDB-30, CALFW.

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
SWEEP_FOLDER = "proposed4_2_ui_aware"

BACKBONE = "r18"
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
BATCH_SIZE = 128
USE_FP16 = True

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
UI_CENTER_MOMENTUM = 0.99
UI_CENTER_UPDATE_INTERVAL = 20

LOSS_NAME = "ui_aware_competition_quality_adaptive_soft_gated_ada_curricular"
EPOCHS = 10
EVAL_EVERY = 1
SAVE_EVERY_EPOCHS = 1
SAVE_EVERY_STEPS = 300
NUM_WORKERS = 2
MAX_TRAIN_MINUTES = 600
MIN_TRAIN_MINUTES_TO_START = 2
RUN_START_TIME = time.time()

EVAL_TARGETS = ["lfw", "cfp_fp", "cplfw", "agedb_30", "calfw", "sllfw", "talfw"]
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
            if any(
                item.is_file() and item.suffix.lower() in IMAGE_EXTS
                for item in class_dir.iterdir()
            ):
                return True
        except OSError:
            continue
    return False


def has_train_data(path):
    return has_recordio_train_data(path) or has_imagefolder_train_data(path)


def looks_like_casia_path(path):
    text = "/".join(part.lower() for part in Path(path).parts)
    return "casia" in text and "webface" in text


def normalize_train_data_dir(path):
    path = Path(path)
    for child in ("casia-webface", "CASIA-WebFace", "webface_112x112"):
        nested = path / child
        if has_train_data(nested):
            print("Detected nested train layout. Using train folder:", nested)
            return nested
    return path


def resolve_train_data_dir(preferred):
    candidates = [
        preferred,
        INPUT_ROOT / "CASIA-WebFace" / "casia-webface",
        INPUT_ROOT / "CASIA-WebFace",
        INPUT_ROOT / "casia-webface" / "casia-webface",
        INPUT_ROOT / "casia-webface",
    ]
    candidates.extend(
        directory for directory in input_dirs(max_depth=2) if looks_like_casia_path(directory)
    )
    for candidate in unique_existing(candidates):
        normalized = normalize_train_data_dir(candidate)
        if has_train_data(normalized):
            return normalized
    raise FileNotFoundError("Could not find CASIA-WebFace train data under /kaggle/input.")


def resolve_eval_dir(train_dir):
    candidates = [
        Path(train_dir).parent / "eval",
        INPUT_ROOT / "CASIA-WebFace" / "eval",
        INPUT_ROOT / "casia-webface" / "eval",
    ]
    candidates.extend(input_dirs(max_depth=2))
    for candidate in unique_existing(candidates):
        eval_dir = candidate if any((candidate / f"{name}.bin").exists() for name in EVAL_TARGETS) else candidate / "eval"
        if any((eval_dir / f"{name}.bin").exists() for name in EVAL_TARGETS):
            return eval_dir
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
            value = int(content.split(",")[0].strip())
            print("Detected num_classes from property:", value)
            return value
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


def exp_dir():
    return (
        OUTPUT_ROOT
        / SWEEP_FOLDER
        / (
            f"{BACKBONE}_proposed4_2_ui_aware"
            f"_uil_{float_tag(UI_LAMBDA)}"
            f"_rho_{float_tag(UI_RHO)}"
            f"_tri_{float_tag(UI_TAU_RI)}"
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
    warnings.warn("No verification .bin folder found. Training will skip per-epoch eval.", RuntimeWarning)
    EVAL_TARGETS = []
else:
    available_eval = [name for name in EVAL_TARGETS if (EVAL_DIR / f"{name}.bin").exists()]
    missing_eval = [name for name in EVAL_TARGETS if name not in available_eval]
    if missing_eval:
        warnings.warn(f"Missing eval bins in {EVAL_DIR}: {missing_eval}", RuntimeWarning)
    EVAL_TARGETS = available_eval

if not PRETRAINED_BACKBONE.exists():
    backbone_candidates = sorted(input_file_candidates("backbone.pth", max_depth=2))
    if backbone_candidates:
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
            raise RuntimeError(
                "Multiple .pth files found, refusing to choose one silently:\n"
                + "\n".join(str(item) for item in pth_candidates)
            )
        PRETRAINED_BACKBONE = pth_candidates[0]

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
SWEEP_ROOT = OUTPUT_ROOT / SWEEP_FOLDER

print("TRAIN_DATA_DIR:", TRAIN_DATA_DIR)
print("EVAL_DIR:", EVAL_DIR)
print("PRETRAINED_BACKBONE:", PRETRAINED_BACKBONE)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("NUM_CLASSES:", NUM_CLASSES)
print("EVAL_TARGETS:", ",".join(EVAL_TARGETS) if EVAL_TARGETS else "(none)")

# %% [markdown]
# ## Restore Previous Outputs

# %%
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
    print("Restoring previous proposed4.2 zip:", zip_candidate)
    with zipfile.ZipFile(zip_candidate, "r") as f:
        f.extractall(OUTPUT_ROOT)
    restored = True
    break
if not restored:
    for folder_candidate in folder_candidates:
        print("Restoring previous proposed4.2 folder:", folder_candidate)
        shutil.copytree(folder_candidate, SWEEP_ROOT, dirs_exist_ok=True)
        restored = True
        break
print("Restored previous proposed4.2 outputs to:" if restored else "No previous proposed4.2 output input found.", SWEEP_ROOT)

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
if (ARCFACE_DIR / "kaggle_proposed_4_2_runner.py").exists():
    compile_files.append("kaggle_proposed_4_2_runner.py")
run([sys.executable, "-m", "py_compile"] + compile_files, cwd=ARCFACE_DIR)

import torch
from backbones import get_model
from soft_gated_losses import UIAwareCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss
from train_phase2_kaggle import MarginSoftmaxHead, amp_autocast, make_grad_scaler

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
use_amp = bool(USE_FP16 and device.type == "cuda")
debug_backbone = get_model(BACKBONE, dropout=0.0, fp16=use_amp, num_features=512).to(device)
debug_margin_loss = UIAwareCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss(
    s=S,
    m=M,
    h=H,
    ui_lambda=UI_LAMBDA,
    ui_rho=UI_RHO,
    ui_tau_ri=UI_TAU_RI,
    ui_tau_easy=UI_TAU_EASY,
    ui_d_margin=UI_D_MARGIN,
    ui_alpha=UI_ALPHA,
    ui_beta=UI_BETA,
    ui_hard_boost=UI_HARD_BOOST,
    ui_dangerous_downweight=UI_DANGEROUS_DOWNWEIGHT,
    ui_sample_weight_min=UI_SAMPLE_WEIGHT_MIN,
    ui_center_momentum=UI_CENTER_MOMENTUM,
)
debug_head = MarginSoftmaxHead(512, NUM_CLASSES, debug_margin_loss, fp16=use_amp).to(device)
debug_optimizer = torch.optim.SGD(
    [
        {"params": debug_backbone.parameters(), "lr": BACKBONE_LR, "name": "backbone"},
        {"params": debug_head.parameters(), "lr": HEAD_LR, "name": "head"},
    ],
    momentum=0.9,
    weight_decay=5e-4,
)
debug_scaler = make_grad_scaler(use_amp)
debug_images = torch.randn(min(8, BATCH_SIZE), 3, 112, 112, device=device)
debug_labels = (torch.arange(debug_images.size(0), device=device) % NUM_CLASSES).long()

debug_margin_loss.update_ui_center(torch.randn(debug_images.size(0), 512, device=device))
debug_optimizer.zero_grad(set_to_none=True)
with amp_autocast(use_amp):
    debug_embeddings = debug_backbone(debug_images)
    debug_loss, debug_logits, debug_norms = debug_head(debug_embeddings, debug_labels)

assert torch.isfinite(debug_loss).item(), "debug loss is NaN or Inf"
assert debug_logits.shape == (debug_images.size(0), NUM_CLASSES), debug_logits.shape
if use_amp:
    debug_scaler.scale(debug_loss).backward()
    debug_scaler.step(debug_optimizer)
    debug_scaler.update()
else:
    debug_loss.backward()
    debug_optimizer.step()

stats = debug_margin_loss.last_stats
assert stats["ui_center_ready"] == 1.0, stats
assert 0.0 <= stats["ui_lambda_i_mean"] <= UI_LAMBDA + 1e-6, stats
assert math.isfinite(stats["ri_mean"]), stats
print("debug_loss:", float(debug_loss.detach().cpu().item()))
print("debug_norm_mean:", float(debug_norms.detach().mean().cpu().item()))
print("last_stats:", json.dumps(stats, indent=2, sort_keys=True))

del debug_backbone, debug_head, debug_margin_loss, debug_optimizer
del debug_images, debug_labels, debug_embeddings, debug_logits, debug_norms, debug_loss
if device.type == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
# ## Train Proposed 4.2

# %%
current_exp_dir = exp_dir()
if is_complete():
    print("[SKIP] proposed4.2 complete:", current_exp_dir)
else:
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
        "--ui_lambda",
        str(UI_LAMBDA),
        "--ui_rho",
        str(UI_RHO),
        "--ui_tau_ri",
        str(UI_TAU_RI),
        "--ui_tau_easy",
        str(UI_TAU_EASY),
        "--ui_d_margin",
        str(UI_D_MARGIN),
        "--ui_alpha",
        str(UI_ALPHA),
        "--ui_beta",
        str(UI_BETA),
        "--ui_hard_boost",
        str(UI_HARD_BOOST),
        "--ui_dangerous_downweight",
        str(UI_DANGEROUS_DOWNWEIGHT),
        "--ui_sample_weight_min",
        str(UI_SAMPLE_WEIGHT_MIN),
        "--ui_center_momentum",
        str(UI_CENTER_MOMENTUM),
        "--ui_center_update_interval",
        str(UI_CENTER_UPDATE_INTERVAL),
        "--train_data",
        str(TRAIN_DATA_DIR),
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
    if EVAL_DIR is not None:
        cmd.extend(["--eval_dir", str(EVAL_DIR)])
    if USE_FP16:
        cmd.append("--fp16")
    if latest.exists():
        print(f"[RESUME] proposed4.2 from {latest}")
        cmd.append("--resume")
    else:
        print("[START] proposed4.2 from pretrained backbone")
        cmd.extend(["--pretrained_backbone", str(PRETRAINED_BACKBONE)])

    run(cmd, cwd=ARCFACE_DIR)

# %% [markdown]
# ## Degraded Eval On 5 CASIA Eval Bins

# %%
if EVAL_DIR is None:
    warnings.warn("No EVAL_DIR found; skipping degraded eval.", RuntimeWarning)
else:
    best_checkpoint = current_exp_dir / "best.pth"
    if not best_checkpoint.exists():
        warnings.warn(f"Missing best.pth for degraded eval: {best_checkpoint}", RuntimeWarning)
    else:
        degraded_cmd = [
            sys.executable,
            "eval_degraded_6phase2.py",
            "--backbone",
            BACKBONE,
            "--checkpoint",
            str(best_checkpoint),
            "--checkpoint-label",
            current_exp_dir.name,
            "--data-dir",
            str(EVAL_DIR),
            "--output",
            str(current_exp_dir / "degraded_eval"),
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

# %% [markdown]
# ## Export

# %%
zip_path = Path("/kaggle/working") / f"{SWEEP_FOLDER}.zip"
if SWEEP_ROOT.exists():
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", OUTPUT_ROOT, SWEEP_FOLDER)
    print("Wrote:", zip_path)
    print("Size MB:", zip_path.stat().st_size / 1024 / 1024)
else:
    print("SWEEP_ROOT does not exist yet:", SWEEP_ROOT)
