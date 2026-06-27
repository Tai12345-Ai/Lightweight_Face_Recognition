#!/usr/bin/env python3
"""Kaggle runner for Proposed 4.3 Full warm-started from a finished Core checkpoint."""
from pathlib import Path
import os, shutil, subprocess, sys, zipfile

REPO_URL = "https://github.com/Tai12345-Ai/Lightweight_Face_Recognition.git"
BRANCH = "main"
CODE_ROOT = Path("/kaggle/working/Lightweight_Face_Recognition")
ARCFACE_DIR = CODE_ROOT / "insightface" / "recognition" / "arcface_torch"
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")


def run(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)))
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)

if not CODE_ROOT.exists():
    run(["git", "clone", "--branch", BRANCH, REPO_URL, str(CODE_ROOT)], cwd=WORK_ROOT)
else:
    run(["git", "fetch", "origin", BRANCH], cwd=CODE_ROOT)
    run(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=CODE_ROOT)

os.chdir(ARCFACE_DIR)
sys.path.insert(0, str(ARCFACE_DIR))
run([sys.executable, "-m", "pip", "install", "-q", "tensorboard", "easydict", "onnx", "opencv-python", "scikit-learn", "pandas", "matplotlib"])

import kaggle_5eval_degraded_common as common

# Force common runner to launch the Full wrapper.
_orig_build_cmd = common.build_proposed_command

def _build_full_cmd(config, current_exp_dir, train_data_dir, eval_dir, num_classes):
    cmd = _orig_build_cmd(config, current_exp_dir, train_data_dir, eval_dir, num_classes)
    for i, item in enumerate(cmd):
        if str(item) == "train_soft_gated_lambda_kaggle.py":
            cmd[i] = "train_proposed_4_3_full_kaggle.py"
            return cmd
    raise RuntimeError("Could not switch trainer to train_proposed_4_3_full_kaggle.py")

common.build_proposed_command = _build_full_cmd
run_5eval_degraded_runner = common.run_5eval_degraded_runner
common_run = common.run


def find_train_data_dir():
    for rec in sorted(INPUT_ROOT.rglob("train.rec")):
        if (rec.parent / "train.idx").exists():
            print("Detected TRAIN_DATA_DIR:", rec.parent)
            return rec.parent
    raise FileNotFoundError("Cannot find train.rec + train.idx under /kaggle/input")


def find_eval_dir():
    required = ["lfw.bin", "cfp_fp.bin", "cplfw.bin", "agedb_30.bin", "calfw.bin"]
    for lfw in sorted(INPUT_ROOT.rglob("lfw.bin")):
        if all((lfw.parent / name).exists() for name in required):
            print("Detected EVAL_DIR:", lfw.parent)
            return lfw.parent
    raise FileNotFoundError("Cannot find 5-eval .bin files under /kaggle/input")


def extract_candidates_from_zips():
    out_dir = WORK_ROOT / "full_from_core_warmstart"
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    for z in sorted(list(INPUT_ROOT.rglob("*.zip")) + list(WORK_ROOT.rglob("*.zip"))):
        try:
            with zipfile.ZipFile(z, "r") as archive:
                members = [m for m in archive.namelist() if m.endswith("best.pth") or m.endswith("latest.pt")]
                if not members:
                    continue
                members = sorted(members, key=lambda m: (0 if "core" in m.lower() else 1, 0 if m.endswith("best.pth") else 1, len(m)))
                member = members[0]
                suffix = "best.pth" if member.endswith("best.pth") else "latest.pt"
                out = out_dir / f"{z.stem}_{suffix}"
                if not out.exists():
                    with archive.open(member) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                print("Extracted checkpoint candidate:", out)
                outs.append(out)
        except zipfile.BadZipFile:
            pass
    return outs


def checkpoint_score(p: Path):
    s = str(p).lower().replace("\\", "/")
    if p.name.lower() == "backbone.pth" or "/backbone/" in s or "full" in s:
        return -9999
    score = 0
    score += 1000 if "core" in s else 0
    score += 300 if "proposed_4_3" in s or "proposed4_3" in s else 0
    score += 80 if p.name == "best.pth" else 30
    score += 20 if "/kaggle/input/" in s else 0
    return score


def find_pretrained_backbone():
    candidates = []
    for root in [INPUT_ROOT, WORK_ROOT]:
        candidates += list(root.rglob("best.pth")) + list(root.rglob("latest.pt"))
    candidates += extract_candidates_from_zips()
    candidates = [p for p in candidates if p.exists() and checkpoint_score(p) > -9999]
    if candidates:
        candidates = sorted(candidates, key=lambda p: (-checkpoint_score(p), len(str(p))))
        print("Checkpoint candidates:")
        for p in candidates[:20]:
            print(f"  score={checkpoint_score(p):4d} | {p}")
        print("Selected Core warm-start:", candidates[0])
        return candidates[0]
    hits = sorted(INPUT_ROOT.rglob("backbone.pth"))
    if not hits:
        raise FileNotFoundError("No Core checkpoint and no fallback backbone.pth found")
    print("[WARN] No Core checkpoint found; fallback to backbone.pth. This is not final Core -> Full comparison.")
    return hits[0]


TRAIN_DATA_DIR = find_train_data_dir()
EVAL_DIR = find_eval_dir()
PRETRAINED_BACKBONE = find_pretrained_backbone()

RUNNER_FILE = "kaggle_proposed_4_3_full_from_core_runner.py"
RUNNER_KIND = "proposed4_3"
OUTPUT_SUBDIR = "proposed_4_3_full_from_core"
BACKUP_ZIP_NAME = "proposed_4_3_full_from_core_20ep_5eval_degraded_s135.zip"
DEGRADED_EVAL_SCRIPT = "eval_degraded_proposed_4_3_full.py"
LOSS_NAME = "proposed_4_3_multi_ui_attention"
BACKBONE = "r18"
EPOCHS = 20
BATCH_SIZE = 128
BACKBONE_LR = 5e-5
HEAD_LR = 5e-4
WARMUP_EPOCHS = 1.0
EVAL_EVERY = 1
SAVE_EVERY_EPOCHS = 1
SAVE_EVERY_STEPS = 300
NUM_WORKERS = 2
USE_FP16 = True
MAX_TRAIN_MINUTES = 660
S = 64.0
M = 0.4
H = 0.333
FULL_TOP_M = 4
FULL_UI_SOFT_TAU = 12.0
FULL_UI_MARGIN = 0.20
FULL_UI_LAMBDA = 0.05
FULL_RI_LAMBDA = 0.05
FULL_ANCHOR_LAMBDA = 0.08
FULL_NEG_LAMBDA = 0.06
FULL_PRESERVE_LAMBDA = 0.03
FULL_DELTA_C = 0.02
FULL_DELTA_N = 0.02
FULL_LABEL_GAMMA = 12.0
FULL_LABEL_MARGIN = 0.05
FULL_UNREC_TAU = 0.35
FULL_UNREC_GAMMA = 8.0
UI_LAMBDA = FULL_UI_LAMBDA
UI_RHO = 0.20
UI_TAU_RI = 1.0
UI_TAU_EASY = 2.0
UI_D_MARGIN = 0.25
UI_ALPHA = 10.0
UI_BETA = 5.0
UI_HARD_BOOST = 0.10
UI_DANGEROUS_DOWNWEIGHT = 0.35
UI_SAMPLE_WEIGHT_MIN = 0.50
ENABLE_ATTENTION = True
ATTENTION_GAMMA = 0.03
ATTENTION_REDUCTION = 16
ATTENTION_ALPHA = 0.25
CENTERED_ATTENTION = True
RI_LAMBDA = FULL_RI_LAMBDA
ATTENTION_SPATIAL_LAMBDA = 1e-4
ATTENTION_CHANNEL_LAMBDA = 1e-4
ATTENTION_TV_LAMBDA = 1e-4
EVAL_TARGETS = ["lfw", "cfp_fp", "cplfw", "agedb_30", "calfw"]
VAL_TARGETS = EVAL_TARGETS
HQ_EVAL_TARGETS = EVAL_TARGETS
RUN_DEGRADED_EVAL = True
DEGRADED_TARGETS = EVAL_TARGETS
DEGRADED_DEGRADATIONS = ["gaussian_blur", "motion_blur", "low_resolution", "jpeg_compression", "low_illumination", "alignment_perturb"]
DEGRADED_SEVERITIES = "1,3,5"
DEGRADED_BATCH_SIZE = 128
UI_CENTER_NUM_SAMPLES = 50000
UI_CENTER_SEVERITIES = "5"
UI_CENTER_OUTPUT = WORK_ROOT / "ui_centers" / "proposed_4_3_full_from_core_multi_ui_centers_s5.pth"


def find_existing_ui_centers():
    hits = []
    for root in [INPUT_ROOT, WORK_ROOT]:
        for p in root.rglob("*.pth"):
            low = str(p).lower()
            if "multi" in low and ("ui" in p.name.lower() or "center" in p.name.lower()):
                hits.append(p)
    if not hits:
        return None
    return sorted(hits, key=lambda p: (0 if "s5" in str(p).lower() else 1, 0 if "full" in str(p).lower() else 1, len(str(p))))[0]


def ensure_multi_ui_centers():
    existing = find_existing_ui_centers()
    if existing is not None:
        print("Found existing multi-UI centers:", existing)
        return existing
    UI_CENTER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "build_multi_ui_centers.py", "--data-dir", str(TRAIN_DATA_DIR), "--pretrained-backbone", str(PRETRAINED_BACKBONE), "--backbone", BACKBONE, "--output", str(UI_CENTER_OUTPUT), "--num-samples", str(UI_CENTER_NUM_SAMPLES), "--batch-size", str(DEGRADED_BATCH_SIZE), "--num-workers", str(NUM_WORKERS), "--degradations", ",".join(DEGRADED_DEGRADATIONS), "--severities", UI_CENTER_SEVERITIES, "--attention-alpha", str(ATTENTION_ALPHA), "--include-global", "--overwrite"]
    if CENTERED_ATTENTION:
        cmd.append("--centered-attention")
    if USE_FP16:
        cmd.append("--fp16")
    common_run(cmd, cwd=ARCFACE_DIR)
    return UI_CENTER_OUTPUT

MULTI_UI_CENTERS = str(ensure_multi_ui_centers())

print("\n===== Proposed 4.3 Full from Core =====")
print("TRAIN_DATA_DIR     :", TRAIN_DATA_DIR)
print("EVAL_DIR           :", EVAL_DIR)
print("PRETRAINED_BACKBONE:", PRETRAINED_BACKBONE)
print("MULTI_UI_CENTERS   :", MULTI_UI_CENTERS)
print("DEGRADED_SEVERITIES:", DEGRADED_SEVERITIES)
print("=======================================\n")

assert (TRAIN_DATA_DIR / "train.rec").exists()
assert (TRAIN_DATA_DIR / "train.idx").exists()
assert (EVAL_DIR / "lfw.bin").exists()
assert Path(PRETRAINED_BACKBONE).exists()
assert Path(MULTI_UI_CENTERS).exists()

run_5eval_degraded_runner(globals())

try:
    from kaggle_proposed_4_3_core_report import make_report
    make_report(backup_zip_name=BACKUP_ZIP_NAME, output_subdir=OUTPUT_SUBDIR, eval_targets=EVAL_TARGETS, degraded_targets=DEGRADED_TARGETS, degraded_degradations=DEGRADED_DEGRADATIONS)
except Exception as exc:
    print("[WARN] Could not generate plots/report:", repr(exc))
