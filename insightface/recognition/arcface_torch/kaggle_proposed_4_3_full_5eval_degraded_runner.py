from pathlib import Path
import os
import subprocess
import sys
import zipfile
import shutil

REPO_URL = "https://github.com/Tai12345-Ai/Lightweight_Face_Recognition.git"
BRANCH = "main"
CODE_ROOT = Path("/kaggle/working/Lightweight_Face_Recognition")
ARCFACE_DIR = CODE_ROOT / "insightface" / "recognition" / "arcface_torch"
INPUT_ROOT = Path("/kaggle/input")

if not CODE_ROOT.exists():
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(CODE_ROOT)], check=True)
else:
    subprocess.run(["git", "pull", "--ff-only"], cwd=CODE_ROOT, check=True)

os.chdir(ARCFACE_DIR)
sys.path.insert(0, str(ARCFACE_DIR))
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "tensorboard", "easydict", "onnx", "opencv-python", "scikit-learn", "pandas", "matplotlib",
], check=True)

from kaggle_5eval_degraded_common import run_5eval_degraded_runner, run


def find_train_data_dir():
    for rec in sorted(INPUT_ROOT.rglob("train.rec")):
        if (rec.parent / "train.idx").exists():
            print("Detected TRAIN_DATA_DIR:", rec.parent)
            return rec.parent
    raise FileNotFoundError("Could not find train.rec + train.idx under /kaggle/input.")


def find_eval_dir():
    required = ["lfw.bin", "cfp_fp.bin", "cplfw.bin", "agedb_30.bin", "calfw.bin"]
    for lfw in sorted(INPUT_ROOT.rglob("lfw.bin")):
        if all((lfw.parent / name).exists() for name in required):
            print("Detected EVAL_DIR:", lfw.parent)
            return lfw.parent
    raise FileNotFoundError("Could not find 5-eval .bin directory under /kaggle/input.")


def extract_core_checkpoint_from_zip():
    for z in sorted(INPUT_ROOT.rglob("*.zip")):
        low = str(z).lower()
        if "core" not in low:
            continue
        try:
            with zipfile.ZipFile(z, "r") as archive:
                members = [m for m in archive.namelist() if m.endswith("best.pth")]
                if not members:
                    continue
                member = sorted(members, key=len)[0]
                out = Path("/kaggle/working/core_warm_start_best.pth")
                with archive.open(member) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print("Extracted Core warm-start:", out)
                return out
        except zipfile.BadZipFile:
            pass
    return None


def find_pretrained_backbone():
    core_hits = []
    for p in list(INPUT_ROOT.rglob("best.pth")) + list(INPUT_ROOT.rglob("latest.pt")):
        low = str(p).lower()
        if "core" in low and "proposed" in low:
            core_hits.append(p)
    if core_hits:
        chosen = sorted(core_hits, key=lambda p: (0 if p.name == "best.pth" else 1, len(str(p))))[0]
        print("Detected Core warm-start:", chosen)
        return chosen
    extracted = extract_core_checkpoint_from_zip()
    if extracted is not None:
        return extracted
    hits = sorted(INPUT_ROOT.rglob("backbone.pth"))
    if not hits:
        raise FileNotFoundError("Could not find backbone.pth under /kaggle/input.")
    chosen = hits[0]
    print("Detected PRETRAINED_BACKBONE:", chosen)
    return chosen


TRAIN_DATA_DIR = find_train_data_dir()
EVAL_DIR = find_eval_dir()
PRETRAINED_BACKBONE = find_pretrained_backbone()

assert (TRAIN_DATA_DIR / "train.rec").exists()
assert (TRAIN_DATA_DIR / "train.idx").exists()
assert (EVAL_DIR / "lfw.bin").exists()
assert PRETRAINED_BACKBONE.exists()

RUNNER_FILE = "kaggle_proposed_4_3_full_5eval_degraded_runner.py"
RUNNER_KIND = "proposed4_3"
OUTPUT_SUBDIR = "proposed_4_3_full"
BACKUP_ZIP_NAME = "proposed_4_3_full_20ep_5eval_degraded_s135.zip"
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
UI_LAMBDA = 0.05
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

EVAL_TARGETS = ["lfw", "cfp_fp", "cplfw", "agedb_30", "calfw"]
VAL_TARGETS = EVAL_TARGETS
HQ_EVAL_TARGETS = EVAL_TARGETS
RUN_DEGRADED_EVAL = True
DEGRADED_TARGETS = EVAL_TARGETS
DEGRADED_DEGRADATIONS = [
    "gaussian_blur", "motion_blur", "low_resolution",
    "jpeg_compression", "low_illumination", "alignment_perturb",
]
DEGRADED_SEVERITIES = "1,3,5"
DEGRADED_BATCH_SIZE = 128
UI_CENTER_NUM_SAMPLES = 50000
UI_CENTER_SEVERITIES = "5"
UI_CENTER_OUTPUT = Path("/kaggle/working/ui_centers/proposed_4_3_full_multi_ui_centers_s5.pth")


def find_existing_ui_centers():
    candidates = []
    for root in [INPUT_ROOT, Path("/kaggle/working")]:
        if not root.exists():
            continue
        for p in root.rglob("*.pth"):
            full = str(p).lower()
            name = p.name.lower()
            if "multi" in full and ("ui" in name or "center" in name or "centers" in name):
                candidates.append(p)
    if candidates:
        chosen = sorted(candidates, key=lambda p: (0 if "full" in str(p).lower() else 1, len(str(p))))[0]
        print("Found existing multi-UI centers:", chosen)
        return chosen
    return None


def ensure_multi_ui_centers():
    existing = find_existing_ui_centers()
    if existing is not None:
        return existing
    UI_CENTER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if UI_CENTER_OUTPUT.exists():
        print("Using existing UI centers:", UI_CENTER_OUTPUT)
        return UI_CENTER_OUTPUT
    cmd = [
        sys.executable, "build_multi_ui_centers.py",
        "--data-dir", str(TRAIN_DATA_DIR),
        "--pretrained-backbone", str(PRETRAINED_BACKBONE),
        "--backbone", BACKBONE,
        "--output", str(UI_CENTER_OUTPUT),
        "--num-samples", str(UI_CENTER_NUM_SAMPLES),
        "--batch-size", str(DEGRADED_BATCH_SIZE),
        "--num-workers", str(NUM_WORKERS),
        "--degradations", ",".join(DEGRADED_DEGRADATIONS),
        "--severities", UI_CENTER_SEVERITIES,
        "--include-global",
        "--overwrite",
    ]
    if USE_FP16:
        cmd.append("--fp16")
    run(cmd, cwd=ARCFACE_DIR)
    return UI_CENTER_OUTPUT


MULTI_UI_CENTERS = str(ensure_multi_ui_centers())
print("MULTI_UI_CENTERS:", MULTI_UI_CENTERS)
run_5eval_degraded_runner(globals())

try:
    from kaggle_proposed_4_3_core_report import make_report
    make_report(
        backup_zip_name=BACKUP_ZIP_NAME,
        output_subdir=OUTPUT_SUBDIR,
        eval_targets=EVAL_TARGETS,
        degraded_targets=DEGRADED_TARGETS,
        degraded_degradations=DEGRADED_DEGRADATIONS,
    )
except Exception as exc:
    print("[WARN] Could not generate plots/report:", repr(exc))
