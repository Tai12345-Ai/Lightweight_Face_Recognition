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
INPUT_ROOT = Path("/kaggle/input")

DEFAULT_TRAIN_DATA_DIR = INPUT_ROOT / "CASIA-WebFace" / "casia-webface"
DEFAULT_EVAL_DIR = INPUT_ROOT / "CASIA-WebFace" / "eval"
DEFAULT_PRETRAINED_BACKBONE = INPUT_ROOT / "backbone" / "backbone.pth"
EXPERIMENTS_ROOT = Path("/kaggle/working/experiments")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run(cmd, cwd=None, check=True):
    print("+", " ".join(str(item) for item in cmd))
    return subprocess.run([str(item) for item in cmd], cwd=cwd, check=check)


def unique_existing(paths):
    seen = set()
    for path in paths:
        path = Path(path)
        if path in seen or not path.exists():
            continue
        seen.add(path)
        yield path


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
            if any(item.is_file() and item.suffix.lower() in IMAGE_EXTS for item in class_dir.iterdir()):
                return True
        except OSError:
            continue
    return False


def has_train_data(path):
    return has_recordio_train_data(path) or has_imagefolder_train_data(path)


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


def eval_dir_has_targets(path, targets):
    return Path(path).is_dir() and all((Path(path) / f"{target}.bin").exists() for target in targets)


def resolve_eval_dir(train_dir, targets):
    candidates = [
        Path(train_dir).parent / "eval",
        INPUT_ROOT / "CASIA-WebFace" / "eval",
        INPUT_ROOT / "casia-webface" / "eval",
    ]
    candidates.extend(input_dirs(max_depth=3))
    for candidate in unique_existing(candidates):
        if eval_dir_has_targets(candidate, targets):
            return candidate
        nested = candidate / "eval"
        if eval_dir_has_targets(nested, targets):
            return nested
    raise FileNotFoundError(
        "Could not find eval directory containing required .bin files: "
        + ", ".join(f"{target}.bin" for target in targets)
    )


def input_file_candidates(filename, max_depth=2):
    for directory in input_dirs(max_depth=max_depth):
        candidate = directory / filename
        if candidate.exists():
            yield candidate


def resolve_pretrained_backbone(preferred):
    preferred = Path(preferred)
    if preferred.exists():
        return preferred
    candidates = sorted(input_file_candidates("backbone.pth", max_depth=2))
    if candidates:
        if len(candidates) > 1:
            warnings.warn(
                "Multiple backbone.pth files found. Using first candidate:\n"
                + "\n".join(str(item) for item in candidates),
                RuntimeWarning,
            )
        return candidates[0]
    pth_candidates = sorted(
        candidate
        for directory in input_dirs(max_depth=2)
        for candidate in directory.glob("*.pth")
    )
    if len(pth_candidates) == 1:
        return pth_candidates[0]
    if not pth_candidates:
        raise FileNotFoundError("No pretrained backbone .pth found under /kaggle/input.")
    raise RuntimeError(
        "Multiple .pth files found, refusing to choose one silently:\n"
        + "\n".join(str(item) for item in pth_candidates)
    )


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


def output_root(config):
    return EXPERIMENTS_ROOT / Path(config["BACKUP_ZIP_NAME"]).stem


def exp_dir(config):
    kind = config["RUNNER_KIND"]
    backbone = config["BACKBONE"]
    backbone_lr = config["BACKBONE_LR"]
    head_lr = config["HEAD_LR"]
    if kind == "phase2":
        name = (
            f"{backbone}_{config['LOSS_NAME']}"
            f"_blr_{float_tag(backbone_lr)}"
            f"_hlr_{float_tag(head_lr)}"
        )
    elif kind == "proposed4_1":
        name = (
            f"{backbone}_proposed4_quality_gate_comp_adaptive_soft_gated_ada_curricular"
            f"_blr_{float_tag(backbone_lr)}"
            f"_hlr_{float_tag(head_lr)}"
        )
    elif kind == "proposed4_2":
        name = (
            f"{backbone}_proposed4_2_ui_aware"
            f"_uil_{float_tag(config['UI_LAMBDA'])}"
            f"_rho_{float_tag(config['UI_RHO'])}"
            f"_tri_{float_tag(config['UI_TAU_RI'])}"
            f"_blr_{float_tag(backbone_lr)}"
            f"_hlr_{float_tag(head_lr)}"
        )
    elif kind == "proposed4_3":
        name = (
            f"{backbone}_proposed4_3_multi_ui_attention"
            f"_uil_{float_tag(config['UI_LAMBDA'])}"
            f"_rho_{float_tag(config['UI_RHO'])}"
            f"_tri_{float_tag(config['UI_TAU_RI'])}"
            f"_ag_{float_tag(config.get('ATTENTION_GAMMA', 0.05))}"
            f"_blr_{float_tag(backbone_lr)}"
            f"_hlr_{float_tag(head_lr)}"
        )
    else:
        raise ValueError(f"Unknown RUNNER_KIND: {kind}")
    return output_root(config) / config["OUTPUT_SUBDIR"] / name


def read_epoch_count(metrics_path):
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return 0
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return len(metrics.get("epochs", []))


def is_complete(config, current_exp_dir):
    return read_epoch_count(Path(current_exp_dir) / "metrics.json") >= int(config["EPOCHS"])


def copy_restored_output(config, source):
    source = Path(source)
    destination = output_root(config) / config["OUTPUT_SUBDIR"]
    if source.name == config["OUTPUT_SUBDIR"]:
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return True
    direct_child = source / config["OUTPUT_SUBDIR"]
    if direct_child.exists():
        shutil.copytree(direct_child, destination, dirs_exist_ok=True)
        return True
    for candidate in source.rglob(config["OUTPUT_SUBDIR"]):
        if candidate.is_dir():
            shutil.copytree(candidate, destination, dirs_exist_ok=True)
            return True
    return False


def restore_previous_outputs(config):
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    backup_zip_name = config["BACKUP_ZIP_NAME"]
    backup_stem = Path(backup_zip_name).stem
    zip_candidates = [INPUT_ROOT / backup_zip_name]
    zip_candidates.extend(directory / backup_zip_name for directory in input_dirs(max_depth=2))

    for zip_candidate in unique_existing(zip_candidates):
        print("Restoring previous backup zip:", zip_candidate)
        restore_dir = Path("/kaggle/working") / f"restore_{backup_stem}"
        if restore_dir.exists():
            shutil.rmtree(restore_dir)
        restore_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_candidate, "r") as archive:
            archive.extractall(restore_dir)
        if not copy_restored_output(config, restore_dir):
            raise RuntimeError(
                f"Backup zip did not contain {config['OUTPUT_SUBDIR']}: {zip_candidate}"
            )
        return True

    folder_candidates = [
        INPUT_ROOT / backup_stem,
        INPUT_ROOT / backup_stem / config["OUTPUT_SUBDIR"],
    ]
    folder_candidates.extend(directory / backup_stem for directory in input_dirs(max_depth=2))
    folder_candidates.extend(
        directory / backup_stem / config["OUTPUT_SUBDIR"]
        for directory in input_dirs(max_depth=2)
    )
    for folder_candidate in unique_existing(folder_candidates):
        print("Restoring previous backup folder:", folder_candidate)
        if copy_restored_output(config, folder_candidate):
            return True

    print("No previous backup found for:", backup_zip_name)
    return False


def complete_accuracy_mean(eval_metrics, targets):
    values = []
    for target in targets:
        item = eval_metrics.get(target)
        if item is None or "accuracy" not in item:
            return None
        values.append(float(item["accuracy"]))
    return float(sum(values) / len(values)) if values else None


def run_degraded_eval(config, current_exp_dir, eval_dir):
    if not config.get("RUN_DEGRADED_EVAL", True):
        return

    best_checkpoint = Path(current_exp_dir) / "best.pth"
    if not best_checkpoint.exists():
        warnings.warn(
            f"Missing best.pth for degraded eval: {best_checkpoint}. Skipping.",
            RuntimeWarning,
        )
        return

    missing_bins = [
        target for target in config["DEGRADED_TARGETS"]
        if not (eval_dir / f"{target}.bin").exists()
    ]
    if missing_bins:
        raise FileNotFoundError(
            "Missing degraded eval bins in "
            f"{eval_dir}: {', '.join(missing_bins)}"
        )

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    degraded_cmd = [
        sys.executable,
        "eval_degraded_6phase2.py",
        "--backbone",
        config["BACKBONE"],
        "--checkpoint",
        str(best_checkpoint),
        "--checkpoint-label",
        Path(current_exp_dir).name,
        "--data-dir",
        str(eval_dir),
        "--output",
        str(Path(current_exp_dir) / "degraded_eval"),
        "--targets",
        ",".join(config["DEGRADED_TARGETS"]),
        "--degradations",
        ",".join(config["DEGRADED_DEGRADATIONS"]),
        "--severities",
        str(config["DEGRADED_SEVERITIES"]),
        "--batch-size",
        str(config["DEGRADED_BATCH_SIZE"]),
        "--device",
        device,
    ]
    if config.get("USE_FP16", True):
        degraded_cmd.append("--fp16")
    run(degraded_cmd, cwd=ARCFACE_DIR)


def add_common_train_args(config, cmd, train_data_dir, eval_dir, num_classes):
    option_names = {
        "phase2": {
            "batch_size": "--batch-size",
            "backbone_lr": "--backbone-lr",
            "head_lr": "--head-lr",
            "warmup_epochs": "--warmup-epochs",
            "eval_every": "--eval-every",
            "save_every": "--save-every",
            "save_every_steps": "--save-every-steps",
            "max_train_minutes": "--max-train-minutes",
            "num_workers": "--num-workers",
            "num_classes": "--num-classes",
            "eval_targets": "--val-targets",
        },
        "proposed": {
            "batch_size": "--batch_size",
            "backbone_lr": "--backbone_lr",
            "head_lr": "--head_lr",
            "warmup_epochs": "--warmup_epochs",
            "eval_every": "--eval_every",
            "save_every": "--save_every",
            "save_every_steps": "--save_every_steps",
            "max_train_minutes": "--max_train_minutes",
            "num_workers": "--num_workers",
            "num_classes": "--num_classes",
            "eval_targets": "--eval_targets",
        },
    }
    names = option_names["phase2" if config["RUNNER_KIND"] == "phase2" else "proposed"]
    cmd.extend([
        names["batch_size"], str(config["BATCH_SIZE"]),
        "--lr", str(config["HEAD_LR"]),
        names["backbone_lr"], str(config["BACKBONE_LR"]),
        names["head_lr"], str(config["HEAD_LR"]),
        names["warmup_epochs"], str(config["WARMUP_EPOCHS"]),
        names["eval_every"], str(config["EVAL_EVERY"]),
        names["save_every"], str(config["SAVE_EVERY_EPOCHS"]),
        names["save_every_steps"], str(config["SAVE_EVERY_STEPS"]),
        names["max_train_minutes"], str(config["MAX_TRAIN_MINUTES"]),
        names["num_workers"], str(config["NUM_WORKERS"]),
        names["num_classes"], str(num_classes),
        names["eval_targets"], ",".join(config["VAL_TARGETS"]),
    ])
    if config["RUNNER_KIND"] == "phase2":
        cmd.extend(["--data-dir", str(train_data_dir), "--eval-dir", str(eval_dir)])
    else:
        cmd.extend(["--train_data", str(train_data_dir), "--eval_dir", str(eval_dir)])
    if config.get("USE_FP16", True):
        cmd.append("--fp16")
    return cmd


def build_phase2_command(config, current_exp_dir, train_data_dir, eval_dir, num_classes):
    cmd = [
        sys.executable,
        "train_phase2_kaggle.py",
        "--loss",
        config["LOSS_NAME"],
        "--backbone",
        config["BACKBONE"],
        "--output-dir",
        str(output_root(config)),
        "--epochs",
        str(config["EPOCHS"]),
    ]
    add_common_train_args(config, cmd, train_data_dir, eval_dir, num_classes)

    latest = Path(current_exp_dir) / "latest.pt"
    if latest.exists():
        print(f"[RESUME] {config['LOSS_NAME']} from {latest}")
        cmd.append("--resume")
    else:
        pretrained_backbone = resolve_pretrained_backbone(config["PRETRAINED_BACKBONE"])
        print(f"[START] {config['LOSS_NAME']} from pretrained backbone: {pretrained_backbone}")
        cmd.extend(["--pretrained-backbone", str(pretrained_backbone)])
    return cmd


def build_proposed_command(config, current_exp_dir, train_data_dir, eval_dir, num_classes):
    cmd = [
        sys.executable,
        "train_soft_gated_lambda_kaggle.py",
        "--loss",
        config["LOSS_NAME"],
        "--network",
        config["BACKBONE"],
        "--s",
        str(config["S"]),
        "--m",
        str(config["M"]),
        "--h",
        str(config["H"]),
        "--output_dir",
        str(output_root(config)),
        "--epochs",
        str(config["EPOCHS"]),
    ]

    if config["RUNNER_KIND"] == "proposed4_2":
        cmd.extend([
            "--ui_lambda", str(config["UI_LAMBDA"]),
            "--ui_rho", str(config["UI_RHO"]),
            "--ui_tau_ri", str(config["UI_TAU_RI"]),
            "--ui_tau_easy", str(config["UI_TAU_EASY"]),
            "--ui_d_margin", str(config["UI_D_MARGIN"]),
            "--ui_alpha", str(config["UI_ALPHA"]),
            "--ui_beta", str(config["UI_BETA"]),
            "--ui_hard_boost", str(config["UI_HARD_BOOST"]),
            "--ui_dangerous_downweight", str(config["UI_DANGEROUS_DOWNWEIGHT"]),
            "--ui_sample_weight_min", str(config["UI_SAMPLE_WEIGHT_MIN"]),
            "--ui_center_momentum", str(config["UI_CENTER_MOMENTUM"]),
            "--ui_center_update_interval", str(config["UI_CENTER_UPDATE_INTERVAL"]),
        ])

    if config["RUNNER_KIND"] == "proposed4_3":
        cmd.extend([
            "--ui_lambda", str(config["UI_LAMBDA"]),
            "--ui_rho", str(config["UI_RHO"]),
            "--ui_tau_ri", str(config["UI_TAU_RI"]),
            "--ui_tau_easy", str(config["UI_TAU_EASY"]),
            "--ui_d_margin", str(config["UI_D_MARGIN"]),
            "--ui_alpha", str(config["UI_ALPHA"]),
            "--ui_beta", str(config["UI_BETA"]),
            "--ui_hard_boost", str(config["UI_HARD_BOOST"]),
            "--ui_dangerous_downweight", str(config["UI_DANGEROUS_DOWNWEIGHT"]),
            "--ui_sample_weight_min", str(config["UI_SAMPLE_WEIGHT_MIN"]),
            "--multi-ui-centers", str(config["MULTI_UI_CENTERS"]),
        ])
        if config.get("ENABLE_ATTENTION", False):
            cmd.append("--enable-attention")
            cmd.extend([
                "--attention-gamma", str(config.get("ATTENTION_GAMMA", 0.05)),
                "--attention-reduction", str(config.get("ATTENTION_REDUCTION", 16)),
            ])

    add_common_train_args(config, cmd, train_data_dir, eval_dir, num_classes)

    latest = Path(current_exp_dir) / "latest.pt"
    if latest.exists():
        print(f"[RESUME] {config['LOSS_NAME']} from {latest}")
        cmd.append("--resume")
    else:
        pretrained_backbone = resolve_pretrained_backbone(config["PRETRAINED_BACKBONE"])
        print(f"[START] {config['LOSS_NAME']} from pretrained backbone: {pretrained_backbone}")
        cmd.extend(["--pretrained_backbone", str(pretrained_backbone)])
    return cmd


def build_train_command(config, current_exp_dir, train_data_dir, eval_dir, num_classes):
    if config["RUNNER_KIND"] == "phase2":
        return build_phase2_command(config, current_exp_dir, train_data_dir, eval_dir, num_classes)
    return build_proposed_command(config, current_exp_dir, train_data_dir, eval_dir, num_classes)


def export_eval_csv(config):
    try:
        import pandas as pd
    except Exception as exc:
        print("Could not import pandas; skipping CSV export:", exc)
        return None

    rows = []
    metrics_root = output_root(config) / config["OUTPUT_SUBDIR"]
    for metrics_path in sorted(metrics_root.glob("*/metrics.json")):
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        for epoch_record in metrics.get("epochs", []):
            evals = epoch_record.get("eval", {}) or {}
            row = {
                "experiment": metrics_path.parent.name,
                "epoch": epoch_record.get("epoch"),
                "loss": epoch_record.get("loss"),
                "mean_norm": epoch_record.get("mean_norm"),
                "lr": epoch_record.get("lr"),
                "backbone_lr": epoch_record.get("backbone_lr"),
                "head_lr": epoch_record.get("head_lr"),
                "Eval5_Avg": complete_accuracy_mean(evals, config["HQ_EVAL_TARGETS"]),
            }
            for name in config["EVAL_TARGETS"]:
                item = evals.get(name, {}) or {}
                row[name] = item.get("accuracy")
                row[f"{name}_std"] = item.get("std")
                row[f"{name}_xnorm"] = item.get("xnorm")
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = Path("/kaggle/working") / f"{Path(config['BACKUP_ZIP_NAME']).stem}_eval_by_epoch.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    try:
        from IPython.display import FileLink, display
        display(FileLink(str(out_csv)))
    except Exception as exc:
        print("Could not render CSV download link:", exc)
    return out_csv


def write_manifest(config, current_exp_dir, train_data_dir, eval_dir, num_classes):
    manifest = {
        "runner_file": config.get("RUNNER_FILE"),
        "runner_kind": config.get("RUNNER_KIND"),
        "loss_name": config.get("LOSS_NAME"),
        "backbone": config.get("BACKBONE"),
        "epochs_requested": int(config.get("EPOCHS", 0)),
        "epochs_recorded": read_epoch_count(Path(current_exp_dir) / "metrics.json"),
        "train_data_dir": str(train_data_dir),
        "eval_dir": str(eval_dir),
        "num_classes": int(num_classes),
        "experiment_dir": str(current_exp_dir),
        "latest_checkpoint": str(Path(current_exp_dir) / "latest.pt"),
        "best_checkpoint": str(Path(current_exp_dir) / "best.pth"),
        "eval_targets": list(config.get("EVAL_TARGETS", [])),
        "degraded_targets": list(config.get("DEGRADED_TARGETS", [])),
        "degraded_degradations": list(config.get("DEGRADED_DEGRADATIONS", [])),
        "degraded_severities": str(config.get("DEGRADED_SEVERITIES", "")),
        "backup_zip_name": config.get("BACKUP_ZIP_NAME"),
    }
    out = Path(current_exp_dir) / "run_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Saved:", out)
    return out


def write_backup(config):
    zip_path = Path("/kaggle/working") / config["BACKUP_ZIP_NAME"]
    if zip_path.exists():
        zip_path.unlink()

    output_subdir = output_root(config) / config["OUTPUT_SUBDIR"]
    if not output_subdir.exists():
        print("No outputs to back up yet:", output_subdir)
        return None

    shutil.make_archive(
        str(zip_path.with_suffix("")),
        "zip",
        str(output_root(config)),
        config["OUTPUT_SUBDIR"],
    )
    print("Saved:", zip_path)
    print("Size MB:", zip_path.stat().st_size / 1024 / 1024)
    try:
        from IPython.display import FileLink, display
        display(FileLink(str(zip_path)))
    except Exception as exc:
        print("Could not render backup download link:", exc)
    return zip_path


def preflight_compile(config):
    compile_files = [
        "degradation/transforms.py",
        "eval_degraded_6phase2.py",
        "losses_extended.py",
        "soft_gated_losses.py",
        "train_phase2_kaggle.py",
        "train_soft_gated_lambda_kaggle.py",
        "recordio_fallback.py",
        "perceptibility_attention.py",
        "build_multi_ui_centers.py",
        "kaggle_proposed_4_3_core_report.py",
        config["RUNNER_FILE"],
        "kaggle_5eval_degraded_common.py",
    ]
    compile_files = [name for name in compile_files if (ARCFACE_DIR / name).exists()]
    run([sys.executable, "-m", "py_compile"] + compile_files, cwd=ARCFACE_DIR)


def generate_report(config):
    if not config.get("GENERATE_REPORT", True):
        return None
    try:
        from kaggle_proposed_4_3_core_report import make_report
        return make_report(
            backup_zip_name=config["BACKUP_ZIP_NAME"],
            output_subdir=config["OUTPUT_SUBDIR"],
            eval_targets=config["EVAL_TARGETS"],
            degraded_targets=config.get("DEGRADED_TARGETS", []),
            degraded_degradations=config.get("DEGRADED_DEGRADATIONS", []),
        )
    except Exception as exc:
        print("[WARN] Could not generate plots/report:", repr(exc))
        return None


def run_5eval_degraded_runner(config):
    start_time = time.time()
    config = dict(config)
    config.setdefault("TRAIN_DATA_DIR", DEFAULT_TRAIN_DATA_DIR)
    config.setdefault("EVAL_DIR", DEFAULT_EVAL_DIR)
    config.setdefault("PRETRAINED_BACKBONE", DEFAULT_PRETRAINED_BACKBONE)
    config.setdefault("OUTPUT_SUBDIR", "phase2_loss")
    config.setdefault("S", 64.0)
    config.setdefault("M", 0.4)
    config.setdefault("H", 0.333)
    config.setdefault("GENERATE_REPORT", True)
    config.setdefault("RUN_DEGRADED_EVAL", True)

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
        "pandas",
        "matplotlib",
    ])

    train_data_dir = resolve_train_data_dir(config["TRAIN_DATA_DIR"])
    eval_dir = resolve_eval_dir(train_data_dir, config["EVAL_TARGETS"])
    num_classes = detect_num_classes(train_data_dir)

    output_root(config).mkdir(parents=True, exist_ok=True)
    restore_previous_outputs(config)
    preflight_compile(config)

    import torch

    print("TRAIN_DATA_DIR:", train_data_dir)
    print("EVAL_DIR:", eval_dir)
    print("OUTPUT_ROOT:", output_root(config))
    print("NUM_CLASSES:", num_classes)
    print("LOSS_NAME:", config["LOSS_NAME"])
    print("EPOCHS:", config["EPOCHS"])
    print("EVAL_TARGETS:", ",".join(config["EVAL_TARGETS"]))
    print("DEGRADED_TARGETS:", ",".join(config["DEGRADED_TARGETS"]))
    print("DEGRADED_DEGRADATIONS:", ",".join(config["DEGRADED_DEGRADATIONS"]))
    print("DEGRADED_SEVERITIES:", str(config["DEGRADED_SEVERITIES"]))
    print("BACKUP_ZIP_NAME:", config["BACKUP_ZIP_NAME"])
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    current_exp_dir = exp_dir(config)
    if is_complete(config, current_exp_dir):
        print(
            f"[SKIP] {config['LOSS_NAME']} already has at least "
            f"{config['EPOCHS']} epochs: {current_exp_dir}"
        )
        run_degraded_eval(config, current_exp_dir, eval_dir)
    else:
        print(
            f"[TRAIN] loss={config['LOSS_NAME']} epochs={config['EPOCHS']} "
            f"backbone_lr={config['BACKBONE_LR']} head_lr={config['HEAD_LR']}"
        )
        train_cmd = build_train_command(config, current_exp_dir, train_data_dir, eval_dir, num_classes)
        run(train_cmd, cwd=ARCFACE_DIR)
        if is_complete(config, current_exp_dir):
            run_degraded_eval(config, current_exp_dir, eval_dir)
        else:
            print(
                f"[STOP] {config['LOSS_NAME']} is not complete yet. "
                f"Resume next session: {current_exp_dir}"
            )

    print("Epochs recorded:", read_epoch_count(current_exp_dir / "metrics.json"))
    print("latest.pt:", (current_exp_dir / "latest.pt").exists())
    print("best.pth:", (current_exp_dir / "best.pth").exists())

    write_manifest(config, current_exp_dir, train_data_dir, eval_dir, num_classes)
    export_eval_csv(config)
    report_zip = generate_report(config)
    backup_zip = write_backup(config)

    elapsed_minutes = (time.time() - start_time) / 60.0
    print("REPORT_ZIP:", report_zip)
    print("BACKUP_ZIP:", backup_zip)
    print(f"Done. Elapsed minutes: {elapsed_minutes:.1f}")
