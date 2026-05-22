#!/usr/bin/env python3
"""Six-degradation verification evaluation for Phase 2 checkpoints."""

import argparse
import csv
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backbones import get_model
from degradation.transforms import DegradationTransform, SUPPORTED_DEGRADATIONS


IMAGE_SIZE = (112, 112)
EMBEDDING_SIZE = 512
METRICS_FIELDS = [
    "checkpoint",
    "target",
    "condition",
    "degradation",
    "severity",
    "accuracy",
    "drop_from_clean",
    "best_threshold",
    "std",
]
SUMMARY_FIELDS = [
    "checkpoint",
    "degradation",
    "clean_avg",
    "degraded_avg",
    "drop",
    "relative_drop",
]


def parse_csv_items(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_severities(value: str) -> List[int]:
    severities = []
    for item in parse_csv_items(value):
        severity = int(item)
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be in [1, 5], got {severity}")
        severities.append(severity)
    if not severities:
        raise ValueError("At least one severity is required.")
    return severities


def torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    prefixes = ("module.", "backbone.", "model.", "net.")
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def is_raw_state_dict(value) -> bool:
    return isinstance(value, dict) and value and all(
        torch.is_tensor(item) for item in value.values()
    )


def extract_backbone_state(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, torch.nn.Module):
        return clean_state_dict_keys(checkpoint.state_dict())

    if isinstance(checkpoint, dict):
        for key in (
            "state_dict_backbone",
            "backbone",
            "backbone_state_dict",
            "model",
            "state_dict",
        ):
            value = checkpoint.get(key)
            if isinstance(value, torch.nn.Module):
                return clean_state_dict_keys(value.state_dict())
            if is_raw_state_dict(value):
                return clean_state_dict_keys(value)

        if is_raw_state_dict(checkpoint):
            return clean_state_dict_keys(checkpoint)

    raise ValueError(
        "Unsupported checkpoint format. Expected Phase 2 checkpoint or raw state_dict."
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def load_backbone(args, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    backbone = get_model(
        args.backbone,
        dropout=0.0,
        fp16=use_fp16,
        num_features=EMBEDDING_SIZE,
    )
    checkpoint = torch_load_cpu(checkpoint_path)
    state_dict = extract_backbone_state(checkpoint)
    result = backbone.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(
            "WARNING: missing backbone keys "
            f"({len(result.missing_keys)}): {result.missing_keys[:10]}"
        )
    if result.unexpected_keys:
        print(
            "WARNING: unexpected checkpoint keys "
            f"({len(result.unexpected_keys)}): {result.unexpected_keys[:10]}"
        )
    return backbone.to(device).eval()


def load_bin_images(path: Path, image_size: Tuple[int, int] = IMAGE_SIZE):
    try:
        with open(path, "rb") as f:
            bins, issame_list = pickle.load(f)
    except UnicodeDecodeError:
        with open(path, "rb") as f:
            bins, issame_list = pickle.load(f, encoding="bytes")

    num_images = len(issame_list) * 2
    if len(bins) < num_images:
        raise ValueError(f"{path} has {len(bins)} images but {num_images} are required.")

    images = np.empty((num_images, image_size[1], image_size[0], 3), dtype=np.uint8)
    for idx in range(num_images):
        image = Image.open(io.BytesIO(bins[idx])).convert("RGB")
        if image.size != image_size:
            image = image.resize(image_size, Image.BICUBIC)
        images[idx] = np.asarray(image, dtype=np.uint8)
        if idx % 1000 == 0:
            print(f"  loading bin: {idx}/{num_images}")
    return images, list(issame_list)


def apply_degradation(
    images: np.ndarray,
    degradation: str,
    severity: int,
    seed: int,
) -> np.ndarray:
    transform = DegradationTransform(degradation, severity=severity, seed=seed)
    degraded = np.empty_like(images)
    for idx in range(images.shape[0]):
        degraded[idx] = transform.apply(images[idx])
        if idx % 1000 == 0:
            print(f"  degrading {degradation}_s{severity}: {idx}/{images.shape[0]}")
    return degraded


def make_batch(images: np.ndarray, start: int, end: int, flip: bool) -> torch.Tensor:
    batch_images = images[start:end]
    if flip:
        batch_images = batch_images[:, :, ::-1, :].copy()
    batch = batch_images.transpose(0, 3, 1, 2).astype(np.float32)
    batch = ((batch / 255.0) - 0.5) / 0.5
    return torch.from_numpy(batch)


@torch.no_grad()
def extract_embeddings(
    images: np.ndarray,
    backbone: torch.nn.Module,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    embeddings_list = []
    for flip in (False, True):
        chunks = []
        for start in range(0, images.shape[0], batch_size):
            end = min(start + batch_size, images.shape[0])
            batch = make_batch(images, start, end, flip).to(device, non_blocking=True)
            output = backbone(batch).detach().cpu().float().numpy()
            chunks.append(output)
        embeddings_list.append(np.concatenate(chunks, axis=0))

    embeddings = embeddings_list[0] + embeddings_list[1]
    return preprocessing.normalize(embeddings)


def calculate_accuracy(threshold: float, dist: np.ndarray, actual_issame: np.ndarray):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def evaluate_embeddings(
    embeddings: np.ndarray,
    issame_list: Sequence[bool],
    nfolds: int = 10,
) -> Tuple[float, float, float]:
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    actual = np.asarray(issame_list).astype(bool)
    n_pairs = min(len(actual), embeddings1.shape[0])
    if n_pairs < 2:
        raise ValueError("Need at least two verification pairs for evaluation.")

    embeddings1 = embeddings1[:n_pairs]
    embeddings2 = embeddings2[:n_pairs]
    actual = actual[:n_pairs]
    dist = np.sum(np.square(np.subtract(embeddings1, embeddings2)), axis=1)
    indices = np.arange(n_pairs)
    fold_count = min(nfolds, n_pairs)
    folds: Iterable[Tuple[np.ndarray, np.ndarray]]
    if fold_count > 1:
        folds = KFold(n_splits=fold_count, shuffle=False).split(indices)
    else:
        folds = [(indices, indices)]

    accuracies = []
    best_thresholds = []
    for train_set, test_set in folds:
        train_acc = np.zeros((len(thresholds),), dtype=np.float32)
        for idx, threshold in enumerate(thresholds):
            _, _, train_acc[idx] = calculate_accuracy(
                threshold, dist[train_set], actual[train_set]
            )
        best_threshold = float(thresholds[int(np.argmax(train_acc))])
        _, _, acc = calculate_accuracy(best_threshold, dist[test_set], actual[test_set])
        accuracies.append(acc)
        best_thresholds.append(best_threshold)

    return (
        float(np.mean(accuracies)),
        float(np.std(accuracies)),
        float(np.mean(best_thresholds)),
    )


def metric_row(
    checkpoint_label: str,
    target: str,
    condition: str,
    degradation: str,
    severity,
    accuracy: float,
    clean_accuracy: float,
    best_threshold: float,
    std: float,
):
    return {
        "checkpoint": checkpoint_label,
        "target": target,
        "condition": condition,
        "degradation": degradation,
        "severity": severity,
        "accuracy": accuracy,
        "drop_from_clean": clean_accuracy - accuracy,
        "best_threshold": best_threshold,
        "std": std,
    }


def evaluate_target(
    target: str,
    bin_path: Path,
    checkpoint_label: str,
    backbone: torch.nn.Module,
    degradations: Sequence[str],
    severities: Sequence[int],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> List[Dict[str, object]]:
    print(f"\n=== {target} ===")
    images, issame_list = load_bin_images(bin_path)
    rows = []

    print("  evaluating clean")
    embeddings = extract_embeddings(images, backbone, batch_size, device)
    clean_acc, clean_std, clean_threshold = evaluate_embeddings(embeddings, issame_list)
    rows.append(
        metric_row(
            checkpoint_label,
            target,
            "clean",
            "clean",
            "",
            clean_acc,
            clean_acc,
            clean_threshold,
            clean_std,
        )
    )
    print(f"  clean acc={clean_acc:.5f} threshold={clean_threshold:.3f} std={clean_std:.5f}")

    for degradation in degradations:
        for severity in severities:
            condition = f"{degradation}_s{severity}"
            print(f"  evaluating {condition}")
            degraded_images = apply_degradation(
                images,
                degradation,
                severity,
                seed + severity * 1009,
            )
            embeddings = extract_embeddings(degraded_images, backbone, batch_size, device)
            acc, std, best_threshold = evaluate_embeddings(embeddings, issame_list)
            rows.append(
                metric_row(
                    checkpoint_label,
                    target,
                    condition,
                    degradation,
                    severity,
                    acc,
                    clean_acc,
                    best_threshold,
                    std,
                )
            )
            print(
                f"  {condition} acc={acc:.5f} "
                f"drop={clean_acc - acc:.5f} threshold={best_threshold:.3f} std={std:.5f}"
            )

    return rows


def summarize_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summaries = []
    checkpoints = []
    for row in rows:
        checkpoint = row["checkpoint"]
        if checkpoint not in checkpoints:
            checkpoints.append(checkpoint)

    for checkpoint in checkpoints:
        checkpoint_rows = [row for row in rows if row["checkpoint"] == checkpoint]
        clean_scores = [
            float(row["accuracy"])
            for row in checkpoint_rows
            if row["condition"] == "clean"
        ]
        clean_avg = float(np.mean(clean_scores)) if clean_scores else None
        degradations = []
        for row in checkpoint_rows:
            degradation = row["degradation"]
            if degradation != "clean" and degradation not in degradations:
                degradations.append(degradation)

        for degradation in degradations:
            degraded_scores = [
                float(row["accuracy"])
                for row in checkpoint_rows
                if row["degradation"] == degradation
            ]
            degraded_avg = float(np.mean(degraded_scores)) if degraded_scores else None
            if clean_avg is not None and degraded_avg is not None:
                drop = clean_avg - degraded_avg
                relative_drop = drop / clean_avg if clean_avg else None
            else:
                drop = None
                relative_drop = None
            summaries.append(
                {
                    "checkpoint": checkpoint,
                    "degradation": degradation,
                    "clean_avg": clean_avg,
                    "degraded_avg": degraded_avg,
                    "drop": drop,
                    "relative_drop": relative_drop,
                }
            )
    return summaries


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in fieldnames})


def write_outputs(output_dir: Path, rows: Sequence[Dict[str, object]], config: Dict[str, object]):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize_rows(rows)
    write_csv(output_dir / "degraded_metrics.csv", rows, METRICS_FIELDS)
    write_csv(output_dir / "degraded_summary.csv", summary_rows, SUMMARY_FIELDS)
    with open(output_dir / "degraded_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "metrics": list(rows),
                "summary": summary_rows,
            },
            f,
            indent=2,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Six-degradation Phase 2 eval")
    parser.add_argument("--backbone", default="r18", choices=["r18"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-label", default=None)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--targets", default="lfw,cfp_fp,cplfw,agedb_30,calfw")
    parser.add_argument(
        "--degradations",
        default=",".join(SUPPORTED_DEGRADATIONS),
        help="Comma-separated degradation names.",
    )
    parser.add_argument("--severities", default="3")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    targets = parse_csv_items(args.targets)
    degradations = parse_csv_items(args.degradations)
    unknown = sorted(set(degradations) - set(SUPPORTED_DEGRADATIONS))
    if unknown:
        raise ValueError(
            f"Unknown degradation(s): {unknown}. Supported: {SUPPORTED_DEGRADATIONS}"
        )
    severities = parse_severities(args.severities)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint_label = args.checkpoint_label or str(checkpoint_path)
    print("checkpoint:", checkpoint_path)
    print("checkpoint_label:", checkpoint_label)
    print("targets:", ",".join(targets))
    print("degradations:", ",".join(degradations))
    print("severities:", ",".join(str(item) for item in severities))
    print("device:", device)

    backbone = load_backbone(args, checkpoint_path, device)
    all_rows: List[Dict[str, object]] = []
    data_dir = Path(args.data_dir)
    for target in targets:
        bin_path = data_dir / f"{target}.bin"
        if not bin_path.exists():
            print(f"WARNING: {bin_path} not found; skipping {target}.")
            continue
        rows = evaluate_target(
            target,
            bin_path,
            checkpoint_label,
            backbone,
            degradations,
            severities,
            args.batch_size,
            device,
            args.seed,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("WARNING: no eval rows were produced. Check --data-dir and --targets.")

    output_dir = Path(args.output)
    write_outputs(
        output_dir,
        all_rows,
        {
            "backbone": args.backbone,
            "checkpoint": str(checkpoint_path),
            "checkpoint_label": checkpoint_label,
            "data_dir": str(data_dir),
            "targets": targets,
            "degradations": degradations,
            "severities": severities,
            "batch_size": args.batch_size,
            "fp16": bool(args.fp16),
            "device": str(device),
            "seed": args.seed,
        },
    )
    print(f"\nWrote degraded eval outputs to {output_dir}")


if __name__ == "__main__":
    main()
