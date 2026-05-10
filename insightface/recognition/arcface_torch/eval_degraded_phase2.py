#!/usr/bin/env python3
"""Degraded verification evaluation for Phase 2 loss comparison.

This script evaluates trained Phase 2 backbones on clean and degraded
verification .bin files. It does not touch the training data or training loop.
"""

import argparse
import csv
import io
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from sklearn import preprocessing
from sklearn.model_selection import KFold
from torchvision import transforms

from backbones import get_model


IMAGE_SIZE = (112, 112)
LOSS_EVAL_ORDER = [
    "arcface",
    "adaface",
    "curricularface",
    "proposed",
    "magface",
    "elasticface",
    "cosface",
]

TO_TENSOR = transforms.ToTensor()
NORMALIZE = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


@dataclass(frozen=True)
class DegradationCase:
    name: str
    severity: str
    value: Optional[float] = None


def parse_csv_items(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> List[int]:
    return [int(item) for item in parse_csv_items(value)]


def parse_float_csv(value: str) -> List[float]:
    return [float(item) for item in parse_csv_items(value)]


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
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def extract_backbone_state(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in (
            "state_dict_backbone",
            "backbone",
            "backbone_state_dict",
            "model",
            "state_dict",
        ):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return clean_state_dict_keys(checkpoint[key])
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return clean_state_dict_keys(checkpoint)
    raise ValueError("Unsupported checkpoint format. Expected Phase 2 checkpoint or raw state_dict.")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA is not available; using CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def load_backbone(args, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    backbone = get_model(
        args.backbone,
        dropout=0,
        fp16=use_fp16,
        num_features=args.embedding_size,
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


def build_degradation_cases(args) -> List[DegradationCase]:
    names = parse_csv_items(args.degradations.lower())
    valid = {"clean", "blur", "lowres", "jpeg", "brightness", "noise"}
    unknown = sorted(set(names) - valid)
    if unknown:
        raise ValueError(f"Unknown degradation(s): {unknown}. Supported: {sorted(valid)}")

    cases = [DegradationCase("clean", "clean")]
    for name in names:
        if name == "clean":
            continue
        if name == "blur":
            for kernel in parse_int_csv(args.blur_kernels):
                if kernel <= 0 or kernel % 2 == 0:
                    raise ValueError("--blur-kernels values must be positive odd integers.")
                cases.append(DegradationCase("blur", f"k{kernel}", float(kernel)))
        elif name == "lowres":
            for scale in parse_float_csv(args.lowres_scales):
                if scale <= 0:
                    raise ValueError("--lowres-scales values must be positive.")
                cases.append(DegradationCase("lowres", f"s{scale:g}", float(scale)))
        elif name == "jpeg":
            for quality in parse_int_csv(args.jpeg_quality):
                if quality < 1 or quality > 100:
                    raise ValueError("--jpeg-quality values must be in [1, 100].")
                cases.append(DegradationCase("jpeg", f"q{quality}", float(quality)))
        elif name == "brightness":
            for factor in parse_float_csv(args.brightness_factors):
                if factor <= 0:
                    raise ValueError("--brightness-factors values must be positive.")
                cases.append(DegradationCase("brightness", f"b{factor:g}", float(factor)))
        elif name == "noise":
            for std in parse_float_csv(args.noise_stds):
                if std < 0:
                    raise ValueError("--noise-stds values must be non-negative.")
                cases.append(DegradationCase("noise", f"std{std:g}", float(std)))
    return cases


def apply_pil_degradation(image: Image.Image, case: DegradationCase) -> Image.Image:
    if case.name == "clean" or case.name == "noise":
        return image
    if case.name == "blur":
        return transforms.GaussianBlur(kernel_size=int(case.value))(image)
    if case.name == "lowres":
        scale = float(case.value)
        width, height = image.size
        small_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(small_size, Image.BICUBIC).resize(image.size, Image.BICUBIC)
    if case.name == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(case.value))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if case.name == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(case.value))
    raise ValueError(f"Unsupported degradation: {case.name}")


def image_to_tensor(image_array: np.ndarray, case: DegradationCase, seed: int, index: int):
    image = Image.fromarray(image_array).convert("RGB")
    image = apply_pil_degradation(image, case)
    tensor = TO_TENSOR(image)
    if case.name == "noise":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + index * 1009 + int(float(case.value) * 100000))
        noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
        tensor = torch.clamp(tensor + noise * float(case.value), 0.0, 1.0)
    return NORMALIZE(tensor)


def make_batch(
    images: np.ndarray,
    start: int,
    end: int,
    case: DegradationCase,
    flip: bool,
    seed: int,
) -> torch.Tensor:
    tensors = []
    for offset, idx in enumerate(range(start, end)):
        tensor = image_to_tensor(images[idx], case, seed, start + offset)
        if flip:
            tensor = torch.flip(tensor, dims=[2])
        tensors.append(tensor)
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def extract_embeddings(
    images: np.ndarray,
    backbone: torch.nn.Module,
    case: DegradationCase,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    embeddings_list = []
    for flip in (False, True):
        chunks = []
        for start in range(0, images.shape[0], batch_size):
            end = min(start + batch_size, images.shape[0])
            batch = make_batch(images, start, end, case, flip, seed).to(
                device, non_blocking=True
            )
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

    return float(np.mean(accuracies)), float(np.std(accuracies)), float(np.mean(best_thresholds))


def collect_checkpoints(args) -> List[Tuple[Path, str]]:
    if bool(args.checkpoint) == bool(args.checkpoint_dir):
        raise ValueError("Pass exactly one of --checkpoint or --checkpoint-dir.")

    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        return [(path, str(path))]

    root = Path(args.checkpoint_dir)
    checkpoints = []
    for loss_name in LOSS_EVAL_ORDER:
        exp_dir = root / f"{args.backbone}_{loss_name}"
        best = exp_dir / "best.pth"
        latest = exp_dir / "latest.pt"
        if best.exists():
            checkpoints.append((best, str(best.relative_to(root))))
        elif latest.exists():
            checkpoints.append((latest, str(latest.relative_to(root))))
        else:
            print(f"WARNING: no best.pth/latest.pt found for {exp_dir}; skipping.")
    if not checkpoints:
        raise FileNotFoundError(f"No Phase 2 checkpoints found under {root}")
    return checkpoints


def evaluate_checkpoint(
    args,
    checkpoint_path: Path,
    checkpoint_label: str,
    cases: Sequence[DegradationCase],
    device: torch.device,
) -> List[Dict[str, object]]:
    print(f"\n=== Checkpoint: {checkpoint_label} ===")
    backbone = load_backbone(args, checkpoint_path, device)
    rows = []
    targets = parse_csv_items(args.targets)
    for target in targets:
        bin_path = Path(args.data_dir) / f"{target}.bin"
        if not bin_path.exists():
            print(f"WARNING: {bin_path} not found; skipping {target}.")
            continue
        print(f"\n--- {target} ---")
        images, issame_list = load_bin_images(bin_path)
        for case in cases:
            print(f"  evaluating {case.name}:{case.severity}")
            embeddings = extract_embeddings(
                images,
                backbone,
                case,
                args.batch_size,
                device,
                args.seed,
            )
            accuracy, std, best_threshold = evaluate_embeddings(embeddings, issame_list)
            rows.append(
                {
                    "checkpoint": checkpoint_label,
                    "target": target,
                    "degradation": case.name,
                    "severity": case.severity,
                    "accuracy": accuracy,
                    "best_threshold": best_threshold,
                    "std": std,
                }
            )
            print(
                f"  {case.name}:{case.severity} "
                f"acc={accuracy:.4f} threshold={best_threshold:.3f} std={std:.4f}"
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
            if row["degradation"] == "clean"
        ]
        degraded_scores = [
            float(row["accuracy"])
            for row in checkpoint_rows
            if row["degradation"] != "clean"
        ]
        clean_avg = float(np.mean(clean_scores)) if clean_scores else None
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


def write_outputs(output_dir: Path, rows: Sequence[Dict[str, object]]):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize_rows(rows)
    metrics_fields = [
        "checkpoint",
        "target",
        "degradation",
        "severity",
        "accuracy",
        "best_threshold",
        "std",
    ]
    summary_fields = [
        "checkpoint",
        "clean_avg",
        "degraded_avg",
        "drop",
        "relative_drop",
    ]
    write_csv(output_dir / "degraded_metrics.csv", rows, metrics_fields)
    write_csv(output_dir / "degraded_summary.csv", summary_rows, summary_fields)
    with open(output_dir / "degraded_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": list(rows), "summary": summary_rows}, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 degraded verification evaluation")
    parser.add_argument("--backbone", default="r18", choices=["r18"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--targets", default="lfw,cfp_fp,agedb_30")
    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--degradations",
        default="clean,blur,lowres,jpeg,brightness,noise",
        help="Comma-separated list from clean,blur,lowres,jpeg,brightness,noise.",
    )
    parser.add_argument("--blur-kernels", default="3,5,7")
    parser.add_argument("--lowres-scales", default="0.75,0.5,0.25")
    parser.add_argument("--jpeg-quality", default="70,50,30")
    parser.add_argument("--brightness-factors", default="0.7,1.3")
    parser.add_argument("--noise-stds", default="0.03,0.06")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    cases = build_degradation_cases(args)
    checkpoints = collect_checkpoints(args)

    all_rows: List[Dict[str, object]] = []
    for checkpoint_path, checkpoint_label in checkpoints:
        rows = evaluate_checkpoint(args, checkpoint_path, checkpoint_label, cases, device)
        all_rows.extend(rows)

    output_dir = Path(args.output)
    write_outputs(output_dir, all_rows)
    print(f"\nWrote degraded metrics to {output_dir}")


if __name__ == "__main__":
    main()
