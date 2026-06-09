#!/usr/bin/env python3
"""Build offline multi-UI centers from CASIA-WebFace + pretrained backbone.

Supports both:
  1. Image-folder layout:
       data_dir/class_x/*.jpg
       data_dir/class_y/*.png

  2. InsightFace/MXNet RecordIO layout:
       data_dir/train.rec
       data_dir/train.idx
       data_dir/property

Creates one normalized center per degradation type (+ optional global) by:
  1. Sampling N images from the training set.
  2. Applying a specific degradation at the given severity.
  3. Extracting embeddings with the pretrained backbone.
  4. Averaging and L2-normalizing to get the center.

Output: a .pth file containing:
  {
    "centers": Tensor[K, 512],
    "names": ["global", "gaussian_blur", ...],
    "backbone": str,
    "num_samples": int,
    "degradations": list[str],
    "severities": list[int],
    "source": str,
  }
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backbones import get_model
from degradation.transforms import DegradationTransform, SUPPORTED_DEGRADATIONS
from recordio_fallback import MXIndexedRecordIOFallback, unpack_image_record


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def is_recordio_dataset(data_dir):
    """Return True if data_dir contains InsightFace RecordIO train files."""
    data_dir = Path(data_dir)
    return (
        (data_dir / "train.rec").exists()
        and (data_dir / "train.idx").exists()
    )


def normalize_image_tensor(img, image_size):
    """BGR uint8 image -> normalized RGB tensor [3, H, W] in [-1, 1]."""
    if img is None:
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    img = cv2.resize(img, (image_size, image_size))

    # BGR -> RGB, HWC -> CHW, normalize to [-1, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)


class DegradedImageDataset(Dataset):
    """Loads image-folder samples, applies a degradation, and returns tensor."""

    def __init__(self, image_paths, degradation_type, severity, image_size=112):
        self.image_paths = list(image_paths)
        self.image_size = image_size

        if degradation_type is not None:
            self.transform = DegradationTransform(
                degradation_type,
                severity=severity,
                seed=42,
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        # cv2.imread gives BGR image
        img = cv2.imread(str(path))

        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.image_size, self.image_size))

        # Keep same behavior as old code: apply degradation before BGR->RGB
        if self.transform is not None:
            img = self.transform.apply(img)

        return normalize_image_tensor(img, self.image_size)


class DegradedRecordIODataset(Dataset):
    """Loads RecordIO samples, applies a degradation, and returns tensor.

    This is needed for Kaggle CASIA-WebFace datasets that provide:
      train.rec, train.idx, property
    instead of extracted jpg/png folders.
    """

    def __init__(self, root_dir, record_indices, degradation_type, severity, image_size=112):
        self.root_dir = Path(root_dir)
        self.record_indices = list(record_indices)
        self.image_size = image_size

        rec_path = self.root_dir / "train.rec"
        idx_path = self.root_dir / "train.idx"

        self.imgrec = MXIndexedRecordIOFallback(idx_path, rec_path)

        if degradation_type is not None:
            self.transform = DegradationTransform(
                degradation_type,
                severity=severity,
                seed=42,
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.record_indices)

    def __getitem__(self, index):
        rec_idx = self.record_indices[index]
        record = self.imgrec.read_idx(rec_idx)

        if record is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            return normalize_image_tensor(img, self.image_size)

        _, image_bytes = unpack_image_record(record)

        # Decode image bytes directly with cv2 -> BGR
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.image_size, self.image_size))

        # Keep same behavior as image-folder path: apply degradation on BGR image
        if self.transform is not None:
            img = self.transform.apply(img)

        return normalize_image_tensor(img, self.image_size)


def collect_image_paths(data_dir, num_samples, seed=42):
    """Collect up to num_samples image paths from data_dir.

    Supports common class-folder layout:
      data_dir/class_name/*.jpg
    Also supports recursive image search as fallback.
    """
    data_dir = Path(data_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    all_paths = []

    # Original behavior: class-folder layout
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in exts:
                all_paths.append(f)

    # Fallback: recursive search if class-folder scan found nothing
    if not all_paths:
        for f in sorted(data_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in exts:
                all_paths.append(f)

    rng = random.Random(seed)
    rng.shuffle(all_paths)
    return all_paths[:num_samples]


def collect_recordio_indices(data_dir, num_samples, seed=42):
    """Collect up to num_samples sample indices from InsightFace RecordIO."""
    data_dir = Path(data_dir)
    rec_path = data_dir / "train.rec"
    idx_path = data_dir / "train.idx"

    imgrec = MXIndexedRecordIOFallback(idx_path, rec_path)

    try:
        header, _ = unpack_image_record(imgrec.read_idx(0))

        if header.flag > 0:
            # InsightFace convention:
            # record 0 stores metadata, label[0] is upper bound for image ids.
            num_images = int(header.label[0])
            indices = list(range(1, num_images))
        else:
            indices = list(imgrec.keys)
            if 0 in indices:
                indices.remove(0)
    finally:
        imgrec.close()

    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[:num_samples]


def build_degraded_dataset(
    data_dir,
    sample_items,
    sample_source,
    degradation_type,
    severity,
    image_size=112,
):
    """Build either image-folder dataset or RecordIO dataset."""
    if sample_source == "recordio":
        return DegradedRecordIODataset(
            data_dir,
            sample_items,
            degradation_type,
            severity,
            image_size=image_size,
        )

    if sample_source == "image_folder":
        return DegradedImageDataset(
            sample_items,
            degradation_type,
            severity,
            image_size=image_size,
        )

    raise ValueError(f"Unknown sample_source: {sample_source}")


@torch.no_grad()
def compute_center(backbone, dataloader, device, use_fp16=False):
    """Compute mean normalized embedding over a dataloader."""
    backbone.eval()
    all_embeddings = []

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)

        if use_fp16 and device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=True):
                emb = backbone(batch)
        else:
            emb = backbone(batch)

        emb = F.normalize(emb.float(), dim=1)
        all_embeddings.append(emb.cpu())

    if not all_embeddings:
        raise RuntimeError("No embeddings computed. Check data_dir.")

    cat = torch.cat(all_embeddings, dim=0)
    center = cat.mean(dim=0, keepdim=True)
    center = F.normalize(center, dim=1)
    return center.squeeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Build offline multi-UI centers")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="CASIA-WebFace root. Supports image-folder or train.rec/train.idx RecordIO.",
    )
    parser.add_argument("--pretrained-backbone", required=True, help="Path to backbone.pth")
    parser.add_argument("--backbone", default="r18", choices=["r18"])
    parser.add_argument("--output", required=True, help="Output .pth path")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--degradations",
        default="gaussian_blur,motion_blur,low_resolution,jpeg_compression,low_illumination,alignment_perturb",
        help="Comma-separated degradation types",
    )
    parser.add_argument("--severities", default="5", help="Comma-separated severity levels")
    parser.add_argument("--include-global", action="store_true", help="Include global center from all degradations combined")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if exists")
    return parser.parse_args()


def load_backbone(backbone_name, pretrained_path, device, use_fp16=False):
    """Load pretrained backbone with several common checkpoint formats."""
    backbone = get_model(
        backbone_name,
        dropout=0.0,
        fp16=use_fp16,
        num_features=512,
    ).to(device)

    ckpt = torch.load(pretrained_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict_backbone" in ckpt:
        state_dict = ckpt["state_dict_backbone"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Remove possible DataParallel / Lightning prefixes.
    if isinstance(state_dict, dict):
        cleaned = {}
        for k, v in state_dict.items():
            new_k = k
            for prefix in ("module.", "backbone."):
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
            cleaned[new_k] = v
        state_dict = cleaned

    backbone.load_state_dict(state_dict, strict=True)
    backbone.eval()

    logger.info("Loaded backbone from %s", pretrained_path)
    return backbone


def main():
    args = parse_args()
    output = Path(args.output)

    if output.exists() and not args.overwrite:
        logger.info("Output already exists, skipping: %s", output)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Normalize data dir
    data_dir = Path(args.data_dir)

    for nested in ("casia-webface", "CASIA-WebFace"):
        if (data_dir / nested).is_dir():
            data_dir = data_dir / nested
            break

    degradations = [d.strip() for d in args.degradations.split(",") if d.strip()]
    severities = [int(s.strip()) for s in args.severities.split(",") if s.strip()]

    for d in degradations:
        if d not in SUPPORTED_DEGRADATIONS:
            raise ValueError(f"Unknown degradation: {d}. Supported: {SUPPORTED_DEGRADATIONS}")

    # Collect samples from either RecordIO or image-folder.
    if is_recordio_dataset(data_dir):
        sample_source = "recordio"
        sample_items = collect_recordio_indices(data_dir, args.num_samples)
        logger.info(
            "Collected %d RecordIO indices from %s",
            len(sample_items),
            data_dir,
        )
    else:
        sample_source = "image_folder"
        sample_items = collect_image_paths(data_dir, args.num_samples)
        logger.info(
            "Collected %d image paths from %s",
            len(sample_items),
            data_dir,
        )

    if not sample_items:
        raise FileNotFoundError(
            f"No usable images found in {data_dir}. "
            "Expected either image-folder layout or train.rec/train.idx RecordIO."
        )

    backbone = load_backbone(
        args.backbone,
        args.pretrained_backbone,
        device,
        use_fp16=args.fp16,
    )

    centers_list = []
    names_list = []

    # Build per-degradation centers for each severity.
    for severity in severities:
        for deg_type in degradations:
            logger.info("Computing center: %s severity=%d ...", deg_type, severity)

            ds = build_degraded_dataset(
                data_dir=data_dir,
                sample_items=sample_items,
                sample_source=sample_source,
                degradation_type=deg_type,
                severity=severity,
                image_size=args.image_size,
            )

            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=(device.type == "cuda"),
            )

            center = compute_center(backbone, dl, device, use_fp16=args.fp16)
            centers_list.append(center)
            names_list.append(deg_type)

            logger.info("  center norm=%.6f", center.norm().item())

    # Global center: mixed degradations.
    if args.include_global:
        logger.info("Computing global center (mixed degradations) ...")

        all_global_embs = []

        global_sample_count = max(1, len(sample_items) // max(1, len(degradations)))
        global_sample_items = sample_items[:global_sample_count]

        for severity in severities:
            for deg_type in degradations:
                ds = build_degraded_dataset(
                    data_dir=data_dir,
                    sample_items=global_sample_items,
                    sample_source=sample_source,
                    degradation_type=deg_type,
                    severity=severity,
                    image_size=args.image_size,
                )

                dl = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=False,
                    pin_memory=(device.type == "cuda"),
                )

                backbone.eval()

                for batch in dl:
                    batch = batch.to(device, non_blocking=True)

                    if args.fp16 and device.type == "cuda":
                        with torch.amp.autocast("cuda", enabled=True):
                            emb = backbone(batch)
                    else:
                        emb = backbone(batch)

                    all_global_embs.append(F.normalize(emb.float(), dim=1).cpu())

        if not all_global_embs:
            raise RuntimeError("No global embeddings computed.")

        global_cat = torch.cat(all_global_embs, dim=0)
        global_center = F.normalize(
            global_cat.mean(dim=0, keepdim=True),
            dim=1,
        ).squeeze(0)

        centers_list.insert(0, global_center)
        names_list.insert(0, "global")

        logger.info("  global center norm=%.6f", global_center.norm().item())

    centers_tensor = torch.stack(centers_list, dim=0)

    payload = {
        "centers": centers_tensor,
        "names": names_list,
        "backbone": args.backbone,
        "num_samples": len(sample_items),
        "sample_source": sample_source,
        "degradations": degradations,
        "severities": severities,
        "source": (
            f"CASIA-WebFace {sample_source} synthetic degraded "
            f"severity={','.join(str(s) for s in severities)}"
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    logger.info("Saved multi-UI centers [%d, %d] to %s", *centers_tensor.shape, output)
    logger.info("Names: %s", names_list)


if __name__ == "__main__":
    main()