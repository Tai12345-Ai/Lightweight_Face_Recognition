#!/usr/bin/env python3
"""Build offline multi-UI centers from CASIA-WebFace + pretrained backbone.

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


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class DegradedImageDataset(Dataset):
    """Loads images, applies a degradation, and returns tensor."""

    def __init__(self, image_paths, degradation_type, severity, image_size=112):
        self.image_paths = image_paths
        self.image_size = image_size
        if degradation_type is not None:
            self.transform = DegradationTransform(degradation_type, severity=severity, seed=42)
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(str(path))
        if img is None:
            # Return a black image if read fails
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.image_size, self.image_size))
        if self.transform is not None:
            img = self.transform.apply(img)
        # BGR->RGB, HWC->CHW, normalize to [-1, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)


def collect_image_paths(data_dir, num_samples, seed=42):
    """Collect up to num_samples image paths from data_dir (class-folder layout)."""
    data_dir = Path(data_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_paths = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in exts:
                all_paths.append(f)
    rng = random.Random(seed)
    rng.shuffle(all_paths)
    return all_paths[:num_samples]


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
    parser.add_argument("--data-dir", required=True, help="CASIA-WebFace image-folder root")
    parser.add_argument("--pretrained-backbone", required=True, help="Path to backbone.pth")
    parser.add_argument("--backbone", default="r18", choices=["r18"])
    parser.add_argument("--output", required=True, help="Output .pth path")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
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

    # Collect images
    image_paths = collect_image_paths(data_dir, args.num_samples)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {data_dir}")
    logger.info("Collected %d image paths from %s", len(image_paths), data_dir)

    # Load backbone
    backbone = get_model(
        args.backbone, dropout=0.0, fp16=args.fp16, num_features=512
    ).to(device)
    ckpt = torch.load(args.pretrained_backbone, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict_backbone" in ckpt:
        backbone.load_state_dict(ckpt["state_dict_backbone"])
    elif isinstance(ckpt, dict) and "model" in ckpt:
        backbone.load_state_dict(ckpt["model"])
    else:
        backbone.load_state_dict(ckpt)
    backbone.eval()
    logger.info("Loaded backbone from %s", args.pretrained_backbone)

    centers_list = []
    names_list = []

    # Build per-degradation centers for each severity
    for severity in severities:
        for deg_type in degradations:
            logger.info("Computing center: %s severity=%d ...", deg_type, severity)
            ds = DegradedImageDataset(image_paths, deg_type, severity)
            dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
            center = compute_center(backbone, dl, device, use_fp16=args.fp16)
            centers_list.append(center)
            names_list.append(deg_type)
            logger.info("  center norm=%.6f", center.norm().item())

    # Global center (all degradations mixed)
    if args.include_global:
        logger.info("Computing global center (mixed degradations) ...")
        all_global_embs = []
        for severity in severities:
            for deg_type in degradations:
                ds = DegradedImageDataset(image_paths[:max(1, len(image_paths) // len(degradations))], deg_type, severity)
                dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
                backbone.eval()
                for batch in dl:
                    batch = batch.to(device, non_blocking=True)
                    if args.fp16 and device.type == "cuda":
                        with torch.amp.autocast("cuda", enabled=True):
                            emb = backbone(batch)
                    else:
                        emb = backbone(batch)
                    all_global_embs.append(F.normalize(emb.float(), dim=1).cpu())
        global_cat = torch.cat(all_global_embs, dim=0)
        global_center = F.normalize(global_cat.mean(dim=0, keepdim=True), dim=1).squeeze(0)
        # Insert global at position 0
        centers_list.insert(0, global_center)
        names_list.insert(0, "global")
        logger.info("  global center norm=%.6f", global_center.norm().item())

    centers_tensor = torch.stack(centers_list, dim=0)  # [K, 512]
    payload = {
        "centers": centers_tensor,
        "names": names_list,
        "backbone": args.backbone,
        "num_samples": len(image_paths),
        "degradations": degradations,
        "severities": severities,
        "source": f"CASIA-WebFace synthetic degraded severity={','.join(str(s) for s in severities)}",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    logger.info("Saved multi-UI centers [%d, %d] to %s", *centers_tensor.shape, output)
    logger.info("Names: %s", names_list)


if __name__ == "__main__":
    main()
