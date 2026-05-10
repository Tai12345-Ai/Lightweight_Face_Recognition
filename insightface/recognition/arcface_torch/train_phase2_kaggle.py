#!/usr/bin/env python3
"""Phase 2 Kaggle training for fixed ResNet18 backbone and loss comparison."""

import argparse
import contextlib
import csv
import io
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backbones import get_model
from losses_extended import available_phase2_losses, get_phase2_loss
from recordio_fallback import MXIndexedRecordIOFallback, unpack_image_record


class SyntheticDataset(Dataset):
    def __init__(self, length=4096, num_classes=16, image_size=112):
        self.length = length
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = torch.rand(3, self.image_size, self.image_size)
        image = (image - 0.5) / 0.5
        label = torch.tensor(index % self.num_classes, dtype=torch.long)
        return image, label


class MXFaceDataset(Dataset):
    """Lazy MXNet RecordIO dataset.

    Uses mxnet when available, with a small pure-Python RecordIO fallback for
    Kaggle/Python images where mxnet wheels are not importable.
    """

    def __init__(self, root_dir, image_size=112):
        self.root_dir = root_dir
        self.image_size = image_size
        self.mx = None
        rec_path = os.path.join(root_dir, "train.rec")
        idx_path = os.path.join(root_dir, "train.idx")

        try:
            import mxnet as mx

            self.mx = mx
            self.imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, "r")
            header, _ = mx.recordio.unpack(self.imgrec.read_idx(0))
        except Exception as exc:
            logging.warning(
                "mxnet RecordIO reader unavailable (%s). Using pure-Python fallback.",
                exc,
            )
            self.imgrec = MXIndexedRecordIOFallback(idx_path, rec_path)
            header, _ = unpack_image_record(self.imgrec.read_idx(0))

        self.num_classes = None
        if header.flag > 0:
            self.num_images = int(header.label[0])
            self.num_classes = int(header.label[1])
            self.imgidx = np.arange(1, self.num_images)
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
            self.num_images = len(self.imgidx)

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.imgidx)

    def __getitem__(self, index):
        idx = self.imgidx[index]
        if self.mx is not None:
            header, image_bytes = self.mx.recordio.unpack(self.imgrec.read_idx(idx))
        else:
            header, image_bytes = unpack_image_record(self.imgrec.read_idx(idx))
        label = header.label
        if not isinstance(label, (int, float)):
            label = label[0]
        if self.mx is not None:
            image = self.mx.image.imdecode(image_bytes).asnumpy()
            image = Image.fromarray(image.astype(np.uint8))
        else:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.transform(image), torch.tensor(int(label), dtype=torch.long)


class MarginSoftmaxHead(nn.Module):
    """Full classification head with normalized weights and margin loss."""

    def __init__(self, embedding_size, num_classes, margin_loss, fp16=False):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.margin_loss = margin_loss
        self.fp16 = fp16
        self.weight = nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))

    def forward(self, embeddings, labels):
        labels = labels.view(-1).long()
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        norm_embeddings = F.normalize(embeddings, dim=1)
        norm_weight = F.normalize(self.weight, dim=1)
        logits = F.linear(norm_embeddings, norm_weight).clamp(-1.0, 1.0)
        logits = self.margin_loss(
            logits, labels.view(-1, 1), embeddings=embeddings, norms=norms
        )
        loss = F.cross_entropy(logits, labels, ignore_index=-1)
        regularization = getattr(self.margin_loss, "_last_mag_reg", None)
        if regularization is not None:
            loss = loss + regularization
        return loss, logits, norms


class DeviceBackbone(nn.Module):
    """Adapter used by eval.verification, which feeds CPU tensors."""

    def __init__(self, backbone, device):
        super().__init__()
        self.backbone = backbone
        self.device = device

    def forward(self, images):
        return self.backbone(images.to(self.device, non_blocking=True))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2 loss comparison fine-tuning on Kaggle"
    )
    parser.add_argument(
        "--loss",
        required=True,
        choices=available_phase2_losses(),
        help="Margin loss to train.",
    )
    parser.add_argument("--backbone", default="r18", choices=["r18"])
    parser.add_argument("--pretrained-backbone", default=None)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=1.0,
        help="Linear LR warmup length in epochs before cosine annealing.",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=500,
        help="Save latest.pt every N optimizer steps. Use 0 to disable step checkpoints.",
    )
    parser.add_argument(
        "--max-train-minutes",
        type=float,
        default=0.0,
        help="Stop cleanly after this many minutes and save a resumable checkpoint. 0 disables.",
    )
    parser.add_argument("--freeze-backbone", action="store_true")

    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--val-targets",
        default="lfw,cfp_fp,agedb_30",
        help="Comma-separated verification .bin names under data-dir. Use empty string to disable.",
    )
    return parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def experiment_dir(args) -> Path:
    return Path(args.output_dir) / "phase2_loss" / f"{args.backbone}_{args.loss}"


def setup_logging(exp_dir: Path):
    exp_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(exp_dir / "train.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def json_safe_config(args, exp_dir: Path) -> Dict:
    config = vars(args).copy()
    config["experiment_dir"] = str(exp_dir)
    config["loss_name"] = args.loss
    return config


def write_json(path: Path, payload: Dict):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def amp_autocast(use_amp):
    if not use_amp:
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=True)
        except TypeError:
            pass
    return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(use_amp):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def build_dataset(args) -> Tuple[Dataset, int]:
    data_dir = args.data_dir
    if data_dir == "synthetic":
        num_classes = args.num_classes or 16
        return SyntheticDataset(num_classes=num_classes, image_size=args.image_size), num_classes

    rec_path = os.path.join(data_dir, "train.rec")
    idx_path = os.path.join(data_dir, "train.idx")
    if os.path.exists(rec_path) and os.path.exists(idx_path):
        dataset = MXFaceDataset(data_dir, image_size=args.image_size)
        num_classes = args.num_classes or dataset.num_classes
        if num_classes is None:
            raise ValueError(
                "Could not infer num_classes from RecordIO. Pass --num-classes."
            )
        return dataset, int(num_classes)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = ImageFolder(data_dir, transform=transform)
    num_classes = args.num_classes or len(dataset.classes)
    return dataset, int(num_classes)


def make_train_loader(args, dataset, device, epoch):
    generator = torch.Generator()
    generator.manual_seed(args.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        generator=generator,
    )


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
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return clean_state_dict_keys(checkpoint)
    raise ValueError("Unsupported pretrained checkpoint format.")


def load_pretrained_backbone(backbone, checkpoint_path: Optional[str]):
    if not checkpoint_path:
        raise ValueError("--pretrained-backbone is required unless --resume is used.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch_load_cpu(checkpoint_path)
    state_dict = extract_backbone_state(checkpoint)
    result = backbone.load_state_dict(state_dict, strict=False)
    logging.info(
        "Loaded pretrained backbone from %s: missing=%d unexpected=%d",
        checkpoint_path,
        len(result.missing_keys),
        len(result.unexpected_keys),
    )
    if result.missing_keys:
        logging.info("Missing keys sample: %s", result.missing_keys[:10])
    if result.unexpected_keys:
        logging.info("Unexpected keys sample: %s", result.unexpected_keys[:10])


def trainable_parameters(backbone, head, freeze_backbone):
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()
        return list(head.parameters())
    return list(backbone.parameters()) + list(head.parameters())


def build_scheduler(optimizer, total_steps, warmup_steps):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))
    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def save_checkpoint(path: Path, payload: Dict):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def make_checkpoint(
    epoch,
    global_step,
    backbone,
    head,
    optimizer,
    scheduler,
    scaler,
    config,
    metrics,
    iteration_in_epoch=0,
    epoch_state=None,
):
    return {
        "epoch": epoch,
        "iteration_in_epoch": iteration_in_epoch,
        "global_step": global_step,
        "state_dict_backbone": backbone.state_dict(),
        "state_dict_head": head.state_dict(),
        "state_optimizer": optimizer.state_dict(),
        "state_lr_scheduler": scheduler.state_dict() if scheduler is not None else None,
        "state_scaler": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "metrics": metrics,
        "epoch_state": epoch_state,
    }


def validate_resume_config(args, checkpoint_config, iteration_in_epoch):
    if not checkpoint_config:
        logging.warning("Checkpoint has no saved config; resume compatibility cannot be checked.")
        return

    if checkpoint_config.get("warmup_steps") is None and getattr(args, "warmup_steps", 0) > 0:
        raise ValueError(
            "This checkpoint was created before warmup scheduling was added. "
            "Resume it with --warmup-epochs 0, or restart this experiment."
        )

    hard_keys = ("loss", "backbone", "embedding_size", "num_classes")
    mismatches = []
    for key in hard_keys:
        saved_value = checkpoint_config.get(key)
        current_value = getattr(args, key, None)
        if saved_value is not None and current_value != saved_value:
            mismatches.append(f"{key}: checkpoint={saved_value!r} current={current_value!r}")
    if mismatches:
        raise ValueError(
            "Refusing to resume from an incompatible checkpoint. "
            + "; ".join(mismatches)
        )

    soft_keys = (
        "epochs",
        "batch_size",
        "lr",
        "seed",
        "image_size",
        "freeze_backbone",
        "warmup_epochs",
        "warmup_steps",
        "total_steps",
    )
    for key in soft_keys:
        saved_value = checkpoint_config.get(key)
        current_value = getattr(args, key, None)
        if saved_value is not None and current_value != saved_value:
            logging.warning(
                "Resume config differs for %s: checkpoint=%r current=%r",
                key,
                saved_value,
                current_value,
            )

    if iteration_in_epoch > 0:
        for key in ("batch_size", "seed", "image_size"):
            saved_value = checkpoint_config.get(key)
            current_value = getattr(args, key, None)
            if saved_value is not None and current_value != saved_value:
                raise ValueError(
                    f"Cannot resume inside an epoch after changing {key}. "
                    "Use the original setting or resume from an epoch-end checkpoint."
                )


def load_latest_if_requested(args, exp_dir, backbone, head, optimizer, scheduler, scaler):
    if not args.resume:
        return 0, 0, 0, None, None

    latest_path = exp_dir / "latest.pt"
    if not latest_path.exists():
        raise FileNotFoundError(f"--resume requested but {latest_path} does not exist")

    checkpoint = torch_load_cpu(latest_path)
    iteration_in_epoch = int(checkpoint.get("iteration_in_epoch", 0))
    validate_resume_config(args, checkpoint.get("config"), iteration_in_epoch)

    backbone.load_state_dict(checkpoint["state_dict_backbone"])
    head.load_state_dict(checkpoint["state_dict_head"])
    optimizer.load_state_dict(checkpoint["state_optimizer"])
    if scheduler is not None and checkpoint.get("state_lr_scheduler") is not None:
        scheduler.load_state_dict(checkpoint["state_lr_scheduler"])
    if scaler is not None and checkpoint.get("state_scaler") is not None:
        scaler.load_state_dict(checkpoint["state_scaler"])

    start_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    metrics = checkpoint.get("metrics")
    epoch_state = checkpoint.get("epoch_state")
    logging.info(
        "Resumed from %s at epoch=%d iteration=%d step=%d",
        latest_path,
        start_epoch,
        iteration_in_epoch,
        global_step,
    )
    return start_epoch, iteration_in_epoch, global_step, metrics, epoch_state


def evaluate_if_available(backbone, data_dir, val_targets, device, batch_size):
    if not val_targets:
        return {}

    try:
        from eval import verification
    except Exception as exc:
        logging.warning(
            "eval.verification unavailable (%s). Using PIL verification fallback.",
            exc,
        )
        verification = None

    results = {}
    eval_model = DeviceBackbone(backbone, device)
    eval_model.eval()

    for target in val_targets:
        bin_path = os.path.join(data_dir, target + ".bin")
        if not os.path.exists(bin_path):
            continue
        logging.info("Evaluating %s", target)
        if verification is not None:
            data_set = verification.load_bin(bin_path, (112, 112))
            _, _, acc, std, xnorm, _ = verification.test(data_set, eval_model, batch_size, 10)
        else:
            from eval_degraded_phase2 import (
                DegradationCase,
                evaluate_embeddings,
                extract_embeddings,
                load_bin_images,
            )

            images, issame_list = load_bin_images(Path(bin_path))
            embeddings = extract_embeddings(
                images,
                backbone,
                DegradationCase("clean", "clean"),
                batch_size,
                device,
                seed=0,
            )
            acc, std, _ = evaluate_embeddings(embeddings, issame_list)
            xnorm = 0.0
        results[target] = {"accuracy": float(acc), "std": float(std), "xnorm": float(xnorm)}
        logging.info("%s accuracy=%.5f std=%.5f xnorm=%.3f", target, acc, std, xnorm)
    return results


def main():
    args = parse_args()
    setup_seed(args.seed)

    exp_dir = experiment_dir(args)
    setup_logging(exp_dir)
    config = json_safe_config(args, exp_dir)
    write_json(exp_dir / "config.json", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")
    logging.info("Device: %s fp16=%s", device, use_amp)

    dataset, num_classes = build_dataset(args)
    args.num_classes = num_classes
    config["num_classes"] = num_classes
    config["num_images"] = len(dataset)
    write_json(exp_dir / "config.json", config)

    train_loader = make_train_loader(args, dataset, device, epoch=0)
    if len(train_loader) == 0:
        raise ValueError("Training dataloader is empty. Reduce --batch-size or check data-dir.")
    steps_per_epoch = len(train_loader)

    backbone = get_model(
        args.backbone,
        dropout=0.0,
        fp16=use_amp,
        num_features=args.embedding_size,
    ).to(device)

    margin_loss, _ = get_phase2_loss(args.loss)
    head = MarginSoftmaxHead(
        embedding_size=args.embedding_size,
        num_classes=num_classes,
        margin_loss=margin_loss,
        fp16=use_amp,
    ).to(device)

    if not args.resume:
        load_pretrained_backbone(backbone, args.pretrained_backbone)

    params = trainable_parameters(backbone, head, args.freeze_backbone)
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(round(steps_per_epoch * max(0.0, args.warmup_epochs)))
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    args.total_steps = total_steps
    args.warmup_steps = warmup_steps
    config["steps_per_epoch"] = steps_per_epoch
    config["total_steps"] = total_steps
    config["warmup_steps"] = warmup_steps
    write_json(exp_dir / "config.json", config)

    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)
    scaler = make_grad_scaler(use_amp)

    (
        start_epoch,
        resume_iteration,
        global_step,
        resumed_metrics,
        resumed_epoch_state,
    ) = load_latest_if_requested(
        args, exp_dir, backbone, head, optimizer, scheduler, scaler
    )
    if resume_iteration >= steps_per_epoch:
        start_epoch += 1
        resume_iteration = 0
        resumed_epoch_state = None

    metrics = resumed_metrics or {
        "best_score": None,
        "best_epoch": None,
        "best_metric": None,
        "epochs": [],
    }
    val_targets = [item.strip() for item in args.val_targets.split(",") if item.strip()]

    logging.info("Experiment dir: %s", exp_dir)
    logging.info(
        "Training loss=%s backbone=%s classes=%d images=%d batch_size=%d lr=%g",
        args.loss,
        args.backbone,
        num_classes,
        len(dataset),
        args.batch_size,
        args.lr,
    )
    logging.info(
        "steps_per_epoch=%d save_every_steps=%d max_train_minutes=%.1f",
        steps_per_epoch,
        args.save_every_steps,
        args.max_train_minutes,
    )
    logging.info(
        "lr_schedule=linear_warmup_cosine total_steps=%d warmup_steps=%d",
        total_steps,
        warmup_steps,
    )

    if args.freeze_backbone:
        logging.info("Backbone is frozen. Training classification head only.")
    else:
        backbone.train()
    head.train()

    run_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        train_loader = make_train_loader(args, dataset, device, epoch=epoch)
        epoch_start = time.time()
        if not args.freeze_backbone:
            backbone.train()
        head.train()

        skip_until_iteration = resume_iteration if epoch == start_epoch else 0
        if skip_until_iteration > 0:
            logging.info(
                "Resuming inside epoch %d: skipping first %d/%d batches",
                epoch + 1,
                skip_until_iteration,
                len(train_loader),
            )

        if epoch == start_epoch and resumed_epoch_state:
            loss_sum = float(resumed_epoch_state.get("loss_sum", 0.0))
            norm_sum = float(resumed_epoch_state.get("norm_sum", 0.0))
            sample_count = int(resumed_epoch_state.get("sample_count", 0))
        else:
            loss_sum = 0.0
            norm_sum = 0.0
            sample_count = 0

        for iteration, (images, labels) in enumerate(train_loader, start=1):
            if iteration <= skip_until_iteration:
                continue

            global_step += 1
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            amp_context = amp_autocast(use_amp) if device.type == "cuda" else contextlib.nullcontext()
            with amp_context:
                if args.freeze_backbone:
                    with torch.no_grad():
                        embeddings = backbone(images)
                else:
                    embeddings = backbone(images)
                loss, _, norms = head(embeddings, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()
            scheduler.step()

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            norm_sum += norms.detach().mean().item() * batch_size
            sample_count += batch_size

            if iteration % args.log_every == 0 or iteration == len(train_loader):
                row = {
                    "event": "iter",
                    "epoch": epoch + 1,
                    "iteration": iteration,
                    "global_step": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss": loss.item(),
                    "epoch_loss": "",
                    "mean_norm": norms.detach().mean().item(),
                    "elapsed_sec": int(time.time() - epoch_start),
                }
                append_csv(exp_dir / "train_log.csv", row)
                logging.info(
                    "epoch=%d/%d iter=%d/%d step=%d loss=%.4f lr=%.6g",
                    epoch + 1,
                    args.epochs,
                    iteration,
                    len(train_loader),
                    global_step,
                    loss.item(),
                    optimizer.param_groups[0]["lr"],
                )

            should_save_step = (
                args.save_every_steps > 0 and global_step % args.save_every_steps == 0
            )
            time_limit_hit = (
                args.max_train_minutes > 0
                and (time.time() - run_start) >= args.max_train_minutes * 60.0
            )
            if should_save_step or time_limit_hit:
                checkpoint = make_checkpoint(
                    epoch=epoch,
                    iteration_in_epoch=iteration,
                    global_step=global_step,
                    backbone=backbone,
                    head=head,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    metrics=metrics,
                    epoch_state={
                        "loss_sum": float(loss_sum),
                        "norm_sum": float(norm_sum),
                        "sample_count": int(sample_count),
                    },
                )
                save_checkpoint(exp_dir / "latest.pt", checkpoint)
                logging.info(
                    "Saved resumable checkpoint at epoch=%d iter=%d step=%d",
                    epoch + 1,
                    iteration,
                    global_step,
                )
                if time_limit_hit:
                    logging.info(
                        "max_train_minutes reached. Stop cleanly; rerun with --resume."
                    )
                    return

        epoch_loss = loss_sum / max(1, sample_count)
        epoch_norm = norm_sum / max(1, sample_count)
        eval_metrics = {}
        should_eval = args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0)
        if should_eval:
            backbone.eval()
            eval_metrics = evaluate_if_available(
                backbone, args.data_dir, val_targets, device, args.batch_size
            )

        if eval_metrics:
            score = float(np.mean([item["accuracy"] for item in eval_metrics.values()]))
            best_metric = "mean_validation_accuracy"
        else:
            score = -float(epoch_loss)
            best_metric = "negative_train_loss"

        is_best = metrics["best_score"] is None or score > metrics["best_score"]
        if is_best:
            metrics["best_score"] = score
            metrics["best_epoch"] = epoch + 1
            metrics["best_metric"] = best_metric

        epoch_record = {
            "epoch": epoch + 1,
            "loss": float(epoch_loss),
            "mean_norm": float(epoch_norm),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": int(time.time() - epoch_start),
            "eval": eval_metrics,
        }
        metrics["epochs"].append(epoch_record)
        write_json(exp_dir / "metrics.json", metrics)

        append_csv(
            exp_dir / "train_log.csv",
            {
                "event": "epoch",
                "epoch": epoch + 1,
                "iteration": 0,
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "loss": "",
                "epoch_loss": epoch_loss,
                "mean_norm": epoch_norm,
                "elapsed_sec": epoch_record["elapsed_sec"],
            },
        )

        checkpoint = make_checkpoint(
            epoch=epoch + 1,
            global_step=global_step,
            backbone=backbone,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            metrics=metrics,
        )
        save_checkpoint(exp_dir / "latest.pt", checkpoint)
        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0):
            save_checkpoint(exp_dir / f"epoch_{epoch + 1:04d}.pt", checkpoint)
        if is_best:
            save_checkpoint(exp_dir / "best.pth", checkpoint)

        logging.info(
            "epoch=%d done loss=%.4f mean_norm=%.3f best_epoch=%s best_metric=%s",
            epoch + 1,
            epoch_loss,
            epoch_norm,
            metrics["best_epoch"],
            metrics["best_metric"],
        )

    logging.info("Training complete. Outputs are in %s", exp_dir)


if __name__ == "__main__":
    main()
