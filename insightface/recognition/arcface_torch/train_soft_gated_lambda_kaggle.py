#!/usr/bin/env python3
"""Standalone Kaggle trainer for Soft-Gated Ada-CurricularFace lambda sweeps."""

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backbones import get_model
from soft_gated_losses import SoftGatedAdaCurricularFaceLoss
from train_phase2_kaggle import (
    MarginSoftmaxHead,
    amp_autocast,
    append_csv,
    build_dataset,
    build_scheduler,
    evaluate_if_available,
    load_pretrained_backbone,
    make_checkpoint,
    make_grad_scaler,
    make_train_loader,
    save_checkpoint,
    setup_logging,
    setup_seed,
    torch_load_cpu,
    trainable_parameters,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Soft-Gated Ada-CurricularFace lambda sweep trainer"
    )
    parser.add_argument(
        "--loss",
        default="soft_gated_ada_curricular",
        choices=["soft_gated_ada_curricular"],
        help="Standalone loss name for config compatibility.",
    )
    parser.add_argument("--network", "--backbone", dest="backbone", default="r18", choices=["r18"])
    parser.add_argument("--s", type=float, default=64.0)
    parser.add_argument("--m", type=float, default=0.4)
    parser.add_argument("--h", type=float, default=0.333)
    parser.add_argument("--lambda_gate", "--lambda-gate", dest="lambda_gate", type=float, required=True)
    parser.add_argument(
        "--train_data",
        "--train-data",
        "--data-dir",
        dest="data_dir",
        required=True,
        help="Training image folder or RecordIO directory. Evaluation bins are not read from here unless eval_dir is omitted.",
    )
    parser.add_argument(
        "--eval_dir",
        "--eval-dir",
        dest="eval_dir",
        default=None,
        help="Directory containing verification .bin files.",
    )
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--pretrained_backbone", "--pretrained-backbone", dest="pretrained_backbone", default=None)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from latest.pt in the lambda experiment, or from a specific checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", "--warmup-epochs", dest="warmup_epochs", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--eval_every", "--eval-every", dest="eval_every", type=int, default=1)
    parser.add_argument("--save_every", "--save-every", dest="save_every", type=int, default=1)
    parser.add_argument(
        "--save_every_steps",
        "--save-every-steps",
        dest="save_every_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--max_train_minutes",
        "--max-train-minutes",
        dest="max_train_minutes",
        type=float,
        default=0.0,
    )
    parser.add_argument("--freeze_backbone", "--freeze-backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--num_classes", "--num-classes", dest="num_classes", type=int, default=None)
    parser.add_argument(
        "--embedding_dim",
        "--embedding-size",
        dest="embedding_size",
        type=int,
        default=512,
    )
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=2)
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--image_size", "--image-size", dest="image_size", type=int, default=112)
    parser.add_argument("--log_every", "--log-every", dest="log_every", type=int, default=50)
    parser.add_argument(
        "--eval_targets",
        "--eval-targets",
        "--val-targets",
        dest="val_targets",
        default="lfw,cfp_ff,cfp_fp,agedb_30,calfw,cplfw,sllfw,talfw",
        help="Comma-separated verification .bin names under eval_dir.",
    )
    parser.add_argument("--t_alpha", "--t-alpha", dest="t_alpha", type=float, default=0.01)
    parser.add_argument(
        "--curriculum_alpha",
        "--curriculum-alpha",
        dest="curriculum_alpha",
        type=float,
        default=0.99,
    )
    parser.add_argument("--eps", type=float, default=1e-3)
    return parser.parse_args()


def lambda_tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text.replace("-", "m").replace(".", "p")


def experiment_dir(args) -> Path:
    return (
        Path(args.output_dir)
        / "soft_gated_lambda_sweep"
        / f"{args.backbone}_{args.loss}_lambda_{lambda_tag(args.lambda_gate)}"
    )


def json_safe_config(args, exp_dir: Path) -> Dict:
    config = vars(args).copy()
    config["experiment_dir"] = str(exp_dir)
    config["loss_name"] = args.loss
    config["eval_dir"] = str(args.eval_dir) if args.eval_dir else None
    return config


def _same_float(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def validate_resume_config(args, checkpoint_config, iteration_in_epoch):
    if not checkpoint_config:
        logging.warning("Checkpoint has no saved config; resume compatibility cannot be checked.")
        return

    hard_keys = ("loss", "backbone", "embedding_size", "num_classes")
    mismatches = []
    for key in hard_keys:
        saved_value = checkpoint_config.get(key)
        current_value = getattr(args, key, None)
        if saved_value is not None and current_value != saved_value:
            mismatches.append(f"{key}: checkpoint={saved_value!r} current={current_value!r}")

    for key in ("s", "m", "h", "lambda_gate"):
        saved_value = checkpoint_config.get(key)
        current_value = getattr(args, key, None)
        if saved_value is not None and not _same_float(saved_value, current_value):
            mismatches.append(f"{key}: checkpoint={saved_value!r} current={current_value!r}")

    if mismatches:
        raise ValueError(
            "Refusing to resume from an incompatible checkpoint. "
            + "; ".join(mismatches)
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


def resolve_resume_path(args, exp_dir: Path) -> Optional[Path]:
    if args.resume is None:
        return None
    if args.resume == "latest":
        return exp_dir / "latest.pt"
    return Path(args.resume)


def load_checkpoint_if_requested(args, exp_dir, backbone, head, optimizer, scheduler, scaler):
    resume_path = resolve_resume_path(args, exp_dir)
    if resume_path is None:
        return 0, 0, 0, None, None
    if not resume_path.exists():
        raise FileNotFoundError(f"--resume requested but checkpoint does not exist: {resume_path}")

    checkpoint = torch_load_cpu(resume_path)
    if "state_dict_head" not in checkpoint:
        raise ValueError(
            f"{resume_path} is not a full soft-gated training checkpoint. "
            "Use --pretrained_backbone for backbone-only checkpoints."
        )

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
        resume_path,
        start_epoch,
        iteration_in_epoch,
        global_step,
    )
    return start_epoch, iteration_in_epoch, global_step, metrics, epoch_state


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
        raise ValueError("Training dataloader is empty. Reduce --batch_size or check train_data.")
    steps_per_epoch = len(train_loader)

    backbone = get_model(
        args.backbone,
        dropout=0.0,
        fp16=use_amp,
        num_features=args.embedding_size,
    ).to(device)

    margin_loss = SoftGatedAdaCurricularFaceLoss(
        s=args.s,
        m=args.m,
        h=args.h,
        lambda_gate=args.lambda_gate,
        t_alpha=args.t_alpha,
        curriculum_alpha=args.curriculum_alpha,
        eps=args.eps,
    )
    head = MarginSoftmaxHead(
        embedding_size=args.embedding_size,
        num_classes=num_classes,
        margin_loss=margin_loss,
        fp16=use_amp,
    ).to(device)

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
    ) = load_checkpoint_if_requested(
        args, exp_dir, backbone, head, optimizer, scheduler, scaler
    )
    if resume_iteration >= steps_per_epoch:
        start_epoch += 1
        resume_iteration = 0
        resumed_epoch_state = None

    if args.resume is None:
        if not args.pretrained_backbone:
            raise ValueError("--pretrained_backbone is required unless --resume is used.")
        load_pretrained_backbone(backbone, args.pretrained_backbone)

    metrics = resumed_metrics or {
        "best_score": None,
        "best_epoch": None,
        "best_metric": None,
        "epochs": [],
    }
    val_targets = [item.strip() for item in args.val_targets.split(",") if item.strip()]
    eval_dir = args.eval_dir or args.data_dir

    logging.info("Experiment dir: %s", exp_dir)
    logging.info(
        "Training loss=%s lambda_gate=%.4f backbone=%s classes=%d images=%d batch_size=%d lr=%g",
        args.loss,
        args.lambda_gate,
        args.backbone,
        num_classes,
        len(dataset),
        args.batch_size,
        args.lr,
    )
    logging.info("train_data=%s", args.data_dir)
    logging.info("eval_dir=%s eval_targets=%s", eval_dir, ",".join(val_targets))
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
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
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
                    "lambda_gate": args.lambda_gate,
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
                    logging.info("max_train_minutes reached. Stop cleanly; rerun with --resume.")
                    return

        epoch_loss = loss_sum / max(1, sample_count)
        epoch_norm = norm_sum / max(1, sample_count)
        eval_metrics = {}
        should_eval = args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0)
        if should_eval:
            backbone.eval()
            eval_metrics = evaluate_if_available(
                backbone, eval_dir, val_targets, device, args.batch_size
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
            "lambda_gate": float(args.lambda_gate),
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
                "lambda_gate": args.lambda_gate,
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
