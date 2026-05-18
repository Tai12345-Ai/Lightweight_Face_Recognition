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
from soft_gated_losses import (
    AdaptiveSoftGatedAdaCurricularFaceV2Loss,
    CompetitionAwareAdaFaceLoss,
    SoftGatedAdaCurricularFaceLoss,
)
from train_phase2_kaggle import (
    MarginSoftmaxHead,
    amp_autocast,
    append_csv,
    build_dataset,
    build_scheduler,
    configure_learning_rates,
    current_learning_rates,
    evaluate_if_available,
    float_tag,
    load_pretrained_backbone,
    make_checkpoint,
    make_grad_scaler,
    make_train_loader,
    save_checkpoint,
    setup_logging,
    setup_seed,
    split_trainable_parameters,
    torch_load_cpu,
    write_json,
)


# Paper-style high-quality average: LFW, CFP-FP, CPLFW, AgeDB, CALFW.
HQ_EVAL_TARGETS = ("lfw", "cfp_fp", "cplfw", "agedb_30", "calfw")
LQ_EVAL_TARGETS = ("sllfw", "talfw")
EVAL7_TARGETS = HQ_EVAL_TARGETS + LQ_EVAL_TARGETS
ALL_EVAL_TARGETS = EVAL7_TARGETS
LOSS_STAT_KEYS = (
    "q_mean",
    "q_std",
    "q_min",
    "q_max",
    "d_mean",
    "d_max",
    "q_star_mean",
    "q_star_std",
    "q_star_min",
    "q_star_max",
    "c_minus_mean",
    "u_pos_star_mean",
    "competition_active_ratio",
    "high_quality_hard_ratio",
    "low_quality_hard_ratio",
    "q_pos_mean",
    "alpha_quality_floor",
    "quality_alpha_mean",
    "quality_alpha_min",
    "quality_alpha_max",
    "lambda_i_mean",
    "lambda_i_max",
    "u_pos_mean",
    "arc_anchor_mean",
    "tau_mean",
    "D_mean",
    "D_max",
    "alpha_mean",
    "alpha_max_actual",
    "soft_hard_ratio",
    "effective_mod_ratio",
    "hard_negative_ratio",
    "curricular_t",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Soft-Gated Ada-CurricularFace lambda sweep trainer"
    )
    parser.add_argument(
        "--loss",
        default="soft_gated_ada_curricular",
        choices=[
            "soft_gated_ada_curricular",
            "adaptive_soft_gated_ada_curricular_v2",
            "competition_aware_adaface",
        ],
        help="Standalone loss name for config compatibility.",
    )
    parser.add_argument("--network", "--backbone", dest="backbone", default="r18", choices=["r18"])
    parser.add_argument("--s", type=float, default=64.0)
    parser.add_argument("--m", type=float, default=0.4)
    parser.add_argument("--h", type=float, default=0.333)
    parser.add_argument("--lambda_gate", "--lambda-gate", dest="lambda_gate", type=float, default=None)
    parser.add_argument("--lambda_max", "--lambda-max", dest="lambda_max", type=float, default=0.3)
    parser.add_argument("--alpha_max", "--alpha-max", dest="alpha_max", type=float, default=0.5)
    parser.add_argument("--gate_gamma", "--gate-gamma", dest="gate_gamma", type=float, default=5.0)
    parser.add_argument(
        "--alpha_quality_floor",
        "--alpha-quality-floor",
        dest="alpha_quality_floor",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--lambda_warmup_epochs",
        "--lambda-warmup-epochs",
        dest="lambda_warmup_epochs",
        type=float,
        default=2.0,
    )
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
    parser.add_argument(
        "--backbone_lr",
        "--backbone-lr",
        dest="backbone_lr",
        type=float,
        default=None,
        help="Learning rate for backbone parameters. Defaults to --lr.",
    )
    parser.add_argument(
        "--head_lr",
        "--head-lr",
        dest="head_lr",
        type=float,
        default=None,
        help="Learning rate for classifier head parameters. Defaults to --lr.",
    )
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
        default="lfw,cfp_fp,cplfw,agedb_30,calfw,sllfw,talfw",
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
    args = parser.parse_args()
    if args.loss == "soft_gated_ada_curricular" and args.lambda_gate is None:
        parser.error("--lambda_gate is required when --loss soft_gated_ada_curricular")
    if (
        args.loss != "adaptive_soft_gated_ada_curricular_v2"
        and args.alpha_quality_floor != 0.5
    ):
        parser.error(
            "--alpha_quality_floor is only valid when "
            "--loss adaptive_soft_gated_ada_curricular_v2"
        )
    return args


def lambda_tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text.replace("-", "m").replace(".", "p")


def experiment_dir(args) -> Path:
    if args.loss == "adaptive_soft_gated_ada_curricular_v2":
        name = (
            f"{args.backbone}_proposed2"
            f"_lmax_{float_tag(args.lambda_max)}"
            f"_amax_{float_tag(args.alpha_max)}"
            f"_gamma_{float_tag(args.gate_gamma)}"
            f"_qfloor_{float_tag(args.alpha_quality_floor)}"
            f"_blr_{float_tag(args.backbone_lr)}"
            f"_hlr_{float_tag(args.head_lr)}"
        )
        return Path(args.output_dir) / "proposed2_sweep" / name
    if args.loss == "competition_aware_adaface":
        name = (
            f"{args.backbone}_proposed3_competition_aware_adaface"
            f"_blr_{float_tag(args.backbone_lr)}"
            f"_hlr_{float_tag(args.head_lr)}"
        )
        return Path(args.output_dir) / "proposed3_competition_aware_adaface" / name

    name = f"{args.backbone}_{args.loss}_lambda_{lambda_tag(args.lambda_gate)}"
    if getattr(args, "use_split_lr", False):
        name += f"_blr_{float_tag(args.backbone_lr)}_hlr_{float_tag(args.head_lr)}"
    return Path(args.output_dir) / "soft_gated_lambda_sweep" / name


def json_safe_config(args, exp_dir: Path) -> Dict:
    config = vars(args).copy()
    if args.loss != "adaptive_soft_gated_ada_curricular_v2":
        config.pop("alpha_quality_floor", None)
    config["experiment_dir"] = str(exp_dir)
    config["loss_name"] = args.loss
    config["eval_dir"] = str(args.eval_dir) if args.eval_dir else None
    config["effective_backbone_lr"] = float(args.backbone_lr)
    config["effective_head_lr"] = float(args.head_lr)
    return config


def normalize_train_data_dir(path) -> str:
    path = Path(path)
    nested_train = path / "casia-webface"
    nested_eval = path / "eval"
    if nested_train.is_dir() and nested_eval.is_dir():
        logging.info(
            "Detected nested CASIA-WebFace layout. Using train folder: %s",
            nested_train,
        )
        return str(nested_train)
    return str(path)


def _complete_accuracy_mean(eval_metrics: Dict, targets) -> Optional[float]:
    values = []
    for target in targets:
        item = eval_metrics.get(target)
        if item is None or "accuracy" not in item:
            return None
        values.append(float(item["accuracy"]))
    return float(np.mean(values)) if values else None


def compute_group_eval(eval_metrics: Dict) -> Dict[str, float]:
    group_metrics = {}
    if not eval_metrics:
        return group_metrics

    hq_avg = _complete_accuracy_mean(eval_metrics, HQ_EVAL_TARGETS)
    lq_avg = _complete_accuracy_mean(eval_metrics, LQ_EVAL_TARGETS)
    eval7_avg = _complete_accuracy_mean(eval_metrics, EVAL7_TARGETS)
    all_avg = _complete_accuracy_mean(eval_metrics, ALL_EVAL_TARGETS)

    if hq_avg is not None:
        group_metrics["HQ_Avg"] = hq_avg
    if lq_avg is not None:
        group_metrics["LQ_Avg"] = lq_avg
    if eval7_avg is not None:
        group_metrics["Eval7_Avg"] = eval7_avg
    if all_avg is not None:
        group_metrics["All_Avg"] = all_avg

    return group_metrics


def select_eval_score(eval_metrics: Dict, group_metrics: Dict[str, float]):
    if "HQ_Avg" in group_metrics:
        return group_metrics["HQ_Avg"], "HQ_Avg"
    if "Eval7_Avg" in group_metrics:
        return group_metrics["Eval7_Avg"], "Eval7_Avg"
    if eval_metrics:
        values = [float(item["accuracy"]) for item in eval_metrics.values()]
        return float(np.mean(values)), "mean_validation_accuracy"
    return None, None


def add_loss_stats(row: Dict, loss_stats: Dict) -> Dict:
    for key in LOSS_STAT_KEYS:
        if key in loss_stats:
            row[key] = loss_stats[key]
    return row


def add_group_metrics(row: Dict, group_metrics: Dict[str, float]) -> Dict:
    for key in ("HQ_Avg", "LQ_Avg", "Eval7_Avg", "All_Avg"):
        row[key] = group_metrics.get(key, "")
    return row


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

    float_keys = ["s", "m", "h"]
    if args.loss == "adaptive_soft_gated_ada_curricular_v2":
        float_keys.extend(
            [
                "lambda_max",
                "alpha_max",
                "gate_gamma",
                "alpha_quality_floor",
                "lambda_warmup_epochs",
            ]
        )
    else:
        float_keys.append("lambda_gate")

    for key in float_keys:
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
    configure_learning_rates(args)
    setup_seed(args.seed)

    exp_dir = experiment_dir(args)
    setup_logging(exp_dir)
    args.data_dir = normalize_train_data_dir(args.data_dir)
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

    if args.loss == "adaptive_soft_gated_ada_curricular_v2":
        margin_loss = AdaptiveSoftGatedAdaCurricularFaceV2Loss(
            s=args.s,
            m=args.m,
            h=args.h,
            lambda_max=args.lambda_max,
            alpha_max=args.alpha_max,
            gate_gamma=args.gate_gamma,
            alpha_quality_floor=args.alpha_quality_floor,
            lambda_warmup_epochs=args.lambda_warmup_epochs,
            t_alpha=args.t_alpha,
            curriculum_alpha=args.curriculum_alpha,
            eps=args.eps,
        )
    elif args.loss == "competition_aware_adaface":
        margin_loss = CompetitionAwareAdaFaceLoss(
            s=args.s,
            m=args.m,
            h=args.h,
            t_alpha=args.t_alpha,
            eps=args.eps,
        )
    else:
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

    optimizer_params, params = split_trainable_parameters(backbone, head, args)
    optimizer = torch.optim.SGD(
        optimizer_params,
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
    if args.loss == "adaptive_soft_gated_ada_curricular_v2":
        logging.info(
            (
                "Training loss=%s lambda_max=%.4f alpha_max=%.4f gate_gamma=%.4f "
                "alpha_quality_floor=%.4f lambda_warmup_epochs=%.3f "
                "backbone=%s classes=%d images=%d "
                "batch_size=%d base_lr=%g backbone_lr=%g head_lr=%g"
            ),
            args.loss,
            args.lambda_max,
            args.alpha_max,
            args.gate_gamma,
            args.alpha_quality_floor,
            args.lambda_warmup_epochs,
            args.backbone,
            num_classes,
            len(dataset),
            args.batch_size,
            args.lr,
            args.backbone_lr,
            args.head_lr,
        )
    elif args.loss == "competition_aware_adaface":
        logging.info(
            (
                "Training loss=%s backbone=%s classes=%d images=%d "
                "batch_size=%d base_lr=%g backbone_lr=%g head_lr=%g"
            ),
            args.loss,
            args.backbone,
            num_classes,
            len(dataset),
            args.batch_size,
            args.lr,
            args.backbone_lr,
            args.head_lr,
        )
    else:
        logging.info(
            (
                "Training loss=%s lambda_gate=%.4f backbone=%s classes=%d images=%d "
                "batch_size=%d base_lr=%g backbone_lr=%g head_lr=%g"
            ),
            args.loss,
            args.lambda_gate,
            args.backbone,
            num_classes,
            len(dataset),
            args.batch_size,
            args.lr,
            args.backbone_lr,
            args.head_lr,
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
        if hasattr(margin_loss, "set_epoch"):
            margin_loss.set_epoch(epoch)
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
        last_loss_stats = {}

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
                loss_stats = getattr(margin_loss, "last_stats", {}) or {}
                last_loss_stats = dict(loss_stats)

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
                lr_values = current_learning_rates(optimizer)
                row = {
                    "event": "iter",
                    "epoch": epoch + 1,
                    "iteration": iteration,
                    "global_step": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "backbone_lr": lr_values["backbone_lr"],
                    "head_lr": lr_values["head_lr"],
                    "loss": loss.item(),
                    "epoch_loss": "",
                    "mean_norm": norms.detach().mean().item(),
                    "lambda_gate": args.lambda_gate if args.lambda_gate is not None else "",
                    "lambda_max": args.lambda_max if args.loss == "adaptive_soft_gated_ada_curricular_v2" else "",
                    "alpha_max": args.alpha_max if args.loss == "adaptive_soft_gated_ada_curricular_v2" else "",
                    "gate_gamma": args.gate_gamma if args.loss == "adaptive_soft_gated_ada_curricular_v2" else "",
                    "alpha_quality_floor": (
                        args.alpha_quality_floor
                        if args.loss == "adaptive_soft_gated_ada_curricular_v2"
                        else ""
                    ),
                    "lambda_warmup_epochs": (
                        args.lambda_warmup_epochs
                        if args.loss == "adaptive_soft_gated_ada_curricular_v2"
                        else ""
                    ),
                    "elapsed_sec": int(time.time() - epoch_start),
                }
                add_loss_stats(row, last_loss_stats)
                append_csv(exp_dir / "train_log.csv", row)
                logging.info(
                    (
                        "epoch=%d/%d iter=%d/%d step=%d loss=%.4f "
                        "backbone_lr=%s head_lr=%s"
                    ),
                    epoch + 1,
                    args.epochs,
                    iteration,
                    len(train_loader),
                    global_step,
                    loss.item(),
                    (
                        f"{lr_values['backbone_lr']:.6g}"
                        if lr_values["backbone_lr"] != ""
                        else "NA"
                    ),
                    (
                        f"{lr_values['head_lr']:.6g}"
                        if lr_values["head_lr"] != ""
                        else "NA"
                    ),
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
        group_metrics = {}
        should_eval = args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0)
        if should_eval:
            backbone.eval()
            eval_metrics = evaluate_if_available(
                backbone, eval_dir, val_targets, device, args.batch_size
            )
            group_metrics = compute_group_eval(eval_metrics)
            if group_metrics:
                logging.info(
                    "group_eval HQ_Avg=%s LQ_Avg=%s Eval7_Avg=%s All_Avg=%s",
                    f"{group_metrics['HQ_Avg']:.5f}" if "HQ_Avg" in group_metrics else "NA",
                    f"{group_metrics['LQ_Avg']:.5f}" if "LQ_Avg" in group_metrics else "NA",
                    f"{group_metrics['Eval7_Avg']:.5f}" if "Eval7_Avg" in group_metrics else "NA",
                    f"{group_metrics['All_Avg']:.5f}" if "All_Avg" in group_metrics else "NA",
                )

        if eval_metrics:
            score, best_metric = select_eval_score(eval_metrics, group_metrics)
        else:
            score = -float(epoch_loss)
            best_metric = "negative_train_loss"

        is_best = metrics["best_score"] is None or score > metrics["best_score"]
        if is_best:
            metrics["best_score"] = score
            metrics["best_epoch"] = epoch + 1
            metrics["best_metric"] = best_metric

        lr_values = current_learning_rates(optimizer)
        epoch_record = {
            "epoch": epoch + 1,
            "loss": float(epoch_loss),
            "mean_norm": float(epoch_norm),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "backbone_lr": lr_values["backbone_lr"],
            "head_lr": lr_values["head_lr"],
            "lambda_gate": float(args.lambda_gate) if args.lambda_gate is not None else None,
            "elapsed_sec": int(time.time() - epoch_start),
            "eval": eval_metrics,
            "group_eval": group_metrics,
        }
        if args.loss == "adaptive_soft_gated_ada_curricular_v2":
            epoch_record.update(
                {
                    "lambda_max": float(args.lambda_max),
                    "alpha_max": float(args.alpha_max),
                    "gate_gamma": float(args.gate_gamma),
                    "alpha_quality_floor": float(args.alpha_quality_floor),
                    "lambda_warmup_epochs": float(args.lambda_warmup_epochs),
                }
            )
        add_loss_stats(epoch_record, last_loss_stats)
        metrics["epochs"].append(epoch_record)
        write_json(exp_dir / "metrics.json", metrics)

        epoch_row = {
            "event": "epoch",
            "epoch": epoch + 1,
            "iteration": 0,
            "global_step": global_step,
            "lr": optimizer.param_groups[0]["lr"],
            "backbone_lr": lr_values["backbone_lr"],
            "head_lr": lr_values["head_lr"],
            "loss": "",
            "epoch_loss": epoch_loss,
            "mean_norm": epoch_norm,
            "lambda_gate": args.lambda_gate if args.lambda_gate is not None else "",
            "lambda_max": args.lambda_max if args.loss == "adaptive_soft_gated_ada_curricular_v2" else "",
            "alpha_max": args.alpha_max if args.loss == "adaptive_soft_gated_ada_curricular_v2" else "",
            "gate_gamma": args.gate_gamma if args.loss == "adaptive_soft_gated_ada_curricular_v2" else "",
            "alpha_quality_floor": (
                args.alpha_quality_floor
                if args.loss == "adaptive_soft_gated_ada_curricular_v2"
                else ""
            ),
            "lambda_warmup_epochs": (
                args.lambda_warmup_epochs
                if args.loss == "adaptive_soft_gated_ada_curricular_v2"
                else ""
            ),
            "elapsed_sec": epoch_record["elapsed_sec"],
        }
        add_group_metrics(epoch_row, group_metrics)
        add_loss_stats(epoch_row, last_loss_stats)
        append_csv(
            exp_dir / "train_log.csv",
            epoch_row,
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
