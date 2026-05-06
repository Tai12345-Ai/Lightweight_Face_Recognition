"""
Lightweight Face Recognition Training Script.

Single-GPU friendly training script for the lightweight FR project.
Supports backbone selection via config, prepares for AdaFace/MagFace in Phase 6.

Usage:
    # Single GPU (auto GLOO fallback)
    python train_lightweight.py configs/lightweight_fr/mbf_arcface.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=2 train_lightweight.py configs/lightweight_fr/mbf_arcface.py
"""

import argparse
import importlib
import importlib.util
import logging
import os
import os.path as osp
from datetime import datetime

import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader

from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch.nn.functional import normalize, linear
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


# =====================================================================
# Config loader — supports configs/lightweight_fr/ subdirectory
# =====================================================================

def get_config_lightweight(config_file):
    """Load config from lightweight_fr subdirectory.

    Loads base_lightweight.py first, then overlays the specific config.
    """
    base_path = osp.join(osp.dirname(osp.abspath(__file__)),
                         "configs", "lightweight_fr", "base_lightweight.py")

    # Load base config
    spec = importlib.util.spec_from_file_location("base_lightweight", base_path)
    base_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_module)
    cfg = base_module.config

    # Load specific config
    config_path = osp.abspath(config_file)
    spec = importlib.util.spec_from_file_location("job_config", config_path)
    job_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job_module)
    cfg.update(job_module.config)

    if cfg.output is None:
        cfg.output = osp.join('work_dirs',
                              osp.splitext(osp.basename(config_file))[0])
    return cfg


# =====================================================================
# PartialFC_V2_Extended — passes norms to loss for AdaFace (Phase 6)
# =====================================================================

class PartialFC_V2_Extended(PartialFC_V2):
    """Extended PartialFC_V2 that passes embeddings and norms to loss.

    Required for AdaFace and MagFace which use feature norms
    to compute adaptive margins.
    """

    def forward(self, local_embeddings, local_labels):
        local_labels.squeeze_()
        local_labels = local_labels.long()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            f"last batch size do not equal current batch size: "
            f"{self.last_batch_size} vs {batch_size}")

        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size)).cuda()
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size).long().cuda()
            for _ in range(self.world_size)
        ]
        _list_embeddings = torch.distributed.nn.all_gather(local_embeddings)
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(list(_list_embeddings))
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive)
        else:
            weight = self.weight

        # Compute norms BEFORE normalizing
        norms = torch.norm(embeddings, dim=1, keepdim=True)

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        # Extended interface: pass embeddings and norms to loss
        logits = self.margin_softmax(
            logits, labels,
            embeddings=norm_embeddings, norms=norms
        )
        loss = self.dist_cross_entropy(logits, labels)
        return loss


# =====================================================================
# Loss factory
# =====================================================================

def create_margin_loss(cfg):
    """Create margin loss based on config."""
    loss_type = getattr(cfg, 'loss_type', 'combined_margin')

    if loss_type == "combined_margin":
        return CombinedMarginLoss(
            64,
            cfg.margin_list[0],
            cfg.margin_list[1],
            cfg.margin_list[2],
            cfg.interclass_filtering_threshold
        )
    elif loss_type == "adaface":
        from losses_extended import AdaFaceLoss
        return AdaFaceLoss(
            s=getattr(cfg, 'adaface_s', 64.0),
            m=getattr(cfg, 'adaface_m', 0.4),
            h=getattr(cfg, 'adaface_h', 0.333),
            t_alpha=getattr(cfg, 'adaface_t_alpha', 0.01),
        )
    elif loss_type == "magface":
        from losses_extended import MagFaceLoss
        return MagFaceLoss(
            s=getattr(cfg, 'magface_s', 64.0),
            l_a=getattr(cfg, 'magface_l_a', 10),
            u_a=getattr(cfg, 'magface_u_a', 110),
            l_m=getattr(cfg, 'magface_l_m', 0.45),
            u_m=getattr(cfg, 'magface_u_m', 0.8),
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def loss_needs_extended_pfc(cfg):
    """Check if the loss type requires PartialFC_V2_Extended."""
    return getattr(cfg, 'loss_type', 'combined_margin') in ('adaface', 'magface')


# =====================================================================
# Distributed init — GLOO fallback for single-GPU / Windows
# =====================================================================

def init_distributed():
    """Initialize distributed training with NCCL/GLOO auto-detection."""
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        distributed.init_process_group(backend)
    except KeyError:
        rank = 0
        local_rank = 0
        world_size = 1
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        distributed.init_process_group(
            backend=backend,
            init_method="tcp://127.0.0.1:12584",
            rank=rank,
            world_size=world_size,
        )
    return rank, local_rank, world_size


# =====================================================================
# Main training function
# =====================================================================

def main(args):
    rank, local_rank, world_size = init_distributed()

    # Config
    cfg = get_config_lightweight(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = None
    if rank == 0 and SummaryWriter is not None:
        summary_writer = SummaryWriter(
            log_dir=os.path.join(cfg.output, "tensorboard"))

    # Data
    train_loader = get_dataloader(
        cfg.rec, local_rank, cfg.batch_size,
        cfg.dali, cfg.dali_aug, cfg.seed, cfg.num_workers
    )

    # Backbone
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16,
        num_features=cfg.embedding_size
    ).cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False,
        device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True
    )
    backbone.train()
    backbone._set_static_graph()

    # Loss + Training head
    margin_loss = create_margin_loss(cfg)

    use_extended = loss_needs_extended_pfc(cfg)
    PFC_Class = PartialFC_V2_Extended if use_extended else PartialFC_V2

    if cfg.optimizer == "sgd":
        module_partial_fc = PFC_Class(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()},
                    {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        module_partial_fc = PFC_Class(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()},
                    {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # LR scheduler
    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    # Resume
    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(
            os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(
            dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(
            dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    # Log config
    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    # Callbacks
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec,
        summary_writer=summary_writer, wandb_logger=None
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    # =====================
    # Training loop
    # =====================
    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)

        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            loss = module_partial_fc(local_embeddings, local_labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16,
                                 lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        # Save checkpoint
        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
            }
            torch.save(checkpoint,
                        os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

    # Final save
    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
        logging.info(f"Training complete. Model saved to {path_module}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Lightweight Face Recognition Training")
    parser.add_argument("config", type=str,
                        help="Path to config file (e.g. configs/lightweight_fr/mbf_arcface.py)")
    main(parser.parse_args())
