#!/usr/bin/env python3
"""Trainer wrapper for Proposed 4.3++ Full.

The base Kaggle trainer is reused for CLI, data loading, logging, resume and
checkpointing.  This wrapper replaces only the Proposed 4.3 runtime pieces:

1) get_model() returns a backbone wrapper that computes F -> M -> F' -> x';
2) the Proposed 4.3 loss is replaced by the Full identity-safe objective;
3) MarginSoftmaxHead.forward passes the wrapper context and normalized class
   weights to the loss.

Clean eval inside the base trainer also receives this wrapper, so both clean
and degraded Full evaluation emit x' rather than the bare backbone embedding x.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import train_soft_gated_lambda_kaggle as base
import soft_gated_losses
from proposed_4_3_full_loss import Proposed43FullIdentitySafeLoss
from proposed_4_3_attention_model import (
    Proposed43AttentionBackbone,
    get_last_attention_context,
    infer_feature_channels,
)


_ORIGINAL_PARSE_ARGS = base.parse_args
_ORIGINAL_GET_MODEL = base.get_model
_FULL_ARGS = None


class _ConfiguredFullLoss(Proposed43FullIdentitySafeLoss):
    def __init__(self, *args, **kwargs):
        if _FULL_ARGS is not None:
            kwargs.setdefault("top_m", int(getattr(_FULL_ARGS, "full_top_m", 4)))
            kwargs.setdefault("ui_soft_tau", float(getattr(_FULL_ARGS, "full_ui_soft_tau", 12.0)))
            kwargs.setdefault("ui_margin", float(getattr(_FULL_ARGS, "full_ui_margin", 0.20)))
            kwargs.setdefault("ri_lambda", float(getattr(_FULL_ARGS, "ri_lambda", 0.05)))
            kwargs.setdefault("anchor_lambda", float(getattr(_FULL_ARGS, "full_anchor_lambda", 0.08)))
            kwargs.setdefault("neg_lambda", float(getattr(_FULL_ARGS, "full_neg_lambda", 0.06)))
            kwargs.setdefault("preserve_lambda", float(getattr(_FULL_ARGS, "full_preserve_lambda", 0.03)))
            kwargs.setdefault("delta_c", float(getattr(_FULL_ARGS, "full_delta_c", 0.02)))
            kwargs.setdefault("delta_n", float(getattr(_FULL_ARGS, "full_delta_n", 0.02)))
            kwargs.setdefault("label_gamma", float(getattr(_FULL_ARGS, "full_label_gamma", 12.0)))
            kwargs.setdefault("label_margin", float(getattr(_FULL_ARGS, "full_label_margin", 0.05)))
            kwargs.setdefault("unrec_tau", float(getattr(_FULL_ARGS, "full_unrec_tau", 0.35)))
            kwargs.setdefault("unrec_gamma", float(getattr(_FULL_ARGS, "full_unrec_gamma", 8.0)))
        super().__init__(*args, **kwargs)


def _full_parse_args():
    global _FULL_ARGS
    args = _ORIGINAL_PARSE_ARGS()
    # The Full wrapper owns the real attention path.  Disable the legacy
    # auxiliary MSE hook in train_soft_gated_lambda_kaggle.py.
    args.enable_attention = False
    args.centered_attention = True
    _FULL_ARGS = args
    return args


def _full_get_model(name, **kwargs):
    raw_backbone = _ORIGINAL_GET_MODEL(name, **kwargs)
    args = _FULL_ARGS
    image_size = int(getattr(args, "image_size", 112)) if args is not None else 112
    feature_channels = infer_feature_channels(raw_backbone, image_size=image_size)
    embedding_dim = int(kwargs.get("num_features", getattr(args, "embedding_size", 512)))
    return Proposed43AttentionBackbone(
        raw_backbone,
        feature_channels=feature_channels,
        embedding_dim=embedding_dim,
        attention_reduction=int(getattr(args, "attention_reduction", 16)),
        attention_alpha=float(getattr(args, "attention_alpha", 0.25)),
        centered_attention=True,
        attention_spatial_lambda=float(getattr(args, "attention_spatial_lambda", 1e-4)),
        attention_channel_lambda=float(getattr(args, "attention_channel_lambda", 1e-4)),
        attention_tv_lambda=float(getattr(args, "attention_tv_lambda", 1e-4)),
    )


def _full_margin_head_forward(self, embeddings, labels):
    labels = labels.view(-1).long()
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    norm_embeddings = F.normalize(embeddings, dim=1)
    norm_weight = F.normalize(self.weight, dim=1)
    logits = F.linear(norm_embeddings, norm_weight).clamp(-1.0, 1.0)
    context = get_last_attention_context()

    if getattr(self.margin_loss, "requires_class_weights", False):
        logits = self.margin_loss(
            logits,
            labels.view(-1, 1),
            embeddings=embeddings,
            norms=norms,
            class_weights=norm_weight,
            context=context,
        )
    else:
        logits = self.margin_loss(
            logits,
            labels.view(-1, 1),
            embeddings=embeddings,
            norms=norms,
        )

    sample_weight = getattr(self.margin_loss, "_last_sample_weight", None)
    if sample_weight is None:
        loss = F.cross_entropy(logits, labels, ignore_index=-1)
    else:
        per_sample_loss = F.cross_entropy(logits, labels, ignore_index=-1, reduction="none")
        valid = labels != -1
        weights = sample_weight.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype).view(-1)
        weights = torch.where(valid, weights.clamp_min(0.0), torch.zeros_like(weights))
        loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)
    base_fr_loss = loss

    regularization = getattr(self.margin_loss, "_last_mag_reg", None)
    if regularization is not None:
        loss = loss + regularization
    extra_loss = getattr(self.margin_loss, "_last_extra_loss", None)
    if extra_loss is not None:
        loss = loss + extra_loss
    loss_stats = getattr(self.margin_loss, "last_stats", None)
    if isinstance(loss_stats, dict):
        loss_stats["base_fr_loss"] = float(base_fr_loss.detach().float().item())
    return loss, logits, norms


# Patch both the imported symbol in the trainer module and the original module.
base.parse_args = _full_parse_args
base.get_model = _full_get_model
base.MultiUIPerceptibilityCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss = _ConfiguredFullLoss
soft_gated_losses.MultiUIPerceptibilityCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss = _ConfiguredFullLoss
base.MarginSoftmaxHead.forward = _full_margin_head_forward


if __name__ == "__main__":
    base.main()
