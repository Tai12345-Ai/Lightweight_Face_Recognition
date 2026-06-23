#!/usr/bin/env python3
"""Trainer wrapper for Proposed 4.3++ Core.

Core uses the true F -> attention -> F' -> x' path, RI predictor, weighted FR,
preserve, identity-anchor and attention regularization.  It keeps
UI-orthogonal and negative-guard disabled so the objective matches the Core
specification.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import train_soft_gated_lambda_kaggle as base
import soft_gated_losses
from proposed_4_3_full_loss import Proposed43CoreIdentitySafeLoss
from proposed_4_3_attention_model import (
    Proposed43AttentionBackbone,
    get_last_attention_context,
    infer_feature_channels,
)


_ORIGINAL_PARSE_ARGS = base.parse_args
_ORIGINAL_GET_MODEL = base.get_model
_CORE_ARGS = None


class _ConfiguredCoreLoss(Proposed43CoreIdentitySafeLoss):
    def __init__(self, *args, **kwargs):
        kwargs["ui_lambda"] = 0.0
        kwargs["neg_lambda"] = 0.0
        if _CORE_ARGS is not None:
            kwargs.setdefault("ri_lambda", float(getattr(_CORE_ARGS, "ri_lambda", 0.05)))
        super().__init__(*args, **kwargs)


def _core_parse_args():
    global _CORE_ARGS
    args = _ORIGINAL_PARSE_ARGS()
    args.enable_attention = False
    args.centered_attention = False
    _CORE_ARGS = args
    return args


def _core_get_model(name, **kwargs):
    raw_backbone = _ORIGINAL_GET_MODEL(name, **kwargs)
    args = _CORE_ARGS
    image_size = int(getattr(args, "image_size", 112)) if args is not None else 112
    feature_channels = infer_feature_channels(raw_backbone, image_size=image_size)
    embedding_dim = int(kwargs.get("num_features", getattr(args, "embedding_size", 512)))
    return Proposed43AttentionBackbone(
        raw_backbone,
        feature_channels=feature_channels,
        embedding_dim=embedding_dim,
        attention_reduction=int(getattr(args, "attention_reduction", 16)),
        attention_alpha=float(getattr(args, "attention_alpha", 0.25)),
        centered_attention=False,
        attention_spatial_lambda=float(getattr(args, "attention_spatial_lambda", 1e-4)),
        attention_channel_lambda=0.0,
        attention_tv_lambda=float(getattr(args, "attention_tv_lambda", 1e-4)),
    )


def _core_margin_head_forward(self, embeddings, labels):
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


base.parse_args = _core_parse_args
base.get_model = _core_get_model
base.MultiUIPerceptibilityCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss = _ConfiguredCoreLoss
soft_gated_losses.MultiUIPerceptibilityCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss = _ConfiguredCoreLoss
base.MarginSoftmaxHead.forward = _core_margin_head_forward


if __name__ == "__main__":
    base.main()
