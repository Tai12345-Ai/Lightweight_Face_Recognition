#!/usr/bin/env python3
"""Trainer wrapper for Proposed 4.3 Full-v1.

This keeps the existing Kaggle trainer intact, but monkey-patches:
1) the Proposed 4.3 margin loss class with Proposed43FullIdentitySafeLoss;
2) MarginSoftmaxHead.forward so the loss receives normalized class weights.

The underlying CLI and checkpoint format remain compatible with
train_soft_gated_lambda_kaggle.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import train_soft_gated_lambda_kaggle as base
import soft_gated_losses
from proposed_4_3_full_loss import Proposed43FullIdentitySafeLoss


def _full_margin_head_forward(self, embeddings, labels):
    labels = labels.view(-1).long()
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    norm_embeddings = F.normalize(embeddings, dim=1)
    norm_weight = F.normalize(self.weight, dim=1)
    logits = F.linear(norm_embeddings, norm_weight).clamp(-1.0, 1.0)

    if getattr(self.margin_loss, "requires_class_weights", False):
        logits = self.margin_loss(
            logits,
            labels.view(-1, 1),
            embeddings=embeddings,
            norms=norms,
            class_weights=norm_weight,
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

    regularization = getattr(self.margin_loss, "_last_mag_reg", None)
    if regularization is not None:
        loss = loss + regularization
    extra_loss = getattr(self.margin_loss, "_last_extra_loss", None)
    if extra_loss is not None:
        loss = loss + extra_loss
    return loss, logits, norms


# Patch both the imported symbol in the trainer module and the original module.
base.MultiUIPerceptibilityCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss = Proposed43FullIdentitySafeLoss
soft_gated_losses.MultiUIPerceptibilityCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss = Proposed43FullIdentitySafeLoss
base.MarginSoftmaxHead.forward = _full_margin_head_forward


if __name__ == "__main__":
    base.main()
