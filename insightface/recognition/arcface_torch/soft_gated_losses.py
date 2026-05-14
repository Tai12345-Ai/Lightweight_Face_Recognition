"""Standalone losses for soft-gated Ada-CurricularFace experiments.

This module is intentionally not wired into ``losses_extended.PHASE2_LOSS_REGISTRY``.
Use it through ``train_soft_gated_lambda_kaggle.py`` while sweeping
``lambda_gate`` before promoting the loss into the main Phase 2 pipeline.
"""

from typing import Tuple

import torch
import torch.nn as nn


def _label_view(labels: torch.Tensor) -> torch.Tensor:
    return labels.view(-1, 1).long()


def _positive_indices(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = _label_view(labels)
    index = torch.where(labels.view(-1) != -1)[0]
    target = labels[index].view(-1)
    return index, target


def _safe_cosine(logits: torch.Tensor, eps: float) -> torch.Tensor:
    return logits.clamp(-1.0 + eps, 1.0 - eps)


class SoftGatedAdaCurricularFaceLoss(nn.Module):
    """AdaFace-positive, CurricularFace-negative with a soft gate threshold.

    Positive:
        q_i = clip((norm_i - mu) / (sigma / h), -1, 1)
        u_pos = cos(theta_y - m * q_i) - (m * q_i + m)

    Negative threshold:
        a_i = cos(theta_y + m)
        tau_i = (1 - lambda_gate) * a_i + lambda_gate * u_pos

    Negative:
        c_ij if c_ij <= tau_i
        c_ij * (t + c_ij) otherwise
    """

    requires_norms = True
    requires_embeddings = False

    def __init__(
        self,
        s: float = 64.0,
        m: float = 0.4,
        h: float = 0.333,
        lambda_gate: float = 0.0,
        t_alpha: float = 0.01,
        curriculum_alpha: float = 0.99,
        eps: float = 1e-3,
    ):
        super().__init__()
        if not 0.0 <= lambda_gate <= 1.0:
            raise ValueError("lambda_gate must be in [0, 1].")
        self.s = s
        self.m = m
        self.h = h
        self.lambda_gate = lambda_gate
        self.t_alpha = t_alpha
        self.curriculum_alpha = curriculum_alpha
        self.eps = eps
        self.last_stats = {}
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)
        self.register_buffer("t", torch.zeros(1))

    def _quality_indicator(self, labels: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        index, _ = _positive_indices(labels)
        safe_norms = norms.view(-1, 1).clamp(min=0.001, max=100.0).detach()

        with torch.no_grad():
            positive_norms = safe_norms[index]
            if positive_norms.numel() > 1:
                batch_mean = positive_norms.mean()
                batch_std = positive_norms.std(unbiased=False).clamp_min(self.eps)
                self.batch_mean.mul_(1.0 - self.t_alpha).add_(batch_mean * self.t_alpha)
                self.batch_std.mul_(1.0 - self.t_alpha).add_(batch_std * self.t_alpha)

        q = (safe_norms[index].view(-1) - self.batch_mean) / (
            self.batch_std + self.eps
        )
        return (q * self.h).clamp(-1.0, 1.0).detach()

    def forward(self, logits, labels, embeddings=None, norms=None):
        if norms is None:
            raise RuntimeError("SoftGatedAdaCurricularFaceLoss requires feature norms.")

        index, target = _positive_indices(labels)
        logits = logits.clone()
        if index.numel() == 0:
            self.last_stats = {}
            return logits * self.s

        rows = logits[index].clone()
        q = self._quality_indicator(labels, norms).to(dtype=rows.dtype)
        target_cos = _safe_cosine(
            rows.gather(1, target.view(-1, 1)).view(-1), eps=self.eps
        )
        theta_y = target_cos.acos()

        u_pos = torch.cos(theta_y - self.m * q)
        u_pos = (u_pos - (self.m * q + self.m)).to(dtype=rows.dtype)
        arc_anchor = torch.cos(theta_y + self.m).to(dtype=rows.dtype)
        tau = (1.0 - self.lambda_gate) * arc_anchor + self.lambda_gate * u_pos

        with torch.no_grad():
            self.t.mul_(self.curriculum_alpha).add_(
                target_cos.detach().mean() * (1.0 - self.curriculum_alpha)
            )

        one_hot = torch.zeros_like(rows, dtype=torch.bool)
        one_hot.scatter_(1, target.view(-1, 1), True)
        hard_mask = (rows > tau.detach().view(-1, 1)) & (~one_hot)
        total_negatives = (~one_hot).sum().clamp_min(1)
        hard_negative_ratio = hard_mask.sum().float() / total_negatives.float()
        with torch.no_grad():
            q_float = q.detach().float()
            self.last_stats = {
                "q_mean": float(q_float.mean().item()),
                "q_std": float(q_float.std(unbiased=False).item()),
                "q_min": float(q_float.min().item()),
                "q_max": float(q_float.max().item()),
                "u_pos_mean": float(u_pos.detach().float().mean().item()),
                "arc_anchor_mean": float(arc_anchor.detach().float().mean().item()),
                "tau_mean": float(tau.detach().float().mean().item()),
                "hard_negative_ratio": float(hard_negative_ratio.item()),
                "curricular_t": float(self.t.detach().item()),
            }
        t = self.t.to(dtype=rows.dtype)
        rows = torch.where(hard_mask, rows * (t + rows), rows).to(dtype=rows.dtype)
        rows.scatter_(1, target.view(-1, 1), u_pos.view(-1, 1))
        logits[index] = rows
        return logits * self.s
