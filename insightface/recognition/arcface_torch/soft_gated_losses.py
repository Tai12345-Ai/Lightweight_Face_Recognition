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
                arc_anchor.detach().mean() * (1.0 - self.curriculum_alpha)
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


class AdaptiveSoftGatedAdaCurricularFaceV2Loss(nn.Module):
    """AdaFace-positive, soft adaptive CurricularFace-negative loss.

    This keeps the AdaFace positive branch and CurricularFace negative branch,
    but replaces the fixed hard negative gate with detached quality and
    difficulty-aware soft modulation.
    """

    requires_norms = True
    requires_embeddings = False

    def __init__(
        self,
        s: float = 64.0,
        m: float = 0.4,
        h: float = 0.333,
        lambda_max: float = 0.3,
        alpha_max: float = 0.5,
        gate_gamma: float = 5.0,
        alpha_quality_floor: float = 0.5,
        lambda_warmup_epochs: float = 2.0,
        t_alpha: float = 0.01,
        curriculum_alpha: float = 0.99,
        eps: float = 1e-3,
    ):
        super().__init__()
        if not 0.0 <= lambda_max <= 1.0:
            raise ValueError("lambda_max must be in [0, 1].")
        if alpha_max < 0.0:
            raise ValueError("alpha_max must be non-negative.")
        if gate_gamma < 0.0:
            raise ValueError("gate_gamma must be non-negative.")
        if not 0.0 <= alpha_quality_floor <= 1.0:
            raise ValueError("alpha_quality_floor must be in [0, 1].")
        if lambda_warmup_epochs < 0.0:
            raise ValueError("lambda_warmup_epochs must be non-negative.")
        self.s = s
        self.m = m
        self.h = h
        self.lambda_max = lambda_max
        self.alpha_max = alpha_max
        self.gate_gamma = gate_gamma
        self.alpha_quality_floor = alpha_quality_floor
        self.lambda_warmup_epochs = lambda_warmup_epochs
        self.t_alpha = t_alpha
        self.curriculum_alpha = curriculum_alpha
        self.eps = eps
        self.current_epoch = 0.0
        self.last_stats = {}
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)
        self.register_buffer("t", torch.zeros(1))

    def set_epoch(self, epoch) -> None:
        self.current_epoch = float(epoch)

    def _warmup_ratio(self) -> float:
        if self.lambda_warmup_epochs <= 0.0:
            return 1.0
        return min(1.0, max(0.0, self.current_epoch / self.lambda_warmup_epochs))

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
            raise RuntimeError(
                "AdaptiveSoftGatedAdaCurricularFaceV2Loss requires feature norms."
            )

        index, target = _positive_indices(labels)
        logits = logits.clone()
        if index.numel() == 0:
            self.last_stats = {}
            return logits * self.s

        rows = logits[index].clone()
        q = self._quality_indicator(labels, norms).to(dtype=rows.dtype)
        q_pos = q.clamp_min(0.0)
        target_cos = _safe_cosine(
            rows.gather(1, target.view(-1, 1)).view(-1), eps=self.eps
        )
        theta_y = target_cos.acos()

        u_pos = torch.cos(theta_y - self.m * q)
        u_pos = (u_pos - (self.m * q + self.m)).to(dtype=rows.dtype)
        arc_anchor = torch.cos(theta_y + self.m).to(dtype=rows.dtype)

        rho = self._warmup_ratio()
        lambda_i = (self.lambda_max * rho * q_pos).detach()
        tau = (
            (1.0 - lambda_i) * arc_anchor.detach()
            + lambda_i * u_pos.detach()
        ).detach()

        with torch.no_grad():
            self.t.mul_(self.curriculum_alpha).add_(
                arc_anchor.detach().mean() * (1.0 - self.curriculum_alpha)
            )

        one_hot = torch.zeros_like(rows, dtype=torch.bool)
        one_hot.scatter_(1, target.view(-1, 1), True)
        negative_mask = ~one_hot

        d_gate = torch.sigmoid(
            self.gate_gamma * (rows.detach() - tau.view(-1, 1))
        ).detach()
        quality_alpha = (
            self.alpha_quality_floor
            + (1.0 - self.alpha_quality_floor) * q_pos
        ).detach()
        alpha = (
            self.alpha_max
            * rho
            * quality_alpha.view(-1, 1)
            * d_gate
        ).detach()

        t = self.t.to(dtype=rows.dtype)
        curricular_neg = rows * (t + rows)
        soft_adaptive_neg = rows + alpha.to(dtype=rows.dtype) * (curricular_neg - rows)
        rows = torch.where(negative_mask, soft_adaptive_neg, rows).to(dtype=rows.dtype)
        rows.scatter_(1, target.view(-1, 1), u_pos.view(-1, 1))
        logits[index] = rows

        with torch.no_grad():
            q_float = q.detach().float()
            q_pos_float = q_pos.detach().float()
            quality_alpha_float = quality_alpha.detach().float()
            lambda_float = lambda_i.detach().float()
            d_neg = d_gate.masked_select(negative_mask).detach().float()
            alpha_neg = alpha.masked_select(negative_mask).detach().float()
            if d_neg.numel() == 0:
                d_mean = d_max = soft_hard_ratio = 0.0
            else:
                d_mean = float(d_neg.mean().item())
                d_max = float(d_neg.max().item())
                soft_hard_ratio = float((d_neg > 0.5).float().mean().item())
            if alpha_neg.numel() == 0:
                alpha_mean = alpha_max_actual = effective_mod_ratio = 0.0
            else:
                alpha_mean = float(alpha_neg.mean().item())
                alpha_max_actual = float(alpha_neg.max().item())
                effective_mod_ratio = float((alpha_neg > 0.05).float().mean().item())

            self.last_stats = {
                "q_mean": float(q_float.mean().item()),
                "q_std": float(q_float.std(unbiased=False).item()),
                "q_min": float(q_float.min().item()),
                "q_max": float(q_float.max().item()),
                "q_pos_mean": float(q_pos_float.mean().item()),
                "alpha_quality_floor": float(self.alpha_quality_floor),
                "quality_alpha_mean": float(quality_alpha_float.mean().item()),
                "quality_alpha_min": float(quality_alpha_float.min().item()),
                "quality_alpha_max": float(quality_alpha_float.max().item()),
                "lambda_i_mean": float(lambda_float.mean().item()),
                "lambda_i_max": float(lambda_float.max().item()),
                "u_pos_mean": float(u_pos.detach().float().mean().item()),
                "arc_anchor_mean": float(arc_anchor.detach().float().mean().item()),
                "tau_mean": float(tau.detach().float().mean().item()),
                "D_mean": d_mean,
                "D_max": d_max,
                "alpha_mean": alpha_mean,
                "alpha_max_actual": alpha_max_actual,
                "soft_hard_ratio": soft_hard_ratio,
                "effective_mod_ratio": effective_mod_ratio,
                "curricular_t": float(self.t.detach().item()),
            }
        return logits * self.s


class CompetitionAwareAdaFaceLoss(nn.Module):
    """AdaFace positive margin refined by detached negative competition.

    Negative logits are not modified. The hardest negative only adjusts the
    detached quality scalar used by the AdaFace positive branch.
    """

    requires_norms = True
    requires_embeddings = False

    def __init__(
        self,
        s: float = 64.0,
        m: float = 0.4,
        h: float = 0.333,
        t_alpha: float = 0.01,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.h = h
        self.t_alpha = t_alpha
        self.eps = eps
        self.last_stats = {}
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)

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
            raise RuntimeError("CompetitionAwareAdaFaceLoss requires feature norms.")

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
        arc_anchor = torch.cos(theta_y + self.m).to(dtype=rows.dtype)

        one_hot = torch.zeros_like(rows, dtype=torch.bool)
        one_hot.scatter_(1, target.view(-1, 1), True)
        c_minus = rows.detach().masked_fill(one_hot, -1.0).max(dim=1).values
        d_i = (
            (c_minus - arc_anchor.detach()).relu()
            / (1.0 - arc_anchor.detach() + self.eps)
        ).clamp(0.0, 1.0).detach()

        q_star = (q * (1.0 + d_i)).clamp(-1.0, 1.0).detach()
        u_pos_star = torch.cos(theta_y - self.m * q_star)
        u_pos_star = (u_pos_star - (self.m * q_star + self.m)).to(dtype=rows.dtype)

        rows = rows.scatter(1, target.view(-1, 1), u_pos_star.view(-1, 1))
        logits[index] = rows

        with torch.no_grad():
            q_float = q.detach().float()
            d_float = d_i.detach().float()
            q_star_float = q_star.detach().float()
            c_minus_float = c_minus.detach().float()
            hard_mask = d_float > 0.0
            high_quality_hard = (q_float > 0.0) & hard_mask
            low_quality_hard = (q_float < 0.0) & hard_mask
            self.last_stats = {
                "q_mean": float(q_float.mean().item()),
                "q_std": float(q_float.std(unbiased=False).item()),
                "q_min": float(q_float.min().item()),
                "q_max": float(q_float.max().item()),
                "d_mean": float(d_float.mean().item()),
                "d_max": float(d_float.max().item()),
                "q_star_mean": float(q_star_float.mean().item()),
                "q_star_std": float(q_star_float.std(unbiased=False).item()),
                "q_star_min": float(q_star_float.min().item()),
                "q_star_max": float(q_star_float.max().item()),
                "c_minus_mean": float(c_minus_float.mean().item()),
                "arc_anchor_mean": float(arc_anchor.detach().float().mean().item()),
                "u_pos_star_mean": float(u_pos_star.detach().float().mean().item()),
                "competition_active_ratio": float(hard_mask.float().mean().item()),
                "high_quality_hard_ratio": float(
                    high_quality_hard.float().mean().item()
                ),
                "low_quality_hard_ratio": float(low_quality_hard.float().mean().item()),
            }
        return logits * self.s
