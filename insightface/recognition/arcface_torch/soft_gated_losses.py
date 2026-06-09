"""Standalone losses for soft-gated Ada-CurricularFace experiments.

This module is intentionally not wired into ``losses_extended.PHASE2_LOSS_REGISTRY``.
Use it through ``train_soft_gated_lambda_kaggle.py`` while sweeping
``lambda_gate`` before promoting the loss into the main Phase 2 pipeline.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CompetitionAdaptiveSoftGatedAdaCurricularFaceLoss(nn.Module):
    """AdaFace-positive, CurricularFace-negative with competition-adaptive gate.

    This keeps the fixed soft-gated Ada-CurricularFace branches, but replaces
    ``lambda_gate`` with a detached per-sample value:

        lambda_i = h * d_i

    where ``d_i`` measures how far the hardest negative has crossed the
    ArcFace anchor for the sample.
    """

    requires_norms = True
    requires_embeddings = False

    def __init__(
        self,
        s: float = 64.0,
        m: float = 0.4,
        h: float = 0.333,
        t_alpha: float = 0.01,
        curriculum_alpha: float = 0.99,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.h = h
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
            raise RuntimeError(
                "CompetitionAdaptiveSoftGatedAdaCurricularFaceLoss requires feature norms."
            )

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

        one_hot = torch.zeros_like(rows, dtype=torch.bool)
        one_hot.scatter_(1, target.view(-1, 1), True)

        c_minus = rows.detach().masked_fill(one_hot, -1.0).max(dim=1).values
        d_i = (
            (c_minus - arc_anchor.detach()).relu()
            / (1.0 - arc_anchor.detach() + self.eps)
        ).clamp(0.0, 1.0).detach()

        lambda_i = (self.h * d_i).detach()
        tau = (
            (1.0 - lambda_i) * arc_anchor.detach()
            + lambda_i * u_pos.detach()
        ).detach()

        with torch.no_grad():
            self.t.mul_(self.curriculum_alpha).add_(
                arc_anchor.detach().mean() * (1.0 - self.curriculum_alpha)
            )

        hard_mask = (rows > tau.view(-1, 1)) & (~one_hot)
        total_negatives = (~one_hot).sum().clamp_min(1)
        hard_negative_ratio = hard_mask.sum().float() / total_negatives.float()

        t = self.t.to(dtype=rows.dtype)
        rows = torch.where(hard_mask, rows * (t + rows), rows).to(dtype=rows.dtype)
        rows.scatter_(1, target.view(-1, 1), u_pos.view(-1, 1))
        logits[index] = rows

        with torch.no_grad():
            q_float = q.detach().float()
            d_float = d_i.detach().float()
            lambda_float = lambda_i.detach().float()
            self.last_stats = {
                "q_mean": float(q_float.mean().item()),
                "q_std": float(q_float.std(unbiased=False).item()),
                "q_min": float(q_float.min().item()),
                "q_max": float(q_float.max().item()),
                "u_pos_mean": float(u_pos.detach().float().mean().item()),
                "arc_anchor_mean": float(arc_anchor.detach().float().mean().item()),
                "c_minus_mean": float(c_minus.detach().float().mean().item()),
                "d_mean": float(d_float.mean().item()),
                "d_max": float(d_float.max().item()),
                "lambda_i_mean": float(lambda_float.mean().item()),
                "lambda_i_max": float(lambda_float.max().item()),
                "tau_mean": float(tau.detach().float().mean().item()),
                "hard_negative_ratio": float(hard_negative_ratio.item()),
                "competition_active_ratio": float((d_float > 0.0).float().mean().item()),
                "curricular_t": float(self.t.detach().item()),
            }
        return logits * self.s


class CompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss(nn.Module):
    """Proposed 4.1: quality-modulated competition-adaptive gate.

    This is the Proposed 4 loss with a fixed, detached quality factor applied
    to the competition gate:

        lambda_i = h * d_i * (0.75 + 0.25 * clamp(q_i, 0, 1))
    """

    requires_norms = True
    requires_embeddings = False

    def __init__(
        self,
        s: float = 64.0,
        m: float = 0.4,
        h: float = 0.333,
        t_alpha: float = 0.01,
        curriculum_alpha: float = 0.99,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.h = h
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
            raise RuntimeError(
                "CompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss "
                "requires feature norms."
            )

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

        one_hot = torch.zeros_like(rows, dtype=torch.bool)
        one_hot.scatter_(1, target.view(-1, 1), True)

        c_minus = rows.detach().masked_fill(one_hot, -1.0).max(dim=1).values
        d_i = (
            (c_minus - arc_anchor.detach()).relu()
            / (1.0 - arc_anchor.detach() + self.eps)
        ).clamp(0.0, 1.0).detach()

        q_pos = q.clamp(0.0, 1.0)
        q_factor = (0.75 + 0.25 * q_pos).detach()
        lambda_i = (self.h * d_i * q_factor).detach()
        tau = (
            (1.0 - lambda_i) * arc_anchor.detach()
            + lambda_i * u_pos.detach()
        ).detach()

        with torch.no_grad():
            self.t.mul_(self.curriculum_alpha).add_(
                arc_anchor.detach().mean() * (1.0 - self.curriculum_alpha)
            )

        hard_mask = (rows > tau.view(-1, 1)) & (~one_hot)
        total_negatives = (~one_hot).sum().clamp_min(1)
        hard_negative_ratio = hard_mask.sum().float() / total_negatives.float()

        t = self.t.to(dtype=rows.dtype)
        rows = torch.where(hard_mask, rows * (t + rows), rows).to(dtype=rows.dtype)
        rows.scatter_(1, target.view(-1, 1), u_pos.view(-1, 1))
        logits[index] = rows

        with torch.no_grad():
            q_float = q.detach().float()
            q_pos_float = q_pos.detach().float()
            q_factor_float = q_factor.detach().float()
            d_float = d_i.detach().float()
            lambda_float = lambda_i.detach().float()
            quality_ratio = lambda_i / (self.h * d_i + self.eps)
            self.last_stats = {
                "q_mean": float(q_float.mean().item()),
                "q_std": float(q_float.std(unbiased=False).item()),
                "q_min": float(q_float.min().item()),
                "q_max": float(q_float.max().item()),
                "q_pos_mean": float(q_pos_float.mean().item()),
                "q_factor_mean": float(q_factor_float.mean().item()),
                "q_factor_min": float(q_factor_float.min().item()),
                "q_factor_max": float(q_factor_float.max().item()),
                "u_pos_mean": float(u_pos.detach().float().mean().item()),
                "arc_anchor_mean": float(arc_anchor.detach().float().mean().item()),
                "c_minus_mean": float(c_minus.detach().float().mean().item()),
                "d_mean": float(d_float.mean().item()),
                "d_max": float(d_float.max().item()),
                "lambda_i_mean": float(lambda_float.mean().item()),
                "lambda_i_max": float(lambda_float.max().item()),
                "tau_mean": float(tau.detach().float().mean().item()),
                "hard_negative_ratio": float(hard_negative_ratio.item()),
                "competition_active_ratio": float((d_float > 0.0).float().mean().item()),
                "quality_modulated_lambda_ratio": float(
                    quality_ratio.detach().float().mean().item()
                ),
                "curricular_t": float(self.t.detach().item()),
            }
        return logits * self.s


class UIAwareCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss(nn.Module):
    """Proposed 4.2: Proposed 4.1 with UI-aware recognizability.

    The classification branch stays close to Proposed 4.1. In addition, the
    loss keeps an EMA center for synthetic unrecognizable-identity samples and
    penalizes embeddings that move too close to that center:

        RI_i = dUI_i * dN_i / (dP_i + eps)
        L_UI_i = max(0, cos(v_i, v_UI) - rho)
        L_total = L_P4.1 + mean(lambda_ui_i * L_UI_i)

    ``lambda_ui_i`` is detached and combines feature-norm quality, negative
    competition, and low RI. Dangerous UI-like samples also receive a mild CE
    down-weight so they are not forced into the labeled class as aggressively as
    hard-but-identifiable samples.
    """

    requires_norms = True
    requires_embeddings = True

    def __init__(
        self,
        s: float = 64.0,
        m: float = 0.4,
        h: float = 0.333,
        ui_lambda: float = 0.05,
        ui_rho: float = 0.2,
        ui_tau_ri: float = 1.0,
        ui_tau_easy: float = 2.0,
        ui_d_margin: float = 0.25,
        ui_alpha: float = 10.0,
        ui_beta: float = 5.0,
        ui_hard_boost: float = 0.1,
        ui_dangerous_downweight: float = 0.35,
        ui_sample_weight_min: float = 0.5,
        ui_center_momentum: float = 0.99,
        ui_center_dim: int = 512,
        t_alpha: float = 0.01,
        curriculum_alpha: float = 0.99,
        eps: float = 1e-3,
    ):
        super().__init__()
        if ui_lambda < 0.0:
            raise ValueError("ui_lambda must be non-negative.")
        if not -1.0 <= ui_rho <= 1.0:
            raise ValueError("ui_rho must be a cosine threshold in [-1, 1].")
        if ui_tau_ri < 0.0 or ui_tau_easy < 0.0:
            raise ValueError("UI RI thresholds must be non-negative.")
        if not 0.0 <= ui_dangerous_downweight <= 1.0:
            raise ValueError("ui_dangerous_downweight must be in [0, 1].")
        if not 0.0 < ui_sample_weight_min <= 1.0:
            raise ValueError("ui_sample_weight_min must be in (0, 1].")
        if not 0.0 <= ui_center_momentum < 1.0:
            raise ValueError("ui_center_momentum must be in [0, 1).")
        if ui_center_dim <= 0:
            raise ValueError("ui_center_dim must be positive.")

        self.s = s
        self.m = m
        self.h = h
        self.ui_lambda = ui_lambda
        self.ui_rho = ui_rho
        self.ui_tau_ri = ui_tau_ri
        self.ui_tau_easy = ui_tau_easy
        self.ui_d_margin = ui_d_margin
        self.ui_alpha = ui_alpha
        self.ui_beta = ui_beta
        self.ui_hard_boost = ui_hard_boost
        self.ui_dangerous_downweight = ui_dangerous_downweight
        self.ui_sample_weight_min = ui_sample_weight_min
        self.ui_center_momentum = ui_center_momentum
        self.ui_center_dim = ui_center_dim
        self.t_alpha = t_alpha
        self.curriculum_alpha = curriculum_alpha
        self.eps = eps
        self.last_stats = {}
        self._last_extra_loss = None
        self._last_sample_weight = None
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)
        self.register_buffer("t", torch.zeros(1))
        self.register_buffer("ui_center", torch.zeros(1, ui_center_dim))
        self.register_buffer("ui_center_ready", torch.zeros(1, dtype=torch.bool))

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

    @torch.no_grad()
    def update_ui_center(self, embeddings: torch.Tensor) -> None:
        if embeddings is None or embeddings.numel() == 0:
            return
        center = F.normalize(embeddings.detach().float(), dim=1).mean(dim=0, keepdim=True)
        center = F.normalize(center, dim=1)
        if self.ui_center.shape != center.shape:
            self.ui_center = center.to(device=self.ui_center.device)
            self.ui_center_ready.fill_(True)
            return
        if not bool(self.ui_center_ready.item()):
            self.ui_center.copy_(center.to(device=self.ui_center.device))
            self.ui_center_ready.fill_(True)
            return
        updated = (
            self.ui_center.float() * self.ui_center_momentum
            + center.to(device=self.ui_center.device) * (1.0 - self.ui_center_momentum)
        )
        self.ui_center.copy_(F.normalize(updated, dim=1).to(dtype=self.ui_center.dtype))

    def forward(self, logits, labels, embeddings=None, norms=None):
        self._last_extra_loss = None
        self._last_sample_weight = None
        if norms is None:
            raise RuntimeError(
                "UIAwareCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss "
                "requires feature norms."
            )
        if embeddings is None:
            raise RuntimeError(
                "UIAwareCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss "
                "requires embeddings."
            )

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

        one_hot = torch.zeros_like(rows, dtype=torch.bool)
        one_hot.scatter_(1, target.view(-1, 1), True)

        c_minus = rows.detach().masked_fill(one_hot, -1.0).max(dim=1).values
        d_i = (
            (c_minus - arc_anchor.detach()).relu()
            / (1.0 - arc_anchor.detach() + self.eps)
        ).clamp(0.0, 1.0).detach()

        q_pos = q.clamp(0.0, 1.0)
        q_factor = (0.75 + 0.25 * q_pos).detach()
        gate_lambda_i = (self.h * d_i * q_factor).detach()
        tau = (
            (1.0 - gate_lambda_i) * arc_anchor.detach()
            + gate_lambda_i * u_pos.detach()
        ).detach()

        with torch.no_grad():
            self.t.mul_(self.curriculum_alpha).add_(
                arc_anchor.detach().mean() * (1.0 - self.curriculum_alpha)
            )

        hard_mask = (rows > tau.view(-1, 1)) & (~one_hot)
        total_negatives = (~one_hot).sum().clamp_min(1)
        hard_negative_ratio = hard_mask.sum().float() / total_negatives.float()

        t = self.t.to(dtype=rows.dtype)
        rows = torch.where(hard_mask, rows * (t + rows), rows).to(dtype=rows.dtype)

        ui_ready = bool(self.ui_center_ready.item())
        if ui_ready:
            valid_embeddings = embeddings[index]
            v = F.normalize(valid_embeddings.float(), dim=1)
            ui_center = F.normalize(
                self.ui_center.to(device=v.device, dtype=v.dtype), dim=1
            )
            cos_ui = (v * ui_center).sum(dim=1).clamp(-1.0 + self.eps, 1.0 - self.eps)

            d_p = (1.0 - target_cos.detach().float()).clamp_min(0.0)
            d_n = (1.0 - c_minus.detach().float()).clamp_min(0.0)
            d_ui = (1.0 - cos_ui.detach()).clamp_min(0.0)
            ri = (d_ui * d_n / (d_p + self.eps)).clamp_min(0.0)

            hard_i = torch.sigmoid(
                self.ui_alpha * (c_minus.detach().float() - target_cos.detach().float())
            )
            ui_like_i = torch.sigmoid(self.ui_beta * (self.ui_tau_ri - ri))
            ui_lambda_i = (
                self.ui_lambda * q_factor.float() * hard_i * ui_like_i
            ).detach()
            ui_loss = F.relu(cos_ui - self.ui_rho)
            self._last_extra_loss = (ui_lambda_i * ui_loss).mean()

            positive_wins = target_cos.detach().float() > c_minus.detach().float()
            ui_like_mask = (ri < self.ui_tau_ri) | (d_ui < self.ui_d_margin)
            easy_mask = (ri >= self.ui_tau_easy) & positive_wins
            hard_identifiable_mask = (
                (ri >= self.ui_tau_ri) & (ri < self.ui_tau_easy) & positive_wins
            )
            dangerous_mask = ui_like_mask & (~positive_wins)

            hard_boost = self.ui_hard_boost * hard_identifiable_mask.float()
            danger_drop = self.ui_dangerous_downweight * dangerous_mask.float()
            sample_weight_valid = (1.0 + hard_boost - danger_drop).clamp(
                self.ui_sample_weight_min, 1.0 + self.ui_hard_boost
            )
            sample_weight = torch.ones(
                labels.view(-1).shape[0],
                device=rows.device,
                dtype=rows.dtype,
            )
            sample_weight[index] = sample_weight_valid.to(dtype=rows.dtype)
            self._last_sample_weight = sample_weight.detach()
        else:
            cos_ui = rows.new_zeros(index.numel()).float()
            d_ui = rows.new_zeros(index.numel()).float()
            ri = rows.new_zeros(index.numel()).float()
            hard_i = rows.new_zeros(index.numel()).float()
            ui_like_i = rows.new_zeros(index.numel()).float()
            ui_lambda_i = rows.new_zeros(index.numel()).float()
            ui_loss = rows.new_zeros(index.numel()).float()
            easy_mask = torch.zeros(index.numel(), device=rows.device, dtype=torch.bool)
            hard_identifiable_mask = torch.zeros_like(easy_mask)
            ui_like_mask = torch.zeros_like(easy_mask)
            dangerous_mask = torch.zeros_like(easy_mask)

        rows.scatter_(1, target.view(-1, 1), u_pos.view(-1, 1))
        logits[index] = rows

        with torch.no_grad():
            q_float = q.detach().float()
            q_pos_float = q_pos.detach().float()
            q_factor_float = q_factor.detach().float()
            d_float = d_i.detach().float()
            gate_lambda_float = gate_lambda_i.detach().float()
            quality_ratio = gate_lambda_i / (self.h * d_i + self.eps)
            ui_extra_loss = (
                0.0
                if self._last_extra_loss is None
                else float(self._last_extra_loss.detach().float().item())
            )
            self.last_stats = {
                "q_mean": float(q_float.mean().item()),
                "q_std": float(q_float.std(unbiased=False).item()),
                "q_min": float(q_float.min().item()),
                "q_max": float(q_float.max().item()),
                "q_pos_mean": float(q_pos_float.mean().item()),
                "q_factor_mean": float(q_factor_float.mean().item()),
                "q_factor_min": float(q_factor_float.min().item()),
                "q_factor_max": float(q_factor_float.max().item()),
                "u_pos_mean": float(u_pos.detach().float().mean().item()),
                "arc_anchor_mean": float(arc_anchor.detach().float().mean().item()),
                "c_minus_mean": float(c_minus.detach().float().mean().item()),
                "d_mean": float(d_float.mean().item()),
                "d_max": float(d_float.max().item()),
                "lambda_i_mean": float(gate_lambda_float.mean().item()),
                "lambda_i_max": float(gate_lambda_float.max().item()),
                "tau_mean": float(tau.detach().float().mean().item()),
                "hard_negative_ratio": float(hard_negative_ratio.item()),
                "competition_active_ratio": float((d_float > 0.0).float().mean().item()),
                "quality_modulated_lambda_ratio": float(
                    quality_ratio.detach().float().mean().item()
                ),
                "curricular_t": float(self.t.detach().item()),
                "ui_center_ready": float(1.0 if ui_ready else 0.0),
                "cos_ui_mean": float(cos_ui.detach().float().mean().item()),
                "d_ui_mean": float(d_ui.detach().float().mean().item()),
                "ri_mean": float(ri.detach().float().mean().item()),
                "ri_min": float(ri.detach().float().min().item()),
                "ri_max": float(ri.detach().float().max().item()),
                "hard_i_mean": float(hard_i.detach().float().mean().item()),
                "ui_like_i_mean": float(ui_like_i.detach().float().mean().item()),
                "ui_lambda_i_mean": float(ui_lambda_i.detach().float().mean().item()),
                "ui_lambda_i_max": float(ui_lambda_i.detach().float().max().item()),
                "ui_loss_mean": float(ui_loss.detach().float().mean().item()),
                "ui_extra_loss": ui_extra_loss,
                "easy_recognizable_ratio": float(easy_mask.float().mean().item()),
                "hard_identifiable_ratio": float(
                    hard_identifiable_mask.float().mean().item()
                ),
                "ui_like_ratio": float(ui_like_mask.float().mean().item()),
                "dangerous_ratio": float(dangerous_mask.float().mean().item()),
            }
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


class MultiUIPerceptibilityCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss(nn.Module):
    """Proposed 4.3: multi-UI centers + perceptibility attention.

    Classification branch is identical to Proposed 4.1 (quality-modulated
    competition-adaptive gate).  The UI-aware branch follows Proposed 4.2
    philosophy but uses **offline multi-UI centers** instead of a single
    online EMA center, and supports an external perceptibility attention
    module via ``compute_ui_suppressed_targets``.

    Loss name: multi_ui_perceptibility_competition_quality_adaptive_soft_gated_ada_curricular
    Alias: proposed_4_3_multi_ui_attention
    """

    requires_norms = True
    requires_embeddings = True

    def __init__(
        self,
        s: float = 64.0,
        m: float = 0.4,
        h: float = 0.333,
        multi_ui_centers: torch.Tensor = None,
        ui_center_names=None,
        ui_lambda: float = 0.05,
        ui_rho: float = 0.2,
        ui_tau_ri: float = 1.0,
        ui_tau_easy: float = 2.0,
        ui_d_margin: float = 0.25,
        ui_alpha: float = 10.0,
        ui_beta: float = 5.0,
        ui_hard_boost: float = 0.1,
        ui_dangerous_downweight: float = 0.35,
        ui_sample_weight_min: float = 0.5,
        t_alpha: float = 0.01,
        curriculum_alpha: float = 0.99,
        eps: float = 1e-3,
    ):
        super().__init__()
        # --- validate multi_ui_centers ---
        if multi_ui_centers is None:
            raise ValueError("multi_ui_centers must be provided (Tensor [K, 512]).")
        if multi_ui_centers.ndim != 2:
            raise ValueError(
                f"multi_ui_centers must be 2-D [K, 512], got shape {multi_ui_centers.shape}"
            )
        if multi_ui_centers.shape[1] != 512:
            raise ValueError(
                f"multi_ui_centers.shape[1] must be 512, got {multi_ui_centers.shape[1]}"
            )
        if multi_ui_centers.shape[0] < 1:
            raise ValueError("multi_ui_centers must have at least 1 center.")

        self.s = s
        self.m = m
        self.h = h
        self.ui_lambda = ui_lambda
        self.ui_rho = ui_rho
        self.ui_tau_ri = ui_tau_ri
        self.ui_tau_easy = ui_tau_easy
        self.ui_d_margin = ui_d_margin
        self.ui_alpha = ui_alpha
        self.ui_beta = ui_beta
        self.ui_hard_boost = ui_hard_boost
        self.ui_dangerous_downweight = ui_dangerous_downweight
        self.ui_sample_weight_min = ui_sample_weight_min
        self.t_alpha = t_alpha
        self.curriculum_alpha = curriculum_alpha
        self.eps = eps
        self.last_stats = {}
        self._last_extra_loss = None
        self._last_sample_weight = None

        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)
        self.register_buffer("t", torch.zeros(1))
        self.register_buffer(
            "multi_ui_centers_buf",
            F.normalize(multi_ui_centers.float(), dim=1),
        )
        self.ui_center_names = list(ui_center_names) if ui_center_names else []

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

    def compute_ui_suppressed_targets(self, embeddings: torch.Tensor):
        """Compute UI-suppressed targets for the attention branch.

        For each sample, subtract the component along its nearest UI center:
            v_prime = v - dot(v, u_star) * u_star
            v_prime = normalize(v_prime)

        Returns:
            v_prime: [B, 512] detached, normalized
            nearest_idx: [B] index of nearest UI center
        """
        with torch.no_grad():
            v = F.normalize(embeddings.detach().float(), dim=1)
            centers = self.multi_ui_centers_buf.to(device=v.device, dtype=v.dtype)
            cos_all = v @ centers.T  # [B, K]
            nearest_idx = cos_all.argmax(dim=1)  # [B]
            u_star = centers[nearest_idx]  # [B, 512]
            dot_vu = (v * u_star).sum(dim=1, keepdim=True)  # [B, 1]
            v_prime = v - dot_vu * u_star
            v_prime = F.normalize(v_prime, dim=1)
        return v_prime.detach(), nearest_idx

    def forward(self, logits, labels, embeddings=None, norms=None):
        self._last_extra_loss = None
        self._last_sample_weight = None

        if norms is None:
            raise RuntimeError(
                "MultiUIPerceptibilityCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss "
                "requires feature norms."
            )
        if embeddings is None:
            raise RuntimeError(
                "MultiUIPerceptibilityCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss "
                "requires embeddings."
            )

        index, target = _positive_indices(labels)
        logits = logits.clone()
        if index.numel() == 0:
            self.last_stats = {}
            return logits * self.s

        rows = logits[index].clone()
        q = self._quality_indicator(labels, norms).to(dtype=rows.dtype)

        # --- Proposed 4.1 classification branch ---
        target_cos = _safe_cosine(
            rows.gather(1, target.view(-1, 1)).view(-1), eps=self.eps
        )
        theta_y = target_cos.acos()

        u_pos = torch.cos(theta_y - self.m * q)
        u_pos = (u_pos - (self.m * q + self.m)).to(dtype=rows.dtype)
        arc_anchor = torch.cos(theta_y + self.m).to(dtype=rows.dtype)

        one_hot = torch.zeros_like(rows, dtype=torch.bool)
        one_hot.scatter_(1, target.view(-1, 1), True)

        c_minus = rows.detach().masked_fill(one_hot, -1.0).max(dim=1).values
        d_i = (
            (c_minus - arc_anchor.detach()).relu()
            / (1.0 - arc_anchor.detach() + self.eps)
        ).clamp(0.0, 1.0).detach()

        q_pos = q.clamp(0.0, 1.0)
        q_factor = (0.75 + 0.25 * q_pos).detach()
        gate_lambda_i = (self.h * d_i * q_factor).detach()
        tau = (
            (1.0 - gate_lambda_i) * arc_anchor.detach()
            + gate_lambda_i * u_pos.detach()
        ).detach()

        with torch.no_grad():
            self.t.mul_(self.curriculum_alpha).add_(
                arc_anchor.detach().mean() * (1.0 - self.curriculum_alpha)
            )

        hard_mask = (rows > tau.view(-1, 1)) & (~one_hot)
        total_negatives = (~one_hot).sum().clamp_min(1)
        hard_negative_ratio = hard_mask.sum().float() / total_negatives.float()

        t_val = self.t.to(dtype=rows.dtype)
        rows = torch.where(hard_mask, rows * (t_val + rows), rows).to(dtype=rows.dtype)

        # --- Multi-UI branch ---
        valid_embeddings = embeddings[index]
        v = F.normalize(valid_embeddings.float(), dim=1)
        centers = self.multi_ui_centers_buf.to(device=v.device, dtype=v.dtype)

        cos_all = v @ centers.T  # [N_valid, K]
        cos_ui_multi, nearest_ui = cos_all.max(dim=1)  # [N_valid]
        cos_ui_multi = cos_ui_multi.clamp(-1.0 + self.eps, 1.0 - self.eps)

        d_ui_multi = (1.0 - cos_ui_multi.detach()).clamp_min(0.0)
        d_p = (1.0 - target_cos.detach().float()).clamp_min(0.0)
        d_n = (1.0 - c_minus.detach().float()).clamp_min(0.0)

        RI_multi = (d_ui_multi * d_n / (d_p + self.eps)).clamp_min(0.0)

        # Multi-UI loss
        L_UI_multi = F.relu(cos_ui_multi - self.ui_rho)

        # Hardness
        hard_i = torch.sigmoid(
            self.ui_alpha * (c_minus.detach().float() - target_cos.detach().float())
        )
        # UI-like
        ui_like_i = torch.sigmoid(self.ui_beta * (self.ui_tau_ri - RI_multi))

        # UI coefficient
        lambda_UI_multi_i = (
            self.ui_lambda * q_factor.float() * hard_i * ui_like_i
        ).detach()

        # Extra loss
        L_extra_multi = (lambda_UI_multi_i * L_UI_multi).mean()
        self._last_extra_loss = L_extra_multi

        # --- Sample weighting ---
        positive_wins = target_cos.detach().float() > c_minus.detach().float()
        ui_like_mask = (RI_multi < self.ui_tau_ri) | (d_ui_multi < self.ui_d_margin)
        hard_identifiable_mask = (
            (RI_multi >= self.ui_tau_ri) & (RI_multi < self.ui_tau_easy) & positive_wins
        )
        dangerous_mask = ui_like_mask & (~positive_wins)

        hard_boost = self.ui_hard_boost * hard_identifiable_mask.float()
        danger_drop = self.ui_dangerous_downweight * dangerous_mask.float()
        sample_weight_valid = (1.0 + hard_boost - danger_drop).clamp(
            self.ui_sample_weight_min, 1.0 + self.ui_hard_boost
        )
        sample_weight = torch.ones(
            labels.view(-1).shape[0],
            device=rows.device,
            dtype=rows.dtype,
        )
        sample_weight[index] = sample_weight_valid.to(dtype=rows.dtype)
        self._last_sample_weight = sample_weight.detach()

        # --- Finalize logits ---
        rows.scatter_(1, target.view(-1, 1), u_pos.view(-1, 1))
        logits[index] = rows

        # --- Stats ---
        with torch.no_grad():
            q_float = q.detach().float()
            d_float = d_i.detach().float()
            gate_lambda_float = gate_lambda_i.detach().float()
            ui_extra_loss_val = float(L_extra_multi.detach().float().item())

            stats = {
                "q_mean": float(q_float.mean().item()),
                "d_mean": float(d_float.mean().item()),
                "gate_lambda_i_mean": float(gate_lambda_float.mean().item()),
                "hard_negative_ratio": float(hard_negative_ratio.item()),
                "curricular_t": float(self.t.detach().item()),
                "cos_ui_multi_mean": float(cos_ui_multi.detach().float().mean().item()),
                "d_ui_multi_mean": float(d_ui_multi.detach().float().mean().item()),
                "ri_multi_mean": float(RI_multi.detach().float().mean().item()),
                "ri_multi_min": float(RI_multi.detach().float().min().item()),
                "ri_multi_max": float(RI_multi.detach().float().max().item()),
                "ui_like_i_mean": float(ui_like_i.detach().float().mean().item()),
                "ui_lambda_i_mean": float(lambda_UI_multi_i.detach().float().mean().item()),
                "ui_loss_mean": float(L_UI_multi.detach().float().mean().item()),
                "ui_extra_loss": ui_extra_loss_val,
                "sample_weight_mean": float(sample_weight.detach().float().mean().item()),
                "sample_weight_min": float(sample_weight.detach().float().min().item()),
                "sample_weight_max": float(sample_weight.detach().float().max().item()),
                "hard_identifiable_ratio": float(
                    hard_identifiable_mask.float().mean().item()
                ),
                "ui_like_ratio": float(ui_like_mask.float().mean().item()),
                "dangerous_ratio": float(dangerous_mask.float().mean().item()),
            }
            # nearest UI center counts
            try:
                K = centers.shape[0]
                counts = torch.zeros(K, device=nearest_ui.device)
                counts.scatter_add_(0, nearest_ui, torch.ones_like(nearest_ui, dtype=counts.dtype))
                for k_idx in range(K):
                    cname = self.ui_center_names[k_idx] if k_idx < len(self.ui_center_names) else f"center_{k_idx}"
                    stats[f"nearest_ui_{cname}"] = int(counts[k_idx].item())
            except Exception:
                pass
            self.last_stats = stats

        return logits * self.s
