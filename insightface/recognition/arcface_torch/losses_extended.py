"""Extended margin losses used by the lightweight FR experiments.

Every loss in this module uses the same interface:

    forward(logits, labels, embeddings=None, norms=None)

``logits`` are cosine similarities computed from L2-normalized features and
class weights. ``embeddings`` are the raw backbone outputs before L2
normalization. ``norms`` are their feature norms and are required by AdaFace
and the proposed CurriculumAwareAdaFace loss.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn


def _label_view(labels: torch.Tensor) -> torch.Tensor:
    return labels.view(-1, 1).long()


def _positive_indices(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = _label_view(labels)
    index = torch.where(labels.view(-1) != -1)[0]
    target = labels[index].view(-1)
    return index, target


def _safe_cosine(logits: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return logits.clamp(-1.0 + eps, 1.0 - eps)


def _arcface_target(target_logit: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
    theta = _safe_cosine(target_logit).acos()
    return torch.cos(theta + margin)


class CombinedMarginLossWrapper(nn.Module):
    """Adapter for the original CombinedMarginLoss with the extended interface."""

    requires_norms = False
    requires_embeddings = False

    def __init__(self, s, m1, m2, m3, interclass_filtering_threshold=0):
        super().__init__()
        try:
            from losses import CombinedMarginLoss
        except ImportError:
            from .losses import CombinedMarginLoss

        self.loss = CombinedMarginLoss(
            s, m1, m2, m3, interclass_filtering_threshold
        )

    def forward(self, logits, labels, embeddings=None, norms=None):
        return self.loss(logits, labels)


class ArcFaceLoss(nn.Module):
    """ArcFace additive angular margin loss."""

    requires_norms = False
    requires_embeddings = False

    def __init__(self, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m

    def forward(self, logits, labels, embeddings=None, norms=None):
        index, target = _positive_indices(labels)
        logits = logits.clone()
        if index.numel() == 0:
            return logits * self.s

        target_logit = logits[index, target]
        logits[index, target] = _arcface_target(
            target_logit, torch.full_like(target_logit, self.m)
        )
        return logits * self.s


class CosFaceLoss(nn.Module):
    """CosFace additive cosine margin loss."""

    requires_norms = False
    requires_embeddings = False

    def __init__(self, s=64.0, m=0.4):
        super().__init__()
        self.s = s
        self.m = m

    def forward(self, logits, labels, embeddings=None, norms=None):
        index, target = _positive_indices(labels)
        logits = logits.clone()
        if index.numel() == 0:
            return logits * self.s

        logits[index, target] = logits[index, target] - self.m
        return logits * self.s


class ElasticFaceLoss(nn.Module):
    """ElasticFace with stochastic per-sample angular margins."""

    requires_norms = False
    requires_embeddings = False

    def __init__(self, s=64.0, m=0.5, std=0.0125, min_m=0.0):
        super().__init__()
        self.s = s
        self.m = m
        self.std = std
        self.min_m = min_m

    def forward(self, logits, labels, embeddings=None, norms=None):
        index, target = _positive_indices(labels)
        logits = logits.clone()
        if index.numel() == 0:
            return logits * self.s

        target_logit = logits[index, target]
        if self.training:
            margins = torch.normal(
                mean=self.m,
                std=self.std,
                size=target_logit.shape,
                device=target_logit.device,
                dtype=target_logit.dtype,
            ).clamp_min(self.min_m)
        else:
            margins = torch.full_like(target_logit, self.m)

        logits[index, target] = _arcface_target(target_logit, margins)
        return logits * self.s


class CurricularFaceLoss(nn.Module):
    """CurricularFace with hard-negative curriculum modulation."""

    requires_norms = False
    requires_embeddings = False

    def __init__(self, s=64.0, m=0.5, alpha=0.99):
        super().__init__()
        self.s = s
        self.m = m
        self.alpha = alpha
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.register_buffer("t", torch.zeros(1))

    def forward(self, logits, labels, embeddings=None, norms=None):
        index, target = _positive_indices(labels)
        logits = logits.clone()
        if index.numel() == 0:
            return logits * self.s

        target_logit = _safe_cosine(logits[index, target])
        sin_theta = torch.sqrt((1.0 - target_logit.pow(2)).clamp_min(0.0))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m
        final_target = torch.where(
            target_logit > self.threshold,
            cos_theta_m,
            target_logit - self.mm,
        )

        with torch.no_grad():
            self.t.mul_(self.alpha).add_(target_logit.mean() * (1.0 - self.alpha))

        rows = logits[index].clone()
        one_hot = torch.zeros_like(rows, dtype=torch.bool)
        one_hot.scatter_(1, target.view(-1, 1), True)
        hard_mask = (rows > cos_theta_m.view(-1, 1)) & (~one_hot)
        rows = torch.where(hard_mask, rows * (self.t + rows), rows)
        rows.scatter_(1, target.view(-1, 1), final_target.view(-1, 1))
        logits[index] = rows
        return logits * self.s


class AdaFaceLoss(nn.Module):
    """AdaFace quality-adaptive margin loss.

    If feature norms are unavailable, this loss falls back to ArcFace. The
    fallback keeps old pipelines usable, while Phase 2 passes raw norms through
    the margin head by default.
    """

    requires_norms = True
    requires_embeddings = False

    def __init__(self, s=64.0, m=0.4, h=0.333, t_alpha=0.01, eps=1e-3):
        super().__init__()
        self.s = s
        self.m = m
        self.h = h
        self.t_alpha = t_alpha
        self.eps = eps
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)

    def _margin_scaler(self, labels, norms):
        index, _ = _positive_indices(labels)
        safe_norms = norms.clamp(min=0.001, max=100.0).detach()

        with torch.no_grad():
            positive_norms = safe_norms[index]
            if positive_norms.numel() > 1:
                batch_mean = positive_norms.mean()
                batch_std = positive_norms.std(unbiased=False).clamp_min(self.eps)
                self.batch_mean.mul_(1.0 - self.t_alpha).add_(batch_mean * self.t_alpha)
                self.batch_std.mul_(1.0 - self.t_alpha).add_(batch_std * self.t_alpha)

        margin_scaler = (
            (safe_norms[index].view(-1) - self.batch_mean)
            / (self.batch_std + self.eps)
        )
        return (margin_scaler * self.h).clamp(-1.0, 1.0)

    def forward(self, logits, labels, embeddings=None, norms=None):
        if norms is None:
            return ArcFaceLoss(s=self.s, m=self.m)(logits, labels)

        index, target = _positive_indices(labels)
        logits = logits.clone()
        if index.numel() == 0:
            return logits * self.s

        margin_scaler = self._margin_scaler(labels, norms)
        rows = logits[index].clone()

        m_arc = torch.zeros_like(rows)
        g_angular = -self.m * margin_scaler
        m_arc.scatter_(1, target.view(-1, 1), g_angular.view(-1, 1))

        theta = _safe_cosine(rows, eps=self.eps).acos()
        rows = torch.cos((theta + m_arc).clamp(self.eps, math.pi - self.eps))

        m_cos = torch.zeros_like(rows)
        g_add = self.m + self.m * margin_scaler
        m_cos.scatter_(1, target.view(-1, 1), g_add.view(-1, 1))
        rows = rows - m_cos

        logits[index] = rows
        return logits * self.s


class CurriculumAwareAdaFaceLoss(AdaFaceLoss):
    """Proposed loss placeholder: AdaFace target margin plus curriculum negatives.

    This is intentionally small and readable so it can be improved later. It
    keeps AdaFace's quality-adaptive target margin, then applies a
    CurricularFace-style hard-negative modulation based on an EMA buffer ``t``.
    """

    requires_norms = True
    requires_embeddings = False

    def __init__(self, s=64.0, m=0.4, h=0.333, t_alpha=0.01, alpha=0.99):
        super().__init__(s=s, m=m, h=h, t_alpha=t_alpha)
        self.alpha = alpha
        self.register_buffer("t", torch.zeros(1))

    def forward(self, logits, labels, embeddings=None, norms=None):
        if norms is None:
            return CurricularFaceLoss(s=self.s, m=0.5, alpha=self.alpha)(logits, labels)

        index, target = _positive_indices(labels)
        raw_logits = logits.clone()
        if index.numel() == 0:
            return raw_logits * self.s

        ada_logits = super().forward(logits, labels, embeddings=embeddings, norms=norms)
        ada_logits = ada_logits / self.s
        target_after_ada = ada_logits[index, target]
        target_before = _safe_cosine(raw_logits[index, target]).detach()

        with torch.no_grad():
            self.t.mul_(self.alpha).add_(target_before.mean() * (1.0 - self.alpha))

        rows = ada_logits[index].clone()
        one_hot = torch.zeros_like(rows, dtype=torch.bool)
        one_hot.scatter_(1, target.view(-1, 1), True)
        hard_threshold = target_after_ada.detach().view(-1, 1)
        hard_mask = (rows > hard_threshold) & (~one_hot)
        rows = torch.where(hard_mask, rows * (self.t + rows), rows)
        rows.scatter_(1, target.view(-1, 1), target_after_ada.view(-1, 1))
        ada_logits[index] = rows
        return ada_logits * self.s


class MagFaceLoss(nn.Module):
    """MagFace magnitude-adaptive margin with feature-norm regularization."""

    requires_norms = True
    requires_embeddings = False

    def __init__(
        self,
        s=64.0,
        l_a=10.0,
        u_a=110.0,
        l_m=0.45,
        u_m=0.8,
        lambda_g=20.0,
    ):
        super().__init__()
        self.s = s
        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m
        self.lambda_g = lambda_g
        self._last_mag_reg = None

    def _calc_margin(self, norms):
        norms = torch.clamp(norms, self.l_a, self.u_a)
        return self.l_m + (self.u_m - self.l_m) * (norms - self.l_a) / (
            self.u_a - self.l_a
        )

    def _calc_reg(self, norms):
        norms = torch.clamp(norms, self.l_a, self.u_a)
        return norms / (self.u_a ** 2) + 1.0 / norms

    def forward(self, logits, labels, embeddings=None, norms=None):
        if norms is None:
            raise ValueError("MagFaceLoss requires feature norms.")

        index, target = _positive_indices(labels)
        logits = logits.clone()
        if index.numel() == 0:
            self._last_mag_reg = logits.new_zeros(())
            return logits * self.s

        positive_norms = norms[index].view(-1)
        margins = self._calc_margin(positive_norms)
        target_logit = logits[index, target]
        logits[index, target] = _arcface_target(target_logit, margins)
        self._last_mag_reg = self.lambda_g * self._calc_reg(positive_norms).mean()
        return logits * self.s


@dataclass(frozen=True)
class LossSpec:
    factory: Callable[[], nn.Module]
    requires_norms: bool = False
    description: str = ""


PHASE2_LOSS_REGISTRY: Dict[str, LossSpec] = {
    "arcface": LossSpec(
        factory=lambda: ArcFaceLoss(s=64.0, m=0.5),
        description="ArcFace additive angular margin",
    ),
    "cosface": LossSpec(
        factory=lambda: CosFaceLoss(s=64.0, m=0.4),
        description="CosFace additive cosine margin",
    ),
    "elasticface": LossSpec(
        factory=lambda: ElasticFaceLoss(s=64.0, m=0.5, std=0.0125),
        description="ElasticFace stochastic angular margin",
    ),
    "curricularface": LossSpec(
        factory=lambda: CurricularFaceLoss(s=64.0, m=0.5),
        description="CurricularFace hard-negative curriculum",
    ),
    "adaface": LossSpec(
        factory=lambda: AdaFaceLoss(s=64.0, m=0.4, h=0.333),
        requires_norms=True,
        description="AdaFace quality-adaptive margin",
    ),
    "magface": LossSpec(
        factory=lambda: MagFaceLoss(
            s=64.0, l_a=10.0, u_a=110.0, l_m=0.45, u_m=0.8, lambda_g=20.0
        ),
        requires_norms=True,
        description="MagFace magnitude-adaptive margin",
    ),
    "proposed": LossSpec(
        factory=lambda: CurriculumAwareAdaFaceLoss(s=64.0, m=0.4, h=0.333),
        requires_norms=True,
        description="Curriculum-aware AdaFace prototype",
    ),
}


def available_phase2_losses():
    return sorted(PHASE2_LOSS_REGISTRY.keys())


def get_phase2_loss(loss_name: str):
    """Return ``(loss_module, requires_norms)`` for a Phase 2 loss name."""

    key = loss_name.lower()
    if key not in PHASE2_LOSS_REGISTRY:
        available = ", ".join(available_phase2_losses())
        raise ValueError(f"Unknown loss '{loss_name}'. Available: {available}")
    spec = PHASE2_LOSS_REGISTRY[key]
    return spec.factory(), spec.requires_norms
