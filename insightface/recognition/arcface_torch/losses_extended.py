"""
Extended loss functions for lightweight face recognition.

All losses use a unified interface:
    forward(logits, labels, embeddings=None, norms=None)

- ArcFace (CombinedMarginLoss): ignores embeddings/norms
- AdaFace: uses norms for quality-adaptive margin
- MagFace: uses norms for magnitude-adaptive margin

This file does NOT modify the original losses.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedMarginLossWrapper(nn.Module):
    """Wrapper around original CombinedMarginLoss for unified interface.

    Allows CombinedMarginLoss to be used with PartialFC_V2_Extended
    which passes (logits, labels, embeddings=..., norms=...).
    """

    def __init__(self, s, m1, m2, m3, interclass_filtering_threshold=0):
        super().__init__()
        from losses import CombinedMarginLoss
        self.loss = CombinedMarginLoss(
            s, m1, m2, m3, interclass_filtering_threshold)

    def forward(self, logits, labels, embeddings=None, norms=None):
        return self.loss(logits, labels)


class AdaFaceLoss(nn.Module):
    """AdaFace: Quality Adaptive Margin for Face Recognition (CVPR 2022).

    Uses feature norm as an image quality indicator to compute
    quality-adaptive angular margins. Low-quality samples get smaller
    margins (easier), high-quality samples get larger margins (harder).

    Reference: https://arxiv.org/abs/2204.00964

    Args:
        s: scale factor (default: 64.0)
        m: base margin (default: 0.4)
        h: AdaFace hyper-parameter controlling margin range (default: 0.333)
        t_alpha: EMA decay for batch norm statistics (default: 0.01)
    """

    def __init__(self, s=64.0, m=0.4, h=0.333, t_alpha=0.01):
        super(AdaFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.h = h
        self.t_alpha = t_alpha

        # Running statistics for norm-based quality indicator
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1) * 20)
        self.register_buffer('batch_std', torch.ones(1) * 100)

        # Precompute ArcFace constants for safe_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(self, logits, labels, embeddings=None, norms=None):
        """
        Args:
            logits: cosine similarities (B, C) — already clamped to [-1, 1]
            labels: (B, 1) with -1 for non-positive classes
            embeddings: normalized embeddings (B, D) — not used directly
            norms: feature norms before normalization (B, 1)
        """
        index_positive = torch.where(labels != -1)[0]

        if norms is None:
            # Fallback: no norms provided, behave like standard ArcFace
            return self._arcface_fallback(logits, labels, index_positive)

        # Update running statistics of feature norms
        with torch.no_grad():
            norms_flat = norms[index_positive].squeeze()
            if norms_flat.numel() > 0:
                batch_mean = norms_flat.mean()
                batch_std = norms_flat.std()
                self.batch_mean = (1 - self.t_alpha) * self.batch_mean + \
                                  self.t_alpha * batch_mean
                self.batch_std = (1 - self.t_alpha) * self.batch_std + \
                                 self.t_alpha * batch_std

        # Compute quality indicator: normalize norms to ~[-1, 1]
        margin_scaler = (norms - self.batch_mean) / (
            self.batch_std + 1e-8)  # z-score
        margin_scaler = margin_scaler * self.h  # scale by h
        margin_scaler = torch.clip(margin_scaler, -1, 1)  # clamp

        # Get target logits (cosine of angle with true class)
        target_logit = logits[index_positive,
                              labels[index_positive].view(-1)]

        with torch.no_grad():
            # Adaptive margin: g_angular (arccos-based) + g_additive
            target_logit_arccos = target_logit.arccos()
            logits.arccos_()

            # Angular margin (scaled by quality)
            # High quality (margin_scaler > 0) → larger margin
            # Low quality (margin_scaler < 0) → smaller margin
            adaptive_m = self.m * (1 + margin_scaler[index_positive].view(-1))
            adaptive_m = torch.clamp(adaptive_m, 0.0, math.pi / 4)

            final_target_logit = target_logit_arccos + adaptive_m
            logits[index_positive,
                   labels[index_positive].view(-1)] = final_target_logit
            logits.cos_()

            # Additive margin (also quality-adaptive)
            g_add = -(self.m + (self.m * margin_scaler[index_positive].view(-1)))
            logits[index_positive,
                   labels[index_positive].view(-1)] += g_add

        logits = logits * self.s
        return logits

    def _arcface_fallback(self, logits, labels, index_positive):
        """Standard ArcFace when norms not available."""
        target_logit = logits[index_positive,
                              labels[index_positive].view(-1)]
        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.m
            logits[index_positive,
                   labels[index_positive].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s
        return logits


class MagFaceLoss(nn.Module):
    """MagFace: A Universal Representation for Face Recognition
    and Quality Assessment (CVPR 2021).

    Uses feature magnitude for adaptive angular margin. Larger magnitudes
    (higher quality) get larger margins.

    Reference: https://arxiv.org/abs/2103.06627

    Args:
        s: scale factor
        l_a: lower bound of magnitude
        u_a: upper bound of magnitude
        l_m: lower margin (for low magnitude)
        u_m: upper margin (for high magnitude)
    """

    def __init__(self, s=64.0, l_a=10, u_a=110, l_m=0.45, u_m=0.8):
        super(MagFaceLoss, self).__init__()
        self.s = s
        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m

    def _calc_margin(self, norms):
        """Linear interpolation of margin based on feature norm magnitude."""
        norms_clamped = torch.clamp(norms, self.l_a, self.u_a)
        margin = (self.u_m - self.l_m) / (self.u_a - self.l_a) * \
                 (norms_clamped - self.l_a) + self.l_m
        return margin

    def forward(self, logits, labels, embeddings=None, norms=None):
        index_positive = torch.where(labels != -1)[0]

        if norms is None:
            raise ValueError(
                "MagFace requires feature norms. "
                "Use PartialFC_V2_Extended or pass norms explicitly.")

        # Compute per-sample adaptive margin
        margins = self._calc_margin(norms[index_positive].view(-1))

        target_logit = logits[index_positive,
                              labels[index_positive].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + margins
            logits[index_positive,
                   labels[index_positive].view(-1)] = final_target_logit
            logits.cos_()

        logits = logits * self.s
        return logits


# =========================================================================
# Stubs for future extensions (Phase 7)
# =========================================================================

class ElasticFaceLoss(nn.Module):
    """ElasticFace: Elastic Margin Loss for Deep Face Recognition (CVPRW 2022).

    Placeholder — not yet implemented.
    Uses random margin sampling from a distribution for regularization.

    Reference: https://arxiv.org/abs/2109.09416
    """

    def __init__(self, s=64.0, m=0.5, std=0.05):
        super(ElasticFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.std = std
        raise NotImplementedError(
            "ElasticFace is not yet implemented. "
            "Planned for Phase 7 extension.")

    def forward(self, logits, labels, embeddings=None, norms=None):
        raise NotImplementedError


class CurricularFaceLoss(nn.Module):
    """CurricularFace: Adaptive Curriculum Learning Loss (CVPR 2020).

    Placeholder — not yet implemented.
    Uses curriculum learning strategy for face recognition training.

    Reference: https://arxiv.org/abs/2004.00288
    """

    def __init__(self, s=64.0, m=0.5):
        super(CurricularFaceLoss, self).__init__()
        self.s = s
        self.m = m
        raise NotImplementedError(
            "CurricularFace is not yet implemented. "
            "Planned for Phase 7 extension.")

    def forward(self, logits, labels, embeddings=None, norms=None):
        raise NotImplementedError
