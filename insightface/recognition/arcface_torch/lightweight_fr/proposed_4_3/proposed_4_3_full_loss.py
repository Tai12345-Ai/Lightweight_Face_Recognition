"""Proposed 4.3 Full identity-safe loss components.

This module is intentionally isolated from the existing soft-gated trainer so that
baseline/Core runners remain unchanged.  The Kaggle Full trainer monkey-patches
`train_soft_gated_lambda_kaggle.MultiUIPerceptibility...` with the class below.

Implemented Full-v1 terms:
- soft top-M UI prototype selection;
- UI-orthogonal projection against the ground-truth class center;
- label-confidence / recoverability / unrecognizable gates;
- identity-anchor proxy loss;
- negative-guard proxy loss;
- preserve proxy loss for easy/non-UI samples;
- diagnostics for Delta C, Delta N, Delta U, rho gates.

The base AdaFace-positive + CurricularFace-negative branch is kept identical in
spirit to the existing Proposed 4.3 loss, while the extra loss is changed from
hard-nearest UI penalty to the identity-safe Full objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .soft_gated_losses import _positive_indices, _safe_cosine


class Proposed43FullIdentitySafeLoss(nn.Module):
    """Full-v1 loss for Proposed 4.3++.

    The class has the same constructor interface as
    MultiUIPerceptibilityCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss
    so it can be dropped into the existing Kaggle trainer.
    """

    requires_norms = True
    requires_embeddings = True
    requires_class_weights = True

    def __init__(
        self,
        s: float = 64.0,
        m: float = 0.4,
        h: float = 0.333,
        multi_ui_centers: torch.Tensor | None = None,
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
        top_m: int = 4,
        ui_soft_tau: float = 12.0,
        ui_margin: float = 0.20,
        label_margin: float = 0.05,
        label_gamma: float = 12.0,
        unrec_tau: float = 0.35,
        unrec_gamma: float = 8.0,
        ri_lambda: float = 0.05,
        attention_lambda: float = 1.0,
        anchor_lambda: float = 0.08,
        neg_lambda: float = 0.06,
        preserve_lambda: float = 0.03,
        delta_c: float = 0.02,
        delta_n: float = 0.02,
    ):
        super().__init__()
        if multi_ui_centers is None:
            raise ValueError("multi_ui_centers must be provided for Proposed43FullIdentitySafeLoss.")
        if multi_ui_centers.ndim != 2:
            raise ValueError(f"multi_ui_centers must be [K,D], got {tuple(multi_ui_centers.shape)}")
        if multi_ui_centers.shape[0] < 1:
            raise ValueError("multi_ui_centers must contain at least one prototype.")

        self.s = float(s)
        self.m = float(m)
        self.h = float(h)
        self.ui_lambda = float(ui_lambda)
        self.ui_rho = float(ui_rho)
        self.ui_tau_ri = float(ui_tau_ri)
        self.ui_tau_easy = float(ui_tau_easy)
        self.ui_d_margin = float(ui_d_margin)
        self.ui_alpha = float(ui_alpha)
        self.ui_beta = float(ui_beta)
        self.ui_hard_boost = float(ui_hard_boost)
        self.ui_dangerous_downweight = float(ui_dangerous_downweight)
        self.ui_sample_weight_min = float(ui_sample_weight_min)
        self.t_alpha = float(t_alpha)
        self.curriculum_alpha = float(curriculum_alpha)
        self.eps = float(eps)

        self.top_m = int(max(1, top_m))
        self.ui_soft_tau = float(ui_soft_tau)
        self.ui_margin = float(ui_margin)
        self.label_margin = float(label_margin)
        self.label_gamma = float(label_gamma)
        self.unrec_tau = float(unrec_tau)
        self.unrec_gamma = float(unrec_gamma)
        self.ri_lambda = float(ri_lambda)
        self.attention_lambda = float(attention_lambda)
        self.anchor_lambda = float(anchor_lambda)
        self.neg_lambda = float(neg_lambda)
        self.preserve_lambda = float(preserve_lambda)
        self.delta_c = float(delta_c)
        self.delta_n = float(delta_n)

        self.last_stats = {}
        self._last_extra_loss = None
        self._last_sample_weight = None

        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)
        self.register_buffer("t", torch.zeros(1))
        self.register_buffer("multi_ui_centers_buf", F.normalize(multi_ui_centers.float(), dim=1))
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
        q = (safe_norms[index].view(-1) - self.batch_mean) / (self.batch_std + self.eps)
        return (q * self.h).clamp(-1.0, 1.0).detach()

    def _soft_topm_ui(self, v: torch.Tensor):
        centers = self.multi_ui_centers_buf.to(device=v.device, dtype=v.dtype)
        cos_all = v @ centers.T
        top_m = min(self.top_m, centers.shape[0])
        top_vals, top_idx = torch.topk(cos_all, k=top_m, dim=1)
        weights = torch.softmax(self.ui_soft_tau * top_vals, dim=1)
        selected = centers[top_idx]
        u_soft = (weights.unsqueeze(-1) * selected).sum(dim=1)
        u_soft = F.normalize(u_soft, dim=1)
        return u_soft, top_vals, top_idx, weights

    def compute_ui_suppressed_targets(self, embeddings: torch.Tensor):
        """Compatibility method for the existing attention auxiliary branch."""
        with torch.no_grad():
            v = F.normalize(embeddings.detach().float(), dim=1)
            u_soft, _, top_idx, _ = self._soft_topm_ui(v)
            dot_vu = (v * u_soft).sum(dim=1, keepdim=True)
            v_prime = F.normalize(v - dot_vu * u_soft, dim=1)
            nearest_idx = top_idx[:, 0]
        return v_prime.detach(), nearest_idx.detach()

    def forward(self, logits, labels, embeddings=None, norms=None, class_weights=None, context=None):
        self._last_extra_loss = None
        self._last_sample_weight = None
        if norms is None:
            raise RuntimeError("Proposed43FullIdentitySafeLoss requires feature norms.")
        if embeddings is None:
            raise RuntimeError("Proposed43FullIdentitySafeLoss requires embeddings.")
        if class_weights is None:
            raise RuntimeError("Proposed43FullIdentitySafeLoss requires normalized class_weights.")

        index, target = _positive_indices(labels)
        logits = logits.clone()
        if index.numel() == 0:
            self.last_stats = {}
            return logits * self.s

        rows = logits[index].clone()
        q = self._quality_indicator(labels, norms).to(dtype=rows.dtype)

        # Base AdaFace-positive + CurricularFace-negative branch.
        target_cos = _safe_cosine(rows.gather(1, target.view(-1, 1)).view(-1), eps=self.eps)
        theta_y = target_cos.acos()
        u_pos = torch.cos(theta_y - self.m * q)
        u_pos = (u_pos - (self.m * q + self.m)).to(dtype=rows.dtype)
        arc_anchor = torch.cos(theta_y + self.m).to(dtype=rows.dtype)

        one_hot = torch.zeros_like(rows, dtype=torch.bool)
        one_hot.scatter_(1, target.view(-1, 1), True)
        c_minus = rows.detach().masked_fill(one_hot, -1.0).max(dim=1).values
        d_i = ((c_minus - arc_anchor.detach()).relu() / (1.0 - arc_anchor.detach() + self.eps)).clamp(0.0, 1.0).detach()
        q_pos = q.clamp(0.0, 1.0)
        q_factor = (0.75 + 0.25 * q_pos).detach()
        gate_lambda_i = (self.h * d_i * q_factor).detach()
        tau = ((1.0 - gate_lambda_i) * arc_anchor.detach() + gate_lambda_i * u_pos.detach()).detach()
        with torch.no_grad():
            self.t.mul_(self.curriculum_alpha).add_(arc_anchor.detach().mean() * (1.0 - self.curriculum_alpha))
        hard_mask = (rows > tau.view(-1, 1)) & (~one_hot)
        total_negatives = (~one_hot).sum().clamp_min(1)
        hard_negative_ratio = hard_mask.sum().float() / total_negatives.float()
        t_val = self.t.to(dtype=rows.dtype)
        rows = torch.where(hard_mask, rows * (t_val + rows), rows).to(dtype=rows.dtype)

        # Full identity-safe branch.  All post-attention terms use x'.
        context = context or {}
        x_prime_all = context.get("prime_x")
        if x_prime_all is None:
            x_prime_all = F.normalize(embeddings.float(), dim=1)
        x_base_all = context.get("base_x")
        if x_base_all is None:
            x_base_all = x_prime_all.detach()

        x_prime = x_prime_all[index].float()
        x_base = x_base_all[index].float()
        w = F.normalize(class_weights.float(), dim=1).to(device=x_prime.device, dtype=x_prime.dtype)
        w_y = w[target]

        base_logits = x_base @ w.T
        base_rows = base_logits
        C = _safe_cosine(base_rows.gather(1, target.view(-1, 1)).view(-1), eps=self.eps)
        base_one_hot = torch.zeros_like(base_rows, dtype=torch.bool)
        base_one_hot.scatter_(1, target.view(-1, 1), True)
        N = base_rows.masked_fill(base_one_hot, -1.0).max(dim=1).values
        C_prime = target_cos
        N_prime = c_minus

        centers = self.multi_ui_centers_buf.to(device=x_prime.device, dtype=x_prime.dtype)
        U = (x_base @ centers.T).max(dim=1).values
        U_prime = (x_prime @ centers.T).max(dim=1).values

        d_ui = (1.0 - U.detach()).clamp_min(0.0)
        d_p = (1.0 - C.detach()).clamp_min(0.0)
        d_n = (1.0 - N.detach()).clamp_min(0.0)
        ri_target = (
            torch.log(d_ui + self.eps)
            + torch.log(d_n + self.eps)
            - torch.log(d_p + self.eps)
        )
        ri_pred_all = context.get("ri_pred")
        if ri_pred_all is None:
            ri_loss = logits.new_zeros(())
            low_ri_true = torch.sigmoid(-ri_target.detach())
        else:
            ri_pred = ri_pred_all[index].float()
            ri_loss = F.smooth_l1_loss(ri_pred, ri_target.detach())
            low_ri_true = torch.sigmoid(-ri_target.detach())

        rho_att_all = context.get("rho_att")
        if rho_att_all is None:
            rho_att = torch.ones_like(C_prime.detach().float())
        else:
            rho_att = rho_att_all[index].float().detach().clamp(0.0, 1.0)
        omega_all = context.get("omega_unrec")
        if omega_all is None:
            omega_unrec = torch.sigmoid(self.unrec_gamma * (self.unrec_tau - ri_target.detach()))
        else:
            omega_unrec = omega_all[index].float().detach().clamp(0.0, 1.0)
        label_gate = torch.sigmoid(
            self.label_gamma * (C.detach().float() - N.detach().float() - self.label_margin)
        )
        rho_ui = (
            low_ri_true
            * q_factor.float()
            * label_gate
            * (1.0 - omega_unrec)
        ).clamp(0.0, 1.0).detach()
        rho_neg = torch.maximum(
            rho_ui,
            (0.5 * rho_att * label_gate).clamp(0.0, 1.0),
        ).detach()

        u_soft, top_vals, top_idx, top_weights = self._soft_topm_ui(x_prime)

        # Project UI direction away from the positive class center.
        u_bar = u_soft - (u_soft * w_y).sum(dim=1, keepdim=True) * w_y
        u_norm = u_bar.norm(dim=1, keepdim=True)
        valid_orth = (u_norm.squeeze(1) > self.eps).float()
        u_perp = F.normalize(u_bar, dim=1)

        cos_ui_orth = (x_prime * u_perp).sum(dim=1).clamp(-1.0 + self.eps, 1.0 - self.eps)
        cos_ui_soft = (x_prime * u_soft).sum(dim=1).clamp(-1.0 + self.eps, 1.0 - self.eps)
        ui_orth_raw = F.relu(cos_ui_orth - self.ui_margin).pow(2) * valid_orth
        ui_orth_loss = (rho_ui * ui_orth_raw).mean()
        anchor_loss = (
            rho_att
            * F.relu(C.detach().float() - C_prime.float() - self.delta_c).pow(2)
        ).mean()
        neg_loss = (
            rho_neg
            * F.relu(N_prime.float() - N.detach().float() - self.delta_n).pow(2)
        ).mean()
        preserve_loss = (
            (1.0 - rho_att)
            * (x_prime - x_base.detach()).pow(2).sum(dim=1)
        ).mean()
        attention_loss = context.get("attention_loss")
        if attention_loss is None:
            attention_loss = logits.new_zeros(())

        extra = (
            self.ri_lambda * ri_loss
            + self.ui_lambda * ui_orth_loss
            + self.anchor_lambda * anchor_loss
            + self.neg_lambda * neg_loss
            + self.preserve_lambda * preserve_loss
            + self.attention_lambda * attention_loss.to(dtype=logits.dtype)
        )
        self._last_extra_loss = extra

        sample_weight_valid = (
            self.ui_sample_weight_min
            + (1.0 - self.ui_sample_weight_min) * (1.0 - omega_unrec)
        ).clamp(self.ui_sample_weight_min, 1.0)
        sample_weight = torch.ones(labels.view(-1).shape[0], device=rows.device, dtype=rows.dtype)
        sample_weight[index] = sample_weight_valid.to(dtype=rows.dtype)
        self._last_sample_weight = sample_weight.detach()

        rows.scatter_(1, target.view(-1, 1), u_pos.view(-1, 1))
        logits[index] = rows

        with torch.no_grad():
            delta_c = C_prime.detach().float() - C.detach().float()
            delta_n = N_prime.detach().float() - N.detach().float()
            delta_u = U_prime.detach().float() - U.detach().float()
            embedding_shift = context.get("embedding_shift")
            if embedding_shift is None:
                embedding_shift = (x_prime - x_base.detach()).norm(dim=1)
            else:
                embedding_shift = embedding_shift[index]
            hard_i = torch.sigmoid(self.ui_alpha * (N.detach().float() - C.detach().float()))
            ui_like_mask = low_ri_true > 0.5
            hard_identifiable_mask = (low_ri_true <= 0.5) & (C.detach().float() > N.detach().float())
            dangerous_mask = ui_like_mask & (C.detach().float() <= N.detach().float())
            self.last_stats = {
                "q_mean": float(q.detach().float().mean().item()),
                "q_factor_mean": float(q_factor.detach().float().mean().item()),
                "d_mean": float(d_i.detach().float().mean().item()),
                "gate_lambda_i_mean": float(gate_lambda_i.detach().float().mean().item()),
                "hard_negative_ratio": float(hard_negative_ratio.item()),
                "curricular_t": float(self.t.detach().item()),
                "cos_ui_multi_mean": float(cos_ui_soft.detach().float().mean().item()),
                "cos_ui_soft_mean": float(cos_ui_soft.detach().float().mean().item()),
                "cos_ui_orth_mean": float(cos_ui_orth.detach().float().mean().item()),
                "d_ui_multi_mean": float(d_ui.detach().float().mean().item()),
                "ri_multi_mean": float(ri_target.detach().float().mean().item()),
                "ri_multi_min": float(ri_target.detach().float().min().item()),
                "ri_multi_max": float(ri_target.detach().float().max().item()),
                "ri_loss": float(ri_loss.detach().float().item()),
                "hard_i_mean": float(hard_i.detach().float().mean().item()),
                "ui_like_i_mean": float(low_ri_true.detach().float().mean().item()),
                "label_gate_mean": float(label_gate.detach().float().mean().item()),
                "omega_unrec_mean": float(omega_unrec.detach().float().mean().item()),
                "rho_att_mean": float(rho_att.detach().float().mean().item()),
                "rho_ui_mean": float(rho_ui.detach().float().mean().item()),
                "rho_neg_mean": float(rho_neg.detach().float().mean().item()),
                "ui_loss_mean": float(ui_orth_raw.detach().float().mean().item()),
                "ui_orth_loss": float(ui_orth_loss.detach().float().item()),
                "anchor_loss": float(anchor_loss.detach().float().item()),
                "negative_guard_loss": float(neg_loss.detach().float().item()),
                "preserve_loss": float(preserve_loss.detach().float().item()),
                "attention_loss": float(attention_loss.detach().float().item()),
                "ui_extra_loss": float(extra.detach().float().item()),
                "delta_c_mean": float(delta_c.detach().float().mean().item()),
                "delta_n_mean": float(delta_n.detach().float().mean().item()),
                "delta_u_mean": float(delta_u.detach().float().mean().item()),
                "embedding_shift_mean": float(embedding_shift.detach().float().mean().item()),
                "sample_weight_mean": float(sample_weight.detach().float().mean().item()),
                "sample_weight_min": float(sample_weight.detach().float().min().item()),
                "sample_weight_max": float(sample_weight.detach().float().max().item()),
                "hard_identifiable_ratio": float(hard_identifiable_mask.float().mean().item()),
                "ui_like_ratio": float(ui_like_mask.float().mean().item()),
                "dangerous_ratio": float(dangerous_mask.float().mean().item()),
                "topm_weight_max_mean": float(top_weights.max(dim=1).values.detach().float().mean().item()),
            }

        return logits * self.s


class Proposed43CoreIdentitySafeLoss(Proposed43FullIdentitySafeLoss):
    """Core 4.3++ objective: RI + attention + preserve + anchor.

    Core intentionally leaves UI-orthogonal and negative-guard disabled.  The
    forward implementation is shared with the Full loss so diagnostics remain
    comparable, but the corresponding weights default to zero.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ui_lambda", 0.0)
        kwargs.setdefault("neg_lambda", 0.0)
        super().__init__(*args, **kwargs)
