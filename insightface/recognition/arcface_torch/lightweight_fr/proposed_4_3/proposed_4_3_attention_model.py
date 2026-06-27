"""Runtime model pieces for Proposed 4.3++ Core/Full.

This module wraps an existing backbone without changing the backbone's public
``forward(x)`` contract elsewhere.  The wrapper is used only by the Core/Full
train/eval scripts that need true feature-map attention:

    F, z, x -> rho_att -> F' -> z', x'

The most recent forward context is kept in a module-level slot so the patched
margin head/loss can compute anchor, preserve, RI and guard terms against the
same batch.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .perceptibility_attention import PerceptibilityAttentionModule, RecoverabilityPredictor


_LAST_ATTENTION_CONTEXT: Optional[Dict[str, torch.Tensor]] = None


def set_last_attention_context(context: Optional[Dict[str, torch.Tensor]]) -> None:
    global _LAST_ATTENTION_CONTEXT
    _LAST_ATTENTION_CONTEXT = context


def get_last_attention_context() -> Optional[Dict[str, torch.Tensor]]:
    return _LAST_ATTENTION_CONTEXT


def infer_feature_channels(backbone: nn.Module, image_size: int = 112) -> int:
    if not hasattr(backbone, "forward_features"):
        raise RuntimeError(
            "Backbone must expose forward_features(x) for Proposed 4.3++ attention."
        )
    was_training = backbone.training
    old_fp16 = getattr(backbone, "fp16", None)
    if old_fp16 is not None:
        backbone.fp16 = False
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, image_size, image_size)
        features = backbone.forward_features(dummy)
    if was_training:
        backbone.train()
    if old_fp16 is not None:
        backbone.fp16 = old_fp16
    if features.ndim != 4:
        raise RuntimeError(f"forward_features returned {tuple(features.shape)}, expected [B,C,H,W].")
    return int(features.shape[1])


class Proposed43AttentionBackbone(nn.Module):
    """Backbone wrapper that outputs the post-attention embedding ``z'``."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: int,
        embedding_dim: int = 512,
        attention_reduction: int = 16,
        attention_alpha: float = 0.25,
        centered_attention: bool = False,
        gate_alpha: float = 0.5,
        gate_beta: float = 1.0,
        gate_eps: float = 1e-3,
        ri_hidden_dim: int = 128,
        attention_spatial_lambda: float = 1e-4,
        attention_channel_lambda: float = 1e-4,
        attention_tv_lambda: float = 1e-4,
    ):
        super().__init__()
        if not hasattr(backbone, "forward_with_features") or not hasattr(backbone, "forward_from_features"):
            raise RuntimeError(
                "Backbone must expose forward_with_features(x) and forward_from_features(F)."
            )
        self.backbone = backbone
        self.attention = PerceptibilityAttentionModule(
            in_channels=feature_channels,
            embedding_dim=embedding_dim,
            reduction=attention_reduction,
        )
        self.ri_predictor = RecoverabilityPredictor(
            in_channels=feature_channels,
            hidden_dim=ri_hidden_dim,
        )
        self.embedding_dim = int(embedding_dim)
        self.attention_alpha = float(attention_alpha)
        self.centered_attention = bool(centered_attention)
        self.gate_alpha = float(gate_alpha)
        self.gate_beta = float(gate_beta)
        self.gate_eps = float(gate_eps)
        self.attention_spatial_lambda = float(attention_spatial_lambda)
        self.attention_channel_lambda = float(attention_channel_lambda)
        self.attention_tv_lambda = float(attention_tv_lambda)

        self.register_buffer("quality_mean", torch.ones(1) * 20.0)
        self.register_buffer("quality_std", torch.ones(1) * 10.0)

    def _quality_gate(self, norms: torch.Tensor) -> torch.Tensor:
        values = norms.detach().float().view(-1)
        if self.training and values.numel() > 1:
            with torch.no_grad():
                mean = values.mean()
                std = values.std(unbiased=False).clamp_min(self.gate_eps)
                self.quality_mean.mul_(0.99).add_(0.01 * mean)
                self.quality_std.mul_(0.99).add_(0.01 * std)
        z = (values - self.quality_mean.to(values.device)) / (
            self.quality_std.to(values.device).clamp_min(self.gate_eps)
        )
        return torch.sigmoid(z)

    def _attention_gate(self, embeddings: torch.Tensor, features: torch.Tensor):
        norms = embeddings.detach().float().norm(dim=1)
        q_gate = self._quality_gate(norms)
        ri_pred = self.ri_predictor(features.detach())
        low_ri = torch.sigmoid(-ri_pred.detach().float())
        omega_unrec = (low_ri * (1.0 - q_gate)).clamp(0.0, 1.0).detach()
        rho_att = (
            (low_ri + self.gate_eps).pow(self.gate_alpha)
            * (q_gate + self.gate_eps).pow(1.0 - self.gate_alpha)
            * (1.0 - omega_unrec).clamp_min(0.0).pow(self.gate_beta)
        ).clamp(0.0, 1.0).detach()
        return ri_pred, q_gate.detach(), low_ri.detach(), omega_unrec, rho_att

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_from_features(features)

    def forward_with_features(self, x: torch.Tensor):
        embedding = self.forward(x)
        context = get_last_attention_context()
        features = context["features_prime"] if context is not None else None
        return embedding, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_embedding, features = self.backbone.forward_with_features(x)
        ri_pred, q_gate, low_ri, omega_unrec, rho_att = self._attention_gate(
            base_embedding,
            features,
        )
        features_prime, maps = self.attention.apply_attention(
            features,
            rho_att=rho_att,
            alpha=self.attention_alpha,
            centered=self.centered_attention,
        )
        embedding_prime = self.backbone.forward_from_features(features_prime)
        attention_loss, attention_stats = self.attention.regularization(
            maps,
            lambda_spatial=self.attention_spatial_lambda,
            lambda_channel=self.attention_channel_lambda,
            lambda_tv=self.attention_tv_lambda,
        )

        x_base = F.normalize(base_embedding.float(), dim=1)
        x_prime = F.normalize(embedding_prime.float(), dim=1)
        context = {
            "base_embeddings": base_embedding,
            "base_norms": base_embedding.float().norm(dim=1, keepdim=True),
            "base_x": x_base,
            "prime_embeddings": embedding_prime,
            "prime_x": x_prime,
            "features": features,
            "features_prime": features_prime,
            "ri_pred": ri_pred,
            "quality_gate": q_gate,
            "low_ri_pred": low_ri,
            "omega_unrec": omega_unrec,
            "rho_att": rho_att,
            "attention_loss": attention_loss,
            "embedding_shift": (x_prime - x_base.detach()).norm(dim=1),
        }
        for key, value in attention_stats.items():
            context[key] = value
        set_last_attention_context(context)
        return embedding_prime

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if state_dict and not any(str(key).startswith("backbone.") for key in state_dict.keys()):
            remapped = {}
            full_prefixes = (
                "attention.",
                "ri_predictor.",
                "quality_mean",
                "quality_std",
            )
            for key, value in state_dict.items():
                key = str(key)
                if key.startswith(full_prefixes):
                    remapped[key] = value
                else:
                    remapped[f"backbone.{key}"] = value
            state_dict = remapped
            strict = False
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)
