"""Perceptibility attention modules for Proposed 4.3.

The original trainer used this module as an auxiliary embedding head.  The
Core/Full 4.3++ path also needs the attention map itself so it can build
``F' = F * (1 + alpha * rho_att * M)`` before the embedding head.  The old
``forward`` behavior is kept for compatibility, while new methods expose the
map, feature amplification and regularization terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptibilityAttentionModule(nn.Module):
    """Channel–spatial attention followed by a linear projection to embedding space.

    Input:  feature map  F  of shape [B, C, H, W]
    Output: normalized embedding v_attn of shape [B, embedding_dim]
    """

    def __init__(self, in_channels: int, embedding_dim: int = 512, reduction: int = 16):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if reduction <= 0 or in_channels < reduction:
            raise ValueError(
                f"reduction must be positive and <= in_channels, got reduction={reduction}, "
                f"in_channels={in_channels}"
            )

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.reduction = reduction

        # ---------- Channel attention (shared MLP) ----------
        mid = max(1, in_channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, kernel_size=1, bias=True),
        )

        # ---------- Spatial attention ----------
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        # ---------- Embedding projection ----------
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, embedding_dim)

    def attention_maps(self, feature_map: torch.Tensor):
        if feature_map.ndim != 4:
            raise ValueError(
                f"PerceptibilityAttentionModule expects 4D input [B,C,H,W], got {feature_map.shape}"
            )

        orig_dtype = feature_map.dtype

        # Attention layers are float32 by default. During fp16 training/eval,
        # feature_map can be float16, so compute maps in fp32 and cast back.
        F_in = feature_map.float() if feature_map.dtype in (torch.float16, torch.bfloat16) else feature_map

        # --- Channel attention ---
        avg_c = F.adaptive_avg_pool2d(F_in, 1)  # [B, C, 1, 1]
        max_c = F.adaptive_max_pool2d(F_in, 1)  # [B, C, 1, 1]
        A_c = torch.sigmoid(self.channel_mlp(avg_c) + self.channel_mlp(max_c))  # [B, C, 1, 1]
        F_c = A_c * F_in  # [B, C, H, W]

        # --- Spatial attention ---
        avg_s = F_c.mean(dim=1, keepdim=True)          # [B, 1, H, W]
        max_s = F_c.max(dim=1, keepdim=True).values     # [B, 1, H, W]
        A_s = torch.sigmoid(
            self.spatial_conv(torch.cat([avg_s, max_s], dim=1))
        )  # [B, 1, H, W]
        M = A_c * A_s  # [B, C, H, W]

        return {
            "channel": A_c.to(dtype=orig_dtype),
            "spatial": A_s.to(dtype=orig_dtype),
            "combined": M.to(dtype=orig_dtype),
        }

    def apply_attention(
        self,
        feature_map: torch.Tensor,
        rho_att: torch.Tensor | None = None,
        alpha: float = 1.0,
        centered: bool = False,
    ):
        maps = self.attention_maps(feature_map)
        M = maps["combined"]
        if centered:
            M_eff = M - M.mean(dim=(1, 2, 3), keepdim=True)
        else:
            M_eff = M

        if rho_att is None:
            rho = torch.ones(
                feature_map.shape[0],
                1,
                1,
                1,
                device=feature_map.device,
                dtype=feature_map.dtype,
            )
        else:
            rho = rho_att.to(device=feature_map.device, dtype=feature_map.dtype).view(-1, 1, 1, 1)

        scale = 1.0 + float(alpha) * rho * M_eff.to(dtype=feature_map.dtype)
        feature_prime = feature_map * scale
        maps["effective"] = M_eff
        maps["scale"] = scale
        return feature_prime, maps

    @staticmethod
    def regularization(
        maps,
        lambda_spatial: float = 0.0,
        lambda_channel: float = 0.0,
        lambda_tv: float = 0.0,
    ):
        spatial = maps["spatial"].float()
        channel = maps["channel"].float()
        l_spatial = spatial.mean()
        l_channel = channel.mean()
        if spatial.shape[-2] > 1:
            tv_h = (spatial[:, :, 1:, :] - spatial[:, :, :-1, :]).abs().mean()
        else:
            tv_h = spatial.new_zeros(())
        if spatial.shape[-1] > 1:
            tv_w = (spatial[:, :, :, 1:] - spatial[:, :, :, :-1]).abs().mean()
        else:
            tv_w = spatial.new_zeros(())
        l_tv = tv_h + tv_w
        total = (
            float(lambda_spatial) * l_spatial
            + float(lambda_channel) * l_channel
            + float(lambda_tv) * l_tv
        )
        stats = {
            "attention_spatial_mean": l_spatial.detach(),
            "attention_channel_mean": l_channel.detach(),
            "attention_tv": l_tv.detach(),
        }
        return total, stats

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W]

        Returns:
            v_attn: [B, embedding_dim], L2-normalized. Kept for the old
            auxiliary MSE branch.
        """
        maps = self.attention_maps(feature_map)
        F_attn = maps["combined"] * feature_map  # [B, C, H, W]

        # --- Embedding projection ---
        pooled = self.gap(F_attn).flatten(1)  # [B, C]
        z_attn = self.fc(pooled.float() if feature_map.dtype != torch.float32 else pooled)
        v_attn = F.normalize(z_attn, dim=1)

        return v_attn


class RecoverabilityPredictor(nn.Module):
    """Predict a scalar RI logit from a backbone feature map."""

    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        hidden_dim = max(1, int(hidden_dim))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4:
            raise ValueError(
                f"RecoverabilityPredictor expects 4D input [B,C,H,W], got {feature_map.shape}"
            )
        pooled = self.gap(feature_map).flatten(1)
        return self.net(pooled.float()).view(-1)
