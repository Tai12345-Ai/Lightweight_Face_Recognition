"""Perceptibility Attention Module for Proposed 4.3.

Implements CBAM-style channel + spatial attention on the backbone feature map,
producing a normalized auxiliary embedding for training-time UI suppression.

Reference: Woo et al., 2304.10066v1 (CBAM-inspired perceptibility attention).
"""

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

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W]

        Returns:
            v_attn: [B, embedding_dim], L2-normalized
        """
        if feature_map.ndim != 4:
            raise ValueError(
                f"PerceptibilityAttentionModule expects 4D input [B,C,H,W], got {feature_map.shape}"
            )

        F_in = feature_map

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
        F_attn = A_s * F_c  # [B, C, H, W]

        # --- Embedding projection ---
        pooled = self.gap(F_attn).flatten(1)  # [B, C]
        z_attn = self.fc(pooled.float() if feature_map.dtype != torch.float32 else pooled)
        v_attn = F.normalize(z_attn, dim=1)

        return v_attn
