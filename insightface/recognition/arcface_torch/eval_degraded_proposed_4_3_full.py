#!/usr/bin/env python3
"""Six-degradation eval for Proposed 4.3++ Full checkpoints.

This mirrors eval_degraded_6phase2.py but loads the Full wrapper
(backbone + attention + RI predictor) and therefore emits post-attention
embeddings x' at inference time.
"""

from pathlib import Path

import torch

import eval_degraded_6phase2 as base
from backbones import get_model
from proposed_4_3_attention_model import Proposed43AttentionBackbone, infer_feature_channels


def _checkpoint_config(checkpoint):
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config")
        if isinstance(config, dict):
            return config
    return {}


def load_full_backbone(args, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    raw_backbone = get_model(
        args.backbone,
        dropout=0.0,
        fp16=use_fp16,
        num_features=base.EMBEDDING_SIZE,
    )
    checkpoint = base.torch_load_cpu(checkpoint_path)
    config = _checkpoint_config(checkpoint)
    feature_channels = infer_feature_channels(raw_backbone, image_size=112)
    model = Proposed43AttentionBackbone(
        raw_backbone,
        feature_channels=feature_channels,
        embedding_dim=base.EMBEDDING_SIZE,
        attention_reduction=int(config.get("attention_reduction", 16)),
        attention_alpha=float(config.get("attention_alpha", 0.25)),
        centered_attention=bool(config.get("centered_attention", True)),
        attention_spatial_lambda=float(config.get("attention_spatial_lambda", 1e-4)),
        attention_channel_lambda=float(config.get("attention_channel_lambda", 1e-4)),
        attention_tv_lambda=float(config.get("attention_tv_lambda", 1e-4)),
    )
    if isinstance(checkpoint, dict) and "state_dict_backbone" in checkpoint:
        state_dict = checkpoint["state_dict_backbone"]
    else:
        state_dict = base.extract_backbone_state(checkpoint)
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(
            "WARNING: missing Full model keys "
            f"({len(result.missing_keys)}): {result.missing_keys[:10]}"
        )
    if result.unexpected_keys:
        print(
            "WARNING: unexpected Full checkpoint keys "
            f"({len(result.unexpected_keys)}): {result.unexpected_keys[:10]}"
        )
    return model.to(device).eval()


base.load_backbone = load_full_backbone


if __name__ == "__main__":
    base.main()
