"""Smoke test for Proposed 4.3 and PerceptibilityAttentionModule."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn.functional as F


def test_proposed_4_3():
    from soft_gated_losses import (
        MultiUIPerceptibilityCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss,
    )

    B, C, EMB = 4, 10, 512
    K = 7

    logits = torch.randn(B, C)
    labels = torch.tensor([0, 1, 2, 3])
    embeddings = F.normalize(torch.randn(B, EMB), dim=1)
    norms = torch.rand(B) * 10 + 10
    centers = F.normalize(torch.randn(K, EMB), dim=1)

    loss_fn = MultiUIPerceptibilityCompetitionQualityAdaptiveSoftGatedAdaCurricularFaceLoss(
        s=64.0,
        m=0.4,
        h=0.333,
        multi_ui_centers=centers,
        ui_center_names=["global", "gb", "mb", "lr", "jc", "li", "ap"],
    )

    scaled_logits = loss_fn(logits, labels, embeddings=embeddings, norms=norms)
    assert scaled_logits.shape == (B, C), f"Shape mismatch: {scaled_logits.shape}"
    print("[PASS] Forward: shape OK")

    assert loss_fn._last_extra_loss is not None, "_last_extra_loss is None!"
    print(f"[PASS] _last_extra_loss = {loss_fn._last_extra_loss.item():.6f}")

    assert loss_fn._last_sample_weight is not None, "_last_sample_weight is None!"
    print(f"[PASS] _last_sample_weight shape = {loss_fn._last_sample_weight.shape}")

    v_prime, nearest_idx = loss_fn.compute_ui_suppressed_targets(embeddings)
    assert v_prime.shape == (B, EMB), f"v_prime shape mismatch: {v_prime.shape}"
    assert nearest_idx.shape == (B,), f"nearest_idx shape mismatch: {nearest_idx.shape}"
    print(f"[PASS] compute_ui_suppressed_targets: v_prime={v_prime.shape}, nearest_idx={nearest_idx.shape}")

    print(f"[PASS] Stats keys: {sorted(loss_fn.last_stats.keys())}")


def test_attention_module():
    from perceptibility_attention import PerceptibilityAttentionModule

    B, C_feat, H, W = 4, 512, 7, 7
    EMB = 512

    feature_map = torch.randn(B, C_feat, H, W)
    module = PerceptibilityAttentionModule(in_channels=C_feat, embedding_dim=EMB, reduction=16)

    v_attn = module(feature_map)
    assert v_attn.shape == (B, EMB), f"Shape mismatch: {v_attn.shape}"

    # Check normalization
    norms = v_attn.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"Not normalized: {norms}"
    print(f"[PASS] PerceptibilityAttentionModule: output={v_attn.shape}, norms OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Smoke Test: Proposed 4.3 Loss")
    print("=" * 60)
    test_proposed_4_3()

    print()
    print("=" * 60)
    print("Smoke Test: PerceptibilityAttentionModule")
    print("=" * 60)
    test_attention_module()

    print()
    print("=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
