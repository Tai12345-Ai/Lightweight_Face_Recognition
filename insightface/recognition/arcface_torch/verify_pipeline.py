"""
End-to-end verification script for lightweight FR pipeline.

Runs all 3 backbones with ArcFace on synthetic data to verify:
1. Backbone forward pass
2. Loss computation
3. Gradient flow
4. Training loop (mini training)
5. Feature extraction
6. Benchmark comparison
7. Degradation transforms

This does NOT require a real dataset or GPU.
"""

import os
import sys
import time
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbones import get_model
from losses import CombinedMarginLoss
from degradation.transforms import DegradationTransform, SUPPORTED_DEGRADATIONS


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# =====================================================================
# TEST 1: Backbone forward pass
# =====================================================================
def test_backbones():
    separator("TEST 1: Backbone Forward Pass")
    networks = {
        'mbf': 'MobileFaceNet',
        'shufflefacenet': 'ShuffleFaceNet',
        'vargfacenet': 'VarGFaceNet',
    }
    results = {}
    x = torch.randn(4, 3, 112, 112)

    for key, name in networks.items():
        model = get_model(key, fp16=False, num_features=512)
        model.eval()
        with torch.no_grad():
            out = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        results[key] = {
            'name': name,
            'output_shape': list(out.shape),
            'params_m': params,
        }
        print(f"  ✓ {name:<20} output={list(out.shape)}  params={params:.2f}M")

    # Verify all outputs are correct shape
    for key, r in results.items():
        assert r['output_shape'] == [4, 512], f"FAIL: {key} shape {r['output_shape']}"
    print("  ✓ All backbone shapes correct: [batch, 512]")
    return results


# =====================================================================
# TEST 2: Loss computation + gradient flow
# =====================================================================
def test_loss_and_gradients():
    separator("TEST 2: Loss Computation & Gradient Flow")
    networks = ['mbf', 'shufflefacenet', 'vargfacenet']

    num_classes = 100
    embedding_size = 512
    batch_size = 8

    x = torch.randn(batch_size, 3, 112, 112)
    labels = torch.randint(0, num_classes, (batch_size,))

    margin_loss = CombinedMarginLoss(64.0, 1.0, 0.5, 0.0, 0)
    classifier_weight = nn.Parameter(
        torch.randn(num_classes, embedding_size))
    nn.init.kaiming_uniform_(classifier_weight)

    for net_name in networks:
        backbone = get_model(net_name, fp16=False, num_features=512)
        backbone.train()

        # Forward
        embeddings = backbone(x)
        norm_emb = nn.functional.normalize(embeddings)
        norm_w = nn.functional.normalize(classifier_weight)
        logits = nn.functional.linear(norm_emb, norm_w)
        logits = logits.clamp(-1, 1)

        # Make labels compatible with CombinedMarginLoss format
        labels_col = labels.view(-1, 1)
        modified_logits = margin_loss(logits, labels_col)

        # Cross entropy
        loss = nn.functional.cross_entropy(modified_logits, labels)

        # Backward
        loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in backbone.parameters())
        print(f"  ✓ {net_name:<20} loss={loss.item():.4f}  "
              f"gradients={'OK' if has_grad else 'MISSING'}")
        assert has_grad, f"FAIL: No gradients for {net_name}"

    print("  ✓ All backbones: loss computed, gradients flow correctly")


# =====================================================================
# TEST 3: Mini training loop (5 steps)
# =====================================================================
def test_mini_training():
    separator("TEST 3: Mini Training Loop (5 steps)")
    networks = ['mbf', 'shufflefacenet', 'vargfacenet']

    num_classes = 50
    embedding_size = 512
    batch_size = 8
    num_steps = 5

    # Synthetic dataset
    images = torch.randn(batch_size * 2, 3, 112, 112)
    labels = torch.randint(0, num_classes, (batch_size * 2,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for net_name in networks:
        backbone = get_model(net_name, fp16=False, num_features=512)
        backbone.train()

        classifier = nn.Linear(embedding_size, num_classes, bias=False)
        margin_loss = CombinedMarginLoss(64.0, 1.0, 0.5, 0.0, 0)

        optimizer = torch.optim.SGD(
            list(backbone.parameters()) + list(classifier.parameters()),
            lr=0.01, momentum=0.9, weight_decay=5e-4)

        losses = []
        step = 0
        for epoch in range(3):
            for img, lbl in loader:
                if step >= num_steps:
                    break
                emb = backbone(img)
                norm_emb = nn.functional.normalize(emb)
                norm_w = nn.functional.normalize(classifier.weight)
                logits = nn.functional.linear(norm_emb, norm_w).clamp(-1, 1)
                modified_logits = margin_loss(logits, lbl.view(-1, 1))
                loss = nn.functional.cross_entropy(modified_logits, lbl)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                step += 1
            if step >= num_steps:
                break

        trend = "↓" if losses[-1] < losses[0] else "→"
        print(f"  ✓ {net_name:<20} "
              f"loss: {losses[0]:.3f} → {losses[-1]:.3f} {trend}  "
              f"({num_steps} steps)")

    print("  ✓ All backbones: training loop runs without errors")


# =====================================================================
# TEST 4: Feature extraction + cosine similarity
# =====================================================================
def test_feature_extraction():
    separator("TEST 4: Feature Extraction & Cosine Similarity")
    networks = ['mbf', 'shufflefacenet', 'vargfacenet']

    batch_size = 8
    images_a = torch.randn(batch_size, 3, 112, 112)
    images_b = torch.randn(batch_size, 3, 112, 112)

    for net_name in networks:
        backbone = get_model(net_name, fp16=False, num_features=512)
        backbone.eval()

        with torch.no_grad():
            emb_a = nn.functional.normalize(backbone(images_a))
            emb_b = nn.functional.normalize(backbone(images_b))
            # Same image should have high similarity
            emb_a_dup = nn.functional.normalize(backbone(images_a))
            sim_same = (emb_a * emb_a_dup).sum(dim=1).mean().item()
            sim_diff = (emb_a * emb_b).sum(dim=1).mean().item()

        print(f"  ✓ {net_name:<20} "
              f"same_input_sim={sim_same:.4f}  "
              f"diff_input_sim={sim_diff:.4f}")
        assert abs(sim_same - 1.0) < 0.01, \
            f"FAIL: Same input should give sim~1.0, got {sim_same}"

    print("  ✓ All backbones: deterministic embeddings, correct similarity")


# =====================================================================
# TEST 5: CPU inference timing
# =====================================================================
def test_cpu_timing():
    separator("TEST 5: CPU Inference Timing")
    networks = ['mbf', 'shufflefacenet', 'vargfacenet']
    x = torch.randn(1, 3, 112, 112)

    for net_name in networks:
        model = get_model(net_name, fp16=False, num_features=512)
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)

        # Timed
        times = []
        with torch.no_grad():
            for _ in range(20):
                t0 = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        mean_ms = np.mean(times)
        std_ms = np.std(times)
        print(f"  ✓ {net_name:<20} {mean_ms:.1f} ms ± {std_ms:.1f} ms")

    print("  ✓ CPU timing complete")


# =====================================================================
# TEST 6: Degradation transforms
# =====================================================================
def test_degradation():
    separator("TEST 6: Degradation Transforms")
    # Create a fake face image (112x112x3)
    img = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)

    for deg_name in SUPPORTED_DEGRADATIONS:
        for severity in [1, 3, 5]:
            deg = DegradationTransform(deg_name, severity=severity, seed=42)
            result = deg.apply(img)
            assert result.shape == (112, 112, 3), \
                f"FAIL: {deg_name} s={severity} shape={result.shape}"
            assert result.dtype == np.uint8, \
                f"FAIL: {deg_name} s={severity} dtype={result.dtype}"
        print(f"  ✓ {deg_name:<25} severities [1,3,5] OK")

    # Test reproducibility
    deg1 = DegradationTransform("gaussian_blur", severity=3, seed=42)
    deg2 = DegradationTransform("gaussian_blur", severity=3, seed=42)
    r1 = deg1.apply(img)
    r2 = deg2.apply(img)
    assert np.array_equal(r1, r2), "FAIL: Not reproducible with same seed"
    print("  ✓ Reproducibility with same seed: OK")


# =====================================================================
# TEST 7: Degradation → embedding pipeline
# =====================================================================
def test_degraded_embedding():
    separator("TEST 7: Degradation → Embedding Pipeline")
    backbone = get_model('mbf', fp16=False, num_features=512)
    backbone.eval()

    # Create synthetic face images
    batch_np = np.random.randint(50, 200, (4, 112, 112, 3), dtype=np.uint8)

    # Clean embeddings
    clean_tensor = torch.from_numpy(
        batch_np.transpose(0, 3, 1, 2).astype(np.float32))
    clean_tensor = ((clean_tensor / 255.0) - 0.5) / 0.5
    with torch.no_grad():
        clean_emb = nn.functional.normalize(backbone(clean_tensor))

    # Degraded embeddings
    deg = DegradationTransform("gaussian_blur", severity=3, seed=42)
    degraded_np = np.stack([deg.apply(img) for img in batch_np])
    deg_tensor = torch.from_numpy(
        degraded_np.transpose(0, 3, 1, 2).astype(np.float32))
    deg_tensor = ((deg_tensor / 255.0) - 0.5) / 0.5
    with torch.no_grad():
        deg_emb = nn.functional.normalize(backbone(deg_tensor))

    # Cosine similarity between clean and degraded (same image)
    sim = (clean_emb * deg_emb).sum(dim=1)
    print(f"  Clean↔Degraded cosine similarity: {sim.mean().item():.4f}")
    print(f"  Range: [{sim.min().item():.4f}, {sim.max().item():.4f}]")
    print("  ✓ Full pipeline: image → degradation → embedding → similarity OK")


# =====================================================================
# SUMMARY
# =====================================================================
def print_summary(backbone_results):
    separator("SUMMARY: Lightweight Backbone Comparison (ArcFace)")

    print(f"\n  {'Network':<20} {'Params(M)':>10} {'Output':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*12}")
    for key, r in backbone_results.items():
        print(f"  {r['name']:<20} {r['params_m']:>10.2f} "
              f"{str(r['output_shape']):>12}")

    print(f"""
  All tests passed:
  ✓ 3 lightweight backbones verified (forward pass, output shape)
  ✓ ArcFace loss computation + gradient flow
  ✓ Mini training loop (SGD, 5 steps)
  ✓ Feature extraction + cosine similarity
  ✓ CPU inference timing
  ✓ 6 degradation transforms × 3 severities
  ✓ End-to-end: degradation → embedding → similarity

  Next steps:
  1. Download dataset (MS1MV3 or CASIA-WebFace)
  2. Update config.rec path in configs/lightweight_fr/
  3. Run: python train_lightweight.py configs/lightweight_fr/mbf_arcface.py
  4. Run: python eval_degraded.py --network mbf --weight model.pt ...
""")


if __name__ == "__main__":
    backbone_results = test_backbones()
    test_loss_and_gradients()
    test_mini_training()
    test_feature_extraction()
    test_cpu_timing()
    test_degradation()
    test_degraded_embedding()
    print_summary(backbone_results)
