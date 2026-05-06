# Lightweight Face Recognition for Low-Quality Images

Research project branch on InsightFace for comparing lightweight face recognition
backbones under degraded/low-quality image conditions.

## Overview

This project adds a modular experimental pipeline on top of InsightFace's
`recognition/arcface_torch/` module without modifying the core repository.

**Research focus:**
- Compare lightweight backbones (MobileFaceNet, ShuffleFaceNet, VarGFaceNet)
- Evaluate robustness under image degradations (blur, low-res, JPEG, low-light)
- Benchmark model efficiency (params, FLOPs, CPU inference time)
- Support adaptive losses (AdaFace, MagFace)

**Fixed components (not modified):**
- Face detection: RetinaFace / SCRFD (in `detection/`)
- Face alignment: coordinate regression (in `alignment/`)
- Verification protocol: LFW, CFP-FP, AgeDB-30 pair verification

## Project Structure

```
recognition/arcface_torch/
├── backbones/
│   ├── shufflefacenet.py       # [NEW] ShuffleNetV2-style backbone
│   ├── vargfacenet.py          # [NEW] VarGFaceNet backbone
│   └── __init__.py             # [MODIFIED] +2 elif for new backbones
├── configs/lightweight_fr/     # [NEW] Project configs
│   ├── base_lightweight.py     # Base config
│   ├── mbf_arcface.py          # MobileFaceNet + ArcFace
│   ├── shuffle_arcface.py      # ShuffleFaceNet + ArcFace
│   ├── vargface_arcface.py     # VarGFaceNet + ArcFace
│   └── mbf_adaface.py          # MobileFaceNet + AdaFace
├── train_lightweight.py        # [NEW] Single-GPU friendly training
├── eval_degraded.py            # [NEW] Degradation evaluation
├── benchmark_model.py          # [NEW] Model efficiency benchmark
├── losses_extended.py          # [NEW] AdaFace, MagFace losses
├── degradation/                # [NEW] Image degradation transforms
│   └── transforms.py
├── extensions/                 # [NEW] Research extension stubs
│   ├── aroface.py              # ARoFace (alignment robustness)
│   ├── crfiqa.py               # CR-FIQA (quality assessment)
│   └── petalface.py            # PETALface (LR adaptation)
└── README_lightweight_fr.md    # This file
```

## Quick Start

### 1. Training

```bash
# MobileFaceNet + ArcFace baseline
python train_lightweight.py configs/lightweight_fr/mbf_arcface.py

# ShuffleFaceNet + ArcFace
python train_lightweight.py configs/lightweight_fr/shuffle_arcface.py

# VarGFaceNet + ArcFace
python train_lightweight.py configs/lightweight_fr/vargface_arcface.py

# MobileFaceNet + AdaFace (Phase 6)
python train_lightweight.py configs/lightweight_fr/mbf_adaface.py
```

**Note:** Update `config.rec` path in config files to point to your dataset
(e.g., MS1MV3 in MXNet RecordIO format).

### 2. Clean Evaluation

Clean evaluation runs automatically during training via `CallBackVerification`
on LFW, CFP-FP, and AgeDB-30.

For standalone clean evaluation:
```bash
python eval_degraded.py \
    --network mbf \
    --weight work_dirs/mbf_arcface/model.pt \
    --rec /path/to/dataset
```

### 3. Degraded Evaluation

```bash
python eval_degraded.py \
    --network mbf \
    --weight work_dirs/mbf_arcface/model.pt \
    --rec /path/to/dataset \
    --targets lfw,cfp_fp,agedb_30 \
    --degradations gaussian_blur,low_resolution,jpeg_compression,low_illumination \
    --severities 1,3,5 \
    --seed 42
```

Supported degradations:
| Degradation | Description |
|-------------|-------------|
| `gaussian_blur` | Gaussian blur with variable σ |
| `motion_blur` | Directional motion blur |
| `low_resolution` | Downsample → upsample back to 112×112 |
| `jpeg_compression` | JPEG quality degradation |
| `low_illumination` | Gamma-based brightness reduction |
| `alignment_perturb` | Synthetic alignment perturbation (post-alignment) |

### 4. Model Benchmark

```bash
# Compare all lightweight backbones
python benchmark_model.py --networks mbf,shufflefacenet,vargfacenet
```

Output includes: parameter count, model size (MB), FLOPs (GFLOPs),
CPU inference time (ms).

## Supported Backbones

| Backbone | Config key | Approx. params | Source |
|----------|-----------|----------------|--------|
| MobileFaceNet | `mbf` | ~1M | Already in repo |
| ShuffleFaceNet | `shufflefacenet` | ~2.3M | New (ShuffleNetV2-style) |
| VarGFaceNet | `vargfacenet` | ~5M | New (ICCVW 2019) |

## Supported Losses

| Loss | Config `loss_type` | Status | Needs norms? |
|------|-------------------|--------|-------------|
| ArcFace | `combined_margin` | ✅ Working | No |
| AdaFace | `adaface` | ✅ Working | Yes |
| MagFace | `magface` | ✅ Working | Yes |
| ElasticFace | — | Stub | — |
| CurricularFace | — | Stub | — |

## Extensions (Not Yet Implemented)

| Extension | Purpose | File |
|-----------|---------|------|
| ARoFace | Alignment robustness training | `extensions/aroface.py` |
| CR-FIQA | Post-embedding quality assessment | `extensions/crfiqa.py` |
| PETALface | Low-resolution adaptation | `extensions/petalface.py` |

These are stub files with integration documentation. They can be implemented
without modifying the core training/evaluation pipeline.

## Backward Compatibility

- Only 1 original file modified: `backbones/__init__.py` (added 2 elif blocks)
- All original scripts (`train_v2.py`, configs, eval) remain functional
- New code is self-contained in new files/directories
