# Phase 2: Loss Function Comparison with Fixed ResNet18 Backbone

## Goal

Create a complete Google Colab notebook (`colab_phase2_loss_comparison.py`) that fine-tunes a pretrained ResNet18 (iResNet18) backbone from InsightFace with **6 different loss functions** under identical conditions, then evaluates clean + degraded accuracy to produce a fair comparison.

## Background

- **Phase 1** compared lightweight *backbones* (MobileFaceNet, ShuffleFaceNet, VarGFaceNet) with ArcFace only.
- **Phase 2** fixes the backbone (ResNet18) and compares *loss functions*: ArcFace, CosFace, CurricularFace, ElasticFace, AdaFace, MagFace.
- Pretrained ResNet18 checkpoint comes from InsightFace model zoo (OneDrive: `ms1mv3_arcface_r18_fp16/backbone.pth`).
- No Phase 1 results are reused — each loss starts from the same pretrained checkpoint.

## User Review Required

> [!IMPORTANT]
> **Pretrained checkpoint source**: The OneDrive folder `ms1mv3_arcface_r18_fp16` (568 MB) should contain a `backbone.pth` file. You need to download this and upload it to Google Drive at the path specified by `PRETRAINED_CKPT`. If the checkpoint file inside has a different name (e.g., `model.pt`), adjust `PRETRAINED_CKPT` accordingly.

> [!IMPORTANT]
> **Dataset**: The notebook defaults to CASIA-WebFace (`faces_webface_112x112`, 10572 classes, ~490K images). If you want to use MS1MV3 or another dataset, update `DATASET_DIR`, `NUM_CLASSES`, and `NUM_IMAGE` in Cell 1.

> [!WARNING]
> **CurricularFace and ElasticFace**: These are currently stubs (`NotImplementedError`) in `losses_extended.py`. This plan includes **full implementations** for both. CurricularFace uses curriculum learning with a running negative cosine estimate; ElasticFace uses random margin sampling from a normal distribution.

## Open Questions

> [!IMPORTANT]
> 1. **Which dataset do you want for Phase 2?** The notebook defaults to CASIA-WebFace (same as Phase 1). If you want MS1MV3, please confirm — this affects `num_classes` and `num_image`.
> 2. **The pretrained R18 from OneDrive was trained on MS1MV3 with ArcFace**. If Phase 2 uses CASIA-WebFace, fine-tuning from MS1MV3 pretrained to CASIA means a domain shift. Is this intentional? (It's valid for the experiment since all losses share the same starting point.)
> 3. **MagFace regularization term**: The original MagFace paper has a magnitude regularization loss `g(a_i)` added to the total loss with weight `lambda_g`. Should I include this as a separate term in the training loss, or just the margin adaptation? (Plan includes both, with `lambda_g=20` default.)

## Proposed Changes

### New Notebook File

#### [NEW] [colab_phase2_loss_comparison.py](file:///d:/projects/Project-2/insightface/recognition/arcface_torch/colab_phase2_loss_comparison.py)

A single `.py` file using `# %%` cell markers (convertible to `.ipynb` via `convert_to_ipynb.py`). Contains 14 cells:

| Cell | Purpose |
|------|---------|
| 0 | Install dependencies (numpy, easydict, ptflops, mxnet, scikit-learn, opencv) |
| 1 | Mount Google Drive + global config (all paths, hyperparams, LOSS_LIST) |
| 2 | Check GPU (nvidia-smi, CUDA check, VRAM) |
| 3 | Clone/setup InsightFace repo, verify `r18` backbone exists |
| 4 | Dataset check (train.rec/train.idx exist, num_classes from property) |
| 5 | Load & verify pretrained R18 backbone (load state_dict, dummy forward, verify raw embedding + norm) |
| 6 | Implement all 6 loss functions (full implementations, no placeholders) |
| 7 | Create per-loss config files |
| 8 | Training loop for all losses (resume/skip, backup per epoch) |
| 9 | Clean evaluation on verification sets |
| 10 | Degraded evaluation (blur, resolution, illumination × severity 1,3,5) |
| 11 | Benchmark model efficiency (params, size, latency) |
| 12 | Final backup to Google Drive |
| 13 | How to run / Troubleshooting guide |

---

### Loss Function Implementations (Cell 6)

All losses will be written into a single file `phase2_losses.py` created by the notebook at runtime. Each loss follows the unified interface: `forward(logits, labels, embeddings=None, norms=None)`.

#### 1. ArcFace
- Uses existing `CombinedMarginLoss` with `margin_list=(1.0, 0.5, 0.0)`, `s=64`
- Wrapped in `CombinedMarginLossWrapper` for unified interface

#### 2. CosFace
- Uses existing `CombinedMarginLoss` with `margin_list=(1.0, 0.0, 0.4)`, `s=64`
- Angular margin = 0, additive cosine margin = 0.4

#### 3. CurricularFace (NEW — full implementation)
- `s=64`, `m=0.5`
- Maintains running estimate `t` of average negative cosine (EMA)
- Hard negative mining: negative logits weighted by `t * cos(θ_j)` when `cos(θ_j) > cos(θ_yi + m)`
- `t` updated via EMA: `t = α*mean(cos(θ_target)) + (1-α)*t` with α=0.99

#### 4. ElasticFace (NEW — full implementation)
- `s=64`, `m_mean=0.5`, `m_std=0.0125`
- Each forward: sample margin `m ~ N(m_mean, m_std)` per batch
- ElasticArc variant: perturbed angular margin `θ + m_sampled`
- At eval time: use fixed `m_mean`

#### 5. AdaFace (existing in `losses_extended.py`, verified working)
- `s=64`, `m=0.4`, `h=0.333`
- Uses feature norms as image quality proxy
- EMA for batch norm statistics
- Quality-adaptive angular + additive margins

#### 6. MagFace (existing in `losses_extended.py` + enhancement)
- `s=64`, `l_a=10`, `u_a=110`, `l_m=0.45`, `u_m=0.8`, `lambda_g=20`
- Linear margin interpolation based on feature magnitude
- **Add**: magnitude regularization term `g(a_i) = 1/(u_a - l_a) * (a_i - l_a/u_a)` weighted by `lambda_g`
- Returns `(logits, mag_reg_loss)` or adds `mag_reg_loss` to training loss

---

### Training Script (Cell 8)

A self-contained training function `train_one_loss()` embedded in the notebook:

```python
def train_one_loss(loss_name, loss_fn, cfg, pretrained_ckpt, output_dir, ...):
    # 1. Create backbone, load pretrained state_dict (backbone ONLY)
    # 2. Create new PartialFC_V2 / PartialFC_V2_Extended head (random init)
    # 3. Create optimizer, scheduler
    # 4. Training loop with logging
    # 5. Save checkpoint per epoch, backup to Drive
    # 6. Return final model path
```

Key design decisions:
- **No DDP**: Single-GPU Colab → use `torch.distributed` with world_size=1, GLOO backend
- **PartialFC_V2_Extended** for AdaFace/MagFace (passes norms), standard `PartialFC_V2` for others
- **Backbone forward modification**: For AdaFace/MagFace, the `PartialFC_V2_Extended.forward()` computes norms from raw embeddings before normalization — this is already implemented in `train_lightweight.py`
- **MagFace regularization**: Added as separate term in loss computation within training loop

---

### Backbone Forward for AdaFace/MagFace

The existing `IResNet.forward()` returns the output *after* the final BN layer (`self.features`), which is **not yet L2-normalized** — that happens inside `PartialFC_V2.forward()` via `normalize(embeddings)`. So:

- `PartialFC_V2_Extended` (already in `train_lightweight.py`) computes `norms = torch.norm(embeddings, dim=1, keepdim=True)` BEFORE calling `normalize(embeddings)`
- This gives us the raw feature norm needed by AdaFace/MagFace
- **No backbone modification needed** — the existing flow already works

---

### Drive Backup Structure

```
/content/drive/MyDrive/phase2_resnet18_loss_comparison/
  pretrained/                    # User uploads R18 checkpoint here
  models/
    arcface/model.pt
    cosface/model.pt
    curricularface/model.pt
    elasticface/model.pt
    adaface/model.pt
    magface/model.pt
  checkpoints/
    arcface/checkpoint_gpu_0.pt
    cosface/checkpoint_gpu_0.pt
    ...
  train_logs/
    arcface.log
    cosface.log
    ...
  eval_logs/
    clean_results.json
  degraded_eval_logs/
    degraded_results.json
  benchmark_logs/
    benchmark_results.json
  configs/
    phase2_losses.py
  final_backup/                  # Tar of everything
```

---

### Resume/Skip Logic

```python
for loss_name in LOSS_LIST:
    model_path = f"{DRIVE_BACKUP}/models/{loss_name}/model.pt"
    ckpt_path = f"{DRIVE_BACKUP}/checkpoints/{loss_name}/checkpoint_gpu_0.pt"
    
    if os.path.exists(model_path) and not FORCE_RETRAIN:
        print(f"[SKIP] {loss_name} — already trained")
        continue
    
    if os.path.exists(ckpt_path) and not FORCE_RETRAIN:
        # Resume from checkpoint
        resume_from = ckpt_path
    else:
        # Fresh start from pretrained
        resume_from = None
    
    train_one_loss(loss_name, ..., resume_from=resume_from)
```

---

### Evaluation (Cells 9-10)

Reuses existing `eval_degraded.py` logic but embedded in notebook:
- Clean eval: LFW, CFP-FP, AgeDB-30
- Degraded eval: gaussian_blur, low_resolution, low_illumination × severity 1,3,5
- Produces comparison tables with robustness drop

---

### Benchmark (Cell 11)

Since all losses use the same backbone (R18), benchmark only runs once:
- Params(M), Size(MB)
- GPU latency batch=1 and batch=16
- Note: inference architecture identical across losses

---

## Verification Plan

### Automated Tests (in notebook)
1. **Cell 5**: Dummy forward pass verifying output shape and feature norm extraction
2. **Cell 6**: Each loss instantiated and tested with dummy logits/labels/norms
3. **Cell 8**: Training produces decreasing loss values (sanity check)
4. **Cell 9-10**: Eval produces accuracy values in expected range

### Manual Verification
- User reviews training logs for convergence
- User checks backup files exist on Google Drive
- User compares accuracy tables across losses
