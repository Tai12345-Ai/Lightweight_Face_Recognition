# Lightweight Face Recognition Robust to Low-Quality Images — Kaggle Report Version

**Project scope**: So sánh lightweight backbone ở **face representation stage**.  
Face detection & alignment được xem là bước tiền xử lý cố định; notebook này tập trung train/evaluate backbone + ArcFace.

**Kaggle version**:
- Chạy mặc định ở `RUN_MODE = "report"`.
- Đọc CASIA-WebFace từ `/kaggle/input/...`.
- Lưu checkpoint/log/output vào `/kaggle/working/...`.
- Không dùng Google Drive / `google.colab`.
---
## Cell 0: Install Dependencies — Kaggle
---
## Cell 1: Global Config + Kaggle Paths + GPU Check
---
## Cell 2: Clone InsightFace + Set Working Directory---
## Cell 3: Create / Patch Backbone Files

- **MobileFaceNet**: Already in repo (`backbones/mobilefacenet.py`).
- **ShuffleFaceNet**: ShuffleNetV2-style face recognition backbone + GDC head. *(Not a faithful reproduction of any specific paper; inspired by ShuffleNetV2 architecture.)*
- **VarGFaceNet**: Simplified VarGFaceNet-style compact backbone with SE modules + GDC head. *(Simplified version; does not implement the full variable-group convolution from the ICCVW 2019 paper.)*---
## Cell 4: Create Configs---
## Cell 5: Create `train_lightweight.py`

Single-GPU T4 compatible. Uses `torchrun --standalone --nproc_per_node=1`.
Reuses `get_dataloader`, `CombinedMarginLoss`, `PartialFC_V2` from repo.---
## Cell 6: Create Degradation Module

**Core degradations** (3 only):
1. `gaussian_blur` — Gaussian blur with variable sigma
2. `low_resolution` — Downsample then upsample back to 112x112
3. `low_illumination` — Gamma correction to darken image---
## Cell 7: Create `eval_degraded.py`---
## Cell 8: Download + Verify CASIA-WebFace Dataset---
## Cell 9: Debug Run — Train MobileFaceNet first (sanity check)

> **Sanity check only**: Verify pipeline works (data loading, loss, training, eval callbacks).
>
> Do NOT use Cell 9 results to conclude which backbone is better.
>
> - When `RUN_MODE="debug"`: train MobileFaceNet into `work_dirs/debug_casia_mbf_arcface`.
> - When `RUN_MODE="report"`: Cell 9 is skipped. Use Cell 10 to train all 3 backbones fairly.---
## Cell 10: Train All 3 Backbones with ArcFace

> **Fair comparison**: All 3 backbones trained from scratch, same dataset, same loss, same epochs,
> under the same `RUN_MODE` namespace (`debug_*` or `report_*`).---
## Cell 11: Clean Evaluation — Compare 3 Backbones---
## Cell 12: Degraded Evaluation — 3 Core Degradations Only

Degradations: `gaussian_blur`, `low_resolution`, `low_illumination`
Severities: 1 (mild), 3 (moderate), 5 (severe)---
## Cell 13: Benchmark Model Efficiency---
## Cell 14: Save Results to Google Drive---
## How to Run This Notebook on Kaggle

### Setup
1. Open Kaggle Notebook.
2. Add CASIA-WebFace dataset containing `train.rec` and `train.idx`.
3. Enable GPU: **Settings → Accelerator → GPU**.
4. Run Cell 0. If `mxnet` import fails, restart the Kaggle session once.
5. Run Cell 1 → Cell 14 in order.

### Default training mode
This notebook is already configured as:

```python
RUN_MODE = "report"
REPORT_EPOCHS = 15
BATCH_SIZE = 64
TRAIN_BACKBONES = ["mbf", "shufflefacenet", "vargfacenet"]
```

If you want faster training, set:

```python
REPORT_EPOCHS = 10
```

but keep the same epoch count for all backbones.

### Outputs
All outputs are saved under:

```text
/kaggle/working/lightweight_fr_results/report/
```

Final zipped file:

```text
/kaggle/working/lightweight_fr_results_report.zip
```

### Evaluation note
If your Kaggle dataset only contains `train.rec/train.idx` and does not contain:

```text
lfw.bin
cfp_fp.bin
agedb_30.bin
```

then leave:

```python
RUN_EVAL = False
```

Train on Kaggle, download `lightweight_fr_results_report.zip`, then evaluate later on Colab or another environment that has the verification `.bin` files.

### Backbone note
- `mbf` is the MobileFaceNet implementation already available in InsightFace.
- `shufflefacenet` and `vargfacenet` in this notebook are lightweight custom/simplified implementations for controlled comparison, not official improved variants.
