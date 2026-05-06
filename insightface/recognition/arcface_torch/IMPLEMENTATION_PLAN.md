# Lightweight Low-Quality Face Recognition — Kế hoạch triển khai

---

## 1. MỤC TIÊU CHUNG

Tạo một nhánh thí nghiệm trên InsightFace với các mục tiêu:

1. So sánh backbone nhẹ cho face recognition
2. Đánh giá độ bền của các backbone này dưới ảnh degraded / low-quality
3. Giữ cố định face detection và face alignment
4. Chỉ tập trung vào face representation / recognition stage
5. Giữ thay đổi ở mức tối thiểu, có tính mô-đun, dễ bảo trì
6. Không refactor toàn bộ repository nếu không cần

---

## 2. PHẠM VI NGHIÊN CỨU

**Pipeline tổng quát:**

```
Face Detection → Face Alignment → Face Representation / Recognition → Matching / Verification
```

**Trong project này:**
- Face detection: **giữ nguyên**
- Face alignment: **giữ nguyên**
- Verification protocol: **giữ nguyên** (repo đã có)
- **Chỉ sửa phần Recognition / Embedding stage**

**Cụ thể phần nghiên cứu:**
- Backbone nhẹ
- Loss / training strategy cho recognition
- Degraded-image evaluation
- Benchmark hiệu quả mô hình
- Chuẩn bị chỗ mở rộng cho ARoFace / CR-FIQA / PETALface

---

## 3. AUDIT REPOSITORY

### Tổng quan

InsightFace là toolbox phân tích khuôn mặt 2D/3D. Khu vực làm việc chính: `recognition/arcface_torch/`.

### Các component đã audit trong `recognition/arcface_torch/`

| Component | File | Phát hiện |
|-----------|------|-----------|
| **Training entrypoint** | `train_v2.py` | Bắt buộc distributed NCCL, hardcode `CombinedMarginLoss`, chọn backbone qua `cfg.network` |
| **Backbone registry** | `backbones/__init__.py` | If-elif chain. Hỗ trợ: `r18/34/50/100/200`, `mbf`, `mbf_large`, `vit_*`. **Chưa có** ShuffleNet, VarGFaceNet |
| **MobileFaceNet** | `backbones/mobilefacenet.py` | ✅ Đã có sẵn. Input 112×112, output embedding (512-d). Có GDC head |
| **IResNet** | `backbones/iresnet.py` | Backbone lớn, fp16 support, BN1d output head |
| **Loss** | `losses.py` | `CombinedMarginLoss(s,m1,m2,m3)`, `ArcFace`, `CosFace`. Chỉ nhận `(logits, labels)`. **Chưa có** AdaFace, MagFace |
| **Training head** | `partial_fc_v2.py` | `PartialFC_V2`: weight + margin_softmax + dist cross entropy. Gọi `self.margin_softmax(logits, labels)`. **Require** `distributed.is_initialized()` |
| **Dataset** | `dataset.py` | `MXFaceDataset` (MXNet RecordIO), `ImageFolder`, DALI. Normalization: `(x/255 - 0.5) / 0.5` |
| **Verification** | `eval/verification.py` | `load_bin()` load .bin verification sets. `test()` extract embeddings → k-fold accuracy. Dùng MXNet IO decode |
| **Config system** | `utils/utils_config.py` | `importlib.import_module("configs.%s")` — chỉ hỗ trợ flat `configs/` namespace |
| **FLOPs** | `flops.py` | Dùng `ptflops`. Chỉ in FLOPs + params, chưa có CPU timing / model size |
| **Callbacks** | `utils/utils_callbacks.py` | `CallBackVerification` (validation trong training), `CallBackLogging` |

### Các module NGOÀI `arcface_torch/` — hoàn toàn tách biệt

| Module | Vai trò | Giữ nguyên? |
|--------|---------|-------------|
| `detection/` (retinaface, scrfd) | Face detector | ✅ Hoàn toàn tách biệt |
| `alignment/` (heatmap, coordinate_reg) | Face alignment | ✅ Hoàn toàn tách biệt |
| `recognition/arcface_mxnet/`, `partial_fc/`, `vpl/`, `subcenter_arcface/`, `idmmd/` | Các variant FR | ✅ Không liên quan |
| `generation/`, `reconstruction/`, `parsing/` | Face gen/recon/parsing | ✅ Không liên quan |
| `python-package/`, `examples/`, `web-demos/`, `challenges/` | SDK, demos | ✅ Không liên quan |

---

## 4. NHỮNG THỨ GIỮ NGUYÊN

| File / Thư mục | Lý do |
|-----------------|-------|
| `detection/` (toàn bộ) | Detector cố định |
| `alignment/` (toàn bộ) | Aligner cố định |
| `recognition/arcface_mxnet/`, `partial_fc/`, `vpl/`, `subcenter_arcface/`, `idmmd/` | Không liên quan |
| `generation/`, `reconstruction/`, `parsing/` | Không liên quan |
| `python-package/`, `examples/`, `web-demos/`, `challenges/` | Không liên quan |
| `train_v2.py` | Script training gốc — giữ tương thích ngược |
| `partial_fc_v2.py` | Training head gốc — không sửa, tạo wrapper riêng |
| `dataset.py` | Dataset loader — reuse nguyên bản |
| `losses.py` | Loss gốc — giữ nguyên, thêm loss mới ở file riêng |
| `lr_scheduler.py` | LR scheduler — reuse |
| `eval/verification.py` | Verification logic — reuse |
| `inference.py`, `torch2onnx.py`, `flops.py` | Utility — giữ nguyên |
| Tất cả config cũ trong `configs/` | Config gốc — giữ nguyên |

---

## 5. FILE CẦN SỬA

| File | Thay đổi | Rủi ro |
|------|----------|--------|
| `backbones/__init__.py` | Thêm 2 elif block (`shufflefacenet`, `vargfacenet`) trước `else: raise ValueError()` | **Rất thấp** — chỉ thêm elif mới, không sửa logic cũ |

> **Chỉ có DUY NHẤT 1 file gốc cần sửa.**

### Về `partial_fc_v2.py` — KHÔNG sửa trực tiếp

`PartialFC_V2` hiện gọi `self.margin_softmax(logits, labels)`. Với AdaFace, cần thêm `embedding` và `norms`.

**Giải pháp**: Tạo `PartialFC_V2_Extended` kế thừa `PartialFC_V2`, override `forward()` để truyền `norms` vào loss. File gốc không bị sửa.

---

## 6. FILE CẦN THÊM MỚI

```
recognition/arcface_torch/
│
├── backbones/
│   ├── shufflefacenet.py              [NEW — Phase 1]
│   └── vargfacenet.py                 [NEW — Phase 1]
│
├── configs/lightweight_fr/            [NEW — Phase 1]
│   ├── __init__.py
│   ├── base_lightweight.py            [base config cho project]
│   ├── mbf_arcface.py                 [MobileFaceNet + ArcFace]
│   ├── shuffle_arcface.py             [ShuffleFaceNet + ArcFace]
│   ├── vargface_arcface.py            [VarGFaceNet + ArcFace]
│   └── mbf_adaface.py                 [Phase 6, MobileFaceNet + AdaFace]
│
├── train_lightweight.py               [NEW — Phase 2]
│
├── eval_degraded.py                   [NEW — Phase 4]
│
├── degradation/                       [NEW — Phase 4]
│   ├── __init__.py
│   └── transforms.py
│
├── benchmark_model.py                 [NEW — Phase 5]
│
├── losses_extended.py                 [NEW — Phase 6]
│
├── extensions/                        [NEW — Phase 7]
│   ├── __init__.py
│   ├── aroface.py                     [stub]
│   ├── crfiqa.py                      [stub]
│   └── petalface.py                   [stub]
│
└── README_lightweight_fr.md           [NEW]
```

---

## 7. PHẠM VI BACKBONE

### Core backbone

| # | Backbone | Config key | ~Params | Trạng thái |
|---|----------|-----------|---------|-----------|
| 1 | MobileFaceNet | `mbf` | ~1M | ✅ Có sẵn trong repo |
| 2 | ShuffleFaceNet (ShuffleNetV2-style) | `shufflefacenet` | ~2.3M | ➕ Thêm mới |
| 3 | VarGFaceNet | `vargfacenet` | ~5M | ➕ Thêm mới |

### Optional backbone (Phase 7+)

| # | Backbone | Trạng thái |
|---|----------|-----------|
| 4 | PocketNet-style | Chưa triển khai |
| 5 | EdgeFace-style | Chưa triển khai |

### Yêu cầu so sánh công bằng
- Cùng input size (3 × 112 × 112)
- Cùng preprocessing (normalization `(x/255 - 0.5) / 0.5`)
- Cùng training head (`PartialFC_V2`)
- Cùng evaluation protocol (LFW, CFP-FP, AgeDB-30)

---

## 8. PHẠM VI LOSS / TRAINING

### Pha ưu tiên

| Pha | Loss | Trạng thái | Cần norms? |
|-----|------|-----------|------------|
| 1 | ArcFace (`combined_margin`) | ✅ Baseline — chạy đầu tiên | Không |
| 2 | AdaFace (`adaface`) | ✅ Đã triển khai | **Có** — dùng feature norm |
| 3 | MagFace (`magface`) | ✅ Đã triển khai | **Có** — dùng feature magnitude |
| 3 | ElasticFace | Stub (Phase 7) | — |
| 3 | CurricularFace | Stub (Phase 7) | — |

### Interface loss thống nhất

```python
forward(logits, labels, embeddings=None, norms=None)
```

- `CombinedMarginLoss` (ArcFace gốc): giữ interface cũ `(logits, labels)`
- `CombinedMarginLossWrapper`: bọc loss cũ, tương thích interface mở
- `AdaFaceLoss`: dùng `norms` để tính adaptive margin
- `MagFaceLoss`: dùng `norms` để tính magnitude-adaptive margin

### Cách truyền norms vào loss

```
PartialFC_V2 (gốc) ──── margin_softmax(logits, labels) ──── CombinedMarginLoss
                                                              (ArcFace, không cần norms)

PartialFC_V2_Extended ── margin_softmax(logits, labels,  ──── AdaFaceLoss / MagFaceLoss
                           embeddings=..., norms=...)          (cần norms)
```

---

## 9. ĐÁNH GIÁ ẢNH CHẤT LƯỢNG THẤP

### Degradation bắt buộc

| # | Loại | Severity 1 | Severity 3 | Severity 5 |
|---|------|-----------|-----------|-----------|
| 1 | Gaussian blur | σ=0.5 | σ=2.0 | σ=5.0 |
| 2 | Motion blur | kernel=3 | kernel=9 | kernel=15 |
| 3 | Low resolution | 56→112 | 28→112 | 14→112 |
| 4 | JPEG compression | q=75 | q=30 | q=10 |
| 5 | Low illumination | γ=1.3 | γ=2.0 | γ=3.5 |

### Degradation optional

| # | Loại | Ghi chú |
|---|------|---------|
| 6 | Alignment perturbation | Synthetic perturbation **sau** alignment. Không thay aligner. |

### Yêu cầu
- Degradation cấu hình được (type + severity)
- Tái lập được bằng random seed
- Giữ nguyên identity pairs / verification protocol
- Đầu ra: clean score, degraded score, performance drop, bảng tổng hợp

---

## 10. HỖ TRỢ PHẦN CỨNG

| Yêu cầu | Chi tiết |
|----------|----------|
| Training | Single-GPU friendly (GLOO fallback cho Windows) |
| Evaluation | CPU-friendly |
| Benchmark | CPU-friendly |
| Distributed | Không cần tối ưu cho multi-node |

---

## 11. NHÁNH MỞ RỘNG TÙY CHỌN

| Extension | Loại | File stub | Trạng thái |
|-----------|------|-----------|-----------|
| ARoFace | Training/augmentation strategy | `extensions/aroface.py` | Stub |
| CR-FIQA | Post-embedding quality assessment | `extensions/crfiqa.py` | Stub |
| PETALface | Low-resolution adaptation | `extensions/petalface.py` | Stub |

Các extension này **không phải backbone**. Mỗi stub chứa:
- Class definition + NotImplementedError
- Docstring mô tả phương pháp
- Hướng dẫn tích hợp vào training/evaluation

---

## 12. KẾ HOẠCH TRIỂN KHAI

### Tóm tắt thứ tự

| Thứ tự | Phase | Nội dung chính | File chính | Trạng thái |
|--------|-------|----------------|------------|-----------|
| 1 | Backbone | ShuffleFaceNet + VarGFaceNet + registry + configs | `shufflefacenet.py`, `vargfacenet.py`, `__init__.py`, `configs/lightweight_fr/` | ✅ Done |
| 2 | Training baseline | Script training single-GPU, ArcFace default | `train_lightweight.py` | ✅ Done |
| 3 | Clean eval | Dùng CallBackVerification / `eval_degraded --degradations none` | (không file mới) | ✅ Done |
| 4 | Degraded eval | Degradation transforms + evaluation pipeline | `degradation/transforms.py`, `eval_degraded.py` | ✅ Done |
| 5 | Benchmark | Param count, FLOPs, model size, CPU timing | `benchmark_model.py` | ✅ Done |
| 6 | AdaFace | Loss interface mở + PartialFC_V2_Extended + AdaFace impl | `losses_extended.py` | ✅ Done |
| 7 | Optional | MagFace + stubs + ARoFace/CR-FIQA/PETALface + README | `extensions/`, `README_lightweight_fr.md` | ✅ Done |

---

### Phase 1 — Backbone Baseline

**Mục tiêu**: Thêm 2 backbone mới, đăng ký vào registry, tạo config.

| File | Mô tả |
|------|-------|
| `backbones/shufflefacenet.py` | ShuffleNetV2 channel shuffle blocks + GDC head. Input 3×112×112 → 512-d. ~2.3M params |
| `backbones/vargfacenet.py` | Variable group conv + SE modules + GDC head. Input 3×112×112 → 512-d. ~5M params |
| `backbones/__init__.py` | +2 elif blocks: `"shufflefacenet"`, `"vargfacenet"` |
| `configs/lightweight_fr/base_lightweight.py` | Base config: batch_size=64, embedding_size=512, loss_type=combined_margin |
| `configs/lightweight_fr/mbf_arcface.py` | MobileFaceNet + ArcFace |
| `configs/lightweight_fr/shuffle_arcface.py` | ShuffleFaceNet + ArcFace |
| `configs/lightweight_fr/vargface_arcface.py` | VarGFaceNet + ArcFace |

---

### Phase 2 — Training Baseline

**Mục tiêu**: Script training single-GPU friendly, tương thích với flow gốc.

| Tính năng | Chi tiết |
|-----------|----------|
| Distributed | Auto-detect, GLOO fallback cho single-GPU / Windows |
| Backbone | Chọn qua `cfg.network` (reuse `get_model()`) |
| Loss | `cfg.loss_type`: `"combined_margin"` (ArcFace mặc định) |
| Training head | `PartialFC_V2` gốc (ArcFace). `PartialFC_V2_Extended` (AdaFace) |
| Dataset | Reuse `get_dataloader()` từ `dataset.py` |
| Validation | Reuse `CallBackVerification` |
| Config loader | Custom loader hỗ trợ `configs/lightweight_fr/` subdirectory |

**Cách chạy:**
```bash
python train_lightweight.py configs/lightweight_fr/mbf_arcface.py
```

---

### Phase 3 — Clean Evaluation

Không cần file mới. Dùng `CallBackVerification` tích hợp trong training loop (LFW/CFP-FP/AgeDB-30).

Standalone: `python eval_degraded.py --network mbf --weight model.pt --rec /path/to/data`

---

### Phase 4 — Degraded Evaluation

**Mục tiêu**: Đánh giá recognition dưới ảnh degraded, cùng protocol.

| File | Mô tả |
|------|-------|
| `degradation/transforms.py` | 6 loại degradation × 5 severity. Deterministic seed. |
| `eval_degraded.py` | Load .bin → apply degradation → extract embeddings → report accuracy + drop |

**Cách chạy:**
```bash
python eval_degraded.py \
    --network mbf --weight model.pt --rec /path/to/data \
    --degradations gaussian_blur,low_resolution,jpeg_compression \
    --severities 1,3,5 --seed 42
```

**Output mẫu:**
```
--- LFW ---
  Condition                        Accuracy       Drop
  Clean                             99.48%        ---
  gaussian_blur_s1                  99.30%      +0.18%
  gaussian_blur_s3                  97.85%      +1.63%
  low_resolution_s5                 89.30%     +10.18%
```

---

### Phase 5 — Benchmark Efficiency

| File | Mô tả |
|------|-------|
| `benchmark_model.py` | Params, model size (MB), FLOPs (GFLOPs), CPU inference time (ms) |

**Cách chạy:**
```bash
python benchmark_model.py --networks mbf,shufflefacenet,vargfacenet
```

**Output mẫu:**
```
  COMPARISON TABLE
  Network              Params(M)   Size(MB)     GFLOPs   CPU b1(ms)   CPU b16(ms)
  mbf                       0.99        3.8      0.440         12.3         85.7
  shufflefacenet            2.30        8.8      0.297         15.1        102.3
  vargfacenet               4.98       19.1      1.022         28.5        195.6
```

---

### Phase 6 — Adaptive Loss (AdaFace)

**Mục tiêu**: Thêm AdaFace, interface mở cho future losses.

| File | Mô tả |
|------|-------|
| `losses_extended.py` | `AdaFaceLoss`, `MagFaceLoss`, `CombinedMarginLossWrapper`, stubs |
| `configs/lightweight_fr/mbf_adaface.py` | MobileFaceNet + AdaFace config |

**Interface:** `forward(logits, labels, embeddings=None, norms=None)`

- ArcFace: bỏ qua embeddings/norms
- AdaFace: dùng norms (feature norm = quality indicator)
- MagFace: dùng norms (magnitude-adaptive margin)

---

### Phase 7 — Optional Extensions

| File | Mô tả |
|------|-------|
| `extensions/aroface.py` | Stub + integration guide — alignment robustness |
| `extensions/crfiqa.py` | Stub + integration guide — quality assessment |
| `extensions/petalface.py` | Stub + integration guide — LR adaptation |
| `README_lightweight_fr.md` | Project README |

---

## 13. RỦI RO KỸ THUẬT / TƯƠNG THÍCH

| # | Rủi ro | Mức độ | Giải pháp |
|---|--------|--------|-----------|
| 1 | `train_v2.py` hardcode NCCL | Cao | `train_lightweight.py` auto-detect NCCL/GLOO |
| 2 | `PartialFC_V2` require distributed | Cao | Init GLOO process group cho single-GPU |
| 3 | `utils_config.py` chỉ hỗ trợ flat `configs/` | Trung bình | Config loader riêng hỗ trợ subdirectory |
| 4 | `CombinedMarginLoss` giữ interface cũ `(logits, labels)`, AdaFace cần interface mở `(logits, labels, embeddings, norms)` | Trung bình | `CombinedMarginLossWrapper` bọc loss cũ + `PartialFC_V2_Extended` truyền norms |
| 5 | `verification.py` dùng MXNet IO | Thấp | Giữ nguyên — MXNet vẫn cần cho `.bin` |
| 6 | `eval_degraded.py` cần modify data từ `.bin` | Thấp | Apply degradation sau decode, trước embedding |
| 7 | Windows không hỗ trợ NCCL | Thấp | GLOO fallback tự động |
| 8 | Script cũ có bị hỏng? | Rất thấp | Chỉ sửa `__init__.py` (thêm elif) |

### Tương thích ngược

```
train_v2.py + configs gốc ────── GIỮ NGUYÊN ────── Chạy như cũ
train_lightweight.py + configs/lightweight_fr/ ────── FILE MỚI ────── Chạy riêng
backbones/__init__.py ────── THÊM elif ────── Cả 2 script đều dùng được
```

---

## 14. ĐIỀU KIỆN TƯƠNG THÍCH NGƯỢC

- ✅ Không làm hỏng script cũ
- ✅ Không thay đổi hành vi cũ ngoài phạm vi project
- ✅ Chỉ sửa 1 file gốc (thêm elif, không sửa logic)
- ✅ Loss interface mở không ép loss cũ phải thay đổi
- ✅ Mọi code mới là file/directory riêng

---

## 15. QUY TẮC LÀM VIỆC ĐÃ TUÂN THỦ

- [x] Audit repo trước khi code
- [x] Không giả định tên file/thư mục — đã kiểm tra thực tế
- [x] Thích nghi với cấu trúc repo thực tế
- [x] Ưu tiên tái sử dụng code gốc (dataset, lr_scheduler, verification, callbacks)
- [x] Chỉ thêm abstraction khi cần (PartialFC_V2_Extended, CombinedMarginLossWrapper)
- [x] Comment vừa đủ
- [x] Baseline chạy trước (ArcFace), adaptive loss sau (AdaFace)

---

## 16. VERIFICATION PLAN

| Phase | Kiểm tra | Lệnh |
|-------|----------|------|
| 1 | Backbone output shape đúng | `python benchmark_model.py --network shufflefacenet` |
| 2 | Training loop chạy được | `python train_lightweight.py configs/lightweight_fr/mbf_arcface.py` |
| 3 | Clean eval hoạt động | Xem output CallBackVerification trong training log |
| 4 | Degraded eval có bảng kết quả | `python eval_degraded.py --degradations gaussian_blur --severities 1,3,5 ...` |
| 5 | Benchmark so sánh được | `python benchmark_model.py --networks mbf,shufflefacenet,vargfacenet` |
| 6 | AdaFace loss giảm | `python train_lightweight.py configs/lightweight_fr/mbf_adaface.py` |

> ⚠️ Cần cài PyTorch + dependencies trước khi chạy kiểm tra.
