# Lightweight FR Project Notes

Tài liệu này mô tả phần mở rộng của đồ án trong
`insightface/recognition/arcface_torch`. README chính của repo nằm ở root.

## Thành Phần Đồ Án

```text
arcface_torch/
|-- configs/lightweight_fr/       # Config cho MobileFaceNet/ShuffleFaceNet/VarGFaceNet
|-- degradation/                  # Synthetic degradation transforms
|-- extensions/                   # Stub nghiên cứu: ARoFace, CR-FIQA, PETALface
|-- lightweight_fr/proposed_4_3/  # Attention, RI/recoverability và loss Proposed 4.3
|-- experiments/kaggle/           # Runner Kaggle và protocol thí nghiệm
|-- notebooks/                    # Notebook/Colab source
|-- tests/                        # Smoke test
|-- train_lightweight.py
|-- train_phase2_kaggle.py
|-- train_soft_gated_lambda_kaggle.py
|-- train_proposed_4_3_core_kaggle.py
|-- train_proposed_4_3_full_kaggle.py
|-- eval_degraded.py
|-- eval_degraded_phase2.py
`-- eval_degraded_proposed_4_3_full.py
```

## Hướng Nghiên Cứu

- Baseline nhẹ: MobileFaceNet, ShuffleFaceNet, VarGFaceNet.
- Loss baseline/phase2: ArcFace, AdaFace, MagFace, CurricularFace.
- Proposed 4.x: soft-gated/adaptive loss, UI-aware, multi-UI attention.
- Proposed 4.3 Core/Full: attention trên feature map, RI/recoverability gate,
  identity-anchor, preserve và negative-guard terms.

## Lệnh Hay Dùng

Train baseline local:

```bash
python train_lightweight.py configs/lightweight_fr/mbf_arcface.py
python train_lightweight.py configs/lightweight_fr/shuffle_arcface.py
python train_lightweight.py configs/lightweight_fr/vargface_arcface.py
```

Eval ảnh suy giảm:

```bash
python eval_degraded.py \
  --network mbf \
  --weight work_dirs/mbf_arcface/model.pt \
  --rec /path/to/eval \
  --targets lfw,cfp_fp,agedb_30 \
  --degradations gaussian_blur,low_resolution,jpeg_compression,low_illumination \
  --severities 1,3,5
```

Smoke test Proposed 4.3:

```bash
python tests/test_proposed_4_3_smoke.py
```

## Tài Liệu Liên Quan

- `IMPLEMENTATION_PLAN.md`: kế hoạch triển khai phase đầu.
- `IMPLEMENTATION_PLAN_PHASE2.md`: kế hoạch phase2/loss comparison.
- `experiments/kaggle/README.md`: cách chạy runner Kaggle.
- `README_LEGACY.md`: bản README cũ được giữ lại để đối chiếu nội dung.
- `docs/PROGRESS.md` ở root: log tiến độ và kết quả thí nghiệm.

