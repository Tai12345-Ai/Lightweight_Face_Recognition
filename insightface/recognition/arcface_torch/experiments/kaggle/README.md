# Kaggle Experiment Runners

Thư mục này chứa các runner phục vụ tái lập thí nghiệm trên Kaggle. Runner sẽ
clone/pull repo, dò dataset trong `/kaggle/input`, chạy train/eval và đóng gói
kết quả nếu cấu hình tương ứng hỗ trợ.

## Nhóm File

```text
experiments/kaggle/
|-- kaggle_5eval_degraded_common.py        # Helper chung cho protocol 5-eval
|-- kaggle_*_5eval_degraded_runner.py      # Runner baseline/loss/proposed
|-- kaggle_proposed_4_3_core_report.py     # Xuất report/plot cho Proposed 4.3
|-- README_PHASE2.md
|-- README_PROPOSED_4_3_CORE.md
|-- README_PROPOSED_4_3_CORE_FULL.md
`-- README_PROPOSED_4_3_FULL_FROM_CORE.md
```

## Cách Chạy

Khi đang ở `insightface/recognition/arcface_torch`:

```bash
python experiments/kaggle/kaggle_arcface_5eval_degraded_runner.py
python experiments/kaggle/kaggle_adaface_5eval_degraded_runner.py
python experiments/kaggle/kaggle_curricularface_5eval_degraded_runner.py
python experiments/kaggle/kaggle_proposed_4_3_core_5eval_degraded_runner.py
python experiments/kaggle/kaggle_proposed_4_3_full_from_core_runner.py
```

Trong Kaggle notebook có thể dùng `%run` với đường dẫn file tương ứng sau khi
repo đã tồn tại trong `/kaggle/working`.

## Input Tối Thiểu

- Dataset train có `train.rec` và `train.idx`.
- Eval bins: `lfw.bin`, `cfp_fp.bin`, `cplfw.bin`, `agedb_30.bin`, `calfw.bin`.
- Pretrained backbone, thường là `backbone.pth`.
- Với Full từ Core, nên cung cấp checkpoint Core `best.pth` hoặc backup zip Core.

Các shim `kaggle_5eval_degraded_common.py` và
`kaggle_proposed_4_3_core_report.py` vẫn được giữ ở `arcface_torch` để notebook
cũ import theo tên file root vẫn chạy được.

