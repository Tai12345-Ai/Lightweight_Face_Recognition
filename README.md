# Lightweight Face Recognition

Đồ án nghiên cứu nhận dạng khuôn mặt nhẹ, tập trung vào khả năng chịu nhiễu
trên ảnh chất lượng thấp. Repository này phát triển trên nền InsightFace, phần
chính của đồ án nằm trong `insightface/recognition/arcface_torch`.

## Mục Tiêu

- So sánh các backbone nhẹ: MobileFaceNet, ShuffleFaceNet, VarGFaceNet và R18.
- Đánh giá độ bền vững trên ảnh bị suy giảm: blur, low resolution, JPEG,
  low illumination và alignment perturbation.
- Thử nghiệm các loss: ArcFace, AdaFace, MagFace, CurricularFace và các hướng
  Proposed 4.x.
- Chuẩn hóa pipeline train/eval để chạy lại trên local, Colab hoặc Kaggle.

## Cấu Trúc Chính

```text
.
|-- README.md
|-- docs/
|   |-- PROGRESS.md
|   `-- scratch_notebook_md.md
`-- insightface/
    |-- requirements.txt
    `-- recognition/arcface_torch/
        |-- configs/lightweight_fr/       # Config train cho đồ án
        |-- degradation/                  # Biến đổi suy giảm ảnh
        |-- docs/lightweight_fr/          # Tài liệu thiết kế và kế hoạch
        |-- experiments/kaggle/           # Runner Kaggle và protocol
        |-- lightweight_fr/proposed_4_3/  # Implementation Proposed 4.3 Core/Full
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

Một số file shim vẫn được giữ ở `arcface_torch` để các script cũ kiểu
`from soft_gated_losses import ...` hoặc runner Kaggle cũ không bị gãy import.
Implementation thật đã được gom vào các thư mục con ở trên.

## Cài Đặt

Khuyến nghị dùng Python 3.10+ và môi trường ảo riêng.

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install tensorboard easydict mxnet onnx opencv-python scikit-learn pandas matplotlib
python -m pip install -r insightface/requirements.txt
```

Nếu chạy CPU hoặc CUDA khác, thay lệnh cài `torch` theo môi trường thực tế.

## Dữ Liệu

Repo không commit dataset, checkpoint hoặc artifact lớn. Các file này đã được
ignore bằng `.gitignore`.

Các script hiện hỗ trợ dữ liệu train dạng MXNet RecordIO:

```text
train.rec
train.idx
property        # nếu có
```

Thư mục eval cần các file `.bin`, ví dụ:

```text
lfw.bin
cfp_fp.bin
cplfw.bin
agedb_30.bin
calfw.bin
```

Với config local trong `configs/lightweight_fr`, cần chỉnh `config.rec` trỏ tới
đường dẫn dataset trên máy của bạn.

## Chạy Nhanh Local

Từ thư mục `arcface_torch`:

```bash
cd insightface/recognition/arcface_torch
python train_lightweight.py configs/lightweight_fr/mbf_arcface.py
python train_lightweight.py configs/lightweight_fr/shuffle_arcface.py
python train_lightweight.py configs/lightweight_fr/vargface_arcface.py
```

Đánh giá ảnh suy giảm:

```bash
python eval_degraded.py ^
  --network mbf ^
  --weight work_dirs/mbf_arcface/model.pt ^
  --rec D:\path\to\eval ^
  --targets lfw,cfp_fp,agedb_30 ^
  --degradations gaussian_blur,low_resolution,jpeg_compression,low_illumination ^
  --severities 1,3,5
```

Benchmark model:

```bash
python benchmark_model.py --networks mbf,shufflefacenet,vargfacenet
```

## Kaggle / Proposed 4.x

Các runner Kaggle đã được gom vào:

```text
insightface/recognition/arcface_torch/experiments/kaggle/
```

Ví dụ khi đang ở `insightface/recognition/arcface_torch`:

```bash
python experiments/kaggle/kaggle_arcface_5eval_degraded_runner.py
python experiments/kaggle/kaggle_adaface_5eval_degraded_runner.py
python experiments/kaggle/kaggle_proposed_4_3_core_5eval_degraded_runner.py
python experiments/kaggle/kaggle_proposed_4_3_full_from_core_runner.py
```

Tài liệu protocol chi tiết nằm trong `experiments/kaggle/README.md` và các file
`README_*.md` cùng thư mục.

## Kiểm Tra

Smoke test nhẹ cho Proposed 4.3:

```bash
cd insightface/recognition/arcface_torch
python tests/test_proposed_4_3_smoke.py
```

Kiểm tra cú pháp các entry point chính:

```bash
python -m py_compile ^
  train_lightweight.py ^
  train_phase2_kaggle.py ^
  train_soft_gated_lambda_kaggle.py ^
  train_proposed_4_3_core_kaggle.py ^
  train_proposed_4_3_full_kaggle.py ^
  eval_degraded.py ^
  eval_degraded_phase2.py ^
  eval_degraded_proposed_4_3_full.py
```

## Ghi Chú Triển Khai

- Không hard-code secret, token hoặc đường dẫn dataset cá nhân vào code.
- Không push checkpoint, dataset, `.bin`, `.rec`, `.idx`, `.pth`, `.pt`, `.onnx`
  hoặc file backup zip.
- Các kết quả thí nghiệm nên ghi vào `docs/PROGRESS.md` hoặc báo cáo riêng,
  kèm config, dataset, seed, epoch và metric.
- Vì repo dựa trên InsightFace, tránh refactor sâu các file gốc nếu không cần;
  phần đồ án nên nằm trong `configs/lightweight_fr`, `degradation`,
  `lightweight_fr`, `experiments` và `notebooks`.

