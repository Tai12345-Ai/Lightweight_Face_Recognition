# Kaggle Phase 2: ResNet18 + ArcFace-family Losses

Phase 2 fixes the backbone to `r18` and compares margin losses. The script loads
only a pretrained ResNet18 backbone checkpoint, usually `backbone.pth`, then
creates a fresh classification head for each loss.

Recommended checkpoint workflow on Kaggle: upload `backbone.pth` as a Kaggle
Dataset and pass its path to `--pretrained-backbone`. Do not use
`rank_x_softmax_weight.pt` unless you intentionally want to restore an old
classifier head.

## 1. Clone and enter arcface_torch

```bash
git clone https://github.com/Tai12345-Ai/Lightweight_Face_Recognition.git
cd Lightweight_Face_Recognition/insightface/recognition/arcface_torch
pip install -r requirement.txt
```

If your dataset is MXNet RecordIO (`train.rec`, `train.idx`), make sure `mxnet`
is installed in the Kaggle environment. ImageFolder datasets also work:

```text
dataset_root/
  person_0001/*.jpg
  person_0002/*.jpg
```

## 2. Paths used below

Adjust these to your Kaggle inputs.

```bash
export DATA_DIR=/kaggle/input/your-face-dataset/faces_webface_112x112
export PRETRAINED=/kaggle/input/your-r18-backbone/backbone.pth
export OUT=/kaggle/working/outputs
```

Outputs are written to:

```text
$OUT/phase2_loss/r18_<loss>/
  latest.pt
  best.pth
  train_log.csv
  metrics.json
  config.json
```

## 3. Train ArcFace

```bash
python train_phase2_kaggle.py \
  --loss arcface \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 5 \
  --batch-size 64 \
  --lr 0.01 \
  --fp16 \
  --eval-every 1 \
  --save-every 1
```

## 4. Train CosFace

```bash
python train_phase2_kaggle.py \
  --loss cosface \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 5 \
  --batch-size 64 \
  --lr 0.01 \
  --fp16
```

## 5. Train ElasticFace

```bash
python train_phase2_kaggle.py \
  --loss elasticface \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 5 \
  --batch-size 64 \
  --lr 0.01 \
  --fp16
```

## 6. Train CurricularFace

```bash
python train_phase2_kaggle.py \
  --loss curricularface \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 5 \
  --batch-size 64 \
  --lr 0.01 \
  --fp16
```

## 7. Train AdaFace

```bash
python train_phase2_kaggle.py \
  --loss adaface \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 5 \
  --batch-size 64 \
  --lr 0.01 \
  --fp16
```

## 8. Train MagFace

```bash
python train_phase2_kaggle.py \
  --loss magface \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 5 \
  --batch-size 64 \
  --lr 0.01 \
  --fp16
```

MagFace includes the magnitude regularization term in addition to the
classification loss.

## 9. Train the proposed loss

`proposed` currently maps to `CurriculumAwareAdaFaceLoss`: AdaFace target margin
plus CurricularFace-style hard negative modulation.

```bash
python train_phase2_kaggle.py \
  --loss proposed \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 5 \
  --batch-size 64 \
  --lr 0.01 \
  --fp16
```

To train only the new head:

```bash
python train_phase2_kaggle.py \
  --loss adaface \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 5 \
  --batch-size 64 \
  --lr 0.01 \
  --fp16 \
  --freeze-backbone
```

## 10. Resume after Kaggle interruption

`--resume` automatically loads:

```text
$OUT/phase2_loss/r18_<loss>/latest.pt
```

Example:

```bash
python train_phase2_kaggle.py \
  --loss adaface \
  --backbone r18 \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 10 \
  --batch-size 64 \
  --lr 0.01 \
  --fp16 \
  --resume
```

The script resumes from the next epoch stored in `latest.pt`.

## 11. Backup outputs as zip

```bash
cd /kaggle/working
zip -r phase2_outputs.zip outputs/phase2_loss
```

## 12. OneDrive checkpoint option

The stable Kaggle option is to upload `backbone.pth` as a Kaggle Dataset. If you
want to try downloading from the public OneDrive link, use:

```bash
python scripts/download_onedrive_checkpoint.py \
  --share-url "https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d" \
  --remote-path "ms1mv3_arcface_r18_fp16/backbone.pth" \
  --output "/kaggle/working/backbone.pth"
```

If the shared URL points to a folder and the remote path differs, change
`--remote-path` to the exact path inside the shared folder.

## 13. Add a new loss to the registry

Edit `losses_extended.py`:

1. Add a class with this interface:

```python
class MyLoss(nn.Module):
    requires_norms = False

    def forward(self, logits, labels, embeddings=None, norms=None):
        ...
        return logits
```

2. Add it to `PHASE2_LOSS_REGISTRY`:

```python
PHASE2_LOSS_REGISTRY["myloss"] = LossSpec(
    factory=lambda: MyLoss(...),
    requires_norms=False,
    description="Short description",
)
```

3. Run:

```bash
python train_phase2_kaggle.py --loss myloss ...
```
