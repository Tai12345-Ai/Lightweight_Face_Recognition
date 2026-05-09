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

## Recommended Kaggle free settings

The `5` epoch examples were only short smoke-test settings. For real Phase 2
fine-tuning on Kaggle, run one loss per session and resume across sessions.

Recommended starting point:

```bash
--epochs 20 \
--batch-size 128 \
--lr 0.01 \
--warmup-epochs 1 \
--fp16 \
--eval-every 2 \
--save-every 1 \
--save-every-steps 300 \
--max-train-minutes 600
```

If Kaggle gives a smaller GPU or you hit OOM, reduce `--batch-size` to `64`.
If one epoch is very slow, use `--eval-every 2` or disable validation with
`--val-targets=` and evaluate later. `--max-train-minutes` should be lower than
your notebook session limit so the script can save and exit cleanly.
`--warmup-epochs 1` warms up the random classification head before cosine LR
decay.

Recommended training queue:

```text
arcface -> cosface -> adaface -> curricularface -> elasticface -> magface -> proposed
```

The Colab runner uses this order by default. If quota runs out, it stops at the
current loss and resumes that same loss in the next session.

## 3. Train ArcFace

```bash
python train_phase2_kaggle.py \
  --loss arcface \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 20 \
  --batch-size 128 \
  --lr 0.01 \
  --warmup-epochs 1 \
  --fp16 \
  --eval-every 2 \
  --save-every 1 \
  --save-every-steps 300 \
  --max-train-minutes 600
```

## 4. Train CosFace

```bash
python train_phase2_kaggle.py \
  --loss cosface \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 20 \
  --batch-size 128 \
  --lr 0.01 \
  --warmup-epochs 1 \
  --fp16 \
  --eval-every 2 \
  --save-every-steps 300 \
  --max-train-minutes 600
```

## 5. Train AdaFace

```bash
python train_phase2_kaggle.py \
  --loss adaface \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 20 \
  --batch-size 128 \
  --lr 0.01 \
  --warmup-epochs 1 \
  --fp16 \
  --eval-every 2 \
  --save-every-steps 300 \
  --max-train-minutes 600
```

## 6. Train CurricularFace

```bash
python train_phase2_kaggle.py \
  --loss curricularface \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 20 \
  --batch-size 128 \
  --lr 0.01 \
  --warmup-epochs 1 \
  --fp16 \
  --eval-every 2 \
  --save-every-steps 300 \
  --max-train-minutes 600
```

## 7. Train ElasticFace

```bash
python train_phase2_kaggle.py \
  --loss elasticface \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 20 \
  --batch-size 128 \
  --lr 0.01 \
  --warmup-epochs 1 \
  --fp16 \
  --eval-every 2 \
  --save-every-steps 300 \
  --max-train-minutes 600
```

## 8. Train MagFace

```bash
python train_phase2_kaggle.py \
  --loss magface \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 20 \
  --batch-size 128 \
  --lr 0.01 \
  --warmup-epochs 1 \
  --fp16 \
  --eval-every 2 \
  --save-every-steps 300 \
  --max-train-minutes 600
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
  --epochs 20 \
  --batch-size 128 \
  --lr 0.01 \
  --warmup-epochs 1 \
  --fp16 \
  --eval-every 2 \
  --save-every-steps 300 \
  --max-train-minutes 600
```

To train only the new head:

```bash
python train_phase2_kaggle.py \
  --loss adaface \
  --backbone r18 \
  --pretrained-backbone "$PRETRAINED" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 2 \
  --batch-size 128 \
  --lr 0.01 \
  --fp16 \
  --freeze-backbone
```

## 10. Resume after Kaggle interruption

`--resume` automatically loads:

```text
$OUT/phase2_loss/r18_<loss>/latest.pt
```

`latest.pt` is saved every `--save-every-steps` optimizer steps and at epoch
end. If training stops during epoch 4 at iteration 3500, the next run skips the
already-trained batches and continues inside epoch 4.

Example:

```bash
python train_phase2_kaggle.py \
  --loss adaface \
  --backbone r18 \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --epochs 20 \
  --batch-size 128 \
  --lr 0.01 \
  --warmup-epochs 1 \
  --fp16 \
  --eval-every 2 \
  --save-every-steps 300 \
  --max-train-minutes 600 \
  --resume
```

Use the same `--output-dir` and `--loss`. Increase `--epochs` only if you want
to continue beyond the original target.

`partial_fc_v2.py` is not used by `train_phase2_kaggle.py`; it is kept for the
separate distributed/multi-GPU InsightFace training path.

## 11. Backup outputs as zip

```bash
cd /kaggle/working
zip -r phase2_outputs.zip outputs/phase2_loss
```

## 12. Continue on Colab

Use `colab_phase2_resume_runner.ipynb` when Kaggle quota runs out. The Colab
runner restores `outputs/phase2_loss` from Google Drive, trains the configured
queue in order, and backs up each loss folder after every run. The matching
`colab_phase2_resume_runner.py` file is the editable percent-format source.

Default Colab queue:

```text
arcface -> cosface -> adaface -> curricularface -> elasticface -> magface -> proposed
```

Workflow:

1. Save Kaggle outputs as `phase2_outputs.zip` or a Kaggle Dataset.
2. Put the extracted `outputs/phase2_loss` folder under your Google Drive
   backup root, for example:

```text
/content/drive/MyDrive/phase2_resnet18_loss/outputs/phase2_loss/
```

3. Open `colab_phase2_resume_runner.ipynb` in Colab.
4. Edit `DATA_DIR`, `PRETRAINED_BACKBONE`, and `DRIVE_BACKUP_ROOT`.
5. Run the notebook. If a loss has `latest.pt`, it resumes that loss. If that
   loss is still incomplete after the time limit, the runner backs it up and
   stops instead of moving to the next loss.

To move back from Colab to Kaggle, zip or upload the same
`outputs/phase2_loss` folder, then copy it into `/kaggle/working/outputs` before
running `--resume`.

## 13. Changing code during experiments

Changing code can affect resume quality and experiment validity.

Safe changes while resuming the same experiment:

- README changes.
- Logging changes.
- Checkpoint/backup frequency changes.
- Bug fixes that do not change model/head/loss state shapes.

Risky changes:

- Changing the backbone architecture or `embedding_size`.
- Changing classifier head parameters or `num_classes`.
- Renaming loss classes or buffers.
- Changing loss behavior in the middle of a run.
- Changing dataset folder names/order for ImageFolder datasets.

If you improve `elasticface`, `magface`, or `proposed`, keep their existing
checkpoint folders separate from earlier runs unless the change is only a bug
fix. Use a new `--output-dir` or rename the experiment folder when the loss
formula changes, otherwise the old and new training curves will be mixed.

## 14. OneDrive checkpoint option

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

## 15. Add a new loss to the registry

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
