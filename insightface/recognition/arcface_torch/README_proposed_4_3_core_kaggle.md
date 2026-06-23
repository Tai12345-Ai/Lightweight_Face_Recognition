# Proposed 4.3 Core Kaggle Runner

Copy these two files into:

```text
insightface/recognition/arcface_torch/
```

Files:

```text
kaggle_proposed_4_3_core_5eval_degraded_runner.py
kaggle_proposed_4_3_core_report.py
```

Run the runner as a Kaggle notebook/script.

## What this Core-v0 does in the current repo

The current repo already has:

- `train_soft_gated_lambda_kaggle.py`
- `kaggle_5eval_degraded_common.py`
- `build_multi_ui_centers.py`
- `eval_degraded_6phase2.py`

This runner reuses that infrastructure. It configures Proposed 4.3 as a practical Core-v0 by:

- using `LOSS_NAME = "proposed_4_3_multi_ui_attention"`;
- setting `UI_LAMBDA = 0.0` to disable explicit UI extra loss;
- enabling attention with `ENABLE_ATTENTION = True`;
- training 20 epochs from `backbone.pth`;
- running the 5 clean evals;
- running synthetic degraded eval at severity 5;
- exporting CSV + plots + zip backup.

## Important note

This is runnable with the current repo design. It is not yet the full mathematical Core from the document because the repo trainer does not yet expose explicit:

- RI predictor loss;
- preserve loss;
- identity-anchor loss;
- Delta C / Delta N / Delta U diagnostics.

Those need a deeper `train_soft_gated_lambda_kaggle.py` patch.

## Minimum metrics to keep

Keep these during Core 20ep:

1. Train loss by epoch.
2. Eval5 average by epoch.
3. Individual LFW / CFP-FP / CPLFW / AgeDB-30 / CALFW accuracy.
4. Synthetic degraded accuracy by degradation.
5. Mean feature norm.
6. Attention loss.
7. RI/UI diagnostic keys if available: `ri_multi_mean`, `cos_ui_multi_mean`, `sample_weight_mean`, `gate_lambda_i_mean`.

For the strict Core patch later, additionally log:

```text
preserve_loss
anchor_loss
ri_pred_loss
rho_att_mean
rho_att_std
delta_C_mean
delta_N_mean
delta_U_mean
```
