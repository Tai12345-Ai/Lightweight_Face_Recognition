# Proposed 4.3 Core/Full Kaggle Protocol

This is the official Kaggle protocol for the Proposed 4.3 Core and Full
experiments in this repository.

## Entry Points

Run Core:

```python
%run experiments/kaggle/kaggle_proposed_4_3_core_5eval_degraded_runner.py
```

Run Full:

```python
%run experiments/kaggle/kaggle_proposed_4_3_full_5eval_degraded_runner.py
```

Both runners auto-detect:

- `train.rec` and `train.idx` under `/kaggle/input`
- the 5 verification bins: `lfw.bin`, `cfp_fp.bin`, `cplfw.bin`, `agedb_30.bin`,
  `calfw.bin`
- `backbone.pth` when no finished Core checkpoint is available
- existing multi-UI centers, or build them automatically

Protocol split:

```text
UI centers: severity 5
Evaluation: severity 1,3,5
```

## Core

Core uses the true attention path:

```text
F_i -> attention map M_i
F'_i = F_i * (1 + alpha * rho_att * centered_or_raw(M_i))
x'_i = embedding(F'_i)
```

Core includes:

- true post-attention embedding `x'`
- RI/recoverability predictor
- recoverability/attention gate
- weighted FR
- identity-anchor loss
- preserve loss
- attention regularization
- diagnostic logging

Core intentionally disables the Full-only penalties:

```text
ui_lambda = 0.0
neg_lambda = 0.0
```

Core uses degraded severities `1,3,5` and writes:

```text
proposed_4_3_core_20ep_5eval_degraded_s135.zip
```

Core UI centers are built from severity `5` only and use:

```text
proposed_4_3_core_multi_ui_centers_s5.pth
```

## Full

Full must be warm-started from a finished Core checkpoint for final comparison.
The Full runner searches in this order:

1. Core `best.pth` or `latest.pt` under `/kaggle/input`
2. Core checkpoint inside a Core backup zip
3. fallback `backbone.pth`

If it falls back to `backbone.pth`, the runner prints:

```text
[WARN] Full is starting from backbone.pth, not from a finished Core checkpoint. This run is for debugging; final comparison should use Core warm-start.
```

Full includes:

- soft top-M UI prototype
- UI-orthogonal projection
- recoverability / RI gate
- label-confidence gate
- unrecognizable gate
- identity-anchor loss
- negative-guard loss
- preserve loss
- attention regularization
- diagnostic logging

Full loss uses the post-attention embedding `x'` for base FR, UI-orthogonal,
identity-anchor, negative-guard and preserve terms. Clean eval runs through the
Full training wrapper, and degraded eval calls:

```text
eval_degraded_proposed_4_3_full.py
```

so inference loads:

```text
backbone + attention module + RI/recoverability predictor
```

and outputs `x'`.

Full default knobs:

```text
FULL_TOP_M = 4
FULL_UI_SOFT_TAU = 12.0
FULL_UI_MARGIN = 0.20

FULL_UI_LAMBDA = 0.05
FULL_RI_LAMBDA = 0.05
FULL_ANCHOR_LAMBDA = 0.08
FULL_NEG_LAMBDA = 0.06
FULL_PRESERVE_LAMBDA = 0.03

FULL_DELTA_C = 0.02
FULL_DELTA_N = 0.02

FULL_LABEL_GAMMA = 12.0
FULL_LABEL_MARGIN = 0.05
FULL_UNREC_TAU = 0.35
FULL_UNREC_GAMMA = 8.0

ATTENTION_ALPHA = 0.25
CENTERED_ATTENTION = True
ATTENTION_SPATIAL_LAMBDA = 1e-4
ATTENTION_CHANNEL_LAMBDA = 1e-4
ATTENTION_TV_LAMBDA = 1e-4
```

Full uses degraded severities `1,3,5` and writes:

```text
proposed_4_3_full_20ep_5eval_degraded_s135.zip
```

Full UI centers are built from severity `5` only and use:

```text
proposed_4_3_full_multi_ui_centers_s5.pth
```

## Logging

`train_log.csv` and `metrics.json` include the required diagnostics:

```text
base_fr_loss
ri_loss
ui_orth_loss
anchor_loss
negative_guard_loss
preserve_loss
attention_loss
rho_att_mean
rho_ui_mean
rho_neg_mean
label_gate_mean
omega_unrec_mean
delta_c_mean
delta_n_mean
delta_u_mean
embedding_shift_mean
cos_ui_soft_mean
cos_ui_orth_mean
sample_weight_mean
```

## Compatibility

Existing runners for ArcFace, AdaFace, CurricularFace, Proposed 4.1,
Proposed 4.2 and Proposed 4.3 Core remain compatible. The Full/Core behavior is
activated only by their dedicated wrapper scripts.
