"""Kaggle report helper for 5-eval + degraded face-recognition runs.

The file name is kept for compatibility with the Proposed 4.3 Core runner, but
the helper is generic: ArcFace, AdaFace, CurricularFace, Proposed 4.1/4.2, and
Proposed 4.3 Core can all use the same report function.

It reads:
  - metrics.json files from the experiment directory;
  - degraded_eval/degraded_metrics.csv if available.

It writes:
  - eval_by_epoch.csv;
  - PNG plots for training, clean verification, diagnostics, and degraded eval;
  - report_summary.json;
  - a report ZIP.
"""
from __future__ import annotations

from pathlib import Path
import json
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _complete_accuracy_mean(evals, targets):
    vals = []
    for target in targets:
        item = (evals or {}).get(target, {}) or {}
        val = _safe_float(item.get("accuracy"))
        if val is None:
            return None
        vals.append(val)
    return sum(vals) / max(1, len(vals))


def _find_metrics_files(root: Path):
    if not root.exists():
        return []
    return sorted(root.glob("*/metrics.json"))


def _load_epoch_rows(metrics_path: Path, eval_targets):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    rows = []
    for rec in metrics.get("epochs", []):
        evals = rec.get("eval", {}) or {}
        row = {
            "experiment": metrics_path.parent.name,
            "epoch": rec.get("epoch"),
            "loss": rec.get("loss"),
            "mean_norm": rec.get("mean_norm"),
            "lr": rec.get("lr"),
            "backbone_lr": rec.get("backbone_lr"),
            "head_lr": rec.get("head_lr"),
            "Eval5_Avg": _complete_accuracy_mean(evals, eval_targets),
            "attention_loss": rec.get("attention_loss"),
            "ri_multi_mean": rec.get("ri_multi_mean"),
            "ri_multi_min": rec.get("ri_multi_min"),
            "ri_multi_max": rec.get("ri_multi_max"),
            "gate_lambda_i_mean": rec.get("gate_lambda_i_mean"),
            "sample_weight_mean": rec.get("sample_weight_mean"),
            "sample_weight_min": rec.get("sample_weight_min"),
            "sample_weight_max": rec.get("sample_weight_max"),
            "cos_ui_multi_mean": rec.get("cos_ui_multi_mean"),
            "d_ui_multi_mean": rec.get("d_ui_multi_mean"),
            "ui_loss_mean": rec.get("ui_loss_mean"),
            "ui_extra_loss": rec.get("ui_extra_loss"),
        }
        for target in eval_targets:
            item = evals.get(target, {}) or {}
            row[target] = item.get("accuracy")
            row[f"{target}_std"] = item.get("std")
            row[f"{target}_xnorm"] = item.get("xnorm")
        rows.append(row)
    return rows


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col not in ("experiment", "target", "condition", "degradation", "source_csv", "checkpoint"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _plot_line(df, x, y, out_path, title, ylabel=None):
    if y not in df.columns or df[y].dropna().empty:
        return False
    plt.figure(figsize=(8, 5))
    for exp_name, sub in df.groupby("experiment"):
        sub = sub.sort_values(x)
        plt.plot(sub[x], sub[y], marker="o", label=exp_name)
    plt.xlabel(x)
    plt.ylabel(ylabel or y)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _plot_clean_targets_by_epoch(df, eval_targets, out_path):
    available = [target for target in eval_targets if target in df.columns and not df[target].dropna().empty]
    if not available:
        return False
    plt.figure(figsize=(9, 5))
    plot_df = df.sort_values("epoch")
    for target in available:
        plt.plot(plot_df["epoch"], plot_df[target], marker="o", label=target)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Clean verification accuracy by target")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _plot_selected_epoch_bar(df, eval_targets, out_path):
    if df.empty:
        return False
    if "Eval5_Avg" in df.columns and not df["Eval5_Avg"].dropna().empty:
        idx = df["Eval5_Avg"].idxmax()
    else:
        idx = df["epoch"].idxmax()
    row = df.loc[idx]
    labels = []
    values = []
    for target in eval_targets:
        val = _safe_float(row.get(target))
        if val is not None:
            labels.append(target)
            values.append(val)
    if not values:
        return False
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylim(max(0.0, min(values) - 0.05), min(1.0, max(values) + 0.02))
    plt.ylabel("accuracy")
    plt.title(f"Clean eval at selected epoch {int(row.get('epoch', 0))}")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _collect_degraded_metrics(exp_root: Path, report_dir: Path):
    rows = []
    for p in exp_root.glob("*/degraded_eval/**/degraded_metrics.csv"):
        try:
            df = pd.read_csv(p)
            df["source_csv"] = str(p)
            rows.append(df)
        except Exception as exc:
            print("[REPORT] Could not read degraded metrics:", p, repr(exc))
    if not rows:
        return None, None

    out = pd.concat(rows, ignore_index=True)
    out = _to_numeric(out)
    out_path = report_dir / "degraded_eval_all_csv_rows.csv"
    out.to_csv(out_path, index=False)
    return out, out_path


def _degraded_only(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "condition" in out.columns:
        out = out[out["condition"].astype(str) != "clean"]
    if "degradation" in out.columns:
        out = out[out["degradation"].astype(str) != "clean"]
    return out


def _plot_bar_series(series: pd.Series, out_path: Path, title: str, ylabel: str):
    series = series.dropna().sort_values(ascending=False)
    if series.empty:
        return False
    plt.figure(figsize=(9, 5))
    plt.bar(series.index.astype(str), series.values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _plot_degraded_heatmap(df: pd.DataFrame, out_path: Path):
    if df.empty or not {"degradation", "target", "accuracy"}.issubset(df.columns):
        return False
    pivot = df.pivot_table(index="degradation", columns="target", values="accuracy", aggfunc="mean")
    if pivot.empty:
        return False
    plt.figure(figsize=(1.3 * len(pivot.columns) + 4, 0.55 * len(pivot.index) + 3))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="accuracy")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Synthetic degraded accuracy heatmap")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _plot_severity_curve(df: pd.DataFrame, out_path: Path):
    if df.empty or "severity" not in df.columns:
        return False
    sev_df = df.dropna(subset=["severity", "accuracy"])
    if sev_df["severity"].nunique() < 2:
        return False
    plt.figure(figsize=(8, 5))
    for degradation, sub in sev_df.groupby("degradation"):
        line = sub.groupby("severity")["accuracy"].mean().sort_index()
        plt.plot(line.index, line.values, marker="o", label=degradation)
    plt.xlabel("severity")
    plt.ylabel("accuracy")
    plt.title("Accuracy by degradation severity")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def make_report(
    backup_zip_name: str,
    output_subdir: str,
    eval_targets,
    degraded_targets=None,
    degraded_degradations=None,
):
    backup_stem = Path(backup_zip_name).stem
    exp_root = Path("/kaggle/working/experiments") / backup_stem / output_subdir
    report_dir = Path("/kaggle/working") / f"{backup_stem}_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_files = _find_metrics_files(exp_root)
    if not metrics_files:
        print("[REPORT] No metrics.json found under", exp_root)
        return None

    all_rows = []
    for metrics_path in metrics_files:
        all_rows.extend(_load_epoch_rows(metrics_path, eval_targets))
    df = pd.DataFrame(all_rows)
    df = _to_numeric(df)

    csv_path = report_dir / "eval_by_epoch.csv"
    df.to_csv(csv_path, index=False)
    print("[REPORT] Saved", csv_path)

    _plot_line(df, "epoch", "loss", report_dir / "01_train_loss.png", "Train loss by epoch", "loss")
    _plot_line(df, "epoch", "Eval5_Avg", report_dir / "02_eval5_avg.png", "5-eval average accuracy", "accuracy")
    _plot_line(df, "epoch", "mean_norm", report_dir / "03_feature_norm.png", "Mean feature norm", "norm")
    _plot_line(df, "epoch", "attention_loss", report_dir / "04_attention_loss.png", "Attention auxiliary loss", "loss")
    _plot_line(df, "epoch", "ri_multi_mean", report_dir / "05_ri_multi_mean.png", "RI/multi-UI mean diagnostic", "ri_multi_mean")
    _plot_line(df, "epoch", "sample_weight_mean", report_dir / "06_sample_weight_mean.png", "Sample-weight mean diagnostic", "sample_weight_mean")
    _plot_line(df, "epoch", "gate_lambda_i_mean", report_dir / "07_gate_lambda_mean.png", "Gate lambda mean diagnostic", "gate_lambda_i_mean")
    _plot_line(df, "epoch", "cos_ui_multi_mean", report_dir / "08_cos_ui_multi_mean.png", "Cosine to multi-UI mean", "cos_ui_multi_mean")
    _plot_clean_targets_by_epoch(df, eval_targets, report_dir / "09_clean_targets_by_epoch.png")
    _plot_selected_epoch_bar(df, eval_targets, report_dir / "09b_clean_eval_selected_epoch_bar.png")

    degraded_df, degraded_csv = _collect_degraded_metrics(exp_root, report_dir)
    degraded_only = _degraded_only(degraded_df)

    if degraded_only is not None and not degraded_only.empty:
        if {"degradation", "accuracy"}.issubset(degraded_only.columns):
            _plot_bar_series(
                degraded_only.groupby("degradation")["accuracy"].mean(),
                report_dir / "10_degraded_mean_by_type.png",
                "Mean synthetic degraded accuracy by type",
                "accuracy",
            )
        if {"target", "accuracy"}.issubset(degraded_only.columns):
            _plot_bar_series(
                degraded_only.groupby("target")["accuracy"].mean(),
                report_dir / "11_degraded_mean_by_target.png",
                "Mean synthetic degraded accuracy by eval target",
                "accuracy",
            )
        _plot_degraded_heatmap(degraded_only, report_dir / "12_degraded_accuracy_heatmap.png")
        _plot_severity_curve(degraded_only, report_dir / "13_degraded_accuracy_by_severity.png")
        if {"degradation", "drop_from_clean"}.issubset(degraded_only.columns):
            _plot_bar_series(
                degraded_only.groupby("degradation")["drop_from_clean"].mean(),
                report_dir / "14_degraded_drop_by_type.png",
                "Mean drop from clean by degradation type",
                "accuracy drop",
            )

    summary = {
        "experiment_root": str(exp_root),
        "metrics_files": [str(x) for x in metrics_files],
        "eval_targets": list(eval_targets),
        "degraded_targets": list(degraded_targets or []),
        "degraded_degradations": list(degraded_degradations or []),
        "main_csv": str(csv_path),
        "degraded_csv": str(degraded_csv) if degraded_csv else None,
        "note": (
            "This is a single-model report. Use a separate compare_all_models script "
            "to aggregate multiple model backup/report folders."
        ),
    }
    with open(report_dir / "report_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    zip_path = Path("/kaggle/working") / f"{backup_stem}_report.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in report_dir.rglob("*"):
            if item.is_file():
                zf.write(item, item.relative_to(report_dir))
    print("[REPORT] Saved", zip_path)

    try:
        from IPython.display import FileLink, display
        display(FileLink(str(csv_path)))
        display(FileLink(str(zip_path)))
    except Exception:
        pass

    return zip_path
