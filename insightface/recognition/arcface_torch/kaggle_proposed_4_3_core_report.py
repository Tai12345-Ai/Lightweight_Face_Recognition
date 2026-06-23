"""Small report helper for Kaggle Proposed 4.3 Core runs.

Put this file next to kaggle_proposed_4_3_core_5eval_degraded_runner.py in
insightface/recognition/arcface_torch. The runner calls make_report() after training.
"""
from __future__ import annotations

from pathlib import Path
import json
import zipfile

import pandas as pd
import matplotlib.pyplot as plt


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
            row[f"{target}_xnorm"] = item.get("xnorm")
        rows.append(row)
    return rows


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


def _collect_degraded_csvs(exp_root: Path, report_dir: Path):
    rows = []
    for p in exp_root.glob("*/degraded_eval/**/*.csv"):
        try:
            df = pd.read_csv(p)
            df["source_csv"] = str(p)
            rows.append(df)
        except Exception:
            pass
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out_path = report_dir / "degraded_eval_all_csv_rows.csv"
        out.to_csv(out_path, index=False)
        return out_path
    return None


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
    for col in df.columns:
        if col not in ("experiment",):
            df[col] = pd.to_numeric(df[col], errors="ignore")

    csv_path = report_dir / "core_eval_by_epoch.csv"
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

    degraded_csv = _collect_degraded_csvs(exp_root, report_dir)
    summary = {
        "experiment_root": str(exp_root),
        "metrics_files": [str(x) for x in metrics_files],
        "eval_targets": list(eval_targets),
        "degraded_targets": list(degraded_targets or []),
        "degraded_degradations": list(degraded_degradations or []),
        "main_csv": str(csv_path),
        "degraded_csv": str(degraded_csv) if degraded_csv else None,
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
