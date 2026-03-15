#!/usr/bin/env python3
"""
plot_proteinoptimizer_results.py

Reads ProteinOptimizer JSON result files with names like:
  results_single_<task>_0_<model>.json

Expected JSON keys:
  - best_score
  - best_scores_history

Outputs three figures using seaborn Set2 style:
  1) Bar plot of final Top 1 per model, averaged across tasks
  2) Convergence plot of Top 1 vs iteration per model, averaged across tasks
  3) Grouped bar plot of final Top 1 by task: each model is one cluster with 5 task bars

Usage:
  python plot_proteinoptimizer_results.py --input_dir /path/to/results --out_dir /path/to/figs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


FNAME_RE = re.compile(r"results_single_(?P<task>[^_]+)_0_(?P<model>.+)\.json$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--title_prefix", type=str, default="")
    return p.parse_args()


def model_display_name(model_key: str) -> str:
    mapping = {
        "baseline": "baseline",
        "gpt-5-mini": "gpt-5-mini",
        "deepseek-reasoner": "deepseek-R1",
        "claude-sonnet-4-5": "claude-sonnet-4.5",
        "gpt-5": "gpt-5",
        "gpt-5-chat-latest": "gpt-5-chat",
    }
    return mapping.get(model_key, model_key)


def preferred_model_order(models_present: List[str]) -> List[str]:
    preferred = [
        "baseline",
        "gpt-5-mini",
        "gpt-5-chat",
        "gpt-5",
        "claude-sonnet-4.5",
        "deepseek-R1",
    ]
    ordered = [m for m in preferred if m in models_present]
    if not ordered:
        ordered = sorted(models_present)
    return ordered


def preferred_task_order(tasks_present: List[str]) -> List[str]:
    # Matches your methods narrative order: GB1, TrpB, Syn-3bfo, AAV, GFP
    preferred = ["gb1", "trpb", "syn-3bfo", "aav", "gfp"]
    ordered = [t for t in preferred if t in tasks_present]
    # Append any extras that appear in files
    extras = [t for t in tasks_present if t not in ordered]
    return ordered + sorted(extras)


def load_records(input_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, "results_single_*_0_*.json")))
    files = [f for f in files if "__MACOSX" not in f and not os.path.basename(f).startswith("._")]

    rows = []
    for fp in files:
        m = FNAME_RE.search(os.path.basename(fp))
        if not m:
            continue

        task = m.group("task")
        model_key = m.group("model")

        with open(fp, "r") as f:
            d = json.load(f)

        hist = d.get("best_scores_history")
        if hist is None:
            best = d.get("best_score")
            if best is None:
                continue
            hist = [float(best)]
        else:
            hist = [float(x) for x in hist]

        for it, v in enumerate(hist, start=1):
            rows.append(
                {
                    "task": task,
                    "model_key": model_key,
                    "model": model_display_name(model_key),
                    "iteration": it,
                    "top1": float(v),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No valid JSON files found in {input_dir}")
    return df


def set_style() -> None:
    sns.set_theme(style="white", context="talk")
    sns.set_palette("Set2")


def apply_common_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_color_map(items_in_order: List[str], palette_name: str = "Set2") -> Dict[str, tuple]:
    palette = sns.color_palette(palette_name, n_colors=len(items_in_order))
    return {k: palette[i] for i, k in enumerate(items_in_order)}


def get_final_per_task_model(df: pd.DataFrame) -> pd.DataFrame:
    # final Top 1 is last iteration for each (task, model)
    last_it = df.groupby(["task", "model"])["iteration"].max().reset_index()
    df_last = df.merge(last_it, on=["task", "model", "iteration"], how="inner")
    return df_last


def plot_bar_overall(df: pd.DataFrame, out_dir: str, title_prefix: str = "") -> None:
    df_last = get_final_per_task_model(df)

    models_present = df_last["model"].unique().tolist()
    model_order = preferred_model_order(models_present)
    cmap = make_color_map(model_order, "Set2")

    bar = df_last.groupby("model")["top1"].mean().reindex(model_order)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.grid(False)  # Remove background grid
    x = np.arange(len(model_order))
    vals = bar.to_numpy()

    bars = ax.bar(
        x,
        vals,
        color=[cmap[m] for m in model_order],
        alpha=0.88,
        width=0.78,
        align="center",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=18, rotation=45, ha='right')
    ax.tick_params(axis="x", labelsize=18, rotation=25)
    ax.tick_params(axis="y", labelsize=18)
    ax.spines["bottom"].set_position(("outward", 10))

    ax.set_xlabel("Model", fontsize=18)
    ax.set_ylabel("Top 1 Score", fontsize=18)

    title = "Top 1 Performance"
    if title_prefix.strip():
        title = f"{title_prefix.strip()} {title}"
    ax.set_title(title, fontsize=18, pad=20)

    apply_common_style(ax)
    ax.set_xlim(-0.6, len(model_order) - 0.4)
    ax.set_ylim(0, min(1.1, float(np.max(vals) * 1.10)))

    for rect, v in zip(bars, vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.01,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=18,
            rotation=0,
        )

    os.makedirs(out_dir, exist_ok=True)
    # fig.tight_layout(rect=[0, 0.08, 1, 0.95])  # Leave space at top for title and bottom for rotated labels
    fig.savefig(os.path.join(out_dir, "ProteinOptimizerResult.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "ProteinOptimizerResult.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_convergence(df: pd.DataFrame, out_dir: str, title_prefix: str = "") -> None:
    models_present = df["model"].unique().tolist()
    model_order = preferred_model_order(models_present)
    cmap = make_color_map(model_order, "Set2")

    agg = df.groupby(["model", "iteration"])["top1"].mean().reset_index()
    max_it = int(df["iteration"].max())

    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.grid(False)  # Remove background grid

    for m in model_order:
        if m not in set(agg["model"].unique()):
            continue
        sub = agg[agg["model"] == m].sort_values("iteration")
        ax.plot(
            sub["iteration"].to_numpy(),
            sub["top1"].to_numpy(),
            marker="o",
            linewidth=2.4,
            markersize=5.5,
            alpha=0.95,
            color=cmap[m],
            label=m,
        )

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Top 1 Score", fontsize=13)
    ax.tick_params(axis="both", labelsize=18)

    title = "Convergence of Top 1 Score"
    if title_prefix.strip():
        title = f"{title_prefix.strip()} {title}"
    ax.set_title(title, fontsize=18, pad=20)

    ax.set_xticks(np.arange(1, max_it + 1))
    ax.set_xlim(0.7, max_it + 0.3)

    y_min = float(agg["top1"].min())
    y_max = float(agg["top1"].max())
    ax.set_ylim(max(0.0, y_min - 0.08), min(1.1, y_max + 0.08))

    apply_common_style(ax)
    ax.legend(frameon=False, ncol=2, fontsize=11, loc="lower right")

    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top for title
    fig.savefig(os.path.join(out_dir, "PO_top1_convergence.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "PO_top1_convergence.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_by_task_grouped(df: pd.DataFrame, out_dir: str, title_prefix: str = "") -> None:
    """
    Big grouped bar plot:
      x-axis: model clusters
      within each cluster: 5 small bars for tasks
      value: final Top 1 for that task-model
    """
    df_last = get_final_per_task_model(df)

    models_present = df_last["model"].unique().tolist()
    tasks_present = df_last["task"].unique().tolist()

    model_order = preferred_model_order(models_present)
    task_order = preferred_task_order(tasks_present)

    # Use Set2 for task colors (better readability within each model cluster)
    task_cmap = make_color_map(task_order, "Set2")

    # Build matrix: models x tasks
    pivot = df_last.pivot_table(index="model", columns="task", values="top1", aggfunc="mean")
    pivot = pivot.reindex(index=model_order, columns=task_order)

    fig, ax = plt.subplots(figsize=(14.0, 6.0))  # Wider and taller figure for rotated labels
    ax.grid(False)  # Remove background grid

    x = np.arange(len(model_order))
    k = len(task_order)
    bar_w = 0.75 / max(k, 1)  # Slightly narrower bars to reduce overlap
    offsets = (np.arange(k) - (k - 1) / 2) * bar_w

    bars_by_task = []
    for j, task in enumerate(task_order):
        vals = pivot[task].to_numpy()
        bars = ax.bar(
            x + offsets[j],
            vals,
            width=bar_w * 0.95,
            alpha=0.90,
            color=task_cmap[task],
            label=task,
        )
        bars_by_task.append((bars, vals))
        
        # Add value labels on top of each bar
        for bar, val in zip(bars, vals):
            if not np.isnan(val) and np.isfinite(val):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.03,  # Increased spacing to avoid overlap
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,  # Slightly smaller to reduce overlap
                    rotation=50,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=18, rotation=35, ha='right')
    ax.tick_params(axis="x", labelsize=18, rotation=15)
    ax.tick_params(axis="y", labelsize=18)
    ax.spines["bottom"].set_position(("outward", 10))

    ax.set_xlabel("Model", fontsize=24)
    ax.set_ylabel("Top 1 Score", fontsize=24)

    title = "Top 1 Performance by Task"
    if title_prefix.strip():
        title = f"{title_prefix.strip()} {title}"
    ax.set_title(title, fontsize=24, pad=24)

    # y-limits from data - add extra space for labels and legend
    finite_vals = np.asarray(pivot.to_numpy(), dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if finite_vals.size == 0:
        y_hi = 1.0
    else:
        y_max = float(np.max(finite_vals))
        y_hi = min(1.1, y_max * 1.35)  # More space for labels to avoid overlap
    ax.set_ylim(0, y_hi)

    apply_common_style(ax)

    # Move legend to avoid overlap - position outside plot area on the right
    ax.legend(frameon=False, ncol=1, fontsize=18, loc="center left", bbox_to_anchor=(0.972, 0.5), handlelength=1.0, handletextpad=0.5)

    os.makedirs(out_dir, exist_ok=True)
    # fig.tight_layout(rect=[0, 0.05, 0.92, 0.95])  # Leave space on the right for legend, bottom for rotated labels, and top for title
    fig.savefig(os.path.join(out_dir, "PO_top1_by_task_grouped.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "PO_top1_by_task_grouped.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_style()

    df = load_records(args.input_dir)

    plot_bar_overall(df, args.out_dir, title_prefix=args.title_prefix)
    plot_convergence(df, args.out_dir, title_prefix=args.title_prefix)
    plot_by_task_grouped(df, args.out_dir, title_prefix=args.title_prefix)

    print("Wrote figures to", args.out_dir)
    print("  ProteinOptimizerResult.png / .pdf")
    print("  PO_top1_convergence.png / .pdf")
    print("  PO_top1_by_task_grouped.png / .pdf")


if __name__ == "__main__":
    main()

