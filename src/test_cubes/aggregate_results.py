#!/usr/bin/env python3
"""
Aggregate test_cubes results from all 9 layer x timestep combinations.
Saves summary CSV, heatmaps (MAE + gap for Az and El), scatter plots, and log.
"""

import json
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches

EVAL_DIR  = Path("probe_exp/test_cubes_eval_all")
PRIOR_DIR = Path("probe_exp/pca256_job_2628839")
OUT_DIR   = Path("probe_exp/test_cubes_eval_all/summary")

LAYERS    = ["layer_04", "layer_10", "layer_18"]
TIMESTEPS = ["t_200", "t_500", "t_800"]

AZ_RANGE = 360.0
EL_RANGE = 45.0   # 75 - 30


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("aggregate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_heatmap(df: pd.DataFrame, col: str, title: str, fname: str,
                 cmap: str = "RdYlGn_r", center=None, fmt: str = ".4f"):
    pivot = df.pivot(index="layer", columns="timestep", values=col)
    pivot = pivot.reindex(index=LAYERS, columns=TIMESTEPS)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        pivot, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5,
        center=center, ax=ax, cbar_kws={"label": col},
        annot_kws={"size": 12},
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / fname
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_scatter_raw(df: pd.DataFrame) -> Path:
    colors_map  = {"layer_04": "#e15759", "layer_10": "#4e79a7", "layer_18": "#59a14f"}
    markers_map = {"t_200": "o", "t_500": "s", "t_800": "D"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in df.iterrows():
        ax.scatter(
            row["cubes_az"], row["cubes_el"],
            color=colors_map[row["layer"]],
            marker=markers_map[row["timestep"]],
            s=140, zorder=3, edgecolors="white", linewidths=1.2,
        )
        ax.annotate(
            f"{row['layer']}\n{row['timestep']}",
            (row["cubes_az"], row["cubes_el"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, color=colors_map[row["layer"]],
        )
    for layer, color in colors_map.items():
        ax.scatter([], [], color=color, marker="o", s=80, label=layer)
    for ts, marker in markers_map.items():
        ax.scatter([], [], color="gray", marker=marker, s=80, label=ts)
    ax.set_xlabel("Azimuth MAE (°)", fontsize=11)
    ax.set_ylabel("Elevation MAE (°)", fontsize=11)
    ax.set_title("Az vs El MAE on test_cubes — all 9 configs", fontsize=13)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "scatter_az_vs_el.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_scatter_normalized(df: pd.DataFrame) -> Path:
    colors_map  = {"layer_04": "#e15759", "layer_10": "#4e79a7", "layer_18": "#59a14f"}
    markers_map = {"t_200": "o", "t_500": "s", "t_800": "D"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in df.iterrows():
        ax.scatter(
            row["norm_az"], row["norm_el"],
            color=colors_map[row["layer"]],
            marker=markers_map[row["timestep"]],
            s=140, zorder=3, edgecolors="white", linewidths=1.2,
        )
        ax.annotate(
            f"{row['layer']}\n{row['timestep']}",
            (row["norm_az"], row["norm_el"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, color=colors_map[row["layer"]],
        )
    for layer, color in colors_map.items():
        ax.scatter([], [], color=color, marker="o", s=80, label=layer)
    for ts, marker in markers_map.items():
        ax.scatter([], [], color="gray", marker=marker, s=80, label=ts)
    ax.set_xlabel(f"Norm. Az MAE  (MAE / {AZ_RANGE:.0f}°)", fontsize=11)
    ax.set_ylabel(f"Norm. El MAE  (MAE / {EL_RANGE:.0f}°)", fontsize=11)
    ax.set_title("Normalized Az vs El MAE on test_cubes", fontsize=13)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "scatter_az_vs_el_normalized.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_score_barplot(df: pd.DataFrame) -> Path:
    colors_map = {"layer_04": "#e15759", "layer_10": "#4e79a7", "layer_18": "#59a14f"}
    df_sorted = df.sort_values("combined_score")
    labels = [f"{r.layer}\n{r.timestep}" for r in df_sorted.itertuples()]
    colors = [colors_map[r.layer] for r in df_sorted.itertuples()]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, df_sorted["combined_score"], color=colors,
                  edgecolor="white", linewidth=1.2, zorder=3)
    for bar, val in zip(bars, df_sorted["combined_score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    legend_handles = [mpatches.Patch(facecolor=color, label=layer)
                  for layer, color in colors_map.items()]
    ax.set_ylabel("Combined score  (Az/360 + El/45)", fontsize=11)
    ax.set_title("Combined normalized score per config — lower is better", fontsize=13)
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left")

    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = OUT_DIR / "barplot_combined_score.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(OUT_DIR / "aggregate.log")

    logger.info("=== Aggregating test_cubes results ===")
    logger.info(f"EVAL_DIR  : {EVAL_DIR}")
    logger.info(f"PRIOR_DIR : {PRIOR_DIR}")
    logger.info(f"OUT_DIR   : {OUT_DIR}")
    logger.info(f"AZ_RANGE  : {AZ_RANGE}°  |  EL_RANGE: {EL_RANGE}°")

    records = []
    for layer in LAYERS:
        for timestep in TIMESTEPS:
            cubes_path = EVAL_DIR / f"{layer}_{timestep}" / "test_cubes_results.json"
            prior_path = PRIOR_DIR / f"{layer}_{timestep}_results.json"

            if not cubes_path.exists():
                logger.warning(f"MISSING: {cubes_path}")
                continue

            with open(cubes_path) as f:
                cubes = json.load(f)

            prior = None
            if prior_path.exists():
                with open(prior_path) as f:
                    prior = json.load(f)
            else:
                logger.warning(f"No prior results for {layer}/{timestep}")

            test_az  = prior["test"]["mae_azimuth_deg"]   if prior else None
            test_el  = prior["test"]["mae_elevation_deg"] if prior else None
            cubes_az = cubes["probe"]["mae_azimuth_deg"]
            cubes_el = cubes["probe"]["mae_elevation_deg"]
            norm_az  = cubes_az / AZ_RANGE
            norm_el  = cubes_el / EL_RANGE
            score    = round(norm_az + norm_el, 6)

            records.append({
                "layer":          layer,
                "timestep":       timestep,
                "test_az":        test_az,
                "test_el":        test_el,
                "cubes_az":       cubes_az,
                "cubes_el":       cubes_el,
                "gap_az":         round(cubes_az - test_az, 4) if test_az is not None else None,
                "gap_el":         round(cubes_el - test_el, 4) if test_el is not None else None,
                "norm_az":        round(norm_az, 6),
                "norm_el":        round(norm_el, 6),
                "combined_score": score,
                "median_az":      cubes["probe"]["median_az_error"],
                "p90_az":         cubes["probe"]["p90_az_error"],
                "median_el":      cubes["probe"]["median_el_error"],
                "p90_el":         cubes["probe"]["p90_el_error"],
            })
            logger.info(
                f"  {layer} / {timestep}"
                f"  cubes_az={cubes_az:.2f}°  cubes_el={cubes_el:.2f}°"
                f"  gap_az={records[-1]['gap_az']:+.2f}°  gap_el={records[-1]['gap_el']:+.2f}°"
                f"  norm_az={norm_az:.4f}  norm_el={norm_el:.4f}  score={score:.4f}"
            )

    df = pd.DataFrame(records).sort_values(["layer", "timestep"]).reset_index(drop=True)

    csv_path = OUT_DIR / "test_cubes_all_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nCSV saved: {csv_path}")
    logger.info("\n" + df.to_string(index=False))

    heatmaps = [
        ("cubes_az",       "Azimuth MAE on test_cubes (°)",                    "heatmap_cubes_az.png",       "RdYlGn_r", None, ".2f"),
        ("cubes_el",       "Elevation MAE on test_cubes (°)",                  "heatmap_cubes_el.png",       "RdYlGn_r", None, ".2f"),
        ("gap_az",         "Generalization gap — Azimuth (cubes − test, °)",   "heatmap_gap_az.png",         "RdYlGn_r", 0,    ".2f"),
        ("gap_el",         "Generalization gap — Elevation (cubes − test, °)", "heatmap_gap_el.png",         "RdYlGn_r", 0,    ".2f"),
        ("combined_score", "Combined score (Az/360 + El/45) — lower is better","heatmap_combined_score.png", "RdYlGn_r", None, ".4f"),
    ]
    for col, title, fname, cmap, center, fmt in heatmaps:
        path = save_heatmap(df, col, title, fname, cmap=cmap, center=center, fmt=fmt)
        logger.info(f"Saved heatmap: {path}")

    path = save_scatter_raw(df)
    logger.info(f"Saved scatter (raw):        {path}")

    path = save_scatter_normalized(df)
    logger.info(f"Saved scatter (normalized): {path}")

    path = save_score_barplot(df)
    logger.info(f"Saved barplot (score):      {path}")

    best = df.loc[df["combined_score"].idxmin()]
    logger.info("\n=== Best config (lowest combined score) ===")
    logger.info(
        f"  {best['layer']} / {best['timestep']}"
        f"  cubes_az={best['cubes_az']:.2f}°  cubes_el={best['cubes_el']:.2f}°"
        f"  gap_az={best['gap_az']:+.2f}°  gap_el={best['gap_el']:+.2f}°"
        f"  combined_score={best['combined_score']:.4f}"
    )

    best_az = df.loc[df["cubes_az"].idxmin()]
    logger.info("\n=== Best config (lowest Az MAE only) ===")
    logger.info(f"  {best_az['layer']} / {best_az['timestep']}  cubes_az={best_az['cubes_az']:.2f}°")

    best_gap = df.loc[df["gap_az"].idxmin()]
    logger.info("\n=== Best config (lowest generalization gap Az) ===")
    logger.info(f"  {best_gap['layer']} / {best_gap['timestep']}  gap_az={best_gap['gap_az']:+.2f}°")

    logger.info("\n=== Done ===")


if __name__ == "__main__":
    main()