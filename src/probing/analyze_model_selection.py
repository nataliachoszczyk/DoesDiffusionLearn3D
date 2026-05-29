#!/usr/bin/env python3
"""
Model selection analysis for PCA/layer/timestep combinations.
Metrics: azimuth MAE (cyclic, 0-360) and elevation MAE.
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

LAYERS = ["layer_04", "layer_10", "layer_18"]
TIMESTEPS = ["t_200", "t_500", "t_800"]
PCA_VALS = [256, 512, 1024]
COLORS_PCA = {256: "#2196F3", 512: "#FF9800", 1024: "#4CAF50"}


def load_results(probe_dir: Path, configs: dict) -> pd.DataFrame:
    records = []
    for n_pca, folder in configs.items():
        for layer in LAYERS:
            for t in TIMESTEPS:
                path = probe_dir / folder / f"{layer}_{t}_results.json"
                if not path.exists():
                    print(f"MISSING: {path}")
                    continue
                with open(path) as f:
                    data = json.load(f)
                for split in ["train", "val", "test"]:
                    records.append({
                        "n_pca": n_pca, "layer": layer, "timestep": t, "split": split,
                        "mae_azimuth": data[split]["mae_azimuth_deg"],
                        "mae_elevation": data[split]["mae_elevation_deg"],
                    })
    return pd.DataFrame(records)


def build_summary(df: pd.DataFrame, az_weight: float, el_weight: float) -> pd.DataFrame:
    splits = {}
    for split in ["train", "val", "test"]:
        s = df[df["split"] == split][["n_pca", "layer", "timestep", "mae_azimuth", "mae_elevation"]]
        splits[split] = s.rename(columns={
            "mae_azimuth": f"{split}_az",
            "mae_elevation": f"{split}_el",
        })
    summary = splits["train"].merge(splits["val"], on=["n_pca", "layer", "timestep"])
    summary = summary.merge(splits["test"], on=["n_pca", "layer", "timestep"])
    summary["overfit_gap_az"] = summary["val_az"] - summary["train_az"]
    summary["overfit_gap_el"] = summary["val_el"] - summary["train_el"]
    total = az_weight + el_weight
    summary["combined_val"] = (
        az_weight * summary["val_az"] + el_weight * summary["val_el"]
    ) / total
    summary["score"] = summary["combined_val"] + 0.5 * (
        (az_weight * summary["overfit_gap_az"] + el_weight * summary["overfit_gap_el"]) / total
    )
    return summary.sort_values("combined_val").reset_index(drop=True)


def plot_heatmaps(summary: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, col, title in [
        (axes[0], "val_az", "Val Azimuth MAE (best n_pca)"),
        (axes[1], "val_el", "Val Elevation MAE (best n_pca)"),
    ]:
        pivot = summary.groupby(["layer", "timestep"])[col].min().unstack()
        pivot = pivot.reindex(index=LAYERS, columns=TIMESTEPS)
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(TIMESTEPS)))
        ax.set_xticklabels(TIMESTEPS, fontsize=12)
        ax.set_yticks(range(len(LAYERS)))
        ax.set_yticklabels(LAYERS, fontsize=12)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}°", ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="black" if pivot.values[i, j] < pivot.values.max() * 0.85 else "white")
        plt.colorbar(im, ax=ax).set_label("MAE (°)", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Timestep", fontsize=11)
        ax.set_ylabel("Layer", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "heatmaps.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'heatmaps.png'}")


def plot_scatter(summary: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    markers_t = {"t_200": "o", "t_500": "s", "t_800": "D"}
    for ax, xcol, ycol, xlabel, ylabel, title in [
        (axes[0], "overfit_gap_az", "val_az",
         "Overfit gap az (°)", "Val Azimuth MAE (°)", "Azimuth: Val MAE vs Overfit Gap"),
        (axes[1], "overfit_gap_el", "val_el",
         "Overfit gap el (°)", "Val Elevation MAE (°)", "Elevation: Val MAE vs Overfit Gap"),
    ]:
        for n_pca in PCA_VALS:
            sub = summary[summary["n_pca"] == n_pca]
            for t, marker in markers_t.items():
                s2 = sub[sub["timestep"] == t]
                ax.scatter(s2[xcol], s2[ycol],
                           color=COLORS_PCA[n_pca], marker=marker,
                           s=80, label=f"n{n_pca}/{t}" if ax == axes[0] else None,
                           edgecolors="white", linewidths=0.5)
        best = summary.iloc[0]
        ax.annotate(
            f"Best\nn{best.n_pca}/{best.layer[6:]}/{best.timestep}",
            xy=(best[xcol], best[ycol]),
            xytext=(best[xcol] + 0.5, best[ycol] + 0.3),
            fontsize=9, color="#333",
            arrowprops=dict(arrowstyle="->", color="#555"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaa"),
        )
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
    handles = [
        plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=COLORS_PCA[n],markersize=9,label=f"n_pca={n}")
        for n in PCA_VALS
    ] + [
        plt.Line2D([0],[0],marker=m,color="gray",markersize=9,label=t,linestyle="None")
        for t, m in markers_t.items()
    ]
    axes[0].legend(handles=handles, fontsize=9, loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_overfit.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'scatter_overfit.png'}")


def plot_top10(summary: pd.DataFrame, output_dir: Path):
    top10 = summary.head(10).copy()
    top10["label"] = [f"n{r.n_pca} · l{r.layer[6:]} · {r.timestep}" for _, r in top10.iterrows()]
    colors = [COLORS_PCA[n] for n in top10["n_pca"].values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col_val, col_test, title in [
        (axes[0], "val_az", "test_az", "Azimuth MAE — Top 10"),
        (axes[1], "val_el", "test_el", "Elevation MAE — Top 10"),
    ]:
        y = range(len(top10))
        ax.barh(y, top10[col_val].values[::-1], color=colors[::-1], label="val", height=0.5)
        ax.barh(y, top10[col_test].values[::-1], color=colors[::-1], alpha=0.4,
                label="test", height=0.3, left=0)
        ax.set_yticks(list(y))
        ax.set_yticklabels(top10["label"].values[::-1], fontsize=10)
        ax.set_xlabel("MAE (°)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
        for i, (v, t) in enumerate(zip(top10[col_val].values[::-1], top10[col_test].values[::-1])):
            ax.text(v + 0.05, i, f"{v:.2f}°", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "top10.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'top10.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_dir", type=Path, default=Path("probe_exp"))
    parser.add_argument("--output_dir", type=Path, default=Path("probe_exp/model_selection"))
    parser.add_argument("--az_weight", type=float, default=1.0, help="Weight for azimuth MAE")
    parser.add_argument("--el_weight", type=float, default=1.0, help="Weight for elevation MAE")
    parser.add_argument("--pca_256", type=str, default="pca256_job_2628839")
    parser.add_argument("--pca_512", type=str, default="pca512_job_2628666")
    parser.add_argument("--pca_1024", type=str, default="pca1024_job_2628888")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        256: args.pca_256,
        512: args.pca_512,
        1024: args.pca_1024,
    }

    df = load_results(args.probe_dir, configs)
    summary = build_summary(df, args.az_weight, args.el_weight)

    summary.to_csv(args.output_dir / "model_selection_summary.csv", index=False, float_format="%.3f")
    print(f"\nSaved CSV: {args.output_dir / 'model_selection_summary.csv'}")

    print("\n=== TOP 10 configs (combined val MAE) ===")
    cols = ["n_pca", "layer", "timestep", "val_az", "val_el", "combined_val", "overfit_gap_az", "overfit_gap_el"]
    print(summary.head(10)[cols].to_string(index=False))

    best = summary.iloc[0]
    print(f"\n   BEST CONFIG: n_pca={best.n_pca} | {best.layer} | {best.timestep}")
    print(f"    val_az={best.val_az:.3f}° | val_el={best.val_el:.3f}° | combined={best.combined_val:.3f}°")
    print(f"    test_az={best.test_az:.3f}° | test_el={best.test_el:.3f}°")

    plot_heatmaps(summary, args.output_dir)
    plot_scatter(summary, args.output_dir)
    plot_top10(summary, args.output_dir)

    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()