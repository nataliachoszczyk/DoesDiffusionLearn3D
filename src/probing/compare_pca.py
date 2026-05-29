import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

PROBE_DIR = Path("probe_exp")
CONFIGS = {256: "pca256_job_2628839", 512: "pca512_job_2628666", 1024: "pca1024_job_2628888"}
LAYERS = ["layer_04", "layer_10", "layer_18"]
TIMESTEPS = ["t_200", "t_500", "t_800"]
OUTPUT_DIR = Path("probe_exp/pca_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = []
for n_pca, folder in CONFIGS.items():
    for layer in LAYERS:
        for t in TIMESTEPS:
            path = PROBE_DIR / folder / f"{layer}_{t}_results.json"
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

df = pd.DataFrame(records)
df.to_csv(OUTPUT_DIR / "results.csv", index=False)

val_df = df[df["split"] == "val"].copy()
val_df["combo"] = val_df["layer"] + "\n" + val_df["timestep"]
combos = [f"{l}\n{t}" for l in LAYERS for t in TIMESTEPS]
colors = {256: "tab:blue", 512: "tab:orange", 1024: "tab:green"}

for col, ylabel, fname in [
    ("mae_azimuth", "Azimuth MAE (deg)", "line_azimuth.png"),
    ("mae_elevation", "Elevation MAE (deg)", "line_elevation.png"),
]:
    fig, ax = plt.subplots(figsize=(12, 5))
    for n_pca in [256, 512, 1024]:
        sub = val_df[val_df["n_pca"] == n_pca].set_index("combo").reindex(combos)
        ax.plot(combos, sub[col].values, marker="o", label=f"n_pca={n_pca}", color=colors[n_pca])
    ax.set_xlabel("Layer / Timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} on val set — PCA comparison")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / fname}")

for n_pca in [256, 512, 1024]:
    for col, title_prefix in [("mae_azimuth", "Azimuth"), ("mae_elevation", "Elevation")]:
        sub = val_df[val_df["n_pca"] == n_pca].pivot(index="layer", columns="timestep", values=col)
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(sub.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(sub.columns)))
        ax.set_xticklabels(sub.columns)
        ax.set_yticks(range(len(sub.index)))
        ax.set_yticklabels(sub.index)
        for i in range(sub.shape[0]):
            for j in range(sub.shape[1]):
                ax.text(j, i, f"{sub.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{title_prefix} MAE val — n_pca={n_pca}")
        plt.tight_layout()
        fname = OUTPUT_DIR / f"heatmap_{col}_pca{n_pca}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved: {fname}")

print("\n=== Best config per n_pca (val azimuth MAE) ===")
print(val_df.loc[val_df.groupby("n_pca")["mae_azimuth"].idxmin()][["n_pca", "layer", "timestep", "mae_azimuth", "mae_elevation"]])