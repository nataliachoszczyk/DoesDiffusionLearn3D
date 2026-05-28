#!/usr/bin/env python3
"""
Evaluate probe on any split and save per-sample predictions CSV + metrics JSON.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


ACTIVATIONS_ROOT = Path("data/activations")
SPLITS_DIR       = Path("data/splits")
DEFAULT_MODEL_DIR = Path("probe_exp/pca256_job_2628839")


def load_activations(df: pd.DataFrame, layer: str, timestep: str) -> np.ndarray:
    arrays = []
    layer_num = layer.split("_")[1]
    t_num     = timestep.split("_")[1]
    for image_id, shape in zip(df["image_id"], df["shape"]):
        fname = f"{image_id}_l{layer_num}_t{t_num}.npy"
        path  = ACTIVATIONS_ROOT / layer / timestep / shape / fname
        arr   = np.load(path).squeeze(0)
        arr   = arr.reshape(32, 32, 2240)
        arr   = arr.reshape(8, 4, 8, 4, 2240).mean(axis=(1, 3))
        arrays.append(arr.reshape(-1))
    return np.stack(arrays, axis=0).astype(np.float32)


def angular_mae(sin_pred, cos_pred, sin_true, cos_true):
    pred_deg = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360
    true_deg = np.degrees(np.arctan2(sin_true, cos_true)) % 360
    diff = np.abs(pred_deg - true_deg)
    diff = np.minimum(diff, 360 - diff)
    return float(np.mean(diff)), pred_deg, true_deg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--layer",     type=str,  default="layer_10")
    parser.add_argument("--timestep",  type=str,  default="t_200")
    parser.add_argument("--split",     type=str,  default="test")
    args = parser.parse_args()

    model_path = args.model_dir / f"{args.layer}_{args.timestep}_model.pkl"
    pca_path   = args.model_dir / f"{args.layer}_{args.timestep}_pca.pkl"
    out_csv    = args.model_dir / f"{args.layer}_{args.timestep}_{args.split}_predictions.csv"
    out_json   = args.model_dir / f"{args.layer}_{args.timestep}_{args.split}_metrics.json"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    df = pd.read_csv(SPLITS_DIR / f"{args.split}.csv")
    print(f"Loaded {len(df)} samples from split '{args.split}'")

    print("Loading activations...")
    X     = load_activations(df, args.layer, args.timestep)
    X_pca = pca.transform(X)

    preds                    = model.predict(X_pca)
    sin_pred, cos_pred, elev_pred = preds[:, 0], preds[:, 1], preds[:, 2]

    mae_az, pred_deg, true_deg = angular_mae(
        sin_pred, cos_pred,
        df["sin_azimuth"].values,
        df["cos_azimuth"].values,
    )
    true_el = df["light_elevation_deg"].values
    mae_el  = float(mean_absolute_error(true_el, elev_pred))

    az_errors = np.minimum(np.abs(pred_deg - true_deg), 360 - np.abs(pred_deg - true_deg))
    el_errors = np.abs(elev_pred - true_el)

    out = df[["image_id", "shape", "light_azimuth_deg", "light_elevation_deg",
              "sin_azimuth", "cos_azimuth"]].copy()
    out["pred_sin_azimuth"]  = sin_pred
    out["pred_cos_azimuth"]  = cos_pred
    out["pred_azimuth_deg"]  = pred_deg
    out["pred_elevation_deg"] = elev_pred
    out["az_error_deg"]      = az_errors
    out["el_error_deg"]      = el_errors
    out = out.round(4)
    out.to_csv(out_csv, index=False)

    metrics = {
        "split":             args.split,
        "layer":             args.layer,
        "timestep":          args.timestep,
        "n_samples":         len(df),
        "mae_azimuth_deg":   mae_az,
        "mae_elevation_deg": mae_el,
        "median_az_error":   float(np.median(az_errors)),
        "median_el_error":   float(np.median(el_errors)),
        "p90_az_error":      float(np.percentile(az_errors, 90)),
        "p90_el_error":      float(np.percentile(el_errors, 90)),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Az MAE : {mae_az:.4f}°")
    print(f"El MAE : {mae_el:.4f}°")
    print(f"Saved predictions : {out_csv}")
    print(f"Saved metrics     : {out_json}")


if __name__ == "__main__":
    main()