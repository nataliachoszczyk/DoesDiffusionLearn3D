#!/usr/bin/env python3
"""
Evaluate best probe (layer_10 / t_200 / n_pca=256) on test_cubes split.
Loads saved Ridge + PCA, runs inference, saves results + plots + CSV + summary txt.
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

ACTIVATIONS_ROOT = Path("data/activations")
SPLITS_DIR = Path("data/splits")
DEFAULT_MODEL_DIR = Path("probe_exp/pca256_job_2628839")
DEFAULT_OUT_DIR = Path("probe_exp/test_cubes_eval")


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("eval_cubes")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def load_activations(df: pd.DataFrame, layer: str, timestep: str) -> np.ndarray:
    arrays = []
    layer_num = layer.split("_")[1]
    t_num = timestep.split("_")[1]
    for image_id, shape in zip(df["image_id"], df["shape"]):
        fname = f"{image_id}_l{layer_num}_t{t_num}.npy"
        path = ACTIVATIONS_ROOT / layer / timestep / shape / fname
        arr = np.load(path).squeeze(0)
        arr = arr.reshape(32, 32, 2240)
        arr = arr.reshape(8, 4, 8, 4, 2240).mean(axis=(1, 3))
        arrays.append(arr.reshape(-1))
    return np.stack(arrays, axis=0).astype(np.float32)


def angular_mae(sin_pred, cos_pred, sin_true, cos_true) -> tuple:
    pred_deg = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360
    true_deg = np.degrees(np.arctan2(sin_true, cos_true)) % 360
    diff = np.abs(pred_deg - true_deg)
    diff = np.minimum(diff, 360 - diff)
    return float(np.mean(diff)), pred_deg, true_deg


def baseline_random_mae(true_deg: np.ndarray, n_trials: int = 100) -> float:
    maes = []
    for _ in range(n_trials):
        rand = np.random.uniform(0, 360, size=len(true_deg))
        diff = np.abs(rand - true_deg)
        diff = np.minimum(diff, 360 - diff)
        maes.append(np.mean(diff))
    return float(np.mean(maes))


def baseline_mean_mae(true_deg: np.ndarray, train_mean_deg: float) -> float:
    diff = np.abs(true_deg - train_mean_deg)
    diff = np.minimum(diff, 360 - diff)
    return float(np.mean(diff))


def plot_scatter(pred_deg, true_deg, pred_el, true_el, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, p, t, title, lim in [
        (axes[0], pred_deg, true_deg, "Azimuth — Predicted vs Ground Truth (SmoothCube_v2)", 360),
        (axes[1], pred_el, true_el, "Elevation — Predicted vs Ground Truth (SmoothCube_v2)", 90),
    ]:
        ax.scatter(t, p, alpha=0.4, s=18, color="#2196F3", edgecolors="none")
        ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Ground Truth (°)", fontsize=12)
        ax.set_ylabel("Predicted (°)", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_predicted_vs_gt.png", dpi=150)
    plt.close()


def plot_error_distribution(az_errors, el_errors, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, errors, title, color in [
        (axes[0], az_errors, "Azimuth Error Distribution (SmoothCube_v2)", "#2196F3"),
        (axes[1], el_errors, "Elevation Error Distribution (SmoothCube_v2)", "#4CAF50"),
    ]:
        ax.hist(errors, bins=40, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(np.mean(errors), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {np.mean(errors):.1f}°")
        ax.axvline(np.median(errors), color="orange", linestyle="--", linewidth=1.5,
                   label=f"Median: {np.median(errors):.1f}°")
        ax.set_xlabel("Absolute Error (°)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "error_distribution.png", dpi=150)
    plt.close()


def load_prior_results(model_dir: Path, layer: str, timestep: str) -> dict:
    path = model_dir / f"{layer}_{timestep}_results.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_predictions_csv(
    df: pd.DataFrame,
    sin_pred: np.ndarray,
    cos_pred: np.ndarray,
    elev_pred: np.ndarray,
    pred_deg: np.ndarray,
    az_errors: np.ndarray,
    el_errors: np.ndarray,
    out_dir: Path,
):
    out = df[["image_id", "shape",
              "light_azimuth_deg", "light_elevation_deg",
              "sin_azimuth", "cos_azimuth"]].copy()
    out["pred_sin_azimuth"]       = sin_pred
    out["pred_cos_azimuth"]       = cos_pred
    out["pred_azimuth_deg"]       = pred_deg
    out["pred_elevation_deg"]     = elev_pred
    out["az_error_deg"]           = az_errors
    out["el_error_deg"]           = el_errors
    out = out.round(4)
    csv_path = out_dir / "test_cubes_predictions.csv"
    out.to_csv(csv_path, index=False)
    return csv_path


def save_summary_txt(
    results: dict,
    mae_az: float,
    mae_el: float,
    rand_mae_az: float,
    mean_mae_az: float,
    rand_mae_el: float,
    mean_mae_el: float,
    az_errors: np.ndarray,
    el_errors: np.ndarray,
    layer: str,
    timestep: str,
    out_dir: Path,
    prior: dict,
):
    median_el = float(np.median(el_errors))
    p90_el    = float(np.percentile(el_errors, 90))

    W = 72

    def row(label, az, el):
        return f"  {label:<28} {az:>12} {el:>12}"

    def fmt(d, split, metric):
        try:
            return f"{d[split][metric]:>11.2f}°"
        except (KeyError, TypeError):
            return f"{'N/A':>12}"

    if prior:
        train_az = fmt(prior, "train", "mae_azimuth_deg")
        train_el = fmt(prior, "train", "mae_elevation_deg")
        val_az   = fmt(prior, "val",   "mae_azimuth_deg")
        val_el   = fmt(prior, "val",   "mae_elevation_deg")
        test_az  = fmt(prior, "test",  "mae_azimuth_deg")
        test_el  = fmt(prior, "test",  "mae_elevation_deg")
    else:
        train_az = train_el = val_az = val_el = test_az = test_el = f"{'N/A':>12}"

    SEP  = "=" * W
    DASH = "-" * W

    lines = [
        SEP,
        "  RESULTS SUMMARY — SANA-1.6B LINEAR PROBING EXPERIMENT",
        SEP,
        "",
        f"  Config   :  {layer} / {timestep} / n_pca=256",
        f"  Split    :  test_cubes  (SmoothCube_v2, unseen during training)",
        f"  N        :  {results['n_samples']} samples",
        "",
        DASH,
        "  PROBE PERFORMANCE ON TEST_CUBES",
        DASH,
        row("Metric", "Azimuth", "Elevation"),
        "  " + "-" * (W - 2),
        row("MAE (mean)",   f"{mae_az:>11.2f}°",                           f"{mae_el:>11.2f}°"),
        row("MAE (median)", f"{results['probe']['median_az_error']:>11.2f}°", f"{median_el:>11.2f}°"),
        row("MAE (p90)",    f"{results['probe']['p90_az_error']:>11.2f}°",    f"{p90_el:>11.2f}°"),
        "",
        DASH,
        "  BASELINE COMPARISON",
        DASH,
        row("Method", "Az MAE", "El MAE"),
        "  " + "-" * (W - 2),
        row("Probe  (Ridge + PCA)",  f"{mae_az:>11.2f}°",   f"{mae_el:>11.2f}°"),
        row("Baseline: random",      f"{rand_mae_az:>11.2f}°", f"{rand_mae_el:>11.2f}°"),
        row("Baseline: always-mean", f"{mean_mae_az:>11.2f}°", f"{mean_mae_el:>11.2f}°"),
        "",
        f"  Improvement vs random  :  {rand_mae_az - mae_az:+.2f}° azimuth",
        f"  Improvement vs mean    :  {mean_mae_az - mae_az:+.2f}° azimuth",
        "",
        DASH,
        "  GENERALIZATION ACROSS SPLITS",
        DASH,
        f"  {'Split':<22} {'Shapes':<24} {'Az MAE':>10} {'El MAE':>10}",
        "  " + "-" * (W - 2),
        f"  {'train':<22} {'Sphere, SmoothCylinder':<24} {train_az} {train_el}",
        f"  {'val':<22} {'Sphere, SmoothCylinder':<24} {val_az} {val_el}",
        f"  {'test':<22} {'Sphere, SmoothCylinder':<24} {test_az} {test_el}",
        f"  {'test_cubes  (*)':<22} {'SmoothCube_v2':<24} {mae_az:>11.2f}° {mae_el:>11.2f}°",
        "",
        "  (*) Zero-shot generalization — SmoothCube_v2 never seen during training",
        "",
        DASH,
        "  INTERPRETATION",
        DASH,
        f"  The probe achieves {mae_az:.1f}° mean azimuth MAE on unseen cube geometry,",
        f"  compared to {rand_mae_az:.1f}° for a random baseline — an improvement of",
        f"  {rand_mae_az - mae_az:.1f}°. The median error is only",
        f"  {results['probe']['median_az_error']:.1f}°.",
        SEP,
    ]

    out_path = out_dir / "results_summary.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out_dir",   type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--layer",     type=str,  default="layer_10")
    parser.add_argument("--timestep",  type=str,  default="t_200")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.out_dir / "eval_cubes.log")

    model_path = args.model_dir / f"{args.layer}_{args.timestep}_model.pkl"
    pca_path   = args.model_dir / f"{args.layer}_{args.timestep}_pca.pkl"

    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    prior = load_prior_results(args.model_dir, args.layer, args.timestep)
    if prior:
        logger.info(f"Loaded prior results from {args.model_dir / f'{args.layer}_{args.timestep}_results.json'}")
    else:
        logger.info("No prior results JSON found — generalization table will show N/A.")

    test_cubes_df = pd.read_csv(SPLITS_DIR / "test_cubes.csv")
    train_df      = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df        = pd.read_csv(SPLITS_DIR / "val.csv")

    logger.info(f"test_cubes samples: {len(test_cubes_df)}")
    logger.info("Loading test_cubes activations...")
    X_cubes     = load_activations(test_cubes_df, args.layer, args.timestep)
    X_cubes_pca = pca.transform(X_cubes)

    preds = model.predict(X_cubes_pca)
    sin_pred, cos_pred, elev_pred = preds[:, 0], preds[:, 1], preds[:, 2]

    mae_az, pred_deg, true_deg = angular_mae(
        sin_pred, cos_pred,
        test_cubes_df["sin_azimuth"].values,
        test_cubes_df["cos_azimuth"].values,
    )
    true_el = test_cubes_df["light_elevation_deg"].values
    mae_el  = float(mean_absolute_error(true_el, elev_pred))

    az_errors = np.minimum(np.abs(pred_deg - true_deg), 360 - np.abs(pred_deg - true_deg))
    el_errors = np.abs(elev_pred - true_el)

    train_full    = pd.concat([train_df, val_df], ignore_index=True)
    train_az_deg  = np.degrees(np.arctan2(
        train_full["sin_azimuth"].values,
        train_full["cos_azimuth"].values,
    )) % 360
    train_el_mean = float(train_full["light_elevation_deg"].mean())

    rand_mae_az = baseline_random_mae(true_deg)
    mean_mae_az = baseline_mean_mae(true_deg, float(np.mean(train_az_deg)))
    rand_mae_el = float(mean_absolute_error(true_el, np.random.uniform(30, 75, size=len(true_el))))
    mean_mae_el = float(mean_absolute_error(true_el, np.full_like(true_el, train_el_mean)))

    logger.info(f"\n{'=' * 55}")
    logger.info(f"  TEST GENERALIZATION: {args.layer} / {args.timestep}")
    logger.info(f"{'=' * 55}")
    logger.info(f"  Probe  — azimuth MAE: {mae_az:.2f}°  | elevation MAE: {mae_el:.2f}°")
    logger.info(f"  Baseline random     : {rand_mae_az:.2f}°  | {rand_mae_el:.2f}°")
    logger.info(f"  Baseline mean       : {mean_mae_az:.2f}°  | {mean_mae_el:.2f}°")
    logger.info(f"{'=' * 55}")
    logger.info(f"  Improvement vs random: {rand_mae_az - mae_az:.2f}° azimuth")
    logger.info(f"  Improvement vs mean:   {mean_mae_az - mae_az:.2f}° azimuth")

    results = {
        "layer": args.layer,
        "timestep": args.timestep,
        "split": "test_cubes",
        "n_samples": len(test_cubes_df),
        "probe": {
            "mae_azimuth_deg":   mae_az,
            "mae_elevation_deg": mae_el,
            "median_az_error":   float(np.median(az_errors)),
            "p90_az_error":      float(np.percentile(az_errors, 90)),
            "median_el_error":   float(np.median(el_errors)),
            "p90_el_error":      float(np.percentile(el_errors, 90)),
        },
        "baselines": {
            "random_mae_azimuth_deg":   rand_mae_az,
            "mean_mae_azimuth_deg":     mean_mae_az,
            "random_mae_elevation_deg": rand_mae_el,
            "mean_mae_elevation_deg":   mean_mae_el,
        },
    }

    with open(args.out_dir / "test_cubes_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.out_dir / 'test_cubes_results.json'}")

    csv_path = save_predictions_csv(
        df=test_cubes_df,
        sin_pred=sin_pred,
        cos_pred=cos_pred,
        elev_pred=elev_pred,
        pred_deg=pred_deg,
        az_errors=az_errors,
        el_errors=el_errors,
        out_dir=args.out_dir,
    )
    logger.info(f"Per-sample predictions saved to {csv_path}")

    txt_path = save_summary_txt(
        results=results,
        mae_az=mae_az,
        mae_el=mae_el,
        rand_mae_az=rand_mae_az,
        mean_mae_az=mean_mae_az,
        rand_mae_el=rand_mae_el,
        mean_mae_el=mean_mae_el,
        az_errors=az_errors,
        el_errors=el_errors,
        layer=args.layer,
        timestep=args.timestep,
        out_dir=args.out_dir,
        prior=prior,
    )
    logger.info(f"Summary TXT saved to {txt_path}")

    plot_scatter(pred_deg, true_deg, elev_pred, true_el, args.out_dir)
    plot_error_distribution(az_errors, el_errors, args.out_dir)
    logger.info("Plots saved.")
    logger.info(f"All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()