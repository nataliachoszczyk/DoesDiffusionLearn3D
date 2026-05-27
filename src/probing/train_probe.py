import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

LAYERS = ["layer_04", "layer_10", "layer_18"]
TIMESTEPS = ["t_200", "t_500", "t_800"]
COMBINATIONS = [(l, t) for l in LAYERS for t in TIMESTEPS]

ACTIVATIONS_ROOT = Path("data/activations")
SPLITS_DIR = Path("data/splits")
RIDGE_ALPHA = 1.0


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("probe")
    logger.setLevel(logging.INFO)
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
        arr = np.load(path).squeeze(0)          # (1024, 2240) -> (1024, 2240)
        arr = arr.reshape(32, 32, 2240)          # spatial grid
        # 4x4 avg pool: (32,32,2240) -> (8,8,2240)
        arr = arr.reshape(8, 4, 8, 4, 2240).mean(axis=(1, 3))
        arrays.append(arr.reshape(-1))           # (143360,)
    return np.stack(arrays, axis=0).astype(np.float32)  # (N, 143360)


def angular_mae_azimuth(sin_pred, cos_pred, sin_true, cos_true) -> float:
    pred_deg = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360
    true_deg = np.degrees(np.arctan2(sin_true, cos_true)) % 360
    diff = np.abs(pred_deg - true_deg)
    diff = np.minimum(diff, 360 - diff)
    return float(np.mean(diff))


def evaluate(model, pca, X, df, logger, split_name: str, already_transformed: bool = False) -> dict:
    X_pca = X if already_transformed else pca.transform(X)
    preds = model.predict(X_pca)
    if hasattr(preds, "to_numpy"):
        preds = preds.to_numpy()

    sin_pred  = preds[:, 0]
    cos_pred  = preds[:, 1]
    elev_pred = preds[:, 2]

    mae_az = angular_mae_azimuth(
        sin_pred, cos_pred,
        df["sin_azimuth"].values,
        df["cos_azimuth"].values,
    )
    mae_el = float(mean_absolute_error(df["light_elevation_deg"].values, elev_pred))

    logger.info(f"  [{split_name}] azimuth MAE: {mae_az:.2f} deg | elevation MAE: {mae_el:.2f} deg")
    return {"mae_azimuth_deg": mae_az, "mae_elevation_deg": mae_el}


def run(task_id: int, exp_dir: Path):
    layer, timestep = COMBINATIONS[task_id]
    log_path = exp_dir / f"{layer}_{timestep}.log"
    logger = setup_logger(log_path)
    logger.info(f"=== task {task_id}: {layer} / {timestep} ===")

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df  = pd.read_csv(SPLITS_DIR / "test.csv")

    logger.info("Loading train activations...")
    X_train = load_activations(train_df, layer, timestep)

    logger.info(f"Fitting PCA (n={N_PCA_COMPONENTS}) on train...")
    pca = IncrementalPCA(n_components=args.n_pca, batch_size=600)
    logger.info(f"PCA components: {args.n_pca} | Ridge alpha: {RIDGE_ALPHA}")
    X_train_pca = pca.fit_transform(X_train)
    del X_train

    Y_train = train_df[["sin_azimuth", "cos_azimuth", "light_elevation_deg"]].values.astype(np.float32)

    logger.info("Training Ridge...")
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train_pca, Y_train)

    results = {"layer": layer, "timestep": timestep}
    results["train"] = evaluate(model, pca, X_train_pca, train_df, logger, "train", already_transformed=True)
    del X_train_pca

    logger.info("Loading val activations...")
    X_val = load_activations(val_df, layer, timestep)
    results["val"] = evaluate(model, pca, X_val, val_df, logger, "val")
    del X_val

    logger.info("Loading test activations...")
    X_test = load_activations(test_df, layer, timestep)
    results["test"] = evaluate(model, pca, X_test, test_df, logger, "test")
    del X_test

    model_path = exp_dir / f"{layer}_{timestep}_model.pkl"
    pca_path   = exp_dir / f"{layer}_{timestep}_pca.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    logger.info(f"Model saved to {model_path}")

    results_path = exp_dir / f"{layer}_{timestep}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--n_pca", type=int, default=512)
    args = parser.parse_args()

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    run(args.task_id, args.exp_dir)