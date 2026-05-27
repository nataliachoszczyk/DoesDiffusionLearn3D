import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error

ACTIVATIONS_ROOT = Path("data/activations")
SPLITS_DIR = Path("data/splits")
LAYER = "layer_10"
TIMESTEP = "t_500"
N_PCA = 512
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
OUTPUT_DIR = Path("probe_exp/ridge_alpha_analysis")


def load_activations(df, layer, timestep):
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


def angular_mae_azimuth(sin_pred, cos_pred, sin_true, cos_true):
    pred_deg = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360
    true_deg = np.degrees(np.arctan2(sin_true, cos_true)) % 360
    diff = np.abs(pred_deg - true_deg)
    return float(np.mean(np.minimum(diff, 360 - diff)))


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")

    print(f"Loading activations: {LAYER} / {TIMESTEP}")
    X_train = load_activations(train_df, LAYER, TIMESTEP)
    X_val = load_activations(val_df, LAYER, TIMESTEP)

    print(f"Fitting PCA (n={N_PCA})...")
    pca = IncrementalPCA(n_components=N_PCA, batch_size=600)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    del X_train, X_val

    Y_train = train_df[["sin_azimuth", "cos_azimuth", "light_elevation_deg"]].values.astype(np.float32)
    Y_val = val_df[["sin_azimuth", "cos_azimuth", "light_elevation_deg"]].values.astype(np.float32)

    print("Running RidgeCV...")
    model_cv = RidgeCV(alphas=ALPHAS, cv=5)
    model_cv.fit(X_train_pca, Y_train)
    print(f"Best alpha (CV): {model_cv.alpha_}")

    results = []
    for alpha in ALPHAS:
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=alpha)
        m.fit(X_train_pca, Y_train)
        preds = m.predict(X_val_pca)
        mae_az = angular_mae_azimuth(preds[:, 0], preds[:, 1], Y_val[:, 0], Y_val[:, 1])
        mae_el = float(mean_absolute_error(Y_val[:, 2], preds[:, 2]))
        print(f"  alpha={alpha:.3f} | azimuth MAE: {mae_az:.2f} deg | elevation MAE: {mae_el:.2f} deg")
        results.append({"alpha": alpha, "mae_azimuth": mae_az, "mae_elevation": mae_el})

    with open(OUTPUT_DIR / "alpha_results.json", "w") as f:
        json.dump({"best_alpha_cv": model_cv.alpha_, "results": results}, f, indent=2)

    alphas_plot = [r["alpha"] for r in results]
    mae_az_vals = [r["mae_azimuth"] for r in results]
    mae_el_vals = [r["mae_elevation"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, vals, title in zip(axes, [mae_az_vals, mae_el_vals], ["Azimuth MAE (deg)", "Elevation MAE (deg)"]):
        ax.semilogx(alphas_plot, vals, marker="o")
        ax.axvline(model_cv.alpha_, color="r", linestyle="--", label=f"CV best: {model_cv.alpha_}")
        ax.set_xlabel("Alpha")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs Ridge alpha — {LAYER}/{TIMESTEP}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ridge_alpha_analysis.png", dpi=150)
    print(f"Plot saved to {OUTPUT_DIR}/ridge_alpha_analysis.png")