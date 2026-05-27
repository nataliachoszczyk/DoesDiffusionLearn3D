import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

ACTIVATIONS_ROOT = Path("data/activations")
SPLITS_DIR = Path("data/splits")
LAYER = "layer_10"
TIMESTEP = "t_500"
MAX_COMPONENTS = 1024
OUTPUT_DIR = Path("probe_exp/pca_analysis")


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


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    print(f"Loading activations: {LAYER} / {TIMESTEP}")
    X = load_activations(train_df, LAYER, TIMESTEP)
    print(f"Shape: {X.shape}")

    pca = PCA(n_components=MAX_COMPONENTS, svd_solver="randomized", random_state=42)
    pca.fit(X)

    cumvar = pca.explained_variance_ratio_.cumsum()
    for thresh in [0.80, 0.90, 0.95, 0.99]:
        n = int((cumvar < thresh).sum()) + 1
        print(f"{thresh*100:.0f}% variance explained by {n} components")

    np.save(OUTPUT_DIR / f"cumvar_{LAYER}_{TIMESTEP}.npy", cumvar)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(pca.explained_variance_ratio_[:200])
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title(f"Scree plot — {LAYER} / {TIMESTEP}")
    axes[0].grid(True)

    axes[1].plot(cumvar)
    axes[1].axhline(0.95, color="r", linestyle="--", label="95%")
    axes[1].axhline(0.90, color="orange", linestyle="--", label="90%")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title("Cumulative variance")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"pca_variance_{LAYER}_{TIMESTEP}.png", dpi=150)
    print(f"Plot saved to {OUTPUT_DIR}/pca_variance_{LAYER}_{TIMESTEP}.png")