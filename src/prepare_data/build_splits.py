# src/prepare_data/build_splits.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

LABELS_CSV  = Path("data/master_labels.csv")
SPLITS_DIR  = Path("data/splits")
RANDOM_SEED = 42

SPLITS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(LABELS_CSV)

# test_cubes: all SmoothCube_v2
test_cubes = df[df["shape"] == "SmoothCube_v2"].copy()

# train/val/test split on Sphere + SmoothCylinder, stratified by shape
seen = df[df["shape"] != "SmoothCube_v2"].copy()

train, temp = train_test_split(seen, test_size=0.2, stratify=seen["shape"], random_state=RANDOM_SEED)
val, test   = train_test_split(temp, test_size=0.5, stratify=temp["shape"], random_state=RANDOM_SEED)

for name, split_df in [("train", train), ("val", val), ("test", test), ("test_cubes", test_cubes)]:
    path = SPLITS_DIR / f"{name}.csv"
    split_df.to_csv(path, index=False)
    print(f"{name}: {len(split_df)} rows -> {path}")
    print(split_df["shape"].value_counts().to_string(), "\n")