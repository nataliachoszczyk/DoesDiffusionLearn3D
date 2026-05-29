import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

SCENES_ROOT = Path("data/final/scenes")
OUTPUT_CSV  = Path("data/master_labels.csv")

SHAPES = ["Sphere", "SmoothCylinder", "SmoothCube_v2"]

records = []

for shape in SHAPES:
    pattern = str(SCENES_ROOT / shape / f"{shape}_new_*.json")
    for json_path in sorted(glob.glob(pattern)):
        with open(json_path) as f:
            meta = json.load(f)

        azimuth   = meta["light_azimuth_deg"]
        elevation = meta["light_elevation_deg"]
        image_id  = Path(json_path).stem

        records.append({
            "image_id":            image_id,
            "shape":               shape,
            "light_azimuth_deg":   azimuth,
            "light_elevation_deg": elevation,
            "sin_azimuth":         np.sin(np.radians(azimuth)),
            "cos_azimuth":         np.cos(np.radians(azimuth)),
        })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df)} rows to {OUTPUT_CSV}")