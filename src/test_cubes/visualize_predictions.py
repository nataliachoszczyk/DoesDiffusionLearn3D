#!/usr/bin/env python3
"""
Overlay predicted and ground-truth shadow direction arrows on all images.
Arrow originates from detected purple object centroid, points toward shadow.
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


IMAGES_DIR = Path("data/final/images")
SPLITS_DIR = Path("data/splits")

AZ_RANGE = 360.0
EL_RANGE = 45.0

_CAM_POS   = np.array([7.358891, -6.925791, 4.959234])
_CAM_FWD   = -_CAM_POS / np.linalg.norm(_CAM_POS)
_CAM_RIGHT = np.cross(_CAM_FWD, np.array([0., 0., 1.]))
_CAM_RIGHT /= np.linalg.norm(_CAM_RIGHT)
_CAM_UP    = np.cross(_CAM_RIGHT, _CAM_FWD)


def az_el_to_shadow_2d(az_deg: float, el_deg: float) -> tuple[float, float]:
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    lx = math.cos(el) * math.cos(az)
    ly = math.cos(el) * math.sin(az)
    lz = math.sin(el)
    shadow = np.array([-lx, -ly, -lz])
    ir = float(np.dot(shadow, _CAM_RIGHT))
    iu = float(np.dot(shadow, _CAM_UP))
    length = math.sqrt(ir**2 + iu**2) + 1e-9
    return ir / length, iu / length


def find_object_centroid(img_array: np.ndarray) -> tuple[float, float] | None:
    """Detect purple object centroid via HSV color mask. Returns (cx, cy) in pixels."""
    r = img_array[:, :, 0].astype(np.float32)
    g = img_array[:, :, 1].astype(np.float32)
    b = img_array[:, :, 2].astype(np.float32)

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc + 1e-6

    # hue
    hue = np.zeros_like(r)
    m = maxc == r
    hue[m] = (60 * ((g[m] - b[m]) / delta[m])) % 360
    m = maxc == g
    hue[m] = 60 * ((b[m] - r[m]) / delta[m]) + 120
    m = maxc == b
    hue[m] = 60 * ((r[m] - g[m]) / delta[m]) + 240

    sat = np.where(maxc > 1e-3, delta / (maxc + 1e-6), 0.0)
    val = maxc / 255.0

    # purple/violet: hue 240-320, sat>0.25, val>0.15
    mask = (hue >= 240) & (hue <= 320) & (sat > 0.25) & (val > 0.15)

    if mask.sum() < 100:
        return None

    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("viz")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_image(row: dict, out_path: Path):
    img = Image.open(row["img_path"]).convert("RGB")
    arr = np.array(img)
    W, H = img.size

    centroid = find_object_centroid(arr)
    cx, cy = centroid if centroid is not None else (W / 2, H / 2)
    radius = W * 0.28

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis("off")

    # mark centroid
    ax.plot(cx, cy, "o", color="white", markersize=5,
            markeredgecolor="black", markeredgewidth=1.0, zorder=5)

    gt_ir,   gt_iu   = az_el_to_shadow_2d(row["gt_az"],   row["gt_el"])
    pred_ir, pred_iu = az_el_to_shadow_2d(row["pred_az"], row["pred_el"])

    for ir, iu, color, label in [
        (gt_ir,   gt_iu,   "#00e676", "GT"),
        (pred_ir, pred_iu, "#ff1744", "Pred"),
    ]:
        ax.annotate(
            "",
            xy=(cx + ir * radius, cy - iu * radius),
            xytext=(cx, cy),
            arrowprops=dict(
                arrowstyle="->, head_width=0.5, head_length=0.4",
                color=color, lw=3.0,
            ),
        )
        ax.plot([], [], color=color, lw=2.5, label=label)

    score = row["az_err"] / AZ_RANGE + row["el_err"] / EL_RANGE
    ax.set_title(
        f"GT  az={row['gt_az']:.1f}  el={row['gt_el']:.1f}\n"
        f"Pred az={row['pred_az']:.1f}  el={row['pred_el']:.1f}  "
        f"err az={row['az_err']:.1f}  el={row['el_err']:.1f}  score={score:.4f}",
        fontsize=8, pad=4,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def process_split(split: str, preds_csv: Path, out_dir: Path, logger: logging.Logger):
    if not preds_csv.exists():
        logger.warning(f"SKIP {split}: {preds_csv} not found")
        return

    df_split = pd.read_csv(SPLITS_DIR / f"{split}.csv")
    df_preds = pd.read_csv(preds_csv)
    df = df_split.merge(
        df_preds[["image_id", "pred_azimuth_deg", "pred_elevation_deg",
                  "az_error_deg", "el_error_deg"]],
        on="image_id", how="inner",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n=== {split}: {len(df)} samples ===")

    missing, saved, no_centroid = 0, 0, 0
    for _, r in df.iterrows():
        img_path = IMAGES_DIR / r["shape"] / f"{r['image_id']}.png"
        if not img_path.exists():
            logger.warning(f"  MISSING image: {img_path}")
            missing += 1
            continue

        az_err = r["az_error_deg"]
        el_err = r["el_error_deg"]
        score  = az_err / AZ_RANGE + el_err / EL_RANGE
        fname  = f"{r['image_id']}_az{az_err:.1f}_el{el_err:.1f}_s{score:.4f}.png"

        img_arr = np.array(Image.open(img_path).convert("RGB"))
        if find_object_centroid(img_arr) is None:
            no_centroid += 1

        save_image({
            "img_path": img_path,
            "gt_az":   r["light_azimuth_deg"],
            "gt_el":   r["light_elevation_deg"],
            "pred_az": r["pred_azimuth_deg"],
            "pred_el": r["pred_elevation_deg"],
            "az_err":  az_err,
            "el_err":  el_err,
        }, out_dir / fname)
        saved += 1

    logger.info(f"  saved={saved}  missing={missing}  no_centroid={no_centroid}  output={out_dir}")

    df_sorted = df.sort_values("az_error_deg").reset_index(drop=True)
    logger.info("  Top 5 best az_err:  " +
                "  ".join(f"{r.image_id}({r.az_error_deg:.1f})"
                          for r in df_sorted.head(5).itertuples()))
    logger.info("  Top 5 worst az_err: " +
                "  ".join(f"{r.image_id}({r.az_error_deg:.1f})"
                          for r in df_sorted.tail(5).itertuples()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",           default="layer_10")
    parser.add_argument("--timestep",        default="t_200")
    parser.add_argument("--cubes_preds_csv", type=Path,
        default=Path("probe_exp/test_cubes_eval_all/layer_10_t_200/test_cubes_predictions.csv"))
    parser.add_argument("--test_preds_csv",  type=Path,
        default=Path("probe_exp/pca256_job_2628839/layer_10_t_200_test_predictions.csv"))
    parser.add_argument("--out_dir",         type=Path,
        default=Path("probe_exp/test_cubes_eval_all/summary/pred_viz"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.out_dir / "visualize.log")
    logger.info(f"Config: {args.layer} / {args.timestep}")

    process_split("test_cubes", args.cubes_preds_csv, args.out_dir / "test_cubes", logger)
    process_split("test",       args.test_preds_csv,  args.out_dir / "test",       logger)

    logger.info("\n=== Done ===")


if __name__ == "__main__":
    main()