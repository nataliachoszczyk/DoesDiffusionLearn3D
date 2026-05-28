#!/usr/bin/env python3
"""
Overlay predicted and ground-truth light direction arrows on all images
from test and test_cubes splits. Saves one PNG per image.
Arrow points FROM object center TOWARD shadow (opposite of lamp position).
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


IMAGES_DIR = Path("data/final/images")
SPLITS_DIR = Path("data/splits")

AZ_RANGE = 360.0
EL_RANGE = 45.0

import numpy as np

_CAM_POS   = np.array([7.358891, -6.925791, 4.959234])
_CAM_FWD   = -_CAM_POS / np.linalg.norm(_CAM_POS)
_CAM_RIGHT = np.cross(_CAM_FWD, np.array([0., 0., 1.]))
_CAM_RIGHT /= np.linalg.norm(_CAM_RIGHT)
_CAM_UP    = np.cross(_CAM_RIGHT, _CAM_FWD)


def az_el_to_shadow_2d(az_deg, el_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    # full 3D lamp direction (unit vector)
    lx = math.cos(el) * math.cos(az)
    ly = math.cos(el) * math.sin(az)
    lz = math.sin(el)
    lamp_3d = np.array([lx, ly, lz])
    # shadow direction = opposite of lamp, then project onto image
    shadow = -lamp_3d
    ir = float(np.dot(shadow, _CAM_RIGHT))
    iu = float(np.dot(shadow, _CAM_UP))
    length = math.sqrt(ir**2 + iu**2) + 1e-9
    return ir / length, iu / length


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
    W, H = img.size

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis("off")

    cx, cy = W / 2, H / 2
    radius = W * 0.28

    gt_dx, gt_dy   = az_el_to_shadow_2d(row["gt_az"],   row["gt_el"])
    pred_dx, pred_dy = az_el_to_shadow_2d(row["pred_az"], row["pred_el"])

    # dy is image-up (+), but matplotlib Y axis points DOWN -> negate dy
    for dx, dy, color, label in [
        (gt_dx,   gt_dy,   "#00e676", "GT"),
        (pred_dx, pred_dy, "#ff1744", "Pred"),
    ]:
        ax.annotate(
            "",
            xy=(cx + dx * radius, cy - dy * radius),   # -dy: image Y is flipped
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

    missing, saved = 0, 0
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

    logger.info(f"  saved={saved}  missing={missing}  output={out_dir}")

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