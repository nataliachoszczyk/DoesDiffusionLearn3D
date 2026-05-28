#!/usr/bin/env python3
"""
Overlay predicted and ground-truth shadow direction arrows on all images.
Uses pixel_coords from scene JSON for arrow origin.
Uses actual 3d_coords of object relative to lamp for correct shadow direction.
"""

import argparse
import json
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


IMAGES_DIR  = Path("data/final/images")
SPLITS_DIR  = Path("data/splits")
SCENES_DIR  = Path("data/final/scenes")

AZ_RANGE = 360.0
EL_RANGE = 45.0

_CAM_POS   = np.array([7.358891, -6.925791, 4.959234])
_CAM_FWD   = -_CAM_POS / np.linalg.norm(_CAM_POS)
_CAM_RIGHT = np.cross(_CAM_FWD, np.array([0., 0., 1.]))
_CAM_RIGHT /= np.linalg.norm(_CAM_RIGHT)
_CAM_UP    = np.cross(_CAM_RIGHT, _CAM_FWD)

LIGHT_RADIUS = 5.0


def az_el_to_lamp_pos(az_deg: float, el_deg: float, r: float = LIGHT_RADIUS) -> np.ndarray:
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    return np.array([
        r * math.cos(el) * math.cos(az),
        r * math.cos(el) * math.sin(az),
        r * math.sin(el),
    ])


def shadow_dir_2d(obj_3d: np.ndarray, lamp_pos: np.ndarray) -> tuple[float, float]:
    """
    Shadow direction = from lamp through object, projected onto ground (z=0),
    then projected onto image axes.
    """
    # vector from lamp to object
    to_obj = obj_3d - lamp_pos
    # project onto ground plane (z=0): the shadow tip on the floor
    # parametric: lamp + t*(to_obj) where z=0 -> t = -lamp_z / to_obj_z
    if abs(to_obj[2]) < 1e-6:
        shadow_ground = np.array([to_obj[0], to_obj[1], 0.0])
    else:
        t = -lamp_pos[2] / to_obj[2]
        shadow_tip = lamp_pos + t * to_obj
        shadow_ground = shadow_tip - np.array([obj_3d[0], obj_3d[1], 0.0])

    # project onto image
    ir = float(np.dot(shadow_ground, _CAM_RIGHT))
    iu = float(np.dot(shadow_ground, _CAM_UP))
    length = math.sqrt(ir**2 + iu**2) + 1e-9
    return ir / length, iu / length


def load_scene(image_id: str, shape: str) -> dict | None:
    # scene JSON: same name as image but .json, under SCENES_DIR/shape/
    p = SCENES_DIR / shape / f"{image_id}.json"
    if not p.exists():
        # fallback: flat scenes dir
        p = SCENES_DIR / f"{image_id}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


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

    # origin of arrow: pixel_coords from JSON if available, else image center
    if row["pixel_coords"] is not None:
        cx, cy = row["pixel_coords"][0], row["pixel_coords"][1]
    else:
        cx, cy = W / 2.0, H / 2.0

    radius = W * 0.28

    obj_3d   = row["obj_3d"]
    gt_lamp  = az_el_to_lamp_pos(row["gt_az"],   row["gt_el"])
    pred_lamp= az_el_to_lamp_pos(row["pred_az"],  row["pred_el"])

    if obj_3d is not None:
        gt_ir,   gt_iu   = shadow_dir_2d(obj_3d, gt_lamp)
        pred_ir, pred_iu = shadow_dir_2d(obj_3d, pred_lamp)
    else:
        # fallback: object at origin
        def _simple(az, el):
            az_r = math.radians(az); el_r = math.radians(el)
            lx = math.cos(el_r)*math.cos(az_r)
            ly = math.cos(el_r)*math.sin(az_r)
            lz = math.sin(el_r)
            s = np.array([-lx, -ly, -lz])
            ir = float(np.dot(s, _CAM_RIGHT))
            iu = float(np.dot(s, _CAM_UP))
            l = math.sqrt(ir**2 + iu**2) + 1e-9
            return ir/l, iu/l
        gt_ir,   gt_iu   = _simple(row["gt_az"],   row["gt_el"])
        pred_ir, pred_iu = _simple(row["pred_az"],  row["pred_el"])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis("off")

    ax.plot(cx, cy, "o", color="white", markersize=5,
            markeredgecolor="black", markeredgewidth=1.0, zorder=5)

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

    missing, saved, no_scene = 0, 0, 0
    for _, r in df.iterrows():
        img_path = IMAGES_DIR / r["shape"] / f"{r['image_id']}.png"
        if not img_path.exists():
            logger.warning(f"  MISSING image: {img_path}")
            missing += 1
            continue

        scene = load_scene(r["image_id"], r["shape"])
        if scene is not None and scene.get("objects"):
            obj      = scene["objects"][0]
            obj_3d   = np.array(obj["3d_coords"])
            px_coords= obj["pixel_coords"]  # [px_x, px_y, depth]
        else:
            obj_3d    = None
            px_coords = None
            no_scene += 1

        az_err = r["az_error_deg"]
        el_err = r["el_error_deg"]
        score  = az_err / AZ_RANGE + el_err / EL_RANGE
        fname  = f"{r['image_id']}_az{az_err:.1f}_el{el_err:.1f}_s{score:.4f}.png"

        save_image({
            "img_path":    img_path,
            "gt_az":       r["light_azimuth_deg"],
            "gt_el":       r["light_elevation_deg"],
            "pred_az":     r["pred_azimuth_deg"],
            "pred_el":     r["pred_elevation_deg"],
            "az_err":      az_err,
            "el_err":      el_err,
            "obj_3d":      obj_3d,
            "pixel_coords":px_coords,
        }, out_dir / fname)
        saved += 1

    logger.info(f"  saved={saved}  missing={missing}  no_scene={no_scene}  output={out_dir}")

    df_sorted = df.sort_values("az_error_deg").reset_index(drop=True)
    logger.info("  Top 5 best az_err:  " +
                "  ".join(f"{r.image_id}({r.az_error_deg:.1f})"
                          for r in df_sorted.head(5).itertuples()))
    logger.info("  Top 5 worst az_err: " +
                "  ".join(f"{r.image_id}({r.az_error_deg:.1f})"
                          for r in df_sorted.tail(5).itertuples()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",    default="layer_10")
    parser.add_argument("--timestep", default="t_200")
    parser.add_argument("--cubes_preds_csv", type=Path,
        default=Path("probe_exp/test_cubes_eval_all/layer_10_t_200/test_cubes_predictions.csv"))
    parser.add_argument("--test_preds_csv", type=Path,
        default=Path("probe_exp/pca256_job_2628839/layer_10_t_200_test_predictions.csv"))
    parser.add_argument("--out_dir", type=Path,
        default=Path("probe_exp/test_cubes_eval_all/summary/pred_viz"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.out_dir / "visualize.log")

    process_split("test_cubes", args.cubes_preds_csv, args.out_dir / "test_cubes", logger)
    process_split("test",       args.test_preds_csv,  args.out_dir / "test",       logger)

    logger.info("\n=== Done ===")


if __name__ == "__main__":
    main()