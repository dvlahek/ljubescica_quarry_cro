#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tifffile
from PIL import Image
from scipy.ndimage import center_of_mass
from scipy.stats import linregress
from skimage.measure import label
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects

AOI_GEOJSON = {
    "type": "Polygon",
    "coordinates": [[
        [16.39916, 46.153192],
        [16.39916, 46.162293],
        [16.410406, 46.162293],
        [16.410406, 46.153192],
        [16.39916, 46.153192],
    ]],
}

TOKEN_URL = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"

START = "2017-01-01"
END = "2025-12-31"
OUT_WIDTH = 1024
OUT_HEIGHT = 1024
MAX_CLOUD_FRAC = 0.30

NDVI_PCTL = 30
BSI_PCTL = 65
NDBI_PCTL = 65
REF_MONTHS = ["2017-07", "2017-10", "2024-08"]
SIGNATURE_DIST_THR = 4.5
KEEP_TOP_COMPONENTS = 3
MIN_OBJECT_PIXELS = 200
CLOSING_RADIUS = 5
REMOVE_SMALL_HOLES_AREA = 15000

MASK_ARTIFACT_FRACTION_THR = 0.45
MAX_REASONABLE_AREA_HA = 40.0
MANUAL_EXCLUDE_MONTHS = {"2018-07", "2022-07"}
KEEP_NDVI_FOR_MASK_ARTIFACT_MONTHS = True

EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B08", "B12", "SCL", "dataMask"],
    output: { bands: 7, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  return [s.B02, s.B03, s.B04, s.B08, s.B12, s.SCL, s.dataMask];
}
""".strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconstruct the Ljubeščica quarry Sentinel-2 workflow.")
    p.add_argument("--out", default="quarry_timelapse_science_qc_v8")
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--width", type=int, default=OUT_WIDTH)
    p.add_argument("--height", type=int, default=OUT_HEIGHT)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def get_token() -> str:
    client_id = os.getenv("SENTINELHUB_CLIENT_ID")
    client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Set SENTINELHUB_CLIENT_ID and SENTINELHUB_CLIENT_SECRET before running.")
    r = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def month_starts(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(
        pd.to_datetime(start).to_period("M").to_timestamp(),
        pd.to_datetime(end).to_period("M").to_timestamp(),
        freq="MS",
    )


def download_month(token: str, date: pd.Timestamp, width: int, height: int) -> np.ndarray:
    payload = {
        "input": {
            "bounds": {
                "geometry": AOI_GEOJSON,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": date.strftime("%Y-%m-%dT00:00:00Z"),
                        "to": (date + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%dT00:00:00Z"),
                    },
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {"width": width, "height": height, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT,
    }
    r = requests.post(
        PROCESS_URL,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    if not r.ok:
        raise RuntimeError(f"Sentinel Hub API error {r.status_code}: {r.text}")
    return tifffile.imread(io.BytesIO(r.content))


def compute_indices(img: np.ndarray):
    blue, green, red, nir, swir, scl, dm = [img[:, :, i] for i in range(7)]
    eps = 1e-9
    ndvi = (nir - red) / (nir + red + eps)
    bsi = ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue) + eps)
    ndbi = (swir - nir) / (swir + nir + eps)
    return blue, green, red, scl, dm, ndvi, bsi, ndbi


def valid_mask(scl: np.ndarray, dm: np.ndarray) -> np.ndarray:
    return np.isin(scl.astype(np.int16), [4, 5, 7]) & (dm > 0.5)


def cloud_fraction(scl: np.ndarray, dm: np.ndarray) -> float:
    bad = np.isin(scl.astype(np.int16), [0, 1, 3, 8, 9, 10, 11])
    valid = dm > 0.5
    return 1.0 if valid.sum() == 0 else float((bad & valid).sum() / valid.sum())


def pixel_area_m2(width: int, height: int) -> float:
    coords = AOI_GEOJSON["coordinates"][0]
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    lat_mid = (min(lats) + max(lats)) / 2.0
    px_w = (max(lons) - min(lons)) * 111_320.0 * np.cos(np.deg2rad(lat_mid)) / width
    px_h = (max(lats) - min(lats)) * 111_320.0 / height
    return float(px_w * px_h)


def keep_largest(mask: np.ndarray, k: int) -> np.ndarray:
    lab = label(mask)
    if lab.max() == 0:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = np.argsort(sizes)[::-1][:k]
    return np.isin(lab, keep)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    out = binary_closing(mask.astype(bool), disk(CLOSING_RADIUS))
    out = remove_small_objects(out, min_size=MIN_OBJECT_PIXELS)
    out = remove_small_holes(out, area_threshold=REMOVE_SMALL_HOLES_AREA)
    return keep_largest(out, KEEP_TOP_COMPONENTS)


def norm_u8(band: np.ndarray, clip_max: float = 0.35) -> np.ndarray:
    return (np.clip(band, 0, clip_max) / clip_max * 255).astype(np.uint8)


def save_preview(out_dir: Path, month: str, red, green, blue, mask, artifact: bool = False) -> None:
    folder = out_dir / "artifact_masks" if artifact else out_dir
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "_artifact" if artifact else ""
    rgb = np.stack([norm_u8(red), norm_u8(green), norm_u8(blue)], axis=-1)
    Image.fromarray(rgb).save(folder / f"{month}_visual_rgb{suffix}.png")
    Image.fromarray((mask.astype(np.uint8) * 255)).save(folder / f"{month}_mask{suffix}.png")


def build_reference(token: str, width: int, height: int):
    feats = []
    ndvi_thr, bsi_thr, ndbi_thr = [], [], []
    for month in REF_MONTHS:
        d = pd.to_datetime(f"{month}-01")
        print(f"reference {month}")
        img = download_month(token, d, width, height)
        _, _, _, scl, dm, ndvi, bsi, ndbi = compute_indices(img)
        valid = valid_mask(scl, dm)
        t_ndvi = float(np.nanpercentile(ndvi[valid], NDVI_PCTL))
        t_bsi = float(np.nanpercentile(bsi[valid], BSI_PCTL))
        t_ndbi = float(np.nanpercentile(ndbi[valid], NDBI_PCTL))
        ndvi_thr.append(t_ndvi)
        bsi_thr.append(t_bsi)
        ndbi_thr.append(t_ndbi)
        cand = clean_mask(valid & (ndvi < t_ndvi) & (bsi > t_bsi) & (ndbi > t_ndbi))
        if cand.any():
            feats.append(np.stack([ndvi[cand], bsi[cand], ndbi[cand]], axis=1))
        time.sleep(1.0)
    if not feats:
        raise RuntimeError("No reference quarry pixels were extracted.")
    X = np.vstack(feats)
    return {
        "mu": np.nanmean(X, axis=0),
        "sigma": np.nanstd(X, axis=0) + 1e-6,
        "ndvi_thr": float(np.mean(ndvi_thr)),
        "bsi_thr": float(np.mean(bsi_thr)),
        "ndbi_thr": float(np.mean(ndbi_thr)),
    }


def fhfm_ndvi(ndvi: np.ndarray, dm: np.ndarray, forest_mask: np.ndarray) -> float:
    vals = ndvi[forest_mask & (dm > 0.5)]
    return float(np.nanmean(vals)) if vals.size else np.nan


def skip_row(date, month, reason, cloud=np.nan, mask_fraction=np.nan, ndvi=np.nan):
    return {
        "date": date,
        "month": month,
        "area_ha": np.nan,
        "cloud_frac": cloud,
        "fhfm_ndvi": ndvi,
        "centroid_x": np.nan,
        "centroid_y": np.nan,
        "mask_fraction": mask_fraction,
        "skip_reason": reason,
    }


def regression(df: pd.DataFrame, column: str) -> dict:
    d = df.dropna(subset=[column, "date"]).copy()
    if len(d) < 3:
        return {"n": int(len(d)), "slope_per_day": None, "r2": None, "pvalue": None}
    x = (d["date"] - d["date"].min()).dt.days.astype(float)
    r = linregress(x, d[column].astype(float))
    return {"n": int(len(d)), "slope_per_day": float(r.slope), "r2": float(r.rvalue ** 2), "pvalue": float(r.pvalue)}


def plot_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    area = df.dropna(subset=["area_ha", "centroid_x"]).copy()
    ndvi = df.dropna(subset=["fhfm_ndvi"]).copy()

    if not area.empty:
        area["smooth"] = area["area_ha"].rolling(3, min_periods=1).mean()
        plt.figure(figsize=(10, 5))
        plt.plot(area["date"], area["area_ha"], "o-", alpha=0.45, label="Monthly area")
        plt.plot(area["date"], area["smooth"], linewidth=2, label="3-month rolling mean")
        plt.ylabel("Area (ha)")
        plt.title("Quarry footprint area")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "plot_1_area.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 6))
        sc = plt.scatter(area["centroid_x"], area["centroid_y"], c=range(len(area)), s=45)
        plt.colorbar(sc, label="Time progression")
        plt.gca().invert_yaxis()
        plt.xlabel("X coordinate (pixels)")
        plt.ylabel("Y coordinate (pixels)")
        plt.title("Quarry centroid shift")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "plot_3_centroid_shift.png", dpi=150)
        plt.close()

    if not ndvi.empty:
        ndvi["smooth"] = ndvi["fhfm_ndvi"].rolling(3, min_periods=1).mean()
        plt.figure(figsize=(10, 5))
        plt.plot(ndvi["date"], ndvi["fhfm_ndvi"], "o-", alpha=0.45, label="Monthly FHFM NDVI")
        plt.plot(ndvi["date"], ndvi["smooth"], linewidth=2, label="3-month rolling mean")
        plt.ylabel("Mean NDVI")
        plt.title("Fixed historical forest-mask NDVI")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "plot_2_fhfm_ndvi.png", dpi=150)
        plt.close()


def write_summary(df: pd.DataFrame, ref: dict, out_dir: Path) -> None:
    summary = {
        "parameters": {
            "NDVI_PCTL": NDVI_PCTL,
            "BSI_PCTL": BSI_PCTL,
            "NDBI_PCTL": NDBI_PCTL,
            "REF_MONTHS": REF_MONTHS,
            "SIGNATURE_DIST_THR": SIGNATURE_DIST_THR,
            "MANUAL_EXCLUDE_MONTHS": sorted(MANUAL_EXCLUDE_MONTHS),
        },
        "fixed_thresholds": {k: float(ref[k]) for k in ["ndvi_thr", "bsi_thr", "ndbi_thr"]},
        "rows": int(len(df)),
        "valid_area_rows": int(df.dropna(subset=["area_ha"]).shape[0]),
        "valid_ndvi_rows": int(df.dropna(subset=["fhfm_ndvi"]).shape[0]),
        "skip_reasons": df["skip_reason"].fillna("").value_counts().to_dict(),
        "regression_area_ha": regression(df, "area_ha"),
        "regression_fhfm_ndvi": regression(df, "fhfm_ndvi"),
        "regression_centroid_x": regression(df, "centroid_x"),
        "regression_centroid_y": regression(df, "centroid_y"),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output folder exists: {out_dir}. Use --overwrite for a fresh run.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_masks").mkdir()

    token = get_token()
    ref = build_reference(token, args.width, args.height)
    print("thresholds:", {k: round(float(ref[k]), 4) for k in ["ndvi_thr", "bsi_thr", "ndbi_thr"]})

    ref_img = download_month(token, pd.to_datetime("2017-07-01"), args.width, args.height)
    _, _, _, _, dm_ref, ndvi_ref, _, _ = compute_indices(ref_img)
    forest_mask = (ndvi_ref > 0.6) & (dm_ref > 0.5)

    rows = []
    pa_m2 = pixel_area_m2(args.width, args.height)
    for date in month_starts(args.start, args.end):
        month = date.strftime("%Y-%m")
        if date.month in [11, 12, 1, 2]:
            rows.append(skip_row(date, month, "winter"))
            continue
        try:
            img = download_month(token, date, args.width, args.height)
            blue, green, red, scl, dm, ndvi, bsi, ndbi = compute_indices(img)
            cloud = cloud_fraction(scl, dm)
            valid = valid_mask(scl, dm)
            if cloud > MAX_CLOUD_FRAC or valid.sum() < 50:
                rows.append(skip_row(date, month, "cloud_or_no_data", cloud=cloud))
                continue

            forest_ndvi = fhfm_ndvi(ndvi, dm, forest_mask)
            brightness = (blue + green + red) / 3.0
            candidate = valid & (ndvi < ref["ndvi_thr"]) & (bsi > ref["bsi_thr"]) & (ndbi > ref["ndbi_thr"]) & (brightness > 0.04)
            X = np.stack([ndvi, bsi, ndbi], axis=-1)
            dist = np.sqrt(np.sum(((X - ref["mu"]) / ref["sigma"]) ** 2, axis=-1))
            quarry = clean_mask(candidate & (dist < SIGNATURE_DIST_THR))

            mask_fraction = float(quarry.sum() / quarry.size)
            area_ha = float(quarry.sum() * pa_m2 / 10000.0)
            mask_artifact = month in MANUAL_EXCLUDE_MONTHS or mask_fraction > MASK_ARTIFACT_FRACTION_THR or area_ha > MAX_REASONABLE_AREA_HA
            if mask_artifact:
                save_preview(out_dir, month, red, green, blue, quarry, artifact=True)
                rows.append(skip_row(
                    date,
                    month,
                    "mask_artifact",
                    cloud=cloud,
                    mask_fraction=mask_fraction,
                    ndvi=forest_ndvi if KEEP_NDVI_FOR_MASK_ARTIFACT_MONTHS else np.nan,
                ))
                print(f"{month}: mask artifact, FHFM_NDVI={forest_ndvi:.3f}")
                continue

            cy, cx = center_of_mass(quarry) if quarry.any() else (np.nan, np.nan)
            save_preview(out_dir, month, red, green, blue, quarry, artifact=False)
            rows.append({
                "date": date,
                "month": month,
                "area_ha": area_ha,
                "cloud_frac": cloud,
                "fhfm_ndvi": forest_ndvi,
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "mask_fraction": mask_fraction,
                "skip_reason": "",
            })
            print(f"{month}: area={area_ha:.2f} ha, FHFM_NDVI={forest_ndvi:.3f}")
            time.sleep(1.0)
        except Exception as exc:
            rows.append(skip_row(date, month, f"error: {exc}"))
            print(f"{month}: ERROR {exc}")

    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(out_dir / "quarry_science_metrics.csv", index=False)
    plot_outputs(df, out_dir)
    write_summary(df, ref, out_dir)
    print(f"done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
