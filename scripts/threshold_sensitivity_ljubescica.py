#!/usr/bin/env python3
from __future__ import annotations

import argparse, io, json, os, time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests, tifffile
from scipy.ndimage import center_of_mass
from scipy.stats import linregress
from skimage.measure import label
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects

AOI = {"type": "Polygon", "coordinates": [[[16.39916, 46.153192], [16.39916, 46.162293], [16.410406, 46.162293], [16.410406, 46.153192], [16.39916, 46.153192]]]}
TOKEN_URL = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"
START, END = "2017-01-01", "2025-12-31"
WIDTH, HEIGHT = 1024, 1024
MAX_CLOUD = 0.30
BASE_NDVI, BASE_BSI, BASE_NDBI, BASE_D = 30, 65, 65, 4.5
REF_MONTHS = ["2017-07", "2017-10", "2024-08"]
KEEP_TOP, MIN_OBJECTS, CLOSE_RADIUS, HOLES_AREA = 3, 200, 5, 15000
WHITE_MASK_FRACTION, MAX_AREA_HA = 0.45, 40.0
MANUAL_EXCLUDE = {"2018-07", "2022-07"}

EVALSCRIPT = """
//VERSION=3
function setup() {
  return {input: ["B02", "B03", "B04", "B08", "B12", "SCL", "dataMask"], output: {bands: 7, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) { return [s.B02, s.B03, s.B04, s.B08, s.B12, s.SCL, s.dataMask]; }
""".strip()

@dataclass(frozen=True)
class Config:
    name: str
    ndvi_pctl: int
    bsi_pctl: int
    ndbi_pctl: int
    dist_thr: float


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Threshold-sensitivity check for the Ljubeščica quarry workflow.")
    p.add_argument("--cache", default="s2_cache_ljubescica")
    p.add_argument("--out", default="threshold_sensitivity_ljubescica")
    p.add_argument("--mode", choices=["base_only", "one_at_a_time", "full_grid"], default="one_at_a_time")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--max-downloads", type=int, default=0)
    p.add_argument("--sleep", type=float, default=1.0)
    return p.parse_args()


def months(start: str = START, end: str = END) -> pd.DatetimeIndex:
    all_months = pd.date_range(pd.to_datetime(start).to_period("M").to_timestamp(), pd.to_datetime(end).to_period("M").to_timestamp(), freq="MS")
    return pd.DatetimeIndex([d for d in all_months if d.month in {3, 4, 5, 6, 7, 8, 9, 10}])


def token() -> str:
    cid, sec = os.getenv("SENTINELHUB_CLIENT_ID"), os.getenv("SENTINELHUB_CLIENT_SECRET")
    if not cid or not sec:
        raise RuntimeError("Set SENTINELHUB_CLIENT_ID and SENTINELHUB_CLIENT_SECRET before downloading scenes.")
    r = requests.post(TOKEN_URL, data={"grant_type": "client_credentials", "client_id": cid, "client_secret": sec}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def cpath(cache: Path, d: pd.Timestamp) -> Path:
    return cache / f"{d:%Y-%m}.tif"


def download_month(tok: str, d: pd.Timestamp) -> np.ndarray:
    payload = {
        "input": {"bounds": {"geometry": AOI, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}}, "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": d.strftime("%Y-%m-%dT00:00:00Z"), "to": (d + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%dT00:00:00Z")}, "mosaickingOrder": "leastCC"}}]},
        "output": {"width": WIDTH, "height": HEIGHT, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT,
    }
    r = requests.post(PROCESS_URL, headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}, json=payload, timeout=180)
    if not r.ok:
        raise RuntimeError(f"Sentinel Hub error for {d:%Y-%m}: {r.status_code}: {r.text}")
    return tifffile.imread(io.BytesIO(r.content))


def ensure_cache(ns: argparse.Namespace, dates: pd.DatetimeIndex) -> None:
    cache = Path(ns.cache)
    cache.mkdir(parents=True, exist_ok=True)
    missing = [d for d in dates if not cpath(cache, d).exists()]
    if not missing:
        print("Cache complete for requested growing-season months.")
        return
    print(f"Missing cached months: {len(missing)}")
    if ns.skip_download:
        print("--skip-download is active; no scenes will be downloaded.")
        return
    tok = token()
    for i, d in enumerate(missing, start=1):
        if ns.max_downloads and i > ns.max_downloads:
            print(f"Reached --max-downloads={ns.max_downloads}; resume later with the same cache folder.")
            break
        print(f"Downloading {d:%Y-%m} ...")
        tifffile.imwrite(cpath(cache, d), download_month(tok, d).astype(np.float32))
        time.sleep(ns.sleep)


def indices(img: np.ndarray):
    blue, green, red, nir, swir, scl, dm = [img[:, :, i] for i in range(7)]
    eps = 1e-9
    ndvi = (nir - red) / (nir + red + eps)
    bsi = ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue) + eps)
    ndbi = (swir - nir) / (swir + nir + eps)
    return blue, green, red, nir, swir, scl, dm, ndvi, bsi, ndbi


def valid(scl: np.ndarray, dm: np.ndarray) -> np.ndarray:
    return np.isin(scl.astype(np.int16), [4, 5, 7]) & (dm > 0.5)


def cloud(scl: np.ndarray, dm: np.ndarray) -> float:
    any_data = dm > 0.5
    bad = np.isin(scl.astype(np.int16), [0, 1, 3, 8, 9, 10, 11])
    return 1.0 if any_data.sum() == 0 else float((bad & any_data).sum() / any_data.sum())


def top_components(mask: np.ndarray, k: int = KEEP_TOP) -> np.ndarray:
    lab = label(mask)
    if lab.max() == 0:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return np.isin(lab, np.argsort(sizes)[::-1][:k])


def clean(mask: np.ndarray) -> np.ndarray:
    out = binary_closing(mask, disk(CLOSE_RADIUS))
    out = remove_small_objects(out, min_size=MIN_OBJECTS)
    out = remove_small_holes(out, area_threshold=HOLES_AREA)
    return top_components(out)


def pixel_area() -> float:
    lon = [p[0] for p in AOI["coordinates"][0]]
    lat = [p[1] for p in AOI["coordinates"][0]]
    lat_mid = (min(lat) + max(lat)) / 2.0
    return float(((max(lon) - min(lon)) * 111_320.0 * np.cos(np.deg2rad(lat_mid)) / WIDTH) * ((max(lat) - min(lat)) * 111_320.0 / HEIGHT))


def configs(mode: str) -> list[Config]:
    base = Config("base", BASE_NDVI, BASE_BSI, BASE_NDBI, BASE_D)
    if mode == "base_only":
        return [base]
    if mode == "one_at_a_time":
        out = [base]
        out += [Config(f"ndvi_{v}", v, BASE_BSI, BASE_NDBI, BASE_D) for v in [25, 35]]
        out += [Config(f"bsi_{v}", BASE_NDVI, v, BASE_NDBI, BASE_D) for v in [60, 70]]
        out += [Config(f"ndbi_{v}", BASE_NDVI, BASE_BSI, v, BASE_D) for v in [60, 70]]
        out += [Config(f"d_{str(v).replace('.', '_')}", BASE_NDVI, BASE_BSI, BASE_NDBI, v) for v in [4.0, 5.0]]
        return out
    return [Config(f"ndvi{n}_bsi{b}_ndbi{i}_d{str(d).replace('.', '_')}", n, b, i, d) for n in [25, 30, 35] for b in [60, 65, 70] for i in [60, 65, 70] for d in [4.0, 4.5, 5.0]]


def scene(cache: Path, d: pd.Timestamp) -> np.ndarray | None:
    return tifffile.imread(cpath(cache, d)) if cpath(cache, d).exists() else None


def reference(cache: Path, cfg: Config) -> dict:
    feats, ndvi_t, bsi_t, ndbi_t = [], [], [], []
    for m in REF_MONTHS:
        img = scene(cache, pd.to_datetime(m + "-01"))
        if img is None:
            raise RuntimeError(f"Missing cached reference month {m}.")
        blue, green, red, nir, swir, scl, dm, ndvi, bsi, ndbi = indices(img)
        v = valid(scl, dm)
        nt, bt, it = np.nanpercentile(ndvi[v], cfg.ndvi_pctl), np.nanpercentile(bsi[v], cfg.bsi_pctl), np.nanpercentile(ndbi[v], cfg.ndbi_pctl)
        ndvi_t.append(float(nt)); bsi_t.append(float(bt)); ndbi_t.append(float(it))
        cand = clean(v & (ndvi < nt) & (bsi > bt) & (ndbi > it))
        if cand.any():
            feats.append(np.stack([ndvi[cand], bsi[cand], ndbi[cand]], axis=1))
    if not feats:
        raise RuntimeError("No reference features extracted.")
    X = np.vstack(feats)
    return {"mu": np.nanmean(X, axis=0), "sigma": np.nanstd(X, axis=0) + 1e-6, "ndvi_thr": float(np.mean(ndvi_t)), "bsi_thr": float(np.mean(bsi_t)), "ndbi_thr": float(np.mean(ndbi_t))}


def forest_mask(cache: Path) -> np.ndarray:
    img = scene(cache, pd.to_datetime("2017-07-01"))
    if img is None:
        raise RuntimeError("Missing cached 2017-07 scene for the fixed historical forest mask.")
    return indices(img)[7] > 0.6


def regress(df: pd.DataFrame, col: str) -> dict:
    ok = df.dropna(subset=[col, "date"])
    if len(ok) < 3:
        return {"slope_per_day": np.nan, "pvalue": np.nan, "r2": np.nan}
    x = (pd.to_datetime(ok["date"]) - pd.to_datetime(ok["date"]).min()).dt.days.astype(float)
    r = linregress(x, ok[col].astype(float))
    return {"slope_per_day": float(r.slope), "pvalue": float(r.pvalue), "r2": float(r.rvalue ** 2)}


def run(cache: Path, dates: pd.DatetimeIndex, cfg: Config, fmask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    sig, px_area, rows = reference(cache, cfg), pixel_area(), []
    for d in dates:
        month = d.strftime("%Y-%m")
        img = scene(cache, d)
        if img is None:
            rows.append({"config": cfg.name, "date": d, "month": month, "area_ha": np.nan, "fhfm_ndvi": np.nan, "centroid_x": np.nan, "centroid_y": np.nan, "skip_reason": "missing_cache"})
            continue
        blue, green, red, nir, swir, scl, dm, ndvi, bsi, ndbi = indices(img)
        base = {"config": cfg.name, "date": d, "month": month, "cloud_frac": cloud(scl, dm), "area_ha": np.nan, "fhfm_ndvi": float(np.nanmean(ndvi[fmask & (dm > 0.5)])) if (fmask & (dm > 0.5)).any() else np.nan, "centroid_x": np.nan, "centroid_y": np.nan, "mask_fraction": np.nan, "skip_reason": ""}
        v = valid(scl, dm)
        if base["cloud_frac"] > MAX_CLOUD or v.sum() < 50:
            rows.append({**base, "skip_reason": "cloud_or_no_data"}); continue
        cand = v & (ndvi < sig["ndvi_thr"]) & (bsi > sig["bsi_thr"]) & (ndbi > sig["ndbi_thr"]) & (((blue + green + red) / 3.0) > 0.04)
        Z = (np.stack([ndvi, bsi, ndbi], axis=-1) - sig["mu"]) / sig["sigma"]
        quarry = clean(cand & (np.sqrt(np.sum(Z * Z, axis=-1)) < cfg.dist_thr))
        frac = float(quarry.sum() / quarry.size)
        area = float(quarry.sum() * px_area / 10000.0)
        if month in MANUAL_EXCLUDE:
            rows.append({**base, "mask_fraction": frac, "skip_reason": "manual_white_mask_artifact"}); continue
        if frac > WHITE_MASK_FRACTION or area > MAX_AREA_HA:
            rows.append({**base, "mask_fraction": frac, "skip_reason": "mask_artifact"}); continue
        cy, cx = center_of_mass(quarry) if quarry.any() else (np.nan, np.nan)
        rows.append({**base, "area_ha": area, "centroid_x": float(cx), "centroid_y": float(cy), "mask_fraction": frac})
    df = pd.DataFrame(rows)
    return df, summarize(df, cfg, sig)


def summarize(df: pd.DataFrame, cfg: Config, sig: dict) -> dict:
    d = df.copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    ay = d.dropna(subset=["area_ha"]).groupby("year")["area_ha"].mean()
    ny = d.dropna(subset=["fhfm_ndvi"]).groupby("year")["fhfm_ndvi"].mean()
    a17, a25 = float(ay.get(2017, np.nan)), float(ay.get(2025, np.nan))
    n17, n25 = float(ny.get(2017, np.nan)), float(ny.get(2025, np.nan))
    growth = float((a25 - a17) / a17 * 100.0) if np.isfinite(a17) and a17 else np.nan
    ndvi_change = float((n25 - n17) / n17 * 100.0) if np.isfinite(n17) and n17 else np.nan
    ar, nr, xr, yr = regress(d, "area_ha"), regress(d, "fhfm_ndvi"), regress(d, "centroid_x"), regress(d, "centroid_y")
    return {**asdict(cfg), "fixed_ndvi_thr": sig["ndvi_thr"], "fixed_bsi_thr": sig["bsi_thr"], "fixed_ndbi_thr": sig["ndbi_thr"], "n_area_months": int(d.dropna(subset=["area_ha"]).shape[0]), "n_ndvi_months": int(d.dropna(subset=["fhfm_ndvi"]).shape[0]), "area_2017_mean": a17, "area_2025_mean": a25, "growth_percent": growth, "fhfm_ndvi_2017_mean": n17, "fhfm_ndvi_2025_mean": n25, "fhfm_ndvi_change_percent": ndvi_change, "area_slope_per_day": ar["slope_per_day"], "area_pvalue": ar["pvalue"], "area_r2": ar["r2"], "fhfm_ndvi_slope_per_day": nr["slope_per_day"], "fhfm_ndvi_pvalue": nr["pvalue"], "centroid_x_slope_per_day": xr["slope_per_day"], "centroid_x_pvalue": xr["pvalue"], "centroid_y_slope_per_day": yr["slope_per_day"], "centroid_y_pvalue": yr["pvalue"], "same_interpretation_as_article": bool(np.isfinite(growth) and growth > 0 and np.isfinite(ar["pvalue"]) and ar["pvalue"] < 0.05 and np.isfinite(xr["slope_per_day"]) and xr["slope_per_day"] > 0 and np.isfinite(yr["slope_per_day"]) and yr["slope_per_day"] < 0 and np.isfinite(ndvi_change) and ndvi_change < 0), "skip_reason_counts": json.dumps(d["skip_reason"].fillna("").value_counts().to_dict(), ensure_ascii=False)}


def main() -> int:
    ns = args()
    dates, cache, out = months(), Path(ns.cache), Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    ensure_cache(ns, dates)
    if ns.download_only:
        return 0
    fmask = forest_mask(cache)
    monthly, summaries = [], []
    for cfg in configs(ns.mode):
        print(f"Config {cfg.name}: NDVI {cfg.ndvi_pctl}, BSI {cfg.bsi_pctl}, NDBI {cfg.ndbi_pctl}, D {cfg.dist_thr}")
        df, sm = run(cache, dates, cfg, fmask)
        monthly.append(df); summaries.append(sm)
    mdf, sdf = pd.concat(monthly, ignore_index=True), pd.DataFrame(summaries)
    mdf.to_csv(out / "threshold_sensitivity_monthly.csv", index=False)
    sdf.to_csv(out / "threshold_sensitivity_summary.csv", index=False)
    payload = {"mode": ns.mode, "n_configs": len(sdf), "cache": str(cache), "outputs": ["threshold_sensitivity_summary.csv", "threshold_sensitivity_monthly.csv"], "interpretation": {"stable_configs": int(sdf["same_interpretation_as_article"].sum()), "total_configs": int(len(sdf))}}
    (out / "threshold_sensitivity_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    cols = ["name", "growth_percent", "area_pvalue", "fhfm_ndvi_change_percent", "fhfm_ndvi_pvalue", "centroid_x_slope_per_day", "centroid_y_slope_per_day", "same_interpretation_as_article"]
    print(sdf[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
