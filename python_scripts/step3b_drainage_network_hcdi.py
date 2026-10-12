#!/usr/bin/env python3
"""
HRRT Dhaka | STEP 3b
Potential natural drainage network from the DEM, compared with the current
dry-season water map (Sentinel-2 MNDWI) to get the retention ratio and the
Hydrological Connectivity Deficit Index (HCDI = 1 - retention).

Inputs  (exported from Google Earth Engine, both EPSG:32646):
    data/raw/Dhaka_DEM_GLO30_30m.tif
    data/raw/Dhaka_MNDWI_2026DrySeason.tif
Outputs:
    data/outputs/drainage_network_thr<N>.gpkg   segments with retention status
    data/outputs/step3b_summary.csv             retention / HCDI, all settings
    figures/step3b_network_vs_water.png         map for the paper

Run from the repository folder:
    python python_scripts/step3b_drainage_network_hcdi.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from scipy import ndimage as ndi
from pysheds.grid import Grid

# ------------------------------------------------------------------ settings
STREAM_THRESHOLDS = [1000, 2500, 5000]  # upstream cells (30 m cells; 2500 cells = 2.25 km2)
MAIN_THRESHOLD = 2500                # setting used for the main map and tables
TOLERANCES_M = [15, 30]              # a path point counts as "retained" if water is this close
MAIN_TOLERANCE = 15
WATER_MNDWI = 0.0                    # MNDWI > 0 = water (same as Step 2)
SAMPLE_SPACING_M = 15                # spacing of sample points along each path
MAJOR_WATER_KM2 = 3.0                # water bodies >= this area are treated as main rivers/lakes
MAIN_RIVER_BUFFER_M = 30             # segment is "main river" if mostly within this distance
SURVIVING_MIN = 0.8                  # share of path retained: >= this -> surviving
LOST_MAX = 0.2                       # <= this -> lost; in between -> partial
DEM_SMOOTH = 3                       # median filter size (cells) to reduce building/tree noise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dem", default="data/raw/Dhaka_DEM_GLO30_30m.tif")
    p.add_argument("--mndwi", default="data/raw/Dhaka_MNDWI_2026DrySeason.tif")
    p.add_argument("--out", default="data/outputs")
    p.add_argument("--fig", default="figures")
    return p.parse_args()


def condition_dem(dem_path, out_path, size):
    """Median-filter the DEM and write it with a clean nodata value."""
    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    bad = ~np.isfinite(arr)
    arr = np.where(bad, np.nanmedian(arr), arr)
    smooth = ndi.median_filter(arr, size=size)
    smooth[bad] = -9999
    profile.update(dtype="float32", nodata=-9999, count=1, compress="deflate")
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(smooth.astype("float32"), 1)
    return profile["crs"]


def load_water(mndwi_path):
    with rasterio.open(mndwi_path) as src:
        arr = src.read(1, masked=True)
        water = arr.filled(-9) > WATER_MNDWI
        return water, src.transform, abs(src.res[0]), src.crs


def distance_maps(water, res):
    """Distance (m) to the nearest water pixel, and to the nearest MAJOR water body."""
    dist_water = ndi.distance_transform_edt(~water, sampling=res).astype("float32")
    labels, n = ndi.label(water, structure=np.ones((3, 3)))
    counts = np.bincount(labels.ravel())
    area_km2 = counts * (res ** 2) / 1e6
    major_ids = [i for i in range(1, n + 1) if area_km2[i] >= MAJOR_WATER_KM2]
    major = np.isin(labels, major_ids)
    if major.any():
        dist_major = ndi.distance_transform_edt(~major, sampling=res).astype("float32")
    else:
        dist_major = np.full(water.shape, 1e9, dtype="float32")
    return dist_water, dist_major, len(major_ids)


def lookup(arr, transform, xs, ys):
    rows, cols = rowcol(transform, xs, ys)
    rows = np.clip(np.asarray(rows), 0, arr.shape[0] - 1)
    cols = np.clip(np.asarray(cols), 0, arr.shape[1] - 1)
    return arr[rows, cols]


def score_segments(gdf, dist_water, dist_major, transform):
    """Per segment: share of its length near water (for each tolerance) and main-river flag."""
    out = {f"ret_{t}": [] for t in TOLERANCES_M}
    main_flag = []
    for line in gdf.geometry:
        length = line.length
        n = max(2, int(np.ceil(length / SAMPLE_SPACING_M)) + 1)
        pts = [line.interpolate(d) for d in np.linspace(0, length, n)]
        xs = np.array([p.x for p in pts])
        ys = np.array([p.y for p in pts])
        dw = lookup(dist_water, transform, xs, ys)
        dm = lookup(dist_major, transform, xs, ys)
        for t in TOLERANCES_M:
            out[f"ret_{t}"].append(float((dw <= t).mean()))
        main_flag.append(bool((dm <= MAIN_RIVER_BUFFER_M).mean() > 0.5))
    for k, v in out.items():
        gdf[k] = v
    gdf["main_river"] = main_flag
    return gdf


def classify(frac):
    if frac >= SURVIVING_MIN:
        return "surviving"
    if frac <= LOST_MAX:
        return "lost"
    return "partial"


def summarise(gdf, thr, tol):
    col = f"ret_{tol}"
    khal = gdf[~gdf["main_river"]]
    rows = []
    for label, d in (("khal_scale", khal), ("all_paths", gdf)):
        total = d["length_m"].sum()
        kept = (d["length_m"] * d[col]).sum()
        retention = kept / total if total > 0 else np.nan
        rows.append({
            "threshold_cells": thr, "tolerance_m": tol, "network": label,
            "segments": len(d), "total_km": round(total / 1000, 2),
            "retained_km": round(kept / 1000, 2),
            "retention": round(retention, 3), "HCDI": round(1 - retention, 3),
        })
    return rows


def make_figure(gdf, water, transform, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    step = 4
    w = water[::step, ::step]
    left, top = transform.c, transform.f
    right = left + water.shape[1] * res
    bottom = top - water.shape[0] * res
    fig, ax = plt.subplots(figsize=(8, 10), dpi=200)
    ax.imshow(np.where(w, 1, np.nan), extent=(left, right, bottom, top),
              cmap="Blues", vmin=0, vmax=1.6, interpolation="nearest")
    colors = {"surviving": "#1a9850", "partial": "#fdae61", "lost": "#d73027"}
    main = gdf[gdf["main_river"]]
    if len(main):
        main.plot(ax=ax, color="#888888", linewidth=0.5)
    for status, c in colors.items():
        d = gdf[(~gdf["main_river"]) & (gdf["status"] == status)]
        if len(d):
            d.plot(ax=ax, color=c, linewidth=0.9, label=f"{status} ({len(d)})")
    ax.legend(loc="lower left", fontsize=8, title="Potential drainage path")
    ax.set_title("Potential natural drainage network vs current water (dry season 2026)")
    ax.set_xlabel("Easting (m, UTM 46N)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    a = parse_args()
    out = Path(a.out)
    fig_dir = Path(a.fig)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) DEM -> conditioned DEM -> flow direction -> flow accumulation
    smooth_path = out / "dem_smoothed.tif"
    crs = condition_dem(a.dem, smooth_path, DEM_SMOOTH)
    grid = Grid.from_raster(str(smooth_path))
    dem = grid.read_raster(str(smooth_path))
    dem = grid.fill_pits(dem)
    dem = grid.fill_depressions(dem)
    dem = grid.resolve_flats(dem)
    fdir = grid.flowdir(dem)
    acc = grid.accumulation(fdir)
    print("flow accumulation done; max upstream cells:", int(np.nanmax(acc)))

    # 2) current water and distance maps
    water, w_transform, res, w_crs = load_water(a.mndwi)
    if w_crs != crs:
        raise SystemExit(f"CRS mismatch: DEM {crs} vs MNDWI {w_crs}. Export both as EPSG:32646.")
    dist_water, dist_major, n_major = distance_maps(water, res)
    print(f"current water: {water.mean() * 100:.1f}% of pixels; major water bodies: {n_major}")

    # 3) extract paths at several thresholds, score them against current water
    summary = []
    main_gdf = None
    for thr in STREAM_THRESHOLDS:
        branches = grid.extract_river_network(fdir, acc > thr)
        gdf = gpd.GeoDataFrame.from_features(branches["features"], crs=crs)
        gdf = gdf[gdf.geometry.length > 0].reset_index(drop=True)
        gdf["seg_id"] = np.arange(1, len(gdf) + 1)
        gdf["length_m"] = gdf.geometry.length
        gdf = score_segments(gdf, dist_water, dist_major, w_transform)
        gdf["status"] = gdf[f"ret_{MAIN_TOLERANCE}"].apply(classify)
        for tol in TOLERANCES_M:
            summary += summarise(gdf, thr, tol)
        keep = ["seg_id", "length_m", "main_river", "status"] + [f"ret_{t}" for t in TOLERANCES_M] + ["geometry"]
        gdf[keep].to_file(out / f"drainage_network_thr{thr}.gpkg", driver="GPKG")
        if thr == MAIN_THRESHOLD:
            main_gdf = gdf

    df = pd.DataFrame(summary)
    df.to_csv(out / "step3b_summary.csv", index=False)
    print("\n=== Retention / HCDI (all settings) ===")
    print(df.to_string(index=False))

    main_row = df[(df.threshold_cells == MAIN_THRESHOLD) & (df.tolerance_m == MAIN_TOLERANCE)
                  & (df.network == "khal_scale")].iloc[0]
    print(f"\nMAIN RESULT (threshold {MAIN_THRESHOLD} cells, tolerance {MAIN_TOLERANCE} m, khal-scale): "
          f"{main_row.segments} segments, {main_row.total_km} km, "
          f"retention {main_row.retention}, HCDI {main_row.HCDI}")

    make_figure(main_gdf, water, w_transform, res, fig_dir / "step3b_network_vs_water.png")
    print("figure saved:", fig_dir / "step3b_network_vs_water.png")


if __name__ == "__main__":
    main()
