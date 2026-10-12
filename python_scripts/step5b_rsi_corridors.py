#!/usr/bin/env python3
"""
HRRT Dhaka | STEP 5b
Restoration Suitability Index (RSI) in a 100 m wide strip along each priority
corridor from Step 4 (10 m grid).

RSI = weighted mean of three criteria, each scaled 0-1:
  proximity : 1 at the DEM flow path, falling to 0 at 50 m from it
  TWI       : Topographic Wetness Index from the DEM, rescaled between its
              2nd and 98th percentile (wetter ground = higher)
  openness  : 1 - built-up probability (Dynamic World, Jan-Mar 2026 mean)
Weights are set by expert judgement (equal by default) and tested for
sensitivity. This is a multi-criteria index, not a trained classifier, and it
has not been validated against ground truth.

Per corridor the script reports the share of strip area with RSI >= 0.6
("restorable"), the share that is clearly built-up, and a feasibility-adjusted
priority (RPI from Step 4 x restorable share).

Inputs : data/outputs/step4_priority_corridors.gpkg   (Step 4)
         data/outputs/dem_smoothed.tif                (Step 3b)
         data/raw/Dhaka_DW_landcover_2026DrySeason.tif (Step 5a, 8 bands:
             built, water, trees, grass, crops, shrub_and_scrub,
             flooded_vegetation, bare)
Outputs: data/outputs/step5_rsi_strip.tif
         data/outputs/step5_corridor_summary.csv
         data/outputs/step5_sensitivity.csv
         figures/step5_rsi_corridors.png

Run from the repository folder:
    python python_scripts/step5b_rsi_corridors.py
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.transform import rowcol
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from pysheds.grid import Grid

# ------------------------------------------------------------------ settings
STRIP_HALF_M = 50                  # strip = cells within 50 m of a corridor (100 m wide)
PROX_ZERO_M = 50                   # proximity score reaches 0 at this distance
WEIGHTS = {                        # (proximity, TWI, openness)
    "equal": (1 / 3, 1 / 3, 1 / 3),
    "proximity-heavy": (0.5, 0.25, 0.25),
    "TWI-heavy": (0.25, 0.5, 0.25),
    "openness-heavy": (0.25, 0.25, 0.5),
}
MAIN_WEIGHTS = "equal"
THRESHOLDS = [0.5, 0.6, 0.7]
MAIN_THRESHOLD = 0.6
BUILT_BARRIER = 0.5                # built-up probability above this = clear barrier
HIGH_MIN, MEDIUM_MIN = 0.6, 0.3    # feasibility classes by restorable share
BUILT_BAND = 0                     # index of the 'built' band in the Dynamic World file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corridors", default="data/outputs/step4_priority_corridors.gpkg")
    p.add_argument("--dem", default="data/outputs/dem_smoothed.tif")
    p.add_argument("--dw", default="data/raw/Dhaka_DW_landcover_2026DrySeason.tif")
    p.add_argument("--out", default="data/outputs")
    p.add_argument("--fig", default="figures")
    return p.parse_args()


def compute_twi(dem_path):
    with rasterio.open(dem_path) as src:
        z = src.read(1).astype("float64")
        nod, transform, crs, res = src.nodata, src.transform, src.crs, abs(src.res[0])
    if nod is not None:
        z[z == nod] = np.nan
    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))
    dem = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    acc = np.array(grid.accumulation(grid.flowdir(dem)), dtype="float64")
    zz = np.where(np.isnan(z), np.nanmedian(z), z)
    dy, dx = np.gradient(zz, res)
    tan_b = np.maximum(np.hypot(dx, dy), 0.001)            # floor for flat ground
    twi = np.log(((acc + 1) * res) / tan_b)
    twi[np.isnan(z)] = np.nan
    return twi.astype("float32"), transform, crs


def read_dw_window(dw_path, bounds, margin):
    with rasterio.open(dw_path) as src:
        if src.count < 8:
            raise SystemExit(f"Expected 8 bands in {dw_path}, found {src.count}. Re-export with Step 5a.")
        minx, miny, maxx, maxy = bounds
        r0, c0 = rowcol(src.transform, minx - margin, maxy + margin)
        r1, c1 = rowcol(src.transform, maxx + margin, miny - margin)
        r0, c0 = max(int(r0), 0), max(int(c0), 0)
        r1, c1 = min(int(r1) + 1, src.height), min(int(c1) + 1, src.width)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        arr = src.read(window=win).astype("float32")
        return arr, src.window_transform(win), src.crs, abs(src.res[0])


def shares(df, rsi_col, thr):
    return (df[rsi_col] >= thr).groupby(df["corridor"]).mean()


def main():
    a = parse_args()
    out, fig_dir = Path(a.out), Path(a.fig)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cor = gpd.read_file(a.corridors)
    cor = cor[cor.corridor > 0].reset_index(drop=True)
    print(f"{cor.corridor.nunique()} corridors, {len(cor)} segments")

    # ---- land cover window around the corridors (10 m grid)
    arr, tr, crs, res = read_dw_window(a.dw, cor.total_bounds, STRIP_HALF_M + 400)
    built = arr[BUILT_BAND]
    built = np.where(np.isfinite(built), built, np.nanmedian(built))
    h, w = built.shape
    print(f"working grid: {w} x {h} cells at {res:.0f} m; mean built-up probability {built.mean():.2f}")

    # ---- TWI from the DEM, resampled onto the 10 m grid
    twi, twi_tr, twi_crs = compute_twi(a.dem)
    lo, hi = np.nanpercentile(twi, [2, 98])
    twi10 = np.full((h, w), np.nan, dtype="float32")
    reproject(twi, twi10, src_transform=twi_tr, src_crs=twi_crs, dst_transform=tr, dst_crs=crs,
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    T = np.clip((twi10 - lo) / (hi - lo), 0, 1)

    # ---- distance to the nearest corridor and which corridor that is
    lab = features.rasterize([(g, int(c)) for g, c in zip(cor.geometry, cor.corridor)],
                             out_shape=(h, w), transform=tr, fill=0, all_touched=True, dtype="int32")
    dist, ind = ndi.distance_transform_edt(lab == 0, sampling=res, return_indices=True)
    nearest = lab[ind[0], ind[1]]
    strip = dist <= STRIP_HALF_M
    P = np.clip(1 - dist / PROX_ZERO_M, 0, 1)
    O = np.clip(1 - built, 0, 1)

    # ---- RSI for every weight set; zonal table
    rsi = {}
    for name, (wp, wt, wo) in WEIGHTS.items():
        rsi[name] = wp * P + wt * T + wo * O
    ok = strip & np.isfinite(T)
    df = pd.DataFrame({"corridor": nearest[ok], "built": built[ok], "TWI": T[ok], "open": O[ok]})
    for name in WEIGHTS:
        df[name] = rsi[name][ok]
    cell_ha = res * res / 10000

    # ---- main result per corridor
    g = df.groupby("corridor")
    summ = pd.DataFrame({
        "strip_ha": (g.size() * cell_ha).round(1),
        "mean_RSI": g[MAIN_WEIGHTS].mean().round(3),
        "restorable_share": shares(df, MAIN_WEIGHTS, MAIN_THRESHOLD).round(3),
        "built_barrier_share": (df.built > BUILT_BARRIER).groupby(df.corridor).mean().round(3),
        "mean_TWI_score": g["TWI"].mean().round(3),
        "mean_openness": g["open"].mean().round(3),
    })
    attrs = cor.groupby("corridor").agg(RPI=("RPI", "max"), gap_km=("gap_m", lambda s: s.sum() / 1000),
                                        length_km=("length_m", lambda s: s.sum() / 1000),
                                        n_segments=("seg_id", "count"))
    summ = attrs.join(summ).round(3)
    summ["feasibility"] = pd.cut(summ.restorable_share, [-0.01, MEDIUM_MIN, HIGH_MIN, 1.01],
                                 labels=["Low", "Medium", "High"], right=False)
    summ["RPI_adjusted"] = (summ.RPI * summ.restorable_share).round(3)
    summ["final_rank"] = summ.RPI_adjusted.rank(ascending=False, method="min").astype(int)
    summ.reset_index().to_csv(out / "step5_corridor_summary.csv", index=False)
    print(f"\n=== Corridor summary (weights '{MAIN_WEIGHTS}', restorable = RSI >= {MAIN_THRESHOLD}) ===")
    print(summ.sort_values("final_rank").to_string())
    tot_ha = summ.strip_ha.sum()
    rest_ha = (summ.strip_ha * summ.restorable_share).sum()
    print(f"\nstrip area {tot_ha:.1f} ha; restorable {rest_ha:.1f} ha ({rest_ha / tot_ha * 100:.1f}%); "
          f"mean built-up barrier share {summ.built_barrier_share.mean():.2f}")

    # ---- sensitivity to weights and threshold
    base = shares(df, MAIN_WEIGHTS, MAIN_THRESHOLD)
    rows = []
    for name in WEIGHTS:
        for thr in THRESHOLDS:
            s = shares(df, name, thr)
            tot = (df[name] >= thr).mean()
            rho = spearmanr(base.reindex(s.index), s)[0] if s.nunique() > 1 else np.nan
            rows.append({"weights": name, "threshold": thr, "restorable_share_all": round(tot, 3),
                         "spearman_vs_main": round(rho, 2) if rho == rho else np.nan})
    sens = pd.DataFrame(rows)
    sens.to_csv(out / "step5_sensitivity.csv", index=False)
    print("\n=== Sensitivity of the restorable share to weights and threshold ===")
    print(sens.to_string(index=False))

    # ---- RSI raster (strip only)
    ras = np.where(ok, rsi[MAIN_WEIGHTS], np.nan).astype("float32")
    with rasterio.open(out / "step5_rsi_strip.tif", "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs=crs, transform=tr, nodata=np.nan, compress="deflate") as dst:
        dst.write(ras, 1)

    # ---- figure: zoom panels for the first six corridors
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = sorted(cor.corridor.unique())[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), dpi=200)
    im = None
    for ax, cid in zip(axes.ravel(), ids):
        sub = cor[cor.corridor == cid]
        minx, miny, maxx, maxy = sub.total_bounds
        m = 300
        r0, c0 = rowcol(tr, minx - m, maxy + m)
        r1, c1 = rowcol(tr, maxx + m, miny - m)
        r0, c0, r1, c1 = max(r0, 0), max(c0, 0), min(r1 + 1, h), min(c1 + 1, w)
        left, top = tr * (c0, r0)
        right, bottom = tr * (c1, r1)
        ext = (left, right, bottom, top)
        ax.imshow(built[r0:r1, c0:c1], extent=ext, cmap="Greys", vmin=0, vmax=1)
        im = ax.imshow(ras[r0:r1, c0:c1], extent=ext, cmap="viridis", vmin=0.3, vmax=0.9, alpha=0.85)
        sub.plot(ax=ax, color="red", linewidth=0.8)
        ax.set_xlim(left, right); ax.set_ylim(bottom, top)
        ax.set_title(f"Corridor {cid}: {summ.loc[cid, 'restorable_share'] * 100:.0f}% restorable", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for ax in axes.ravel()[len(ids):]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label="RSI (strip 100 m wide); grey = built-up probability")
    fig.suptitle("Restoration Suitability Index along the top corridors (red = DEM flow path)")
    fig.savefig(fig_dir / "step5_rsi_corridors.png")
    plt.close(fig)
    print("\nfigure saved:", fig_dir / "step5_rsi_corridors.png")


if __name__ == "__main__":
    main()
