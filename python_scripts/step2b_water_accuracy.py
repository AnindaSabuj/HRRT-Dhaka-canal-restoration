#!/usr/bin/env python3
"""
HRRT Dhaka | STEP 2b
Accuracy of the current water map (Sentinel-2 MNDWI > 0, Jan-Mar 2026) from the
120 blind photo-interpreted validation points.

Labels: W = water, L = land, U = undefined (excluded).
The 120 points were drawn by stratified random sampling on the MAP classes
(60 mapped water, 60 mapped non-water), so besides the plain sample statistics the
script also gives area-weighted estimates (Olofsson et al., 2014), which are the
unbiased ones.

Inputs : data/raw/HRRT_Validation.xlsx      columns: ID | Lat / Lon | Label
         data/raw/Dhaka_MNDWI_2026DrySeason.tif
Outputs: data/outputs/step2b_accuracy.csv
         data/outputs/step2b_confusion_matrix.csv
         figures/step2b_confusion_matrix.png

Run from the repository folder:
    python python_scripts/step2b_water_accuracy.py
(needs: pip install openpyxl)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

OTSU = -0.076          # alternative threshold found in Step 2 (sample statistics only)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="data/raw/HRRT_Validation.xlsx")
    p.add_argument("--mndwi", default="data/raw/Dhaka_MNDWI_2026DrySeason.tif")
    p.add_argument("--out", default="data/outputs")
    p.add_argument("--fig", default="figures")
    return p.parse_args()


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def kappa(cm):
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe)


def main():
    a = parse_args()
    out, fig_dir = Path(a.out), Path(a.fig)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(a.labels)
    df.columns = ["id", "latlon", "label"]
    ll = df.latlon.astype(str).str.split(",", expand=True).astype(float)
    df["lat"], df["lon"] = ll[0], ll[1]                      # the sheet is written as "lat, lon"
    df["label"] = df.label.astype(str).str.strip().str.upper()
    print("labels:", df.label.value_counts().to_dict())

    with rasterio.open(a.mndwi) as src:
        arr = src.read(1)
        x, y = Transformer.from_crs(4326, src.crs, always_xy=True).transform(df.lon.values, df.lat.values)
        r, c = rasterio.transform.rowcol(src.transform, x, y)
        r, c = np.clip(r, 0, arr.shape[0] - 1), np.clip(c, 0, arr.shape[1] - 1)
        df["mndwi"] = arr[r, c]
        valid = np.isfinite(arr)
        w_water = float((arr[valid] > 0).mean())             # share of the window mapped as water
        px_km2 = abs(src.res[0] * src.res[1]) / 1e6
        area_km2 = valid.sum() * px_km2
    df["map_water"] = (df.mndwi.fillna(-9) > 0).astype(int)  # masked pixels count as non-water, as in Step 2
    print(f"points mapped water: {df.map_water.sum()} of {len(df)} (design: 60 / 60)")

    d = df[df.label.isin(["W", "L"])].copy()
    d["ref_water"] = (d.label == "W").astype(int)
    n_used, n_u = len(d), int((df.label == "U").sum())

    # confusion matrix: rows = map (non-water, water), columns = reference (land, water)
    cm = np.zeros((2, 2), dtype=int)
    for m_, r_ in zip(d.map_water, d.ref_water):
        cm[m_, r_] += 1
    n_map = cm.sum(1)                                        # points per map class
    oa = np.trace(cm) / n_used
    lo, hi = wilson(np.trace(cm), n_used)
    ua_w = cm[1, 1] / n_map[1] if n_map[1] else np.nan       # user's accuracy, water
    ua_l = cm[0, 0] / n_map[0] if n_map[0] else np.nan
    pa_w = cm[1, 1] / cm[:, 1].sum() if cm[:, 1].sum() else np.nan   # producer's accuracy, water (sample)
    pa_l = cm[0, 0] / cm[:, 0].sum() if cm[:, 0].sum() else np.nan
    f1 = 2 * ua_w * pa_w / (ua_w + pa_w) if (ua_w + pa_w) > 0 else np.nan
    kap = kappa(cm)

    # area-weighted estimates (stratified by map class)
    W = np.array([1 - w_water, w_water])
    U = np.array([ua_l, ua_w])
    p = W[:, None] * cm / n_map[:, None]                      # estimated area proportions
    oa_w = float(np.trace(p))
    se_oa = float(np.sqrt(np.sum(W ** 2 * U * (1 - U) / np.maximum(n_map - 1, 1))))
    pa_w_area = p[1, 1] / p[:, 1].sum() if p[:, 1].sum() else np.nan
    pa_l_area = p[0, 0] / p[:, 0].sum() if p[:, 0].sum() else np.nan
    f1_area = 2 * ua_w * pa_w_area / (ua_w + pa_w_area)
    ref_water = float(p[:, 1].sum())
    q = cm[:, 1] / n_map
    se_ref = float(np.sqrt(np.sum(W ** 2 * q * (1 - q) / np.maximum(n_map - 1, 1))))

    # sensitivity of the plain sample statistics to the threshold
    sens = {}
    for name, thr in (("MNDWI > 0", 0.0), (f"MNDWI > {OTSU}", OTSU)):
        mp = (d.mndwi.fillna(-9) > thr).astype(int)
        c2 = np.zeros((2, 2), dtype=int)
        for m_, r_ in zip(mp, d.ref_water):
            c2[m_, r_] += 1
        sens[name] = (np.trace(c2) / c2.sum(), kappa(c2))

    res = {
        "points_total": len(df), "points_undefined_excluded": n_u, "points_used": n_used,
        "reference_water": int(d.ref_water.sum()), "reference_land": int((1 - d.ref_water).sum()),
        "overall_accuracy_sample_pct": round(oa * 100, 1),
        "OA_95CI_low_pct": round(lo * 100, 1), "OA_95CI_high_pct": round(hi * 100, 1),
        "kappa_sample": round(kap, 3),
        "users_accuracy_water_pct": round(ua_w * 100, 1), "users_accuracy_land_pct": round(ua_l * 100, 1),
        "producers_accuracy_water_sample_pct": round(pa_w * 100, 1),
        "producers_accuracy_land_sample_pct": round(pa_l * 100, 1),
        "F1_water_sample": round(f1, 3),
        "overall_accuracy_area_weighted_pct": round(oa_w * 100, 1),
        "OA_area_weighted_SE_pct": round(se_oa * 100, 1),
        "producers_accuracy_water_area_weighted_pct": round(pa_w_area * 100, 1),
        "producers_accuracy_land_area_weighted_pct": round(pa_l_area * 100, 1),
        "F1_water_area_weighted": round(f1_area, 3),
        "mapped_water_share_of_window_pct": round(w_water * 100, 2),
        "window_area_km2": round(area_km2, 1),
        "mapped_water_area_km2": round(w_water * area_km2, 1),
        "bias_adjusted_water_area_km2": round(ref_water * area_km2, 1),
        "bias_adjusted_water_area_95CI_km2": round(1.96 * se_ref * area_km2, 1),
    }
    pd.Series(res).to_csv(out / "step2b_accuracy.csv", header=["value"])
    pd.DataFrame(cm, index=["map: non-water", "map: water"], columns=["reference: land", "reference: water"]
                 ).to_csv(out / "step2b_confusion_matrix.csv")

    print("\n=== Confusion matrix (counts) ===")
    print(pd.DataFrame(cm, index=["map: non-water", "map: water"], columns=["reference: land", "reference: water"]))
    print("\n=== Accuracy ===")
    for k, v in res.items():
        print(f"{k:48s} {v}")
    print("\nthreshold sensitivity (sample statistics only, not area-weighted):")
    for k, (o, kk) in sens.items():
        print(f"  {k:18s} OA {o * 100:.1f}%  kappa {kk:.3f}")

    # figure for the paper
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"]})
    fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=300)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1) * 1.15)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Land", "Water"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Non-water", "Water"])
    ax.set_xlabel("Reference (photo-interpretation)"); ax.set_ylabel("Map (Sentinel-2 MNDWI)")
    ax.set_title(f"n = {n_used}; OA = {oa * 100:.1f}%, kappa = {kap:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "step2b_confusion_matrix.png")
    print("\nfigure saved:", fig_dir / "step2b_confusion_matrix.png")


if __name__ == "__main__":
    main()
