#!/usr/bin/env python3
"""
HRRT Dhaka | STEP 6b
Temporal indicators for 2000, 2005, 2010, 2015 and 2020, linked to the 2026 results.

(A) Channel condition per year: does water appear along the potential drainage
    network? Source: JRC Global Surface Water yearly classes (Landsat, 30 m).
    A path point is "retained" if water lies within one pixel (30 m) of it.
(B) Encroachment per year: is the land along the network built up?
    Source: GHSL built-up surface (100 m). A point is "encroached" if the
    built-up share of its 100 m cell is >= 0.5; "new" if it was below 0.5 in the
    previous 5-year epoch.
(C) Link to 2026: how the 2026 status (surviving / partial / lost, Sentinel-2,
    Step 3b) relates to the water and built-up histories, which paths held water
    in 2000 but not in 2020, and the history of the 15 priority corridors.

Landsat-based water is coarse (30 m): narrow khals are under-detected, so the
retention values are LOWER BOUNDS and are comparable between years only, never
with the 10 m Sentinel-2 figure for 2026.

Inputs : data/outputs/drainage_network_thr2500.gpkg      (Step 3b)
         data/outputs/step4_priority_corridors.gpkg       (Step 4)
         data/raw/Dhaka_JRC_yearly_water_1990_2021.tif    (Step 6a, 8 bands)
         data/raw/Dhaka_GHSL_builtup_1975_2020.tif        (Step 6a, ~10 bands)
Outputs: figures/step6_channel_condition_<year>.png (5 maps) + step6_channel_condition_panel.png
         figures/step6_encroachment_<year>.png      (5 maps) + step6_encroachment_panel.png
         figures/step6_trends.png, figures/step6_corridors_heatmap.png
         data/outputs/step6_temporal_summary.csv, step6_status_groups.csv,
         step6_segments_temporal.csv, step6_corridors_temporal.csv

Run from the repository folder:
    python python_scripts/step6b_temporal_maps.py
"""
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu

# ------------------------------------------------------------------ settings
YEARS = [2000, 2005, 2010, 2015, 2020]
GSW_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2021]   # band order of the water file
GHSL_FIRST_EPOCH = 1975                                         # band order of the built-up file
WATER_TOL_M = 30                  # one Landsat pixel
WATER_SPACING_M = 30
BUILT_SPACING_M = 50
BUILT_THR = 0.5
SURVIVING_MIN, LOST_MAX = 0.8, 0.2
MIN_VALID = 0.5                   # a segment needs this share of valid (observed) points
HELD_WATER_2000_MIN = 0.5         # "held water in 2000" if retained share >= this
COL = {"surviving": "#1a9850", "partial": "#fdae61", "lost": "#d73027", "nodata": "#bbbbbb"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="data/outputs/drainage_network_thr2500.gpkg")
    p.add_argument("--corridors", default="data/outputs/step4_priority_corridors.gpkg")
    p.add_argument("--gsw", default="data/raw/Dhaka_JRC_yearly_water_1990_2021.tif")
    p.add_argument("--ghsl", default="data/raw/Dhaka_GHSL_builtup_1975_2020.tif")
    p.add_argument("--out", default="data/outputs")
    p.add_argument("--fig", default="figures")
    return p.parse_args()


def read_stack(path, crs_expected):
    with rasterio.open(path) as src:
        if str(src.crs) != str(crs_expected):
            raise SystemExit(f"CRS mismatch in {path}: {src.crs} vs network {crs_expected}")
        return src.read().astype("float32"), src.transform, abs(src.res[0])


def lookup(arr, transform, xs, ys):
    r, c = rowcol(transform, np.asarray(xs), np.asarray(ys))
    r = np.clip(np.asarray(r), 0, arr.shape[0] - 1)
    c = np.clip(np.asarray(c), 0, arr.shape[1] - 1)
    return arr[r, c]


def sample_points(gdf, spacing):
    seg, xs, ys = [], [], []
    for r in gdf.itertuples():
        n = max(2, int(np.ceil(r.geometry.length / spacing)) + 1)
        for d in np.linspace(0, r.geometry.length, n):
            p = r.geometry.interpolate(d)
            seg.append(r.seg_id); xs.append(p.x); ys.append(p.y)
    return pd.DataFrame({"seg_id": seg, "x": xs, "y": ys})


def classify(v):
    if not np.isfinite(v):
        return "nodata"
    return "surviving" if v >= SURVIVING_MIN else ("lost" if v <= LOST_MAX else "partial")


def extent(tr, shape, res):
    left, top = tr.c, tr.f
    return (left, left + shape[1] * res, top - shape[0] * res, top)


def km_ticks(ax):
    xt = ax.get_xticks(); ax.set_xticks(xt); ax.set_xticklabels([f"{t / 1000:.0f}" for t in xt])
    yt = ax.get_yticks(); ax.set_yticks(yt); ax.set_yticklabels([f"{t / 1000:.0f}" for t in yt])


def main():
    a = parse_args()
    out, fig_dir = Path(a.out), Path(a.fig)
    fig_dir.mkdir(parents=True, exist_ok=True)

    net = gpd.read_file(a.network)
    cor = gpd.read_file(a.corridors)
    gsw, g_tr, g_res = read_stack(a.gsw, net.crs)
    ghsl, h_tr, h_res = read_stack(a.ghsl, net.crs)
    if gsw.shape[0] != len(GSW_YEARS):
        raise SystemExit(f"Expected {len(GSW_YEARS)} bands in the water file, found {gsw.shape[0]}.")
    epochs = [GHSL_FIRST_EPOCH + 5 * i for i in range(ghsl.shape[0])]
    print(f"water file years: {GSW_YEARS}\nbuilt-up file epochs (assumed chronological): {epochs}")
    for y in YEARS + [YEARS[0] - 5]:
        if y not in epochs:
            raise SystemExit(f"Built-up epoch {y} missing; found {epochs}. Send me the band names printed in Earth Engine.")
    gidx = {y: i for i, y in enumerate(GSW_YEARS)}
    bidx = {y: i for i, y in enumerate(epochs)}

    info = net[["seg_id", "length_m", "main_river", "status"]].rename(columns={"status": "status_2026"})
    khal = ~net.main_river

    # ================= (A) water along the network, per year
    pw = sample_points(net, WATER_SPACING_M)
    water_masks = {}
    for y in YEARS:
        cls = gsw[gidx[y]]
        water = cls >= 2
        water_masks[y] = water
        dist = ndi.distance_transform_edt(~water, sampling=g_res)
        pw[f"valid{y}"] = lookup(cls, g_tr, pw.x, pw.y) > 0
        pw[f"ret{y}"] = lookup(dist, g_tr, pw.x, pw.y) <= WATER_TOL_M
    seg = info.copy()
    for y in YEARS:
        v = pw.groupby("seg_id")[f"valid{y}"].mean()
        r = (pw[f"ret{y}"] & pw[f"valid{y}"]).groupby(pw.seg_id).sum() / pw.groupby("seg_id")[f"valid{y}"].sum().replace(0, np.nan)
        seg[f"w{y}"] = seg.seg_id.map(r.where(v >= MIN_VALID))
        seg[f"status{y}"] = seg[f"w{y}"].apply(classify)

    # ================= (B) built-up along the network, per epoch
    pb = sample_points(net, BUILT_SPACING_M)
    for y in YEARS + [YEARS[0] - 5]:
        pb[f"b{y}"] = lookup(np.clip(ghsl[bidx[y]] / 10000.0, 0, 1), h_tr, pb.x, pb.y)
    pb = pb.merge(info[["seg_id", "main_river"]], on="seg_id")
    for y in YEARS:
        pb[f"enc{y}"] = pb[f"b{y}"] >= BUILT_THR
        pb[f"new{y}"] = pb[f"enc{y}"] & ~(pb[f"b{y - 5}"] >= BUILT_THR)
        seg[f"b{y}"] = seg.seg_id.map(pb.groupby("seg_id")[f"b{y}"].mean())
    seg["d_built_2000_2020"] = seg.b2020 - seg.b2000

    # ================= yearly summary (khal-scale network)
    kseg = seg[~seg.main_river]
    kpb = pb[~pb.main_river]
    rows = []
    for y in YEARS:
        ok = kseg[f"w{y}"].notna()
        tot = kseg.loc[ok, "length_m"].sum()
        ret = (kseg.loc[ok, "length_m"] * kseg.loc[ok, f"w{y}"]).sum() / tot if tot else np.nan
        rows.append({"year": y, "network_km_with_data": round(tot / 1000, 1),
                     "retention_30m": round(ret, 3), "HCDI_30m": round(1 - ret, 3),
                     "water_km_surviving": round(kseg.loc[kseg[f"status{y}"] == "surviving", "length_m"].sum() / 1000, 1),
                     "built_share_of_path_points": round(kpb[f"enc{y}"].mean(), 3),
                     "km_newly_encroached": round(kpb[f"new{y}"].sum() * BUILT_SPACING_M / 1000, 1),
                     "mean_built_fraction": round(kpb[f"b{y}"].mean(), 3)})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "step6_temporal_summary.csv", index=False)
    print("\n=== Yearly summary, khal-scale potential network (Landsat water 30 m; GHSL 100 m) ===")
    print(summ.to_string(index=False))

    # ================= (C) link with the 2026 analysis
    grp = []
    for st in ("surviving", "partial", "lost"):
        d = kseg[kseg.status_2026 == st]
        row = {"status_2026": st, "segments": len(d), "km": round(d.length_m.sum() / 1000, 1)}
        for y in YEARS:
            row[f"water{y}"] = round(d[f"w{y}"].mean(), 3)
        for y in YEARS:
            row[f"built{y}"] = round(d[f"b{y}"].mean(), 3)
        row["built_change_2000_2020"] = round(d.d_built_2000_2020.mean(), 3)
        grp.append(row)
    grp = pd.DataFrame(grp)
    grp.to_csv(out / "step6_status_groups.csv", index=False)
    print("\n=== Water and built-up history by 2026 status ===")
    print(grp.to_string(index=False))
    lo = kseg[kseg.status_2026 == "lost"].d_built_2000_2020.dropna()
    su = kseg[kseg.status_2026 == "surviving"].d_built_2000_2020.dropna()
    if len(lo) >= 5 and len(su) >= 5:
        p = mannwhitneyu(lo, su, alternative="greater").pvalue
        print(f"\nbuilt-up increase 2000-2020: lost segments {lo.mean():.3f} vs surviving {su.mean():.3f} "
              f"(Mann-Whitney, lost > surviving: p = {p:.3f}; descriptive only - neighbouring segments are not independent)")

    held = kseg[(kseg.w2000 >= HELD_WATER_2000_MIN) & (kseg.w2020 <= LOST_MAX)]
    gained = kseg[(kseg.w2000 <= LOST_MAX) & (kseg.w2020 >= SURVIVING_MIN)]
    print(f"\npaths with water in 2000 but not 2020: {len(held)} segments, {held.length_m.sum() / 1000:.1f} km")
    print(f"paths without water in 2000 but with water in 2020 (noise / new water): {len(gained)} segments, "
          f"{gained.length_m.sum() / 1000:.1f} km")
    seg["lost_since_2000"] = seg.seg_id.isin(held.seg_id)
    seg.to_csv(out / "step6_segments_temporal.csv", index=False)

    ct = cor[["corridor", "seg_id"]].merge(seg, on="seg_id", how="left")
    ctab = ct.groupby("corridor").agg(
        **{f"water{y}": (f"w{y}", "mean") for y in YEARS},
        **{f"built{y}": (f"b{y}", "mean") for y in YEARS}).round(3)
    ctab["built_change_2000_2020"] = (ctab.built2020 - ctab.built2000).round(3)
    ctab.reset_index().to_csv(out / "step6_corridors_temporal.csv", index=False)
    print("\n=== Priority corridors: water and built-up history ===")
    print(ctab.to_string())

    # ================= figures
    bounds = net.total_bounds
    pad = 1500
    ext_g = {y: extent(g_tr, gsw[gidx[y]].shape, g_res) for y in YEARS}

    def draw_channel(ax, y):
        ax.imshow(np.where(water_masks[y], 1, np.nan), extent=ext_g[y], cmap="Blues", vmin=0, vmax=1.6,
                  interpolation="nearest")
        m = net.merge(seg[["seg_id", f"status{y}"]], on="seg_id")
        m[m.main_river].plot(ax=ax, color="#777777", linewidth=0.6)
        for st, c in COL.items():
            d = m[(~m.main_river) & (m[f"status{y}"] == st)]
            if len(d):
                d.plot(ax=ax, color=c, linewidth=0.9)
        r = summ.loc[summ.year == y].iloc[0]
        ax.set_title(f"{y}: HCDI {r.HCDI_30m:.2f} (30 m)", fontsize=9)
        ax.set_xlim(bounds[0] - pad, bounds[2] + pad); ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
        km_ticks(ax)

    def draw_enc(ax, y):
        ax.imshow(np.clip(ghsl[bidx[y]] / 10000.0, 0, 1), extent=extent(h_tr, ghsl[0].shape, h_res),
                  cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        net[net.main_river].plot(ax=ax, color="#4393c3", linewidth=0.7)
        net[~net.main_river].plot(ax=ax, color="#9ecae1", linewidth=0.4)
        e = pb[pb[f"enc{y}"] & ~pb[f"new{y}"]]
        n = pb[pb[f"new{y}"]]
        ax.scatter(e.x, e.y, s=2, c="#d73027", linewidths=0)
        ax.scatter(n.x, n.y, s=10, c="#ff7f00", edgecolors="k", linewidths=0.3)
        r = summ.loc[summ.year == y].iloc[0]
        ax.set_title(f"{y}: {r.built_share_of_path_points * 100:.0f}% of path points built-up; "
                     f"{r.km_newly_encroached:.1f} km new", fontsize=8)
        ax.set_xlim(bounds[0] - pad, bounds[2] + pad); ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
        km_ticks(ax)

    leg_ch = [Line2D([0], [0], color=COL[k], lw=2, label=l) for k, l in
              [("surviving", "water along path (>=80%)"), ("partial", "partial"), ("lost", "no water (<=20%)"),
               ("nodata", "no Landsat data")]] + [Line2D([0], [0], color="#777777", lw=2, label="main river/lake")]
    leg_en = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#d73027", markersize=5, label="built-up path point"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f00", markeredgecolor="k", markersize=7,
                     label="newly built-up since previous epoch"),
              Line2D([0], [0], color="#9ecae1", lw=2, label="potential drainage path")]

    for kind, draw, leg, name in (("channel_condition", draw_channel, leg_ch, "channel condition"),
                                  ("encroachment", draw_enc, leg_en, "built-up encroachment")):
        for y in YEARS:
            fig, ax = plt.subplots(figsize=(6.5, 8), dpi=200)
            draw(ax, y)
            ax.legend(handles=leg, loc="lower left", fontsize=7)
            ax.set_xlabel("Easting (km, UTM 46N)"); ax.set_ylabel("Northing (km)")
            ax.set_title(f"Potential drainage network: {name}, {y}\n" + ax.get_title(), fontsize=9)
            fig.tight_layout()
            fig.savefig(fig_dir / f"step6_{kind}_{y}.png")
            plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(15, 13), dpi=150)
        for ax, y in zip(axes.ravel(), YEARS):
            draw(ax, y)
        axes.ravel()[5].axis("off")
        axes.ravel()[5].legend(handles=leg, loc="center", fontsize=10, title=name.capitalize())
        fig.tight_layout()
        fig.savefig(fig_dir / f"step6_{kind}_panel.png")
        plt.close(fig)

    # trends
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), dpi=200)
    ax[0, 0].plot(summ.year, summ.HCDI_30m, "o-", color="#d73027")
    ax[0, 0].set_title("(a) Connectivity deficit (HCDI) from Landsat water, 30 m", fontsize=9)
    ax[0, 0].set_ylabel("HCDI (1 - retention)")
    ax[0, 1].plot(summ.year, summ.built_share_of_path_points * 100, "o-", color="#333333")
    ax2 = ax[0, 1].twinx()
    ax2.bar(summ.year, summ.km_newly_encroached, width=1.6, color="#ff7f00", alpha=0.5)
    ax2.set_ylabel("km newly built-up per epoch", color="#ff7f00")
    ax[0, 1].set_title("(b) Path points with built-up land (GHSL)", fontsize=9)
    ax[0, 1].set_ylabel("% of path points built-up")
    for st in ("surviving", "partial", "lost"):
        ax[1, 0].plot(YEARS, grp.loc[grp.status_2026 == st, [f"water{y}" for y in YEARS]].values[0], "o-",
                      color=COL[st], label=f"{st} in 2026")
        ax[1, 1].plot(YEARS, grp.loc[grp.status_2026 == st, [f"built{y}" for y in YEARS]].values[0], "o-",
                      color=COL[st], label=f"{st} in 2026")
    ax[1, 0].set_title("(c) Water along paths, by 2026 status", fontsize=9)
    ax[1, 0].set_ylabel("mean share of path with water")
    ax[1, 1].set_title("(d) Built-up fraction along paths, by 2026 status", fontsize=9)
    ax[1, 1].set_ylabel("mean built-up fraction")
    ax[1, 0].legend(fontsize=8); ax[1, 1].legend(fontsize=8)
    for axx in ax.ravel():
        axx.set_xticks(YEARS)
    fig.tight_layout()
    fig.savefig(fig_dir / "step6_trends.png")
    plt.close(fig)

    # corridor heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(11, 6), dpi=200)
    for axx, pref, title, cmap in ((axs[0], "water", "Water along corridor path", "Blues"),
                                   (axs[1], "built", "Built-up fraction along corridor", "Reds")):
        M = ctab[[f"{pref}{y}" for y in YEARS]].values
        im = axx.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        axx.set_xticks(range(len(YEARS))); axx.set_xticklabels(YEARS)
        axx.set_yticks(range(len(ctab))); axx.set_yticklabels([f"Corridor {c}" for c in ctab.index], fontsize=7)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    axx.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                             color="white" if M[i, j] > 0.6 else "black")
        axx.set_title(title, fontsize=10)
        fig.colorbar(im, ax=axx, shrink=0.7)
    fig.tight_layout()
    fig.savefig(fig_dir / "step6_corridors_heatmap.png")
    plt.close(fig)
    print("\nfigures saved in", fig_dir)


if __name__ == "__main__":
    main()
