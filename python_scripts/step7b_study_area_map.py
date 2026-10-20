#!/usr/bin/env python3
"""
HRRT Dhaka | STEP 7b
Study-area map. The study area is the rectangular window used in every step of the
analysis (23.62-23.92 N, 90.28-90.52 E).

Layout
  * main map: current surface water (Sentinel-2, Jan-Mar 2026) and the Buriganga, Turag,
    Balu and Shitalakshya rivers; coordinate ticks and labels on all four sides
    (degrees-minutes-seconds), faint grid lines, compass rose (top right) and
    scale bar (bottom right)
  * right column: satellite view of Bangladesh with the study window in a red box,
    and below it a legend box aligned with the left edge of that map
  * all text in Times New Roman

It also measures how much of each river's OpenStreetMap-mapped length lies inside the
window (the rivers are longer than the window) and saves that to a CSV.

Inputs : data/raw/Dhaka_MNDWI_2026DrySeason.tif      (Step 1 export, EPSG:32646)
         data/raw/Bangladesh_divisions.geojson       (Step 7a export)
         OpenStreetMap river lines (downloaded on the first run and cached in
         data/raw/osm_main_rivers_v2.json)
         Esri World Imagery tiles for the satellite inset (needs internet and the
         package contextily; without them the inset uses a plain grey background)
Output : figures/step7_study_area_map.png (300 dpi) and .pdf
         data/outputs/step7_river_coverage.csv

Run from the repository folder:
    python python_scripts/step7b_study_area_map.py
Options: --paths  also draws the 2026 status of the potential drainage paths
"""
import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle, Wedge
from pyproj import Transformer
from shapely.geometry import LineString, box

# ------------------------------------------------------------------ settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

BBOX = (23.62, 90.28, 23.92, 90.52)         # south, west, north, east = the study window
BBOX_WIDE = (23.30, 90.00, 24.50, 91.00)    # wider box used only to find the full river courses
CRS_UTM = "EPSG:32646"
WATER_COLOR = "#6baed6"
RIVERS = {                                  # order = order in the legend
    "Buriganga": {"color": "#d95f02", "regex": r"buriganga|বুড়িগঙ্গা"},
    "Turag": {"color": "#1b9e77", "regex": r"turag|তুরাগ"},
    "Balu": {"color": "#7570b3", "regex": r"\bbalu\b|বালু"},
    "Shitalakshya": {"color": "#e7298a", "regex": r"shital|sital|lakhya|lakkhya|শীতল"},
}
# short name fragments used to ask OpenStreetMap (spelling varies a lot in OSM)
QUERY_PATTERNS = {
    "Buriganga": "buriganga|বুড়িগঙ্গা",
    "Turag": "turag|তুরাগ",
    "Balu": "balu|বালু",
    "Shitalakshya": "shital|sital|lakhya|lakkhya|শীতল",
}
NAME_KEYS = "^(name|alt_name|old_name|name:.*)$"
TITLE = "Location of the Buriganga, Turag, Balu and Shitalakshya river system surrounding Dhaka, 2026"
SUBTITLE = "Extent: 23.62-23.92°N, 90.28-90.52°E (about 24.5 km x 33 km)"
SCALE_BAR_KM = 4                 # bar length; labels at 0, 1/4, 1/2 and the full length
GRID_STEP_DEG = 0.05             # 0.05 degrees = 3 arc-minutes
GRID_LINES = True                # False = ticks and labels only, no lines across the map
SAT_ZOOM = 8                     # detail of the satellite inset
STATUS_COLORS = {"surviving": "#1a9850", "partial": "#fdae61", "lost": "#d73027"}
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
FIG_W, FIG_H = 11.5, 9.4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--water", default="data/raw/Dhaka_MNDWI_2026DrySeason.tif")
    p.add_argument("--divisions", default="data/raw/Bangladesh_divisions.geojson")
    p.add_argument("--rivers-cache", default="data/raw/osm_main_rivers_v2.json")
    p.add_argument("--fallback-cache", default="data/raw/osm_main_rivers_fallback.json")
    p.add_argument("--network", default="data/outputs/drainage_network_thr2500.gpkg")
    p.add_argument("--out", default="data/outputs")
    p.add_argument("--paths", action="store_true")
    p.add_argument("--fig", default="figures")
    return p.parse_args()


# ------------------------------------------------------------------ OpenStreetMap
def overpass(query, cache=None):
    if cache and Path(cache).exists():
        print("using cached file:", cache)
        return json.loads(Path(cache).read_text(encoding="utf-8"))
    body = urllib.parse.urlencode({"data": query}).encode()
    last = None
    for url in OVERPASS_URLS:
        try:
            print("downloading from", url)
            req = urllib.request.Request(url, data=body, headers={"User-Agent": "HRRT-Dhaka-research/1.0"})
            with urllib.request.urlopen(req, timeout=240) as r:
                text = r.read().decode("utf-8")
            js = json.loads(text)
            if cache and js.get("elements"):            # never cache an empty answer
                Path(cache).parent.mkdir(parents=True, exist_ok=True)
                Path(cache).write_text(text, encoding="utf-8")
            return js
        except Exception as ex:
            last = ex
            print("  failed:", ex)
    print(f"Could not download ({last}).")
    return {"elements": []}


def river_query(pattern, waterways, bbox):
    s, w, n, e = bbox
    bb = f"{s},{w},{n},{e}"
    return ('[out:json][timeout:180];('
            f'way["waterway"~"^({waterways})$"][~"{NAME_KEYS}"~"{pattern}",i]({bb});'
            f'relation["waterway"~"^({waterways})$"][~"{NAME_KEYS}"~"{pattern}",i]({bb});'
            f'relation["type"="waterway"][~"{NAME_KEYS}"~"{pattern}",i]({bb});'
            ');out geom;')


def classify_name(tags, wanted):
    text = " ".join(str(v) for k, v in tags.items() if k == "name" or k == "alt_name"
                    or k == "old_name" or k.startswith("name:"))
    for key in wanted:
        if re.search(RIVERS[key]["regex"], text, re.I):
            return key
    return None


def collect_lines(js, wanted, parts):
    """Add river lines from an Overpass answer to `parts` (ways first, then relation members)."""
    seen = set()
    for el in js.get("elements", []):
        if el.get("type") != "way":
            continue
        g = el.get("geometry")
        key = classify_name(el.get("tags", {}), wanted)
        if key and g and len(g) >= 2:
            parts[key].append(LineString([(p["lon"], p["lat"]) for p in g]))
            seen.add(el.get("id"))
    for el in js.get("elements", []):
        if el.get("type") != "relation":
            continue
        key = classify_name(el.get("tags", {}), wanted)
        if not key:
            continue
        for m in el.get("members", []):
            g = m.get("geometry")
            if m.get("type") == "way" and g and len(g) >= 2 and m.get("ref") not in seen:
                parts[key].append(LineString([(p["lon"], p["lat"]) for p in g]))
                seen.add(m.get("ref"))


def get_rivers(cache, fallback_cache):
    parts = {k: [] for k in RIVERS}
    pattern = "|".join(QUERY_PATTERNS.values())
    collect_lines(overpass(river_query(pattern, "river", BBOX_WIDE), cache), list(RIVERS), parts)

    missing = [k for k in RIVERS if not parts[k]]
    if missing:
        print("Not found as waterway=river:", ", ".join(missing), "- trying canals, streams and drains")
        pat = "|".join(QUERY_PATTERNS[k] for k in missing)
        collect_lines(overpass(river_query(pat, "river|canal|stream|drain|ditch", BBOX_WIDE), fallback_cache),
                      missing, parts)

    still = [k for k in RIVERS if not parts[k]]
    for key in still:
        print(f"WARNING: no OSM river lines found for {key}")
    if still:
        s, w, n, e = BBOX_WIDE
        q = (f'[out:json][timeout:60];way["waterway"~"^(river|canal)$"]["name"]'
             f'({s + 0.3},{w + 0.4},{n - 0.55},{e - 0.3});out tags;')
        names = sorted({el["tags"]["name"] for el in overpass(q).get("elements", []) if "tags" in el})
        print("Named rivers and canals in OSM around Dhaka (send me this list):")
        print(", ".join(names[:60]) if names else "(none returned)")

    out = {}
    for key, lines in parts.items():
        if lines:
            out[key] = gpd.GeoSeries(lines, crs="EPSG:4326").to_crs(CRS_UTM)
    return out


# ------------------------------------------------------------------ cartography
def dms(value, hemi):
    total = int(round(abs(value) * 3600))
    d, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{d}°{m}'{s}\"{hemi}"


def frame_ticks(frame):
    """Where each meridian and parallel meets the four sides of the map frame."""
    left, right, bottom, top = frame
    tr = Transformer.from_crs(4326, CRS_UTM, always_xy=True)
    s, w, n, e = BBOX
    lons = np.arange(np.ceil(w / GRID_STEP_DEG) * GRID_STEP_DEG, e + 1e-9, GRID_STEP_DEG)
    lats = np.arange(np.ceil(s / GRID_STEP_DEG) * GRID_STEP_DEG, n + 1e-9, GRID_STEP_DEG)
    la = np.linspace(s - 0.1, n + 0.1, 400)
    lo = np.linspace(w - 0.1, e + 0.1, 400)
    res = {"bottom": [], "top": [], "left": [], "right": [], "lines": []}
    for lon in lons:
        xs, ys = tr.transform(np.full_like(la, lon), la)
        res["lines"].append((xs, ys))
        xb, xt = float(np.interp(bottom, ys, xs)), float(np.interp(top, ys, xs))
        lab = dms(lon, "E")
        if left <= xb <= right:
            res["bottom"].append((xb, lab))
        if left <= xt <= right:
            res["top"].append((xt, lab))
    for lat in lats:
        xs, ys = tr.transform(lo, np.full_like(lo, lat))
        res["lines"].append((xs, ys))
        yl, yr = float(np.interp(left, xs, ys)), float(np.interp(right, xs, ys))
        lab = dms(lat, "N")
        if bottom <= yl <= top:
            res["left"].append((yl, lab))
        if bottom <= yr <= top:
            res["right"].append((yr, lab))
    return res


def draw_grid_and_ticks(ax, frame):
    t = frame_ticks(frame)
    if GRID_LINES:
        for xs, ys in t["lines"]:
            ax.plot(xs, ys, color="0.4", lw=0.4, ls=(0, (2, 3)), zorder=2)
    fs = 7
    ax.set_xticks([p for p, _ in t["bottom"]])
    ax.set_xticklabels([l for _, l in t["bottom"]], fontsize=fs, fontweight="bold")
    ax.set_yticks([p for p, _ in t["left"]])
    ax.set_yticklabels([l for _, l in t["left"]], fontsize=fs, fontweight="bold", rotation=90, va="center")
    ax.tick_params(direction="out", length=5, width=0.9, pad=3)
    top = ax.secondary_xaxis("top")
    top.set_xticks([p for p, _ in t["top"]])
    top.set_xticklabels([l for _, l in t["top"]], fontsize=fs, fontweight="bold")
    top.tick_params(direction="out", length=5, width=0.9, pad=3)
    right = ax.secondary_yaxis("right")
    right.set_yticks([p for p, _ in t["right"]])
    right.set_yticklabels([l for _, l in t["right"]], fontsize=fs, fontweight="bold", rotation=90, va="center")
    right.tick_params(direction="out", length=5, width=0.9, pad=3)
    for a in (ax, top, right):
        for sp in a.spines.values():
            sp.set_linewidth(1.8)


def draw_scale_bar(ax, frame, km):
    """Line scale bar with ticks and labels above, bottom right (style of a GIS layout)."""
    left, right, bottom, top = frame
    W, H = right - left, top - bottom
    L = km * 1000
    x0 = right - 0.24 * W - L                       # leaves room for the word "Kilometers"
    y0 = bottom + 0.045 * H
    ax.add_patch(Rectangle((x0 - 0.04 * W, y0 - 0.025 * H), L + 0.26 * W, 0.105 * H,
                           fc="white", ec="0.25", lw=0.6, alpha=0.93, zorder=6))
    ax.plot([x0, x0 + L], [y0, y0], color="black", lw=1.3, zorder=7, solid_capstyle="butt")
    for i in range(9):                                # minor ticks every L/8
        x = x0 + i * L / 8
        major = i in (0, 2, 4, 8)
        ax.plot([x, x], [y0, y0 - (0.014 if major else 0.008) * H], color="black", lw=1.1, zorder=7)
    for i, frac in zip((0, 2, 4, 8), (0, 0.25, 0.5, 1.0)):
        lab = f"{km * frac:g}"
        ax.text(x0 + frac * L, y0 + 0.012 * H, lab, ha="center", va="bottom", fontsize=9,
                fontweight="bold", zorder=8)
    ax.text(x0 + L + 0.012 * W, y0 + 0.012 * H, "Kilometers", ha="left", va="bottom", fontsize=9,
            fontweight="bold", zorder=8)


def draw_compass(ax):
    """Four-point compass rose with N, E, S, W, top right of the map."""
    w = 0.20
    h = w * ax.get_position().width * FIG_W / (ax.get_position().height * FIG_H)   # keep it square
    axc = ax.inset_axes([0.975 - w, 0.975 - h, w, h])
    axc.set_xlim(-1.75, 1.75); axc.set_ylim(-1.75, 1.75)
    axc.set_aspect("equal"); axc.set_xticks([]); axc.set_yticks([])
    axc.patch.set_facecolor("white"); axc.patch.set_alpha(0.93)
    for sp in axc.spines.values():
        sp.set_linewidth(0.6); sp.set_color("0.25")
    for ang in (0, 90, 180, 270):
        rot = np.array([[np.cos(np.radians(ang)), -np.sin(np.radians(ang))],
                        [np.sin(np.radians(ang)), np.cos(np.radians(ang))]])
        tip = rot @ np.array([0.0, 1.05])
        lf, rt = rot @ np.array([-0.2, 0.0]), rot @ np.array([0.2, 0.0])
        axc.add_patch(Polygon([tip, lf, (0, 0)], fc="black", ec="black", lw=0.7, zorder=3))
        axc.add_patch(Polygon([tip, rt, (0, 0)], fc="white", ec="black", lw=0.7, zorder=3))
    axc.add_patch(Circle((0, 0), 0.21, fc="white", ec="black", lw=0.8, zorder=4))
    for th, col in ((0, "black"), (180, "black")):
        axc.add_patch(Wedge((0, 0), 0.21, th + 90, th + 180, fc=col, ec="black", lw=0.5, zorder=5))
    for txt, (x, y) in (("N", (0, 1.38)), ("S", (0, -1.42)), ("W", (-1.38, -0.02)), ("E", (1.4, -0.02))):
        axc.text(x, y, txt, ha="center", va="center", fontsize=11, fontweight="bold", zorder=6)


def add_satellite(axi):
    try:
        import contextily as ctx
        ctx.add_basemap(axi, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, zoom=SAT_ZOOM,
                        attribution=False)
        return True
    except Exception as ex:
        print(f"Satellite basemap not available ({ex}); using a plain background. "
              "Install it with: pip install contextily (and check your internet connection).")
        axi.set_facecolor("#e6e6e6")
        return False


def main():
    a = parse_args()
    fig_dir, out_dir = Path(a.fig), Path(a.out)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- current water (downsampled 2x with max, so thin rivers survive)
    with rasterio.open(a.water) as src:
        arr = src.read(1)
        tr, res = src.transform, abs(src.res[0])
    water = np.isfinite(arr) & (arr > 0)
    f = 2
    nr, nc = water.shape[0] // f, water.shape[1] // f
    w2 = water[:nr * f, :nc * f].reshape(nr, f, nc, f).max(axis=(1, 3))
    frame = (tr.c, tr.c + nc * f * res, tr.f - nr * f * res, tr.f)    # left, right, bottom, top
    window = box(frame[0], frame[2], frame[1], frame[3])
    aspect = (frame[1] - frame[0]) / (frame[3] - frame[2])

    # ---- rivers: full mapped courses, then the part inside the window
    rivers_full = get_rivers(a.rivers_cache, a.fallback_cache)
    rivers, cov = {}, []
    for key, g in rivers_full.items():
        inside = g.intersection(window)
        inside = inside[~inside.is_empty].explode(index_parts=False)
        inside = inside[inside.geom_type == "LineString"]
        total_km, in_km = g.length.sum() / 1000, inside.length.sum() / 1000
        cov.append({"river": key, "mapped_length_km_osm": round(total_km, 1),
                    "inside_window_km": round(in_km, 1),
                    "share_inside_pct": round(in_km / total_km * 100, 1) if total_km else np.nan})
        if len(inside):
            rivers[key] = inside
    if cov:
        cov = pd.DataFrame(cov)
        cov.to_csv(out_dir / "step7_river_coverage.csv", index=False)
        print("\nShare of each river's OSM-mapped length inside the study window:")
        print(cov.to_string(index=False))

    # ---- page layout, built from its parts (inches) so everything lines up
    global FIG_W
    div = gpd.read_file(a.divisions).to_crs(4326)
    div = div.explode(index_parts=False)                      # split multi-part and mixed geometries
    div = div[div.geom_type == "Polygon"].reset_index(drop=True)
    if div.empty:
        raise SystemExit("No polygons found in " + str(a.divisions) + ". Re-export it with Step 7a.")
    minx, miny, maxx, maxy = div.total_bounds
    pad = 0.55
    ratio = ((maxx - minx + 2 * pad) * np.cos(np.radians(23.7))) / (maxy - miny + 2 * pad)   # inset width / height
    map_h_in = 0.80 * FIG_H
    map_w_in = map_h_in * aspect
    inset_h_in = 3.4
    inset_w_in = inset_h_in * ratio
    left_in, gap_in, right_in = 0.85, 0.95, 0.35
    FIG_W = left_in + map_w_in + gap_in + inset_w_in + right_in
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150)
    bottom_in = 0.075 * FIG_H
    ax = fig.add_axes([left_in / FIG_W, bottom_in / FIG_H, map_w_in / FIG_W, map_h_in / FIG_H])
    ax.imshow(np.ma.masked_where(~w2, w2), extent=(frame[0], frame[1], frame[2], frame[3]),
              cmap=ListedColormap([WATER_COLOR]), interpolation="nearest", zorder=1)
    ax.set_xlim(frame[0], frame[1]); ax.set_ylim(frame[2], frame[3])

    if a.paths and Path(a.network).exists():
        net = gpd.read_file(a.network)
        net = net[~net.main_river]
        for st, c in STATUS_COLORS.items():
            d = net[net.status == st]
            if len(d):
                d.plot(ax=ax, color=c, linewidth=0.8, zorder=3)

    halo = [pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()]
    for key, geoms in rivers.items():
        geoms.plot(ax=ax, color=RIVERS[key]["color"], linewidth=1.8, zorder=5, path_effects=halo)
        big = geoms.iloc[int(np.argmax(geoms.length.values))]
        p = big.interpolate(0.5, normalized=True)
        Wd = frame[1] - frame[0]
        x_lab, ha = p.x, "center"
        if p.x > frame[1] - 0.12 * Wd:
            x_lab, ha = p.x - 0.015 * Wd, "right"
        elif p.x < frame[0] + 0.12 * Wd:
            x_lab, ha = p.x + 0.015 * Wd, "left"
        ax.text(x_lab, p.y, key, fontsize=9, fontweight="bold", color=RIVERS[key]["color"], zorder=9,
                ha=ha, va="center", path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    draw_grid_and_ticks(ax, frame)
    ax.set_xlabel(""); ax.set_ylabel("")
    draw_scale_bar(ax, frame, SCALE_BAR_KM)
    draw_compass(ax)

    # ---- right column: satellite inset of Bangladesh, then the legend box under it (same left edge and width)
    col_x = (left_in + map_w_in + gap_in) / FIG_W
    inset_y = (bottom_in + map_h_in - inset_h_in) / FIG_H
    axi = fig.add_axes([col_x, inset_y, inset_w_in / FIG_W, inset_h_in / FIG_H])
    axi.set_xlim(minx - pad, maxx + pad); axi.set_ylim(miny - pad, maxy + pad)
    add_satellite(axi)
    div.plot(ax=axi, facecolor="none", edgecolor="white", linewidth=0.5, alpha=0.8, zorder=3, aspect=None)
    country = div.union_all() if hasattr(div, "union_all") else div.unary_union
    gpd.GeoSeries([country], crs=4326).plot(ax=axi, facecolor="none", edgecolor="#ffd92f", linewidth=1.2,
                                             zorder=4, aspect=None)
    s, w, n, e = BBOX
    axi.add_patch(Rectangle((w, s), e - w, n - s, fc="none", ec="red", lw=1.8, zorder=6))
    axi.set_xlim(minx - pad, maxx + pad); axi.set_ylim(miny - pad, maxy + pad)
    axi.set_aspect(1 / np.cos(np.radians(23.7)))
    axi.set_xticks([]); axi.set_yticks([]); axi.set_xlabel(""); axi.set_ylabel("")
    axi.annotate("Study area", xy=((w + e) / 2, (s + n) / 2), xytext=(minx - pad + 0.25, s - 1.1),
                 fontsize=10, color="white", fontweight="bold", zorder=7,
                 path_effects=[pe.withStroke(linewidth=2.4, foreground="red")],
                 arrowprops=dict(arrowstyle="-", color="red", lw=1.1))
    axi.text(0.5, 0.97, "BANGLADESH", transform=axi.transAxes, ha="center", va="top", fontsize=10,
             fontweight="bold", color="white", zorder=7,
             path_effects=[pe.withStroke(linewidth=2.6, foreground="black")])
    for sp in axi.spines.values():
        sp.set_linewidth(1.8)

    leg_top = inset_y - 0.025
    leg_bottom = bottom_in / FIG_H + 0.10
    axl = fig.add_axes([col_x, leg_bottom, inset_w_in / FIG_W, leg_top - leg_bottom])
    axl.axis("off")
    handles = [Patch(fc=WATER_COLOR, ec="none", label="Surface water, 2026\n(Sentinel-2 MNDWI > 0)")]
    for key in RIVERS:
        if key in rivers:
            handles.append(Line2D([0], [0], color=RIVERS[key]["color"], lw=2.6, label=f"{key} River"))
    if a.paths:
        for st, c in STATUS_COLORS.items():
            handles.append(Line2D([0], [0], color=c, lw=1.6, label=f"Potential path:\n{st} (2026)"))
    leg = axl.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1, 1, 0), bbox_transform=axl.transAxes,
                     borderaxespad=0, mode="expand",
                     title="Legend", fontsize=9.5, title_fontsize=12, frameon=True, fancybox=False,
                     edgecolor="black", borderpad=0.8, labelspacing=0.85, handlelength=2.2)
    leg.get_title().set_fontweight("bold")
    leg.get_frame().set_linewidth(1.8)
    fig.text(col_x, bottom_in / FIG_H, "River courses continue beyond the map frame.\n"
             "Sources: Sentinel-2 (Copernicus);\nriver lines (c) OpenStreetMap contributors;\n"
             "satellite view: Esri, Maxar, Earthstar\nGeographics; divisions: FAO GAUL 2015.\n"
             "Projection: UTM zone 46N;\ncoordinates in degrees, minutes, seconds.",
             fontsize=7, va="bottom", ha="left", color="0.2")

    fig.suptitle(TITLE, fontsize=13.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, SUBTITLE, ha="center", fontsize=10.5, color="0.15")
    fig.savefig(fig_dir / "step7_study_area_map.png", dpi=300)
    fig.savefig(fig_dir / "step7_study_area_map.pdf")
    plt.close(fig)
    print("\nsaved:", fig_dir / "step7_study_area_map.png")


if __name__ == "__main__":
    main()
