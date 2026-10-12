#!/usr/bin/env python3
"""
HRRT Dhaka | STEP 3c
Plausibility check of the DEM-derived drainage paths against OpenStreetMap
waterways (canal / drain / stream / ditch), used to choose the main threshold.

Needs the Step 3b outputs (data/outputs/drainage_network_thr*.gpkg) and an
internet connection (first run downloads OSM waterways from the Overpass API
and caches them in data/raw/osm_waterways.json).

Outputs:
    data/outputs/step3c_osm_agreement.csv   recall / precision / F1 per threshold and buffer
    data/outputs/step3c_named_khals.csv     coverage of each named OSM waterway
Data (c) OpenStreetMap contributors, ODbL.

Run from the repository folder:
    python python_scripts/step3c_osm_check.py
"""
import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.affinity import translate
from shapely.geometry import LineString
from shapely.ops import unary_union

# ------------------------------------------------------------------ settings
BBOX = (23.62, 90.28, 23.92, 90.52)      # south, west, north, east (= Earth Engine AOI)
THRESHOLDS = [1000, 2500, 5000]          # same thresholds as Step 3b
BUFFERS_M = [30, 50, 100]                # matching distance between a DEM path and an OSM line
MAIN_BUFFER = 50
SHIFT_M = 750                            # null test: shift the DEM network this far
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
CRS_UTM = "EPSG:32646"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/outputs")
    p.add_argument("--osm-cache", default="data/raw/osm_waterways.json")
    return p.parse_args()


def get_osm(cache):
    cache = Path(cache)
    if cache.exists():
        print("using cached OSM file:", cache)
        return json.loads(cache.read_text(encoding="utf-8"))
    s, w, n, e = BBOX
    query = (
        '[out:json][timeout:180];'
        f'(way["waterway"~"^(canal|drain|stream|ditch|river)$"]({s},{w},{n},{e}););'
        'out geom;'
    )
    body = urllib.parse.urlencode({"data": query}).encode()
    last = None
    for url in OVERPASS_URLS:
        try:
            print("downloading OSM waterways from", url)
            req = urllib.request.Request(url, data=body, headers={"User-Agent": "HRRT-Dhaka-research/1.0"})
            with urllib.request.urlopen(req, timeout=240) as r:
                text = r.read().decode("utf-8")
            js = json.loads(text)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
            return js
        except Exception as ex:  # try the next server
            last = ex
            print("  failed:", ex)
    raise SystemExit(f"Could not download OSM data ({last}). Wait a minute and run again.")


def osm_to_gdf(js):
    rows, geoms = [], []
    for el in js.get("elements", []):
        g = el.get("geometry")
        if not g or len(g) < 2:
            continue
        tags = el.get("tags", {})
        rows.append({"osm_id": el.get("id"), "waterway": tags.get("waterway", ""),
                     "name": tags.get("name:en") or tags.get("name") or ""})
        geoms.append(LineString([(p["lon"], p["lat"]) for p in g]))
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326").to_crs(CRS_UTM)
    return gdf


def share_within(lines, zone):
    """Share of the total length of `lines` that lies inside `zone` (a polygon)."""
    total = lines.length.sum()
    if total == 0:
        return float("nan")
    return float(lines.intersection(zone).length.sum() / total)


def zone(lines, buf):
    return unary_union(list(lines.buffer(buf)))


def main():
    a = parse_args()
    out = Path(a.out)

    osm = osm_to_gdf(get_osm(a.osm_cache))
    print("\nOSM waterways in the study area (km):")
    osm["length_m"] = osm.length
    print((osm.groupby("waterway")["length_m"].sum() / 1000).round(1).to_string())
    osm_k = osm[osm.waterway != "river"].reset_index(drop=True)   # khal-scale: canal/drain/stream/ditch
    print(f"khal-scale OSM length: {osm_k.length.sum() / 1000:.1f} km; "
          f"named: {(osm_k.name != '').sum()} of {len(osm_k)} ways")

    osm_zones = {b: zone(osm_k, b) for b in BUFFERS_M}
    rows = []
    for thr in THRESHOLDS:
        dem = gpd.read_file(out / f"drainage_network_thr{thr}.gpkg")
        dem = dem[~dem["main_river"]].reset_index(drop=True)
        for b in BUFFERS_M:
            recall = share_within(osm_k, zone(dem, b))        # OSM length matched by a DEM path
            precision = share_within(dem, osm_zones[b])       # DEM length matched by an OSM line
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else float("nan")
            row = {"threshold_cells": thr, "buffer_m": b, "dem_km": round(dem.length.sum() / 1000, 1),
                   "osm_km": round(osm_k.length.sum() / 1000, 1),
                   "recall": round(recall, 3), "precision": round(precision, 3), "F1": round(f1, 3)}
            if b == MAIN_BUFFER:
                # null test: the same network shifted sideways should match OSM much less
                sh = []
                for dx, dy in ((SHIFT_M, 0), (-SHIFT_M, 0), (0, SHIFT_M), (0, -SHIFT_M)):
                    moved = dem.copy()
                    moved["geometry"] = [translate(g, xoff=dx, yoff=dy) for g in dem.geometry]
                    sh.append(share_within(osm_k, zone(moved, b)))
                row["recall_if_shifted"] = round(sum(sh) / len(sh), 3)
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "step3c_osm_agreement.csv", index=False)
    print("\n=== Agreement of DEM paths with OSM khal-scale waterways ===")
    print(res.to_string(index=False))

    main_rows = res[res.buffer_m == MAIN_BUFFER]
    best = main_rows.loc[main_rows["F1"].idxmax()]
    print(f"\nBEST threshold by F1 at {MAIN_BUFFER} m: {int(best.threshold_cells)} cells "
          f"(recall {best.recall}, precision {best.precision}, F1 {best.F1}; "
          f"recall if shifted {best.recall_if_shifted})")

    # named waterways: how much of each is matched by the DEM paths of the best threshold
    dem_best = gpd.read_file(out / f"drainage_network_thr{int(best.threshold_cells)}.gpkg")
    dem_best = dem_best[~dem_best["main_river"]]
    z = zone(dem_best, MAIN_BUFFER)
    named = osm_k[osm_k.name != ""].copy()
    named["covered_m"] = named.geometry.intersection(z).length
    named["length_m"] = named.length
    tab = named.groupby("name")[["length_m", "covered_m"]].sum()
    tab["coverage"] = (tab.covered_m / tab.length_m).round(2)
    tab = tab.sort_values("length_m", ascending=False)
    tab[["length_m", "covered_m"]] = tab[["length_m", "covered_m"]].round(0)
    tab.to_csv(out / "step3c_named_khals.csv")
    print("\nLongest named OSM waterways and share matched by DEM paths:")
    print(tab.head(15).to_string())


if __name__ == "__main__":
    main()
