#!/usr/bin/env python3
"""
HRRT Dhaka | STEP 4
Graph analysis of the potential drainage network (Step 3b) and ranking of
restoration corridors.

Idea
  * The network is a directed graph (node = segment end point, edge = segment,
    direction = flow, upstream -> downstream).
  * A segment is FUNCTIONAL if >= 80% of it lies within 15 m of current water,
    or if it belongs to a main river.
  * A segment is CONNECTED if it and every segment downstream of it, up to a
    main river, are functional.
  * Restoring one broken segment helps only if the broken segments downstream of
    it are restored too. So a restoration CORRIDOR = a broken segment plus the
    broken segments between it and the nearest connected point.
  * Restoration Priority Index (RPI) = km of network drained through the
    corridor's entry segment per km of channel that must be restored.
  * Corridors are chosen greedily: pick the best, mark it restored, recompute.
  * Directed edge betweenness centrality is reported for every segment as a
    descriptive network metric.
  * Gain in connected network length is compared with restoring the same number
    of corridors chosen at random.

Input : data/outputs/drainage_network_thr2500.gpkg   (from Step 3b)
Output: data/outputs/step4_segments_ranked.gpkg
        data/outputs/step4_priority_corridors.gpkg   (top PRIORITY_N, used in Step 5)
        data/outputs/step4_priority_corridors.csv
        data/outputs/step4_connectivity_gain.csv
        figures/step4_priority_map.png

Run from the repository folder:
    python python_scripts/step4_graph_priority.py
"""
import argparse
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ------------------------------------------------------------------ settings
THRESHOLD = 2500                  # network from Step 3b to analyse (main setting)
RET_COL = "ret_15"                # share of a segment within 15 m of current water
FUNCTIONAL_MIN = 0.8              # functional (water-filled) if share >= this
PRIORITY_N = 15                   # corridors passed on to Step 5
K_LIST = [5, 10, 15, 20, 30, 50]  # numbers of corridors tested for connectivity gain
N_RANDOM = 100                    # random selections used for the comparison
MIN_COST_KM = 0.05                # floor for the restoration length (avoids dividing by ~0)
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/outputs")
    p.add_argument("--fig", default="figures")
    return p.parse_args()


def node_key(xy):
    return (round(xy[0], 1), round(xy[1], 1))


def build_graph(gdf):
    G = nx.MultiDiGraph()
    for r in gdf.itertuples():
        coords = list(r.geometry.coords)
        u, v = node_key(coords[0]), node_key(coords[-1])
        if u == v:
            continue
        G.add_edge(u, v, key=int(r.seg_id), seg_id=int(r.seg_id),
                   length_m=float(r.length_m), main=bool(r.main_river))
    return G


def out_edge(G, v):
    nxt = [d["seg_id"] for _, _, d in G.out_edges(v, data=True)]
    return nxt[0] if nxt else None


def connected_set(G, E, functional):
    """Segments that are functional with a fully functional path down to a main river."""
    cache = {}

    def is_conn(seg):
        path, cur = [], seg
        while True:
            if cur in cache:
                res = cache[cur]
                break
            if cur in path or cur not in functional:
                res = False
                break
            path.append(cur)
            _, v, d = E[cur]
            if d["main"]:
                res = True
                break
            nxt = out_edge(G, v)
            if nxt is None:                 # leaves the study area without reaching a river
                res = False
                break
            cur = nxt
        for s in path:
            cache[s] = res
        return res

    return {s for s in E if is_conn(s)}


def chain_to_river(G, E, e, functional, conn):
    """Broken segments from e downstream to the first connected point; ok=False if none exists."""
    chain, cur, seen = [], e, set()
    while True:
        if cur in seen:
            return chain, False
        seen.add(cur)
        _, v, d = E[cur]
        if d["main"]:
            return chain, True
        if cur in functional:
            if cur in conn:
                return chain, True
        else:
            chain.append(cur)
        nxt = out_edge(G, v)
        if nxt is None:
            return chain, False
        cur = nxt


def total_km(E, ids):
    return sum(E[s][2]["length_m"] for s in ids) / 1000


def make_figure(gdf, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 10), dpi=200)
    gdf[gdf.main_river].plot(ax=ax, color="#4393c3", linewidth=1.6)
    gdf[(~gdf.main_river) & (~gdf.priority)].plot(ax=ax, color="#bbbbbb", linewidth=0.5)
    pr = gdf[gdf.priority].copy()
    if len(pr):
        pr["inv_rank"] = pr["corridor"].max() + 1 - pr["corridor"]
        pr.sort_values("corridor", ascending=False).plot(
            ax=ax, column="inv_rank", cmap="YlOrRd", linewidth=2.8)
        for c, grp in pr[pr.corridor <= 10].groupby("corridor"):
            mid = grp.geometry.iloc[0].interpolate(0.5, normalized=True)
            ax.annotate(str(int(c)), (mid.x, mid.y), fontsize=7, weight="bold", ha="center", va="center",
                        bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_title(f"Priority restoration corridors (top {PRIORITY_N}); numbers = rank of the top 10")
    xt = ax.get_xticks(); ax.set_xticks(xt); ax.set_xticklabels([f"{t / 1000:.0f}" for t in xt])
    yt = ax.get_yticks(); ax.set_yticks(yt); ax.set_yticklabels([f"{t / 1000:.0f}" for t in yt])
    ax.set_xlabel("Easting (km, UTM 46N)")
    ax.set_ylabel("Northing (km)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    a = parse_args()
    out, fig_dir = Path(a.out), Path(a.fig)
    fig_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(out / f"drainage_network_thr{THRESHOLD}.gpkg").reset_index(drop=True)
    G = build_graph(gdf)
    E = {d["seg_id"]: (u, v, d) for u, v, k, d in G.edges(keys=True, data=True)}
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (from {len(gdf)} segments); "
          f"weakly connected components: {nx.number_weakly_connected_components(G)}")

    # ---- per-segment metrics
    ebc = nx.edge_betweenness_centrality(G, normalized=True, weight="length_m")
    bc = {k[2]: v for k, v in ebc.items()}
    up = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        sub = G.subgraph(nx.ancestors(G, u) | {u})
        up[d["seg_id"]] = (sum(dd["length_m"] for _, _, dd in sub.edges(data=True)) + d["length_m"]) / 1000
    gdf["betweenness"] = gdf.seg_id.map(bc).fillna(0.0)
    gdf["upstream_km"] = gdf.seg_id.map(up).fillna(0.0)
    gdf["gap_m"] = gdf.length_m * (1 - gdf[RET_COL])
    up_km = gdf.set_index("seg_id")["upstream_km"].to_dict()
    gap_m = gdf.set_index("seg_id")["gap_m"].to_dict()

    functional0 = set(gdf.loc[(gdf[RET_COL] >= FUNCTIONAL_MIN) | gdf.main_river, "seg_id"]) & set(E)
    elig = [s for s in gdf.loc[~gdf.main_river & (gdf[RET_COL] < FUNCTIONAL_MIN), "seg_id"] if s in E]
    conn0 = connected_set(G, E, functional0)
    all_km = total_km(E, E)
    base_km = total_km(E, conn0)
    n_rho = gdf[gdf.seg_id.isin(elig)]
    rho = spearmanr(n_rho.betweenness, n_rho.upstream_km)[0]
    print(f"\nbroken segments: {len(elig)} ({total_km(E, elig):.1f} km, no-water gap {sum(gap_m[s] for s in elig) / 1000:.1f} km)")
    print(f"network {all_km:.1f} km; connected to a main river today: {base_km:.1f} km ({base_km / all_km * 100:.1f}%)")
    print(f"Spearman correlation, betweenness vs upstream length (broken segments): {rho:.2f}")

    # ---- greedy corridor selection
    n_max = max(max(K_LIST), PRIORITY_N)
    F, picks, cum_gap, cum_gain, prev = set(functional0), [], 0.0, 0.0, base_km
    greedy_curve = []
    for it in range(1, n_max + 1):
        conn = connected_set(G, E, F)
        best = None
        for e in elig:
            if e in F:
                continue
            chain, ok = chain_to_river(G, E, e, F, conn)
            if not ok or not chain:
                continue
            cost = sum(gap_m[c] for c in chain) / 1000
            score = up_km[e] / max(cost, MIN_COST_KM)
            if best is None or score > best[0]:
                best = (score, e, chain, cost)
        if best is None:
            break
        score, e, chain, cost = best
        F |= set(chain)
        new_km = total_km(E, connected_set(G, E, F))
        cum_gap += cost
        cum_gain = new_km - base_km
        picks.append({"corridor": it, "entry_seg": e, "segments": chain, "RPI": score,
                      "gap_km": cost, "upstream_km": up_km[e], "marginal_gain_km": new_km - prev})
        greedy_curve.append((it, cum_gap, cum_gain))
        prev = new_km
    print(f"\ncorridors found: {len(picks)} (stopped when no broken segment has a path to a main river)")
    if not picks:
        raise SystemExit("No broken segment has a continuous path down to a main river inside the study area. "
                         "Check the Step 3b map (main-river flag) and send me the output.")

    # ---- random comparison (same chain logic, random order)
    rng = np.random.default_rng(SEED)
    rand = {K: [] for K in K_LIST}
    for _ in range(N_RANDOM):
        Fr, done, gap_c = set(functional0), 0, 0.0
        order = list(rng.permutation(elig))
        conn = connected_set(G, E, Fr)
        for e in order:
            if done >= max(K_LIST):
                break
            if e in Fr:
                continue
            chain, ok = chain_to_river(G, E, e, Fr, conn)
            if not ok or not chain:
                continue
            Fr |= set(chain)
            gap_c += sum(gap_m[c] for c in chain) / 1000
            conn = connected_set(G, E, Fr)
            done += 1
            if done in rand:
                rand[done].append((gap_c, total_km(E, conn) - base_km))

    rows = []
    for K in K_LIST:
        g = [x for x in greedy_curve if x[0] == K]
        if not g:
            continue
        _, gap_k, gain_k = g[0]
        rr = np.array(rand[K]) if rand[K] else np.array([[np.nan, np.nan]])
        rows.append({"K_corridors": K, "gap_to_restore_km": round(gap_k, 1), "gain_km_RPI": round(gain_k, 1),
                     "gain_per_gap_km_RPI": round(gain_k / gap_k, 2) if gap_k > 0 else np.nan,
                     "random_gap_km": round(float(rr[:, 0].mean()), 1),
                     "random_gain_km": round(float(rr[:, 1].mean()), 1),
                     "random_gain_per_gap_km": round(float(rr[:, 1].mean() / rr[:, 0].mean()), 2)
                     if rr[:, 0].mean() > 0 else np.nan})
    gain = pd.DataFrame(rows)
    if len(gain):
        gain.insert(1, "connected_km_today", round(base_km, 1))
        gain.to_csv(out / "step4_connectivity_gain.csv", index=False)
        print("\n=== Connectivity gain: greedy RPI corridors vs random corridors (mean of "
              f"{N_RANDOM} runs) ===")
        print(gain.to_string(index=False))
    else:
        print(f"fewer than {min(K_LIST)} corridors available, so the gain table was skipped")

    # ---- write segment-level results
    gdf["corridor"] = 0
    gdf["RPI"] = 0.0
    for p in picks:
        for s in p["segments"]:
            gdf.loc[gdf.seg_id == s, ["corridor", "RPI"]] = [p["corridor"], p["RPI"]]
    gdf["priority"] = (gdf.corridor > 0) & (gdf.corridor <= PRIORITY_N)
    keep = ["seg_id", "corridor", "RPI", "betweenness", "upstream_km", "gap_m", "length_m",
            RET_COL, "status", "main_river", "priority", "geometry"]
    gdf[keep].to_file(out / "step4_segments_ranked.gpkg", driver="GPKG")
    pri = gdf[gdf.priority].sort_values(["corridor", "seg_id"])
    pri[keep].to_file(out / "step4_priority_corridors.gpkg", driver="GPKG")
    pri.drop(columns="geometry")[keep[:-1]].round(4).to_csv(out / "step4_priority_corridors.csv", index=False)

    tab = pd.DataFrame([{k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.items() if k != "segments"}
                        | {"n_segments": len(p["segments"])} for p in picks[:PRIORITY_N]])
    print(f"\ntop {min(10, len(tab))} corridors (RPI = km of network drained per km restored):")
    print(tab.head(10).to_string(index=False))
    make_figure(gdf, fig_dir / "step4_priority_map.png")
    print("\nfigure saved:", fig_dir / "step4_priority_map.png")


if __name__ == "__main__":
    main()
