"""
visualize_combined_barplot.py

Fig. 3: change of every margin segment as a bar along the LIA margin (clockwise from its northernmost point),
four periods stacked. Run after calculate_retreat_with_flowlines.py and screen_margin_artefacts.py.
Takes the glacier names from visualize_retreat.py, so keep both in the same folder.
"""

import os

import geopandas as gpd
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox
from scipy.spatial import cKDTree

from visualize_retreat import GLACIER_TEXT

mpl.rcParams['svg.fonttype'] = 'none'

# SET THESE PATHS to where your data lives
DATA_DIR = "data"
INPUT_DIR = os.path.join(DATA_DIR, "Analysis_Inputs")  # Zenodo deposit
IMBIE_PATH = os.path.join(INPUT_DIR, "Greenland_Basins_PS_v1.4.2_extended.gpkg")
GLACIER_NAMES_PATH = os.path.join(DATA_DIR, "glacier_names", "GreenlandGlacierNames_GGNv01_WGS84.shp")
BASE_DIR = os.path.join("outputs", "ice_margin_analysis")
ARTEFACT_CSV = os.path.join(BASE_DIR, "artefact_screen", "artefact_segments.csv")
OUTPUT_DIR = os.path.join("outputs", "figures")

TARGET_CRS = "EPSG:3413"

PERIODS = {"LIA_to_1980s": "LIA to 1980s", "1980s_to_2000s": "1980s to 2000s",
           "2000s_to_2022": "2000s to 2022", "LIA_to_2022": "LIA to 2022"}   # panels, top to bottom
ZERO_POSITIVE = ("LIA_to_1980s", "LIA_to_2022")   # LIA is the max extent, so gains are drawn as 0
YLIM = {k: (-40, 10) if k == "LIA_to_2022" else (-30, 10) for k in PERIODS}

REGION_COLORS = {'CE': "#55dde0", 'CW': "#326186", 'NE': "#1d2c35", 'NO': "#f8bd4f",
                 'NW': "#009e73", 'SE': "#e75a0d", 'SW': "#5d2609"}
BAR_W = 0.4   # % of the margin
GRID = dict(color='#888888', linestyle='--', linewidth=0.7, alpha=0.9)
GLACIER_LINE = dict(color='#333333', linewidth=0.8, alpha=0.8)


def load_data():
    imbie = gpd.read_file(IMBIE_PATH).to_crs(TARGET_CRS)
    g = gpd.read_file(GLACIER_NAMES_PATH)
    g = g[g.ID.isin(list(GLACIER_TEXT))].to_crs(TARGET_CRS)
    names = g.ID.map(lambda i: GLACIER_TEXT[i].replace("\n", " "))

    # x axis: outer ring of the LIA mask at 250 m spacing, clockwise from the northernmost vertex
    lia = gpd.read_file(os.path.join(BASE_DIR, "processed_masks", "LIA_processed.shp"))
    poly = lia.to_crs(TARGET_CRS).geometry.union_all()
    poly = max(poly.geoms, key=lambda p: p.area) if poly.geom_type == "MultiPolygon" else poly
    ring = poly.exterior.segmentize(250.0)
    if ring.is_ccw:
        ring = type(ring)(list(ring.coords)[::-1])
    c = np.asarray(ring.coords)[:-1, :2]
    coords = np.roll(c, -int(np.argmax(c[:, 1])), axis=0)
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(coords, axis=0).T))]
    total = d[-1] + float(np.hypot(*(coords[0] - coords[-1])))
    tree = cKDTree(coords)

    glaciers = {}   # name -> position as a fraction of the margin
    for name, p in zip(names, g.geometry):
        glaciers[name] = d[tree.query([p.x, p.y])[1]] / total

    artefacts = pd.read_csv(ARTEFACT_CSV)
    periods, source = {}, []
    for key in PERIODS:
        seg = gpd.read_file(os.path.join(BASE_DIR, "change", f"{key}.shp")).to_crs(TARGET_CRS).reset_index(drop=True)
        drop = artefacts.loc[artefacts.period == key, "seg_index"]
        seg = seg.loc[seg.index.difference(drop)]
        # every segment sits inside the LIA margin, so it takes the margin point nearest its midpoint
        cen = seg.geometry.centroid
        i = tree.query(np.c_[cen.x.values, cen.y.values])[1]
        pts = gpd.GeoDataFrame(dict(change_m=seg.change_m.values, pos=d[i]), geometry=cen.values, crs=seg.crs)
        j = gpd.sjoin(pts, imbie[["SUBREGION1", "geometry"]], how="left", predicate="within")
        j = j[~j.index.duplicated()].dropna(subset=["SUBREGION1"])
        ms = pd.DataFrame(dict(coastline_position=j.pos.values, normalized_position=j.pos.values / total * 100,
                               change_m=j.change_m.values, region=j.SUBREGION1.values))
        ms = ms.sort_values("normalized_position")
        true_km = ms.change_m.values / 1000.0
        if key in ZERO_POSITIVE:
            ms = ms.assign(change_m=ms.change_m.clip(upper=0.0))
        periods[key] = ms
        print(f"  {key}: {len(seg)} segments ({len(drop)} artefacts dropped), {len(ms)} bars")
        source.append(ms.assign(period=key, change_km=np.round(true_km, 4), plotted_km=(ms.change_m / 1000).round(4))[
            ["period", "region", "normalized_position", "coastline_position", "change_km", "plotted_km"]])
    return periods, pd.concat(source), glaciers


def draw_panels(periods, glaciers):
    fig, axes = plt.subplots(len(PERIODS), 1, figsize=(18, 12), sharex=True)
    for ax, key in zip(axes, PERIODS):
        ms = periods[key]
        for y in [10, -10, -20, -30]:
            ax.axhline(y=y, zorder=0, **GRID)
        ax.axhline(y=0, zorder=1, color='black', linewidth=0.8, alpha=0.7)
        for region in ms['region'].unique():
            rd = ms[ms['region'] == region]
            ax.bar(rd['normalized_position'], rd['change_m'] / 1000.0, width=BAR_W, color=REGION_COLORS[region],
                   edgecolor='none', zorder=2, rasterized=True)   # rasterized keeps the svg small
        ax.set_ylim(YLIM[key])
        ax.tick_params(axis='y', which='both', right=False, left=True, labelsize=14)
        # rotated label for now so tight_layout leaves room for it; moved inside the panel later
        ax.text(1.02, 0.5, PERIODS[key], transform=ax.transAxes, fontsize=16, fontweight='bold', va='center',
                ha='left', rotation=90)
        for s in ['top', 'right', 'bottom']:
            ax.spines[s].set_visible(False)
    axes[2].set_ylabel('Ice Margin Change (km)', fontsize=18)

    # glacier lines through all panels, names under the bottom one
    for x in glaciers.values():
        for ax in axes:
            ax.axvline(x=x * 100.0, linestyle=(0, (5, 10)), zorder=3, **GLACIER_LINE)
    for name, x in glaciers.items():
        axes[-1].text(x * 100.0, -36, name, fontsize=13, ha='right', va='top', rotation=45, rotation_mode='anchor')
    for ax in axes:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout()

    # ticks every 10 km, and the -40 grid line in the LIA to 2022 panel
    for ax, key in zip(axes, PERIODS):
        lo, hi = YLIM[key]
        ax.set_yticks(np.arange(lo, hi + 1, 10))
        if key == "LIA_to_2022":
            ax.axhline(y=-40, zorder=0, **GRID)
        for ln in ax.get_lines():   # grid line on the panel edge would get half clipped
            yd = np.asarray(ln.get_ydata(), float)
            if yd.size == 2 and yd[0] == yd[1] and yd[0] in (lo, hi):
                ln.set_clip_on(False)
    return fig, axes


def _box(a, rend):
    # display box of a text (its white background if it has one) or of a marker
    if hasattr(a, "get_bbox_patch") and a.get_bbox_patch() is not None:
        a.update_bbox_position_size(rend)
        return a.get_bbox_patch().get_window_extent(rend)
    return a.get_window_extent(rend)


def hits(ax, box, bars, others, pad=3.0):
    # True if the box (plus pad px) touches a drawn bar of the panel or one of the other boxes
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([[box.x0 - pad, box.y0 - pad],
                                                            [box.x1 + pad, box.y1 + pad]])
    x, v = bars
    lo, hi = ax.get_ylim()
    v = np.clip(v, lo, hi)
    half = BAR_W / 2.0
    sel = (x + half >= x0) & (x - half <= x1) & (v != 0)
    if np.any((np.maximum(0, v[sel]) >= y0) & (np.minimum(0, v[sel]) <= y1)):
        return True
    g = Bbox.from_extents(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)
    return any(g.overlaps(o) for o in others)


def mark_clipped_bars(fig, axes, periods, join_pct=1.0, group_pct=5.0):
    # bars cut by the y-limits: a triangle at the most extreme bar of each cluster, and one value per group of
    # nearby clusters, right of the group, or left if that hits a bar or another mark
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    clip = {}
    for ax, key in zip(axes, PERIODS):
        lo, hi = ax.get_ylim()
        ms = periods[key]
        km = ms.change_m.values / 1000.0
        bars = (ms.normalized_position.values, km)
        marks, texts = [], []
        for side, sel in (("below", km < lo), ("above", km > hi)):
            s = ms[sel].assign(km=km[sel]).sort_values("normalized_position")
            if s.empty:
                continue
            x = s.normalized_position.values
            ym = lo + 0.035 * (hi - lo) if side == "below" else hi - 0.035 * (hi - lo)
            gcut = np.r_[0, np.where(np.diff(x) > group_pct)[0] + 1, len(x)]
            for ga, gb in zip(gcut[:-1], gcut[1:]):
                grp = s.iloc[ga:gb]
                gx = grp.normalized_position.values
                cut = np.r_[0, np.where(np.diff(gx) > join_pct)[0] + 1, len(gx)]
                for a, b in zip(cut[:-1], cut[1:]):
                    c = grp.iloc[a:b]
                    r = c.loc[c.km.idxmin()] if side == "below" else c.loc[c.km.idxmax()]
                    marks += ax.plot([float(r.normalized_position)], [ym], marker="v" if side == "below" else "^",
                                     ms=8, color="black", markeredgecolor="white", markeredgewidth=0.6, zorder=6,
                                     clip_on=False)
                r = grp.loc[grp.km.idxmin()] if side == "below" else grp.loc[grp.km.idxmax()]
                val = float(r.km)
                txt = f"{val:.1f} km".replace("-", "\u2212") if val < 0 else f"+{val:.1f} km"
                t = ax.text(float(gx.max()) + 0.9, ym, txt, fontsize=11, ha="left", va="center", zorder=6,
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
                others = [_box(o, rend) for o in marks + texts]
                if hits(ax, _box(t, rend), bars, others):
                    t.set_position((float(gx.min()) - 0.9, ym))
                    t.set_ha("right")
                    if hits(ax, _box(t, rend), bars, others):
                        t.set_position((float(gx.max()) + 0.9, ym))
                        t.set_ha("left")
                texts.append(t)
        clip[key] = marks + texts
    return clip


def period_labels_inside(fig, axes, periods, clip, step=0.25):
    # period labels inside the panels near the top, right edges at one common x: the right-most x where no
    # label touches a bar or a clipped-bar mark in any panel
    labels = []
    for ax, key in zip(axes, PERIODS):
        for t in list(ax.texts):
            if t.get_text() == PERIODS[key]:
                t.remove()
        labels.append(ax.text(92.5, 0.97, PERIODS[key], transform=ax.get_xaxis_transform(), ha="right", va="top",
                              fontsize=16, fontweight="bold", zorder=5,
                              bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9)))
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    x = 92.5
    while x > 0:
        ok = True
        for ax, key, t in zip(axes, PERIODS, labels):
            t.set_x(x)
            bars = (periods[key].normalized_position.values, periods[key].change_m.values / 1000.0)
            if hits(ax, _box(t, rend), bars, [_box(a, rend) for a in clip[key]]):
                ok = False
                break
        if ok:
            return
        x -= step
    raise RuntimeError("no common x for the period labels clears every panel")


def add_region_brackets(ax, ms, min_pct=0.5):
    # basin brackets above the top panel. runs shorter than 0.5 % of the margin (a few mixed bars at a basin
    # boundary) are merged into the neighbour
    s = ms.sort_values("normalized_position")
    r, x = s.region.values, s.normalized_position.values
    cut = np.r_[0, np.where(r[1:] != r[:-1])[0] + 1, len(r)]
    merged = []
    for reg, a, b in [[r[a], x[a], x[b - 1]] for a, b in zip(cut[:-1], cut[1:])]:
        if merged and (b - a < min_pct or merged[-1][0] == reg):
            merged[-1][2] = b
        else:
            merged.append([reg, a, b])
    tr = ax.get_xaxis_transform()
    for i, (reg, a, b) in enumerate(merged):
        # neighbouring brackets meet half way
        lo = (a if i == 0 else (merged[i - 1][2] + a) / 2) + 0.25
        hi = (b if i == len(merged) - 1 else (b + merged[i + 1][1]) / 2) - 0.25
        mid = (lo + hi) / 2
        ax.plot([lo, lo, hi, hi], [1.03, 1.07, 1.07, 1.03], transform=tr, color="black", lw=1.2, clip_on=False)
        ax.plot([mid, mid], [1.07, 1.095], transform=tr, color="black", lw=1.2, clip_on=False)
        ax.text(mid, 1.11, reg, transform=tr, ha="center", va="bottom", fontsize=14, clip_on=False)


def spread_glacier_labels(fig, ax, names, y_label=-45.0):
    # glacier names too close together are spread evenly around their cluster centre, with a thin leader back to
    # their dashed line
    texts = [t for t in ax.texts if t.get_text() in names]
    fig.canvas.draw()
    x0, x1 = ax.get_xlim()
    ax_w_in = ax.get_position().width * fig.get_figwidth()
    fs_in = texts[0].get_fontsize() / 72.0
    gap = 1.5 * fs_in / np.sin(np.radians(45)) / ax_w_in * (x1 - x0)   # 1.5 rotated text heights
    xs = np.array([t.get_position()[0] for t in texts])
    order = np.argsort(xs)
    p = xs[order].astype(float)
    clusters = [[i] for i in range(len(p))]

    def place(cl):
        k = np.arange(len(cl))
        return p[cl].mean() + (k - k.mean()) * gap

    # merge neighbouring clusters until none overlap
    merged = True
    while merged:
        merged = False
        for a in range(len(clusters) - 1):
            if place(clusters[a])[-1] + gap > place(clusters[a + 1])[0] + 1e-9:
                clusters[a:a + 2] = [clusters[a] + clusters[a + 1]]
                merged = True
                break
    new = np.concatenate([place(cl) for cl in clusters])
    ybot = ax.get_ylim()[0]
    for idx, xn in zip(order, new):
        t = texts[idx]
        xt = xs[idx]
        t.set_position((xn, y_label))
        ax.add_line(mlines.Line2D([xt, xt, xn], [ybot, ybot - 1.5, y_label + 0.9], clip_on=False, zorder=3,
                                  **GLACIER_LINE))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    periods, source, glaciers = load_data()
    fig, axes = draw_panels(periods, glaciers)
    clip = mark_clipped_bars(fig, axes, periods)
    period_labels_inside(fig, axes, periods, clip)
    add_region_brackets(axes[0], periods["LIA_to_2022"])
    spread_glacier_labels(fig, axes[-1], set(glaciers))
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"Fig3_combined_barplots.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    source.to_csv(os.path.join(OUTPUT_DIR, "Fig3_source_data.csv"), index=False)
    print(f"Saved Fig3_combined_barplots.png / .svg and Fig3_source_data.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
