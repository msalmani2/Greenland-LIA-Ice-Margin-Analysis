"""
nw_four_bar_panels_only.py

Fig. 4: NW Greenland ice margin change in four bar panels along the 1985 margin, plus a context map.
Needs calculate_retreat_with_flowlines.py in the same folder, but not its outputs.
"""

import os
import sys

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib import transforms as mtransforms
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, substring, unary_union
from shapely.strtree import STRtree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calculate_retreat_with_flowlines import calculate_ice_margin_change_gdf  # noqa: E402

mpl.rcParams.update({"svg.fonttype": "none", "font.size": 13, "axes.titlesize": 18, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11})

# SET THESE PATHS to where your data lives
DATA_DIR = "data"
KJELDSEN_DIR = os.path.join(DATA_DIR, "PROMICE_NW")
LIA_DIR = os.path.join(DATA_DIR, "GRISHM")             # Zenodo deposit
MASK_DIR = os.path.join(DATA_DIR, "Ice_Masks")         # Zenodo deposit
INPUT_DIR = os.path.join(DATA_DIR, "Analysis_Inputs")  # Zenodo deposit
# (path, regional). The Kjeldsen NW masks are regional, so their outline also has an inland clip edge
MASK_FILES = {
    "1985": (os.path.join(KJELDSEN_DIR, "PROMICE_NW_1985.shp"), True),
    "LIA": (os.path.join(LIA_DIR, "GRISHM_nunataks_filled.shp"), False),
    "2022": (os.path.join(MASK_DIR, "Greenland_2022_ice_mask_PROMICE_modified.shp"), False),
    "2000": (os.path.join(KJELDSEN_DIR, "GIMP_NW_1999_2002.shp"), True),
}
GLACIER_NAMES_PATH = os.path.join(DATA_DIR, "glacier_names", "GreenlandGlacierNames_GGNv01_WGS84.shp")
FLOWLINES_PATH = os.path.join(INPUT_DIR, "glacier_flowlines.gpkg")
OUTPUT_DIR = os.path.join("outputs", "figures")

TARGET_CRS = "EPSG:32624"  # UTM 24N, as in the main analysis
MAP_CRS = "EPSG:3413"      # context map only

# transect anchors (EPSG:3413): start (N), middle (picks the right arc), end (S)
ANCHORS = [(-538035.9442518415, -1330770.0244489769),
           (-531885.699898103, -1393365.5745355159),
           (-254498.14369713535, -1979667.5495359409)]

PARAMS = dict(sample_distance=500.0,         # m (the main analysis uses 1000 m)
              simplify_tol=100.0, flowline_search_radius=2000.0,
              bin_km=5.0,                    # each bin keeps its largest-magnitude value
              comparison_window_m=100000.0,  # LIA and 2022 are clipped to this window around the transect
              clip_edge_tol_m=1.0, glacier_max_dist_m=25000.0,
              unit_min_km2=1.0,              # smallest patch of ice that is in neither Kjeldsen mask
              unit_on_tol_m=150.0, unit_on_frac=0.5)
# ice in our LIA and 2022 masks but in neither Kjeldsen mask gives a retreat to 1985 and an advance
# to 2022 that cancel, so bins on it are left empty in these two panels
EXCLUDE_ON_UNITS = ("LIA_to_1985", "2000_to_2022")
# the LIA margin is the max extent, so positive changes from it are drawn as zero
ZERO_POSITIVE = ("LIA_to_1985", "LIA_to_2022")

Y_LIMITS = {"LIA_to_1985": (-30.0, 10.0), "1985_to_2000": (-30.0, 10.0),
            "2000_to_2022": (-30.0, 10.0), "LIA_to_2022": (-40.0, 10.0)}
Y_TICK_STEP = 10.0
PERIODS = [("LIA_to_1985", "LIA", "1985", "Change from LIA to 1985"),
           ("1985_to_2000", "1985", "2000", "Change from 1985 to 2000"),
           ("2000_to_2022", "2000", "2022", "Change from 2000 to 2022"),
           ("LIA_to_2022", "LIA", "2022", "Cumulative change from LIA to 2022")]
MASK_LABELS = {"LIA": "LIA maximum", "1985": "PROMICE 1985 (Kjeldsen et al.)",
               "2000": "GIMP_NW 1999–2002 (Kjeldsen et al.)", "2022": "PROMICE 2022"}

# glaciers labelled on the transect (GGN names), plus a few more by ID to fill the gaps
GLACIERS_TO_INCLUDE = {
    "Kangerlussuup Sermersua", "Salliarutsip Sermia", "Sermeq", "Akullikassaap Sermia",
    "Kakiffaat Sermiat", "Sermeq Avannarleq", "Ikertuup Sermersua", "Illulip Sermia",
    "Nunatakassaap Sermia", "Kjer Gletsjer", "Sermersuaq", "Dietrichson Gletsjer",
    "Sverdrup Gletsjer", "Nansen Gletsjer", "Nordenskiöld Gletsjer", "Nuussuup Sermia",
    "Issuussasull Sermiat", "Rink Gletsjer", "Qeqertat Timaanni Sermeq", "Morell Gletsjer",
    "Innaqqissorsuup Oqquani Sermeq", "Yngvar Nielsen Gletsjer", "Helland Gletsjer",
    "Sermipaluk", "Savissup Sermia", "Naajarsorfik", "Neptune Gletsjer",
    "Paakitsup Sermersua", "Ullip Sermia", "Søndre Tvillinggletsjer",
}
EXTRA_GLACIER_IDS = {"GGN0579", "GGN0559", "GGN0570", "GGN0557", "GGN0539", "GGN0529", "GGN0503"}
# familiar name as a smaller second line. None = take it from GGN (Harald Moltke Brae, Kong Oscar,
# Steenstrup, Upernavik Isstrom, Hayes); GGN has none for Nunatakassaap Sermia
WELL_KNOWN_GLACIERS = {"GGN0581": None, "GGN0553": None, "GGN0542": None, "GGN0531": "Alison Glacier",
                       "GGN0452": None, "GGN0539": None}
# also named on the map, bold in the bar panels
MAP_LABEL_IDS = {"GGN0581", "GGN0571", "GGN0576", "GGN0572", "GGN0553", "GGN0542",
                 "GGN0531", "GGN0486", "GGN0461", "GGN0452", "GGN0390"}

BAR_COLOR = "#2b2b2b"
MAP_COLORS = {"LIA": mpl.colormaps["viridis"](0.08), "1985": mpl.colormaps["viridis"](0.38),
              "2000": mpl.colormaps["viridis"](0.66), "2022": "#e3e3e3"}
GLACIER_LINE = dict(color="#333333", lw=1.4, alpha=0.8)  # same as Fig. 3


def drawn_km(key, change_m):
    v = np.asarray(change_m, dtype=float) / 1000.0
    return np.minimum(v, 0.0) if key in ZERO_POSITIVE else v


# geometry helpers

def remove_holes(geometry):
    if isinstance(geometry, Polygon):
        return Polygon(geometry.exterior)
    elif geometry.geom_type == "MultiPolygon":
        return type(geometry)([Polygon(poly.exterior) for poly in geometry.geoms])
    return geometry


def pick_arc(ring, start_pt, mid_pt, end_pt):
    # cut the ring at start and end, return the arc closest to mid_pt and the other one
    L = ring.length
    s0, s1 = ring.project(start_pt), ring.project(end_pt)

    def sub(a, b):
        g = substring(ring, a, b, normalized=False)
        return g if isinstance(g, LineString) else linemerge(g)

    def join(a, b):
        return LineString(list(a.coords) + list(b.coords)[1:])

    if s0 <= s1:
        arc_a, arc_b = sub(s0, s1), join(sub(s1, L), sub(0, s0))
    else:
        arc_b, arc_a = sub(s1, s0), join(sub(s0, L), sub(0, s1))
    return (arc_a, arc_b) if mid_pt.distance(arc_a) <= mid_pt.distance(arc_b) else (arc_b, arc_a)


def orient_start_to_end(line, start_pt, end_pt):
    p0, p1 = Point(line.coords[0]), Point(line.coords[-1])
    if p0.distance(start_pt) + p1.distance(end_pt) > p0.distance(end_pt) + p1.distance(start_pt):
        return LineString(list(line.coords)[::-1])
    return line


def polygon_parts(geom):
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    return sorted(parts, key=lambda p: p.area, reverse=True)  # largest first


def polygons_only(geom):
    # overlays can leave lines and points behind
    if geom.is_empty or geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        return unary_union(polys) if polys else Polygon()
    return Polygon()


def single_polygons(geom):
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif hasattr(geom, "geoms"):
        for sub in geom.geoms:
            yield from single_polygons(sub)


def load_mask(path, regional):
    # mask in UTM with all holes filled. fill, dissolve, then fill again since the union can close new rings
    if regional:
        gdf = gpd.read_file(path)
    else:
        # our masks: fill every part in EPSG:3413 and union them first
        filled = []
        for geom in gpd.read_file(path).to_crs(MAP_CRS).geometry:
            if geom is not None and not geom.is_empty:
                filled.extend(Polygon(p.exterior) for p in
                              single_polygons(geom if geom.is_valid else shapely.make_valid(geom)))
        u = shapely.make_valid(shapely.union_all(filled))
        gdf = gpd.GeoDataFrame(geometry=list(single_polygons(u)), crs=MAP_CRS)
    gdf = gdf[gdf.geometry.notna()].to_crs(TARGET_CRS)
    geoms = []
    for g in gdf.geometry:
        if not g.is_valid:
            g = g.buffer(0)
        if not g.is_empty:
            geoms.append(remove_holes(g))
    u = remove_holes(unary_union(geoms))
    if not u.is_valid:
        u = u.buffer(0)
    return shapely.force_2d(u)


# NW masks and the transect

def build_nw_mask(label, path, regional, anchors):
    s, m, e = anchors
    geom = load_mask(path, regional)
    parts = polygon_parts(geom)
    arc, other = pick_arc(LineString(parts[0].exterior.coords), s, m, e)
    print(f"  {label:5s} {len(parts)} parts, arc {arc.length / 1e3:,.1f} km")
    return dict(label=label, regional=regional, geom=geom, detached=parts[1:],
                arc=orient_start_to_end(arc, s, e),
                clip_edge=orient_start_to_end(other, s, e) if regional else None)


def build_display_domain(clip_edge, arcs, ray_len=150000.0, step=10000.0):
    # footprint of the NW masks: the inland clip edge, extended seaward at both ends,
    # closed across the sea by the convex hull of the margin arcs
    E = clip_edge

    def ray_end(at_start):
        if at_start:
            p0, p1 = Point(E.coords[0]), E.interpolate(step)
        else:
            p0, p1 = Point(E.coords[-1]), E.interpolate(E.length - step)
        v = np.array([p0.x - p1.x, p0.y - p1.y])
        v = v / np.linalg.norm(v)
        return (p0.x + ray_len * v[0], p0.y + ray_len * v[1])

    far_s, far_e = ray_end(True), ray_end(False)
    D = Polygon([far_s] + list(E.coords) + [far_e])
    if not D.is_valid:
        D = D.buffer(0)
    pts = [c for a in arcs for c in a.coords] + [far_s, far_e]
    return polygon_parts(unary_union([D, shapely.MultiPoint(pts).convex_hull]))[0]


def find_domain_mismatch_units(masks, domain, ref_arc, min_km2):
    # ice in both LIA and 2022 but in neither Kjeldsen mask
    ours = masks["LIA"]["geom"].intersection(masks["2022"]["geom"]).intersection(domain)
    kj = unary_union([masks["1985"]["geom"], masks["2000"]["geom"]])
    d = polygons_only(ours.difference(kj))
    parts = [] if d.is_empty else [g for g in polygon_parts(d) if g.area >= min_km2 * 1e6]
    u = gpd.GeoDataFrame({"x_km": [ref_arc.project(g.representative_point()) / 1e3 for g in parts]},
                         geometry=parts, crs=TARGET_CRS).sort_values("x_km").reset_index(drop=True)
    u.insert(0, "unit_id", [f"U{i+1}" for i in range(len(u))])
    return u


def samples_on_units(seg_geoms, units, tol_m, min_frac):
    # a segment is on a unit if at least min_frac of it is within tol_m
    out = np.full(len(seg_geoms), "", dtype=object)
    share = np.zeros(len(seg_geoms))
    seg_len = np.maximum(shapely.length(seg_geoms), 1e-9)
    for uid, g in zip(units["unit_id"], units.geometry):
        f = shapely.length(shapely.intersection(seg_geoms, g.buffer(tol_m))) / seg_len
        better = (f >= min_frac) & (f > share)
        out[better] = uid
        share = np.maximum(share, f)
    return out


def exclude_unit_bins(bins, df, period_key, bin_km):
    # empty the whole bin, the samples next to a unit are measured across it too
    b = bins.copy()
    ok = df["change_m"].notna().values & (df["unit_id"].values != "")
    k = np.clip(np.floor(df["x_km"].values[ok] / bin_km).astype(int), 0, len(b) - 1)
    n_on = np.bincount(k, minlength=len(b))
    b["excluded"] = (period_key in EXCLUDE_ON_UNITS) & (n_on > 0)
    b.loc[b["excluded"], "change_m"] = np.nan
    b["status"] = np.where(b["excluded"], "not shown: ice outside both Kjeldsen masks",
                           np.where(b["change_m"].notna(), "shown", "no data"))
    return b


# measurement

def run_period(period_key, older, newer, flowlines_gdf, ref_arc, domain, params):
    # older margin = main arc + outline of every detached part inside the domain
    lines = [older["arc"]] + [LineString(p.exterior.coords) for p in older["detached"] if p.intersects(domain)]
    older_gdf = gpd.GeoDataFrame(geometry=lines, crs=TARGET_CRS)
    if newer["regional"]:
        newer_geom, newer_clip = newer["geom"], newer["clip_edge"]
    else:
        newer_geom, newer_clip = newer["geom_window"], newer["window"].exterior
    newer_gdf = gpd.GeoDataFrame(geometry=[newer_geom], crs=TARGET_CRS)

    # fix_invalid off: buffer(0) would empty the older margin, which goes in as lines
    kw = dict(sample_distance=params["sample_distance"], simplification_tolerance=params["simplify_tol"],
              keep_main_island=False, fix_invalid=False, line_buffer_if_line=20.0,
              flowline_search_radius=params["flowline_search_radius"])
    print(f"\n[{period_key}] {older['label']} -> {newer['label']}")
    res = calculate_ice_margin_change_gdf(older_gdf, newer_gdf, flowlines_gdf=flowlines_gdf, use_flowlines=True, **kw)
    # same run without flowlines; wherever the value differs, the flowline distance was used
    base = calculate_ice_margin_change_gdf(older_gdf, newer_gdf, flowlines_gdf=None, use_flowlines=False, **kw)

    seg = res.geometry.values
    mids = shapely.line_interpolate_point(seg, 0.5, normalized=True)
    method = np.where(np.isclose(res["change_m"].values, base["change_m"].values, rtol=0, atol=1e-9),
                      "nearest", "flowline")

    # drop nearest-distance samples that hit the newer mask's clip edge instead of its margin
    newer_bnd = unary_union([LineString(p.exterior.coords) for p in polygon_parts(newer_geom)])
    d_all = shapely.distance(mids, newer_bnd)
    d_clip = shapely.distance(mids, newer_clip)
    clip_flag = (method == "nearest") & (d_clip <= d_all + params["clip_edge_tol_m"])

    # x along the 1985 arc: the segment midpoint, or for flowline values the point where
    # that flowline crosses the 1985 arc, so a glacier lines up in every panel
    x_used = shapely.line_locate_point(ref_arc, mids)
    fl_geoms = flowlines_gdf.to_crs(TARGET_CRS).geometry.values
    fl_tree = STRtree(fl_geoms)
    for i in np.where(method == "flowline")[0]:
        cands = sorted(int(c) for c in fl_tree.query(seg[i], predicate="intersects"))
        chosen = next((c for c in cands if fl_geoms[c].intersects(newer_bnd)), cands[0] if cands else None)
        if chosen is None:
            continue
        cross = fl_geoms[chosen].intersection(ref_arc)
        if cross.is_empty:
            continue
        pts = list(cross.geoms) if hasattr(cross, "geoms") else [cross]
        best = min(pts, key=lambda q: q.distance(mids[i]))
        if best.geom_type != "Point":
            best = best.interpolate(best.project(mids[i]))
        x_used[i] = ref_arc.project(best)

    df = pd.DataFrame({"x_km": x_used / 1000.0, "change_m": res["change_m"].values.astype(float)})
    df.loc[clip_flag, "change_m"] = np.nan
    print(f"[{period_key}] {len(df)} segments, {int((method == 'flowline').sum())} on flowlines, "
          f"{int(clip_flag.sum())} dropped at a clip edge")
    return df, seg


def bin_max_magnitude(df, arc_len_km, bin_km):
    edges = np.arange(0.0, arc_len_km + bin_km, bin_km)
    nb = len(edges) - 1
    ok = df["change_m"].notna().values
    x = df["x_km"].values[ok]
    v = df["change_m"].values[ok]
    k = np.clip(np.floor(x / bin_km).astype(int), 0, nb - 1)
    rows = []
    for b in range(nb):
        sel = np.where(k == b)[0]
        lo, hi = edges[b], min(edges[b + 1], arc_len_km)
        val = float(v[sel[np.argmax(np.abs(v[sel]))]]) if len(sel) else np.nan
        rows.append((b, lo, hi, (lo + hi) / 2, val))
    return pd.DataFrame(rows, columns=["bin", "bin_start_km", "bin_end_km", "bin_center_km", "change_m"])


# glaciers

def infer_glacier_name(row):
    for col in ["New_Greenl", "Official_n", "Old_Greenl", "Alternativ", "Name", "GLACNAME"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip() not in ("", "None"):
            return str(row[col])
    return None


def prepare_glaciers(ref_arc, max_dist_m):
    g = gpd.read_file(GLACIER_NAMES_PATH).to_crs(TARGET_CRS)
    g["inferred_name"] = g.apply(infer_glacier_name, axis=1)
    g = g[g["inferred_name"].isin(GLACIERS_TO_INCLUDE) | g["Official_n"].isin(GLACIERS_TO_INCLUDE)
          | g["ID"].isin(EXTRA_GLACIER_IDS)].copy()
    g["dist_m"] = g.geometry.distance(ref_arc)
    g["x_km"] = [ref_arc.project(p) / 1000.0 for p in g.geometry]
    L = ref_arc.length / 1000.0
    g = g[(g["dist_m"] <= max_dist_m) & (g["x_km"] > 1.0) & (g["x_km"] < L - 1.0)].copy()
    g["label"] = [o if isinstance(o, str) and o.strip() else n for o, n in zip(g["Official_n"], g["inferred_name"])]

    def familiar(row):
        if row["ID"] not in WELL_KNOWN_GLACIERS:
            return ""
        if WELL_KNOWN_GLACIERS[row["ID"]]:
            return WELL_KNOWN_GLACIERS[row["ID"]]
        f = row.get("Foreign_na")
        return f if isinstance(f, str) and f.strip() and f != row["label"] else ""

    g["familiar"] = g.apply(familiar, axis=1)
    g["on_map"] = g["ID"].isin(MAP_LABEL_IDS)
    return g.sort_values("x_km").reset_index(drop=True)


# figure

def graticule(ax, L, B, R, T, crs, n_lat=5, n_lon=4):
    # lat/lon lines labelled where they leave the frame (grid north is not true north in EPSG:3413)
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    from_ll = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ex, ey = [], []
    for t in np.linspace(0, 1, 60):
        ex += [L + t * (R - L), L + t * (R - L), L, R]
        ey += [B, T, B + t * (T - B), B + t * (T - B)]
    lons, lats = to_ll.transform(ex, ey)
    lo0, lo1, la0, la1 = min(lons), max(lons), min(lats), max(lats)

    def nice(span, target):
        return next((s for s in (0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0) if s >= span / target), 10.0)

    d_lat, d_lon = nice(la1 - la0, n_lat), nice(lo1 - lo0, n_lon)
    style = dict(color="#4b5563", lw=0.6, alpha=0.55, zorder=4, ls=(0, (3, 2)))
    tprops = dict(fontsize=10, color="black", zorder=8,
                  bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    tol_x, tol_y = 0.02 * (R - L), 0.02 * (T - B)
    placed = []

    def label_at_edge(xs, ys, text, prefer):
        for edge in prefer:
            for x, y in [(xs[0], ys[0]), (xs[-1], ys[-1])]:
                hit = {"left": x - L < tol_x, "right": R - x < tol_x, "top": T - y < tol_y, "bottom": y - B < tol_y}[edge]
                fx, fy = (x - L) / (R - L), (y - B) / (T - B)
                # no label within 7 % of a corner or of another label
                corner = min(fx, 1 - fx) < 0.07 and min(fy, 1 - fy) < 0.07
                if not hit or corner or not all(np.hypot(fx - a, fy - b) > 0.07 for a, b in placed):
                    continue
                placed.append((fx, fy))
                px, py, ha, va = {"left": (L + 0.01 * (R - L), y, "left", "center"),
                                  "right": (R - 0.01 * (R - L), y, "right", "center"),
                                  "top": (x, T - 0.008 * (T - B), "center", "top"),
                                  "bottom": (x, B + 0.008 * (T - B), "center", "bottom")}[edge]
                ax.text(px, py, text, ha=ha, va=va, **tprops)
                return

    def draw(lon, lat, text, prefer):
        xs, ys = (np.array(a) for a in from_ll.transform(lon, lat))
        m = (xs >= L) & (xs <= R) & (ys >= B) & (ys <= T)
        if m.sum() > 1:
            ax.plot(xs[m], ys[m], **style)
            label_at_edge(xs[m], ys[m], text, prefer)

    v = np.ceil(la0 / d_lat) * d_lat
    while v <= la1:
        gl = np.linspace(lo0 - d_lon, lo1 + d_lon, 400)
        draw(gl, np.full_like(gl, v), f"{abs(v):g}°N", ["left", "right"])
        v += d_lat
    v = np.ceil(lo0 / d_lon) * d_lon
    while v <= lo1:
        gl = np.linspace(la0 - d_lat, la1 + d_lat, 400)
        draw(np.full_like(gl, v), gl, f"{abs(v):g}°W", ["top", "bottom"])
        v += d_lon


def spread_positions(x, gaps):
    # push crowded labels apart (at least gaps[i]) but keep each cluster centred on its glaciers
    x = np.asarray(x, dtype=float)
    blocks = []
    for i in range(len(x)):
        blocks.append([[i], x[i]])
        while len(blocks) > 1:
            (ma, sa), (mb, sb) = blocks[-2], blocks[-1]
            if sa + sum(gaps[j] for j in ma[:-1]) + gaps[ma[-1]] <= sb:
                break
            mem = ma + mb
            offs = np.concatenate([[0.0], np.cumsum([gaps[j] for j in mem[:-1]])])
            blocks[-2:] = [[mem, float(np.mean(x[mem] - offs))]]
    out = np.empty_like(x)
    for mem, st in blocks:
        out[mem] = st + np.concatenate([[0.0], np.cumsum([gaps[j] for j in mem[:-1]])])
    return out


def end_marker(ax, x, y, letter, fontsize=22):
    ax.text(x, y, letter, ha="center", va="center", fontsize=fontsize, fontweight="bold", color="black",
            zorder=12, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=1.6))


def plot_fig4(bins, arc_len_km, glaciers, map_layers, anchors_map, out_base, bin_km):
    spans_y = [Y_LIMITS[k][1] - Y_LIMITS[k][0] for k, _, _, _ in PERIODS]
    fig = plt.figure(figsize=(30, 15.5), facecolor="white")
    # panel heights follow the y range so all panels have the same vertical scale
    gs = fig.add_gridspec(4, 2, width_ratios=[3.0, 1.0], height_ratios=spans_y, wspace=0.035, hspace=0.17,
                          left=0.045, right=0.995, top=0.80, bottom=0.065)
    axes = [fig.add_subplot(gs[0, 0])]
    for i in range(1, 4):
        axes.append(fig.add_subplot(gs[i, 0], sharex=axes[0]))
    ax_map = fig.add_subplot(gs[:, 1])

    xpad = 110.0
    for ax, (key, older, newer, title) in zip(axes, PERIODS):
        b = bins[key]
        lo, hi = Y_LIMITS[key]
        ax.set_ylim(lo, hi)
        ax.set_yticks(np.arange(lo, hi + 1e-9, Y_TICK_STEP))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
        for x in glaciers["x_km"].values:
            ax.axvline(x, ls="--", zorder=1, **GLACIER_LINE)
        v = drawn_km(key, b["change_m"].values)
        ax.bar(b["bin_center_km"].values, v, width=8.0, color=BAR_COLOR, edgecolor="none", align="center", zorder=2)
        for xc, val in zip(b["bin_center_km"].values, v):
            if np.isfinite(val) and (val < lo or val > hi):
                ax.text(xc, lo if val < lo else hi, f"{val:.0f}", ha="center", va="bottom" if val < lo else "top",
                        fontsize=9, color="#b91c1c", zorder=7)
        ax.axhline(0, color="black", lw=1.6, zorder=3)
        ax.text(0.008, 0.95, f"{title} (negative = retreat)", transform=ax.transAxes, ha="left", va="top",
                fontsize=15, fontweight="bold", zorder=6,
                bbox=dict(facecolor="white", alpha=0.95, edgecolor="none", pad=0.6))
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.spines["left"].set_linewidth(1.8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.grid(False, axis="x")
        ax.set_xlim(-xpad, arc_len_km + xpad)

    for ax in (axes[0], axes[-1]):
        end_marker(ax, -xpad * 0.55, 0.0, "N")
        end_marker(ax, arc_len_km + xpad * 0.55, 0.0, "S")

    for ax in axes:
        ax.set_xticks(np.arange(0, arc_len_km + 1e-9, 200.0))
        ax.xaxis.set_minor_locator(MultipleLocator(bin_km))
        ax.tick_params(axis="x", which="major", length=5, width=1.0, labelbottom=False)
    axes[-1].tick_params(axis="x", which="major", length=6, width=1.2, labelbottom=True)
    axes[-1].set_xlabel(f"Distance along the 1985 ice margin, N to S (km); transect length {arc_len_km:,.1f} km",
                        fontsize=15, fontweight="bold")
    y_top, y_bot = axes[0].get_position().y1, axes[-1].get_position().y0
    fig.text(0.012, 0.5 * (y_top + y_bot), "Ice margin change (km)", rotation=90, ha="center", va="center",
             fontsize=16, fontweight="bold")

    # glacier labels above the top panel, tied to their line; bold if also on the map,
    # familiar name as a smaller second line
    ax0 = axes[0]
    base = mtransforms.blended_transform_factory(ax0.transData, ax0.transAxes)
    width_in = ax0.get_position().width * fig.get_figwidth()
    km_per_pt = (ax0.get_xlim()[1] - ax0.get_xlim()[0]) / (width_in * 72.0)
    two = glaciers["familiar"].astype(bool).values  # two-line labels reach further left
    up_pt = np.where(two, 22.0, 9.0)
    down_pt = np.where(two, 7.0, 9.0)
    gaps_km = (down_pt[:-1] + up_pt[1:] + 3.0) / np.sin(np.radians(45)) * km_per_pt
    x_lab = spread_positions(glaciers["x_km"].values, gaps_km)
    y_lab = 1.06
    for (_, r), xl in zip(glaciers.iterrows(), x_lab):
        weight = "bold" if r["on_map"] else "normal"
        ax0.plot([r["x_km"], r["x_km"], xl], [1.0, 1.02, y_lab - 0.005], transform=base, clip_on=False, zorder=5,
                 **GLACIER_LINE)
        # with a second line, the main name is shifted up and left
        tr = mtransforms.offset_copy(base, fig=fig, x=-10.0, y=10.0, units="points") if r["familiar"] else base
        ax0.text(xl, y_lab, r["label"], transform=tr, rotation=45, rotation_mode="anchor",
                 ha="left", va="bottom", fontsize=12, fontweight=weight, clip_on=False)
        if r["familiar"]:
            ax0.text(xl, y_lab, r["familiar"], transform=base, rotation=45, rotation_mode="anchor",
                     ha="left", va="bottom", fontsize=10, fontstyle="italic", color="#374151", clip_on=False)

    # context map (EPSG:3413)
    order = ["LIA", "1985", "2000", "2022"]
    for k in order:
        gpd.GeoSeries([map_layers[k]], crs=MAP_CRS).plot(ax=ax_map, color=MAP_COLORS[k], edgecolor="none",
                                                         zorder=2 + order.index(k))
    s_pt, e_pt = anchors_map
    bx0, by0, bx1, by1 = map_layers["extent"]
    ax_map.set(xlim=(bx0, bx1), ylim=(by0, by1), aspect="equal", facecolor="#f4f7fb", xticks=[], yticks=[])
    graticule(ax_map, bx0, by0, bx1, by1, MAP_CRS, n_lat=8, n_lon=5)
    ax_map.plot([s_pt.x, e_pt.x], [s_pt.y, e_pt.y], "o", color="black", ms=6, zorder=11)
    end_marker(ax_map, s_pt.x - 18000, s_pt.y + 14000, "N", fontsize=18)
    end_marker(ax_map, e_pt.x - 20000, e_pt.y - 12000, "S", fontsize=18)
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    for _, r in glaciers[glaciers["on_map"]].iterrows():
        p = r["pt_map"]
        ax_map.plot(p.x, p.y, "o", color="black", ms=2.5, zorder=11)
        room_left = (p.x - bx0) / (bx1 - bx0) > 0.012 * len(r["label"])  # does the name fit on the left?
        ax_map.text(p.x + (-6000 if room_left else 6000), p.y, r["label"], ha="right" if room_left else "left",
                    va="center", fontsize=10, zorder=12, path_effects=halo, clip_on=True)
    # scale bar (EPSG:3413 is true scale at 70 N)
    sb_x0, sb_y0 = bx0 + 0.06 * (bx1 - bx0), by0 + 0.035 * (by1 - by0)
    for i, c in enumerate(["black", "white"]):
        ax_map.add_patch(Rectangle((sb_x0 + i * 50000, sb_y0), 50000, 5000, fc=c, ec="black", lw=0.8, zorder=12))
    for d in (0, 50, 100):
        ax_map.text(sb_x0 + d * 1000, sb_y0 + 8000, f"{d}", ha="center", va="bottom", fontsize=10, zorder=12)
    ax_map.text(sb_x0 + 105000, sb_y0 + 2500, "km", ha="left", va="center", fontsize=10, zorder=12)
    handles = [Patch(fc=MAP_COLORS[k], ec="#6b7280", lw=0.6, label=MASK_LABELS[k]) for k in ["2022", "2000", "1985", "LIA"]]
    handles.append(Patch(fc="white", ec="black", label="N, S = transect ends"))
    leg = ax_map.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.075), frameon=True, fontsize=10.5,
                        framealpha=1.0)
    leg.set_zorder(20)
    ins = ax_map.inset_axes([0.02, 0.36, 0.30, 0.30])
    gpd.GeoSeries([map_layers["greenland"]], crs=MAP_CRS).plot(ax=ins, color="#d1d5db", edgecolor="#6b7280", lw=0.3)
    gpd.GeoSeries([map_layers["domain"].intersection(map_layers["greenland"])], crs=MAP_CRS).plot(ax=ins, color="#1f2937")
    ins.set(xticks=[], yticks=[], aspect="equal")

    fig.savefig(out_base + ".png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(params=PARAMS):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    flowlines_gdf = gpd.read_file(FLOWLINES_PATH)
    pts = gpd.GeoSeries(gpd.points_from_xy(*zip(*ANCHORS)), crs=MAP_CRS).to_crs(TARGET_CRS)
    anchors = tuple(Point(p.x, p.y) for p in pts)
    s, m, e = anchors

    print("Loading masks (holes filled) and cutting the N->S arcs...")
    masks = {k: build_nw_mask(k, p, reg, anchors) for k, (p, reg) in MASK_FILES.items()}
    ref_arc = masks["1985"]["arc"]
    arc_len_km = ref_arc.length / 1000.0
    domain = build_display_domain(masks["1985"]["clip_edge"], [masks[k]["arc"] for k in ("1985", "LIA", "2022")])

    # clip LIA and 2022 to a window around the arcs (buffering the simplified arcs is much faster
    # than unioning the raw ones)
    window = unary_union([masks[k]["arc"].simplify(1000.0).buffer(params["comparison_window_m"] + 1000.0, quad_segs=4)
                          for k in ("1985", "LIA", "2022", "2000")])
    window = polygon_parts(window)[0]
    for k in ("LIA", "2022"):
        masks[k]["window"] = window
        gw = masks[k]["geom"].intersection(window)
        masks[k]["geom_window"] = gw if gw.is_valid else gw.buffer(0)

    glaciers = prepare_glaciers(ref_arc, params["glacier_max_dist_m"])
    per = {key: run_period(key, masks[o], masks[n], flowlines_gdf, ref_arc, domain, params)
           for key, o, n, _ in PERIODS}

    units = find_domain_mismatch_units(masks, domain, ref_arc, params["unit_min_km2"])
    bins = {}
    for key, (df, seg) in per.items():
        df["unit_id"] = samples_on_units(seg, units, params["unit_on_tol_m"], params["unit_on_frac"])
        bins[key] = exclude_unit_bins(bin_max_magnitude(df, arc_len_km, params["bin_km"]), df, key, params["bin_km"])

    def to_map(g):
        return gpd.GeoSeries([g], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]

    layers = {k: to_map(polygons_only(masks[k]["geom"].intersection(domain))) for k in ("LIA", "1985", "2000", "2022")}
    bx0, by0, bx1, by1 = to_map(ref_arc).bounds
    padx, pady = 0.10 * (bx1 - bx0), 0.04 * (by1 - by0)
    layers["extent"] = (bx0 - padx, by0 - pady, bx1 + padx * 1.6, by1 + 2.0 * pady)
    layers["greenland"] = to_map(masks["LIA"]["geom"].simplify(3000))
    layers["domain"] = to_map(domain)
    glaciers["pt_map"] = list(gpd.GeoSeries(glaciers.geometry, crs=TARGET_CRS).to_crs(MAP_CRS))

    out_base = os.path.join(OUTPUT_DIR, "Fig4_NW_margin_change")
    plot_fig4(bins, arc_len_km, glaciers, layers, (to_map(s), to_map(e)), out_base, params["bin_km"])

    # source data: one row per bin, measured change and the bar drawn in each panel
    b0 = bins[PERIODS[0][0]]
    src = pd.DataFrame({"bin": b0["bin"].astype(int), "bin_start_km": b0["bin_start_km"].round(3),
                        "bin_end_km": b0["bin_end_km"].round(3), "bin_center_km": b0["bin_center_km"].round(3)})
    gx, gl = glaciers["x_km"].values, glaciers["label"].values
    src["nearest_labelled_glacier_within_10km"] = [
        gl[int(np.argmin(np.abs(gx - x)))] if len(gx) and np.min(np.abs(gx - x)) <= 10.0 else ""
        for x in src["bin_center_km"]]
    for key, _, _, _ in PERIODS:
        src[f"{key}_change_km"] = (bins[key]["change_m"] / 1000.0).round(4).values
        src[f"{key}_drawn_km"] = np.round(drawn_km(key, bins[key]["change_m"].values), 4)
        src[f"{key}_status"] = bins[key]["status"].values
    src.to_csv(os.path.join(OUTPUT_DIR, "Fig4_source_data.csv"), index=False)
    print(f"\nSaved Fig4_NW_margin_change.png / .svg and Fig4_source_data.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
