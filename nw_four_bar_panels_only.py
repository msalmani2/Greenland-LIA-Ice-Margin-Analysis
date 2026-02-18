"""
nw_four_bar_panels_only.py

High-resolution ice-margin change transect for Northwest Greenland (Fig. 4).
Unlike the whole-Greenland bar plots, this works at 500 m sample spacing grouped
into 5 km bins, which is fine enough to pick up individual outlet glaciers.
Four stacked panels show LIA-1985, 1985-2000, 2000-2022, and LIA-2022. The
reference margin is the PROMICE NW 1985 mask; a manually placed start/mid/end
triplet defines which arc of that polygon to traverse.

Where glacier flowlines are available the distance calculation follows the flow
path instead of using straight-line nearest-neighbor, which matters a lot in the
narrow Upernavik fjords.

Used in:
    Salmani et al., "Nearly half of Greenland's post-Little Ice Age area loss
    occurred since 2000"
    - Fig. 4 (Northwest Greenland four-panel bar plot)
    - Supplementary Table S5 (glacier-specific retreat statistics)

Inputs:  NW ice masks, start/mid/end shapefile, glacier names, flowlines
Outputs: NW_four_bar_panels_same_scale_v5.png / .svg
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring, linemerge
from shapely.strtree import STRtree

warnings.filterwarnings("ignore")
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 13
mpl.rcParams["axes.titlesize"] = 22
mpl.rcParams["axes.labelsize"] = 17
mpl.rcParams["xtick.labelsize"] = 13
mpl.rcParams["ytick.labelsize"] = 13
mpl.rcParams["legend.fontsize"] = 14

TARGET_CRS = "EPSG:32624"  # UTM Zone 24N


# Geometry helpers

def remove_holes(geometry):
    """Drop interior rings, we only care about the outer boundary."""
    if isinstance(geometry, Polygon):
        return Polygon(geometry.exterior)
    elif geometry.geom_type == "MultiPolygon":
        return type(geometry)([Polygon(poly.exterior) for poly in geometry.geoms])
    return geometry


def extract_largest_exterior_linestring(geom):
    if isinstance(geom, Polygon):
        return LineString(geom.exterior.coords)
    if geom.geom_type == "MultiPolygon":
        largest = max(list(geom.geoms), key=lambda g: g.area)
        return LineString(largest.exterior.coords)
    raise ValueError(f"Unsupported geometry type for margin extraction: {geom.geom_type}")


def ensure_linestring(geom):
    if isinstance(geom, LineString):
        return geom
    return linemerge(geom)


def concat_lines(ls1, ls2):
    c1 = list(ls1.coords)
    c2 = list(ls2.coords)
    return LineString(c1 + c2[1:])


def pick_arc_by_midpoint(reference_ring, start_pt, mid_pt, end_pt):
    """A closed ring has two arcs between any two points. Pick the one
    that passes closer to mid_pt so we traverse the right side."""
    L = reference_ring.length
    s0 = reference_ring.project(start_pt)
    s1 = reference_ring.project(end_pt)

    if s0 <= s1:
        arc_a = ensure_linestring(substring(reference_ring, s0, s1, normalized=False))
        tail = ensure_linestring(substring(reference_ring, s1, L, normalized=False))
        head = ensure_linestring(substring(reference_ring, 0, s0, normalized=False))
        arc_b = concat_lines(tail, head)
    else:
        arc_b = ensure_linestring(substring(reference_ring, s1, s0, normalized=False))
        tail = ensure_linestring(substring(reference_ring, s0, L, normalized=False))
        head = ensure_linestring(substring(reference_ring, 0, s1, normalized=False))
        arc_a = concat_lines(tail, head)

    d_a = mid_pt.distance(arc_a)
    d_b = mid_pt.distance(arc_b)
    return arc_a if d_a <= d_b else arc_b


def sample_points_along_line(line, sample_distance=500.0):
    """Evenly spaced points and their cumulative distances along the line."""
    total = line.length
    n_steps = int(np.floor(total / sample_distance))
    points = [line.interpolate(i * sample_distance) for i in range(n_steps + 1)]
    if points[-1].distance(line.interpolate(total)) > 1e-6:
        points.append(line.interpolate(total))
    dists = [i * sample_distance for i in range(len(points))]
    if len(dists) > 0:
        dists[-1] = total
    return points, dists


# Loading

def load_ice_masks(data_dir, mask_files, target_crs=TARGET_CRS):
    masks = {}
    for name, fname in mask_files.items():
        path = os.path.join(data_dir, fname)
        gdf = gpd.read_file(path).to_crs(target_crs)
        gdf = gdf[gdf.geometry.notna()]
        gdf["geometry"] = gdf["geometry"].apply(remove_holes)
        gdf = gdf.dissolve().reset_index(drop=True)
        geom = gdf.geometry.iloc[0]
        masks[name] = gpd.GeoDataFrame(geometry=[geom], crs=target_crs)
        print(f"Loaded mask: {name} -> {fname}")
    return masks


def load_start_end_midpoint(shp_path, target_crs=TARGET_CRS):
    """Shapefile with 'id' column: 1=start, 2=midpoint, 3=end. Pins the NW arc."""
    gdf = gpd.read_file(shp_path).to_crs(target_crs)
    if "id" not in gdf.columns:
        raise ValueError("NW_startend shapefile must have an 'id' column with values 1,2,3.")

    def get_pt(val):
        rows = gdf[gdf["id"] == val]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one point with id={val}, found {len(rows)}.")
        geom = rows.geometry.iloc[0]
        if geom.geom_type != "Point":
            raise ValueError(f"id={val} feature must be a Point.")
        return geom

    return get_pt(1), get_pt(2), get_pt(3)


# Distance calculations

def extract_ice_margins(ice_masks):
    margins = {}
    for name, gdf in ice_masks.items():
        geom = gdf.geometry.iloc[0]
        margins[name] = extract_largest_exterior_linestring(geom)
    return margins


def calculate_margin_changes(reference_points, comparison_margin, comparison_mask_geom):
    """Signed distance: inside comparison mask = advance (negative), outside = retreat (positive)."""
    distances = []
    for pt in reference_points:
        nearest = comparison_margin.interpolate(comparison_margin.project(pt))
        d = pt.distance(nearest)
        inside = comparison_mask_geom.covers(pt)
        distances.append(-d if inside else d)
    return distances


def calculate_all_changes(reference_points, ice_margins, ice_masks, reference_name):
    all_d = {}
    for name, margin in ice_margins.items():
        if name == reference_name:
            all_d[name] = [0.0] * len(reference_points)
        else:
            comp_geom = ice_masks[name].geometry.iloc[0]
            all_d[name] = calculate_margin_changes(reference_points, margin, comp_geom)
    return all_d


# Flowline-enhanced distance

def _build_nw_flowlines(flowlines_shp, clip_geom, target_crs=TARGET_CRS, buffer_m=500000.0):
    if flowlines_shp is None or (not os.path.exists(flowlines_shp)):
        return None, None
    fl_gdf = gpd.read_file(flowlines_shp)
    if fl_gdf.empty:
        return None, None
    fl_gdf = fl_gdf.to_crs(target_crs)
    clip_buff = gpd.GeoSeries([clip_geom], crs=target_crs).buffer(buffer_m).iloc[0]
    fl_nw = fl_gdf[fl_gdf.geometry.intersects(clip_buff) | fl_gdf.geometry.within(clip_buff)].copy()
    if fl_nw.empty:
        return None, None
    geoms = list(fl_nw.geometry.values)
    tree = STRtree(geoms)
    return fl_nw, tree


def _flowline_distance_for_point(pt, comparison_margin, comparison_mask_geom,
                                 flowline_tree, flowline_geoms, search_radius=5000.0):
    """Try to measure along-flowline distance. Returns None if no flowline nearby."""
    if flowline_tree is None or not flowline_geoms:
        return None
    search_geom = pt.buffer(search_radius)
    candidates = flowline_tree.query(search_geom)
    if candidates is None or len(candidates) == 0:
        return None

    # STRtree sometimes returns indices, sometimes geometries
    min_d = float("inf")
    best_fl = None
    for fl in candidates:
        if isinstance(fl, (int, np.integer)):
            if fl < 0 or fl >= len(flowline_geoms):
                continue
            geom = flowline_geoms[fl]
        else:
            geom = fl
        if geom is None:
            continue
        d = geom.distance(pt)
        if d < min_d:
            min_d = d
            best_fl = geom

    if best_fl is None:
        return None
    inter = best_fl.intersection(comparison_margin)
    if inter.is_empty:
        return None

    if isinstance(inter, Point):
        ip = inter
    elif inter.geom_type == "MultiPoint":
        ip = min(inter.geoms, key=lambda g: g.distance(pt))
    elif inter.geom_type == "LineString":
        ip = inter.interpolate(inter.project(pt))
    else:
        return None

    s_pt = best_fl.project(pt)
    s_ip = best_fl.project(ip)
    dist_along = abs(s_ip - s_pt)
    inside = comparison_mask_geom.covers(pt)
    return -dist_along if inside else dist_along


def calculate_all_changes_with_flowlines(reference_points, ice_margins, ice_masks,
                                         reference_name, flowlines_shp=None,
                                         flowline_search_radius=5000.0):
    """Like calculate_all_changes but tries flowline method first, falls back to nearest-distance."""
    reference_geom = ice_masks[reference_name].geometry.iloc[0]
    flowlines_gdf, flowline_tree = _build_nw_flowlines(flowlines_shp, reference_geom)
    flowline_geoms = list(flowlines_gdf.geometry.values) if flowlines_gdf is not None else []

    all_d = {}
    for name, margin in ice_margins.items():
        if name == reference_name:
            all_d[name] = [0.0] * len(reference_points)
            continue
        comp_geom = ice_masks[name].geometry.iloc[0]
        dists = []
        for pt in reference_points:
            d_fl = _flowline_distance_for_point(pt, comparison_margin=margin,
                                                comparison_mask_geom=comp_geom,
                                                flowline_tree=flowline_tree,
                                                flowline_geoms=flowline_geoms,
                                                search_radius=flowline_search_radius)
            if d_fl is not None:
                dists.append(float(d_fl))
            else:
                nearest = margin.interpolate(margin.project(pt))
                d = pt.distance(nearest)
                inside = comp_geom.covers(pt)
                dists.append(-d if inside else d)
        all_d[name] = dists
    return all_d


# Grouping and period math

def group_segments_max_magnitude(values, segment_size):
    """Collapse values into groups, keeping the element with biggest |value| in each."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return [], []
    if n <= segment_size:
        j = int(np.argmax(np.abs(values)))
        return [float(values[j])], [j]
    grouped_vals, grouped_idx = [], []
    for i in range(0, n, segment_size):
        seg = values[i:i + segment_size]
        if len(seg) > 0:
            j = int(np.argmax(np.abs(seg)))
            grouped_vals.append(float(seg[j]))
            grouped_idx.append(i + j)
    return grouped_vals, grouped_idx


def build_period_series_500m(dist_rel_ref, name_lia="Trimline_NW_LIA",
                             name_2000="GIMP_NW_1999_2002", name_2022="PROMICE_2022_NW"):
    """From per-epoch signed distances relative to 1985 reference, build the four period series."""
    lia = np.asarray(dist_rel_ref.get(name_lia, []), dtype=float)
    d2000 = np.asarray(dist_rel_ref.get(name_2000, []), dtype=float)
    d2022 = np.asarray(dist_rel_ref.get(name_2022, []), dtype=float)
    n = max(len(lia), len(d2000), len(d2022))

    def pad_to(arr, m):
        if len(arr) == 0:
            return np.full(m, np.nan)
        if len(arr) < m:
            out = np.full(m, np.nan)
            out[:len(arr)] = arr
            return out
        return arr[:m]

    lia = pad_to(lia, n)
    d2000 = pad_to(d2000, n)
    d2022 = pad_to(d2022, n)

    # flip signs so retreat shows as negative on the plot
    return {
        "LIA_to_1985": lia,
        "1985_to_2000": -d2000,
        "2000_to_2022": d2000 - d2022,
        "LIA_to_2022": lia - d2022,
    }


# Glacier labels

def infer_glacier_name(row):
    for col in ["New_Greenl", "Official_n", "Old_Greenl", "Alternativ", "Name", "GLACNAME"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip() not in ("", "None"):
            return str(row[col])
    return None


def prepare_glacier_positions(glacier_shp, clip_geom, target_crs=TARGET_CRS, buffer_m=10000):
    """Load glacier points and filter to the NW outlets we want labelled."""
    if glacier_shp is None or not os.path.exists(glacier_shp):
        return None
    glaciers_to_include = {
        "Kangerlussuup Sermersua", "Salliarutsip Sermia", "Sermeq", "Akullikassaap Sermia",
        "Kakiffaat Sermiat", "Sermeq Avannarleq", "Ikertuup Sermersua", "Illulip Sermia",
        "Nunatakassaap Sermia", "Kjer Gletsjer", "Sermersuaq", "Dietrichson Gletsjer",
        "Sverdrup Gletsjer", "Nansen Gletsjer", "Nordenskiöld Gletsjer", "Nuussuup Sermia",
        "Issuussasull Sermiat", "Rink Gletsjer", "Qeqertat Timaanni Sermeq", "Morell Gletsjer",
        "Innaqqissorsuup Oqquani Sermeq", "Yngvar Nielsen Gletsjer", "Helland Gletsjer",
        "Sermipaluk", "Savissup Sermia", "Naajarsorfik", "Neptune Gletsjer",
        "Paakitsup Sermersua", "Ullip Sermia", "Søndre Tvillinggletsjer"
    }
    gdf = gpd.read_file(glacier_shp).to_crs(target_crs)
    clip_boundary = gpd.GeoSeries([clip_geom], crs=target_crs).buffer(buffer_m).iloc[0]
    gdf = gdf[gdf.geometry.intersects(clip_boundary) | gdf.geometry.within(clip_boundary)]
    if gdf.empty:
        return None
    gdf['inferred_name'] = gdf.apply(infer_glacier_name, axis=1)
    return gdf[gdf['inferred_name'].isin(glaciers_to_include)].copy()


def glacier_lines_and_labels(axes, glacier_gdf, ref_points_filtered, x_km_filtered,
                             line_color="#6b7280", line_alpha=0.35, line_lw=0.9,
                             label_rotation=45, label_fontsize=13):
    """Vertical glacier lines on all panels, labels on the top panel only."""
    if glacier_gdf is None or glacier_gdf.empty or len(ref_points_filtered) == 0:
        return
    ref_pts = np.array([(p.x, p.y) for p in ref_points_filtered])
    x_positions = []
    labels = []
    for _, row in glacier_gdf.iterrows():
        name = row['inferred_name']
        if not name:
            continue
        gx, gy = row.geometry.x, row.geometry.y
        d = np.hypot(ref_pts[:, 0] - gx, ref_pts[:, 1] - gy)
        k = int(np.nanargmin(d))
        x_positions.append(x_km_filtered[k])
        labels.append(name)

    if len(x_positions) == 0:
        return
    x_positions = np.array(x_positions)
    order = np.argsort(x_positions)
    x_positions = x_positions[order]
    labels = [labels[i] for i in order]

    for ax in axes:
        y0, y1 = ax.get_ylim()
        ax.vlines(x_positions, y0, y1, colors=line_color, alpha=line_alpha, linewidth=line_lw, zorder=2)

    ax_top = axes[0]
    ylim = ax_top.get_ylim()
    y_text = ylim[1] - 0.05 * (ylim[1] - ylim[0])
    for x, lab in zip(x_positions, labels):
        ax_top.text(x, y_text, lab, ha="right", va="top", rotation=label_rotation,
                    fontsize=label_fontsize, fontweight="bold", rotation_mode="anchor",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.6))


# Four-panel plot

def plot_four_bar_panels_same_scale(x_km_grouped, period_grouped_dict,
                                    title="NW Greenland Ice Margin Change by Period (5-km bins)",
                                    figsize=(18, 14), glacier_gdf=None, ref_points_filtered=None,
                                    x_km_filtered=None, x_major_tick_interval_km=200,
                                    x_minor_tick_interval_km=5, bar_width_km=8.0):
    """Four stacked bars sharing x-axis, each with its own y-limits but a common positive ceiling."""
    plt.style.use("seaborn-v0_8-whitegrid")
    panel_info = [
        ("LIA_to_1985",   "#326186", "Change from LIA to 1985 (negative = retreat)"),
        ("1985_to_2000",  "#E75A0D", "Change from 1985 to 2000 (negative = retreat)"),
        ("2000_to_2022",  "#5D2609", "Change from 2000 to 2022 (negative = retreat)"),
        ("LIA_to_2022",   "#2F855A", "Cumulative change from LIA to 2022 (negative = retreat)"),
    ]

    # shared positive y-limit across panels
    all_vals_km_by_panel = {key: np.asarray(period_grouped_dict[key], dtype=float) / 1000.0
                           for key, _, _ in panel_info}
    pos_max_km = 0.0
    for arr in all_vals_km_by_panel.values():
        if arr.size:
            m = np.nanmax(arr)
            if np.isfinite(m):
                pos_max_km = max(pos_max_km, m)
    pos_max_km = max(pos_max_km, 0.1)
    pos_lim = max(1.05 * pos_max_km, 10.0)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, sharey=False,
                             gridspec_kw=dict(hspace=0.01), facecolor="white")

    for ax, (key, color, label) in zip(axes, panel_info):
        y_km = np.asarray(period_grouped_dict[key], dtype=float) / 1000.0
        neg_min_km = 0.0
        if y_km.size and np.any(np.isfinite(y_km)):
            y_neg = y_km[np.isfinite(y_km) & (y_km < 0)]
            if y_neg.size:
                neg_min_km = float(np.nanmin(y_neg))
        neg_lim = max(1.05 * abs(neg_min_km), 30.0)
        ax.set_ylim(-neg_lim, pos_lim)

        candidate_ticks = np.array([-30.0, -20.0, -10.0, 0.0, 10.0])
        y0, y1 = ax.get_ylim()
        ticks = candidate_ticks[(candidate_ticks >= y0) & (candidate_ticks <= y1)]
        if (0.0 >= y0) and (0.0 <= y1) and (0.0 not in ticks):
            ticks = np.sort(np.append(ticks, 0.0))
        ax.set_yticks(ticks)

        ax.bar(x_km_grouped, y_km, width=bar_width_km, color=color, alpha=0.95,
               edgecolor="#333333", linewidth=0.4, align="center")
        ax.axhline(0, color="black", linewidth=2.0, zorder=3)
        ax.text(0.01, 0.92, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=15, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.95, edgecolor="none", pad=0.6))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_linewidth(2.4)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.grid(False, axis="x")
        ax.set_ylabel("km", fontsize=15, fontweight="bold")

    xmin, xmax = float(np.nanmin(x_km_grouped)), float(np.nanmax(x_km_grouped))
    major_ticks = np.arange(
        np.floor(xmin / x_major_tick_interval_km) * x_major_tick_interval_km,
        np.ceil(xmax / x_major_tick_interval_km) * x_major_tick_interval_km + 1e-9,
        x_major_tick_interval_km)
    for ax in axes:
        ax.set_xticks(major_ticks)
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor_tick_interval_km))
    for ax in axes[:-1]:
        ax.tick_params(axis="x", which="major", length=5.0, width=1.0, labelbottom=False)
        ax.tick_params(axis="x", which="minor", length=3.0, width=0.8)
    axes[-1].tick_params(axis="x", which="major", length=6.0, width=1.2, labelbottom=True)
    axes[-1].tick_params(axis="x", which="minor", length=3.0, width=0.8)
    axes[-1].set_xlabel("Distance along reference ice margin (km)", fontsize=16, fontweight="bold")

    if glacier_gdf is not None and ref_points_filtered is not None and x_km_filtered is not None:
        glacier_lines_and_labels(axes, glacier_gdf, ref_points_filtered, x_km_filtered,
                                 line_color="#4b5563", line_alpha=0.35, line_lw=0.9,
                                 label_rotation=45, label_fontsize=13)

    fig.suptitle("NW Greenland Ice Margin Change by Period relative to PROMICE_NW_1985 (5-km bins)",
                 fontsize=22, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    return fig, axes


# -------------------------------------------------------

def run_four_panel_plot(data_dir, ice_mask_files, startend_shp, glacier_shp=None,
                        flowlines_shp=None, reference_mask_name="PROMICE_NW_1985",
                        sample_distance=500, group_bin_km=5.0, range_km=(0, 2700),
                        output_dir="nw_four_panel_output", x_major_tick_interval_km=200,
                        x_minor_tick_interval_km=5, bar_width_km=8.0,
                        flowline_search_radius=5000.0):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading ice masks and extracting margins...")
    ice_masks = load_ice_masks(data_dir, ice_mask_files, target_crs=TARGET_CRS)
    ice_margins = extract_ice_margins(ice_masks)

    print("Loading start/mid/end anchors...")
    start_pt, mid_pt, end_pt = load_start_end_midpoint(startend_shp, target_crs=TARGET_CRS)

    ref_geom = ice_masks[reference_mask_name].geometry.iloc[0]
    ref_ring = extract_largest_exterior_linestring(ref_geom)
    print("Selecting NW arc and sampling points...")
    ref_arc = pick_arc_by_midpoint(ref_ring, start_pt, mid_pt, end_pt)
    ref_points, ref_dists_m = sample_points_along_line(ref_arc, sample_distance=sample_distance)
    x_km_full = np.array(ref_dists_m, dtype=float) / 1000.0

    lo, hi = range_km
    mask_range = (x_km_full >= lo) & (x_km_full <= hi)
    x_km_full_f = x_km_full[mask_range]
    ref_points_f = [ref_points[i] for i, keep in enumerate(mask_range) if keep]

    print("Computing signed distances (flowlines where available)...")
    if flowlines_shp is not None and os.path.exists(flowlines_shp):
        margin_distances = calculate_all_changes_with_flowlines(
            reference_points=ref_points, ice_margins=ice_margins, ice_masks=ice_masks,
            reference_name=reference_mask_name, flowlines_shp=flowlines_shp,
            flowline_search_radius=flowline_search_radius)
    else:
        margin_distances = calculate_all_changes(ref_points, ice_margins, ice_masks, reference_mask_name)
    original_by_mask_filtered = {
        name: np.asarray(vals, dtype=float)[mask_range] for name, vals in margin_distances.items()}

    print("Building period series and grouping to 5-km bins...")
    period_500m = build_period_series_500m(original_by_mask_filtered)
    seg_size = int(round((group_bin_km * 1000.0) / float(sample_distance)))

    some_key = "LIA_to_1985"
    _, gidx_tmp = group_segments_max_magnitude(period_500m[some_key], seg_size)
    x_km_grouped = x_km_full_f[gidx_tmp]

    period_grouped = {}
    for key, arr in period_500m.items():
        gvals, _ = group_segments_max_magnitude(arr, seg_size)
        period_grouped[key] = gvals

    glacier_gdf = prepare_glacier_positions(glacier_shp, ref_geom) if glacier_shp else None

    print("Creating four-panel bar plot...")
    fig, axes = plot_four_bar_panels_same_scale(
        x_km_grouped=x_km_grouped, period_grouped_dict=period_grouped,
        title="NW Greenland Ice Margin Change by Period relative to PROMICE_NW_1985",
        figsize=(18, 11), glacier_gdf=glacier_gdf, ref_points_filtered=ref_points_f,
        x_km_filtered=x_km_full_f, x_major_tick_interval_km=x_major_tick_interval_km,
        x_minor_tick_interval_km=x_minor_tick_interval_km, bar_width_km=bar_width_km)

    png_out = os.path.join(output_dir, "NW_four_bar_panels_same_scale_v5.png")
    svg_out = os.path.join(output_dir, "NW_four_bar_panels_same_scale_v5.svg")
    fig.savefig(png_out, dpi=350, bbox_inches="tight")
    fig.savefig(svg_out, bbox_inches="tight")
    print(f"Saved: {png_out}\nSaved: {svg_out}")

    return {"reference_arc": ref_arc, "x_km_full_filtered": x_km_full_f,
            "x_km_grouped": x_km_grouped, "period_grouped": period_grouped,
            "glacier_gdf": glacier_gdf}


# -------------------------------------------------------
if __name__ == "__main__":
    # SET THESE PATHS to where your NW data lives (Kjeldsen, 2022)
    data_dir = os.path.join("data", "NorthWest_Ice_masks")
    ice_mask_files = {
        "PROMICE_NW_1985": "PROMICE_NW_1985.shp",
        "Trimline_NW_LIA": "Trimline_NW_LIA.shp",
        "GIMP_NW_1999_2002": "GIMP_NW_1999_2002.shp",
        "PROMICE_2022_NW": "PROMICE_2022_NW.shp",
    }
    startend_shp = os.path.join(data_dir, "NW_startend.shp")
    glacier_shp = os.path.join("data", "glacier_names", "GreenlandGlacierNames_GGNv01_WGS84.shp")
    flowlines_shp = os.path.join("data", "flowlines", "glacier_flowlines_for_margin_change.shp")

    run_four_panel_plot(
        data_dir=data_dir, ice_mask_files=ice_mask_files, startend_shp=startend_shp,
        glacier_shp=glacier_shp, flowlines_shp=flowlines_shp,
        reference_mask_name="PROMICE_NW_1985",
        sample_distance=500, group_bin_km=5.0, range_km=(0, 3000),
        output_dir="nw_four_panel_output",
        x_major_tick_interval_km=200, x_minor_tick_interval_km=5,
        bar_width_km=8.0, flowline_search_radius=2000.0)
