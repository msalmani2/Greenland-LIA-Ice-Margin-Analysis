"""
visualize_retreat.py

Produces the main map figures and optional coastline bar-plots for the paper.
The maps show Greenland's IMBIE basins coloured by area-change rate with
proportionally-sized circles along the coast indicating local retreat/advance
magnitude (cf. Fig. 2 in the manuscript). An optional bar-plot mode linearises
the coastline and plots per-segment change values (early drafts of Fig. 3).

Used in:
    Salmani et al., "Nearly half of Greenland's post-Little Ice Age area loss
    occurred since 2000"
    - Fig. 2 (circle-marker maps, four panels)

Inputs:  Change-result shapefiles (from calculate_retreat_with_flowlines.py), processed ice-mask boundaries, IMBIE basins,
         glacier name CSV, hillshade raster, colour CSV from area_comparison script
Outputs: PNG + SVG maps per period, PNG + SVG bar plots per period (when enabled)
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from shapely.geometry import Point
from tqdm import tqdm
import matplotlib as mpl
import rasterio
from rasterio.plot import show as rioshow

mpl.rcParams['svg.fonttype'] = 'none'  # text stays as text in SVGs

# SET THESE PATHS to where your data lives
IMBIE_BASE_PATH = os.path.join("data", "IMBIE_basins", "IMBIE basins - cleaned_extended.shp")
GLACIER_CSV_PATH = os.path.join("data", "glacier_names", "glacier_names_IDs.csv")
HILLSHADE_PATH = os.path.join("data", "basemap", "Greenland_Hillshade_Basemap.tif")

BASE_DIR = "data/ice_margin_analysis"
CHANGE_RESULTS_DIR = os.path.join(BASE_DIR, "outputs")

RESULT_FILES = {
    "LIA_to_1980s": "LIA_to_1980s.shp",
    "1980s_to_2000": "1980s_to_2000.shp",
    "2000_to_2022": "2000_to_2022.shp",
    "LIA_to_2022": "LIA_to_2022.shp",
    "1980s_to_2022": "1980s_to_2022.shp",
}

PROCESSED_MASKS_DIR = os.path.join(BASE_DIR, "processed_masks")

EARLIER_BOUNDARY_PATHS = {
    "LIA_to_1980s": os.path.join(PROCESSED_MASKS_DIR, "LIA_processed.shp"),
    "1980s_to_2000": os.path.join(PROCESSED_MASKS_DIR, "PROMICE_1980s_processed_v2.shp"),
    "2000_to_2022": os.path.join(PROCESSED_MASKS_DIR, "CCI_2000_processed.shp"),
    "LIA_to_2022": os.path.join(PROCESSED_MASKS_DIR, "LIA_processed.shp"),
    "1980s_to_2022": os.path.join(PROCESSED_MASKS_DIR, "PROMICE_1980s_processed_v2.shp"),
}

TARGET_CRS = "EPSG:3413"    # polar stereographic for the maps
COMPUTE_CRS = "EPSG:32624"  # UTM 24N, what the change results are stored in

# toggle which periods to generate (handy when re-running just one panel)
RUN_LIA_1980S_MAP = True
RUN_LIA_1980S_BAR = False
RUN_1980S_2000_MAP = True
RUN_1980S_2000_BAR = False
RUN_2000_2022_MAP = True
RUN_2000_2022_BAR = False
RUN_LIA_2022_MAP = True
RUN_LIA_2022_BAR = False
RUN_1980S_2022_MAP = True
RUN_1980S_2022_BAR = False

# per-period basin colours, loaded from the area-comparison CSV at runtime
PERIOD_REGION_COLORS = {}

# fallback if the CSV isn't around
REGION_COLORS = {
    'NO': "#cb997e", 'NE': "#eddcd2", 'CE': "#a5a58d",
    'SE': "#F2EAE3", 'SW': "#ddbea9", 'CW': "#fff1e6", 'NW': "#b7b7a4"
}


def load_period_colors_from_csv(csv_path):
    global PERIOD_REGION_COLORS

    if not os.path.exists(csv_path):
        print(f"Warning: Color CSV not found: {csv_path}")
        print("Using fallback colors.")
        return

    try:
        df = pd.read_csv(csv_path)
        period_mapping = {
            "LIA_to_1980s": "LIA_to_1980s",
            "1980s_to_2000": "1980s_to_2000",
            "2000_to_2022": "2000_to_2022",
            "LIA_to_2022": "LIA_to_2022",
            "1980s_to_2022": "1980s_to_2022"
        }

        for period_key, csv_col_prefix in period_mapping.items():
            color_col = f"{csv_col_prefix}_color_hex"
            if color_col in df.columns:
                period_colors = {}
                for _, row in df.iterrows():
                    region = row['Region']
                    if region != 'Total' and pd.notna(row[color_col]) and row[color_col] != '':
                        period_colors[region] = row[color_col]
                if period_colors:
                    PERIOD_REGION_COLORS[period_key] = period_colors

        print(f"Loaded colors for {len(PERIOD_REGION_COLORS)} periods from CSV.")
    except Exception as e:
        print(f"Error loading colors from CSV: {e}")
        print("Using fallback colors.")


# retreat/advance colour ramp for the circle markers
CMAP_COLORS = ["#490cac", "#6831bf", "#8656d2", "#c2a0f8", "#D89E7C",
               "#ee9b00", "#ce5e09", "#ae2012", "#9b2226"]
RETREAT_ADVANCE_CMAP = LinearSegmentedColormap.from_list("retreat_advance", CMAP_COLORS)

# style knobs, all in one place so the plotting code doesn't get cluttered
MAP_STYLE_PARAMS = {
    "figsize": (10, 12), "facecolor": 'white',
    "imbie_region_alpha": 1.0,
    "glacier_marker_size": 8, "glacier_marker_color": 'black',
    "glacier_marker_edgecolor": 'white', "glacier_marker_edgewidth": 0.7,
    "north_arrow_fontsize": 14, "scale_bar_fontsize": 12, "title_fontsize": 18,
    "dpi": 300, "rasterize_heavy": True,
    "coastline_color": '#333333', "coastline_linewidth": 1.5, "coastline_alpha": 0.7,
    "hillshade_alpha": 0.5, "hillshade_cmap": 'gray'
}

BUFFER_STYLE_PARAMS = {
    "min_change_threshold_km": 0.5, "sampling_interval_m": 2000, "grouping_size": 10,
    "buffer_scale_factor": 4, "buffer_alpha": 0.9,
    "buffer_edge_color": 'black', "buffer_edge_width": 0.4,
    "legend_circle_sizes": [-20, -10, -5, 5, 10, 20],
    "legend_position": (0.85, 0.0)
}

BAR_PLOT_STYLE_PARAMS = {
    "figsize": (12, 7), "bar_width_normalized": 0.4, "bar_alpha": 1.0,
    "bar_edgecolor": 'none',
    "hline_y": 0, "hline_color": 'black', "hline_linewidth": 0.6, "hline_alpha": 0.6,
    "extra_hlines": [10, -10, -20],
    "extra_hlines_style": {'color': '#888888', 'linestyle': '--', 'linewidth': 0.7, 'alpha': 0.9},
    "ylabel": 'Ice Margin Change (km)', "ylabel_fontsize": 14, "title_fontsize": 16,
    "dpi": 300, "y_axis_limits": (-35, 20), "use_fixed_y_limits": True,
    "glacier_line_color": 'black', "glacier_line_width": 0.5, "glacier_line_alpha": 0.3,
    "glacier_line_y_start": 0, "glacier_line_y_end": -30, "glacier_label_y": -32,
    "glacier_text_fontsize": 12, "glacier_name_rotation": 45, "show_region_legend": False
}

PERIOD_META = {
    "LIA_to_1980s": {"title_suffix": "LIA to 1980s", "prefix": "lia_to_1980s"},
    "1980s_to_2000": {"title_suffix": "1980s to 2000", "prefix": "1980s_to_2000"},
    "2000_to_2022": {"title_suffix": "2000 to 2022", "prefix": "2000_to_2022"},
    "LIA_to_2022": {"title_suffix": "LIA to 2022", "prefix": "lia_to_2022"},
    "1980s_to_2022": {"title_suffix": "1980s to 2022", "prefix": "1980s_to_2022"},
}


def get_period_colors(period_key):
    return PERIOD_REGION_COLORS.get(period_key, REGION_COLORS)


# Data prep

def process_coastline_data_for_buffers(change_data_gdf, min_threshold_km=1.0,
                                       sampling_interval_m=1000, grouping_size=5):
    """Thin out the change segments into grouped representative points for the map circles."""
    min_threshold_m = min_threshold_km * 1000.0
    significant = change_data_gdf[np.abs(change_data_gdf['change_m']) >= min_threshold_m].copy()

    if len(significant) == 0:
        return {"buffer_points": [], "max_abs_change": 0,
                "num_original_segments": len(change_data_gdf),
                "num_filtered_segments": 0, "num_final_points": 0}

    sampled = []
    for idx, seg in significant.iterrows():
        geom = seg.geometry
        val = seg['change_m']
        if geom.geom_type == 'LineString':
            L = geom.length
            if L > sampling_interval_m:
                n = int(L / sampling_interval_m)
                for i in range(n + 1):
                    d = min(i * sampling_interval_m, L)
                    sampled.append({'geometry': geom.interpolate(d), 'change_m': val,
                                    'segment_id': idx, 'distance': d})
            else:
                sampled.append({'geometry': geom.centroid, 'change_m': val,
                                'segment_id': idx, 'distance': 0})

    if not sampled:
        return {"buffer_points": [], "max_abs_change": 0,
                "num_original_segments": len(change_data_gdf),
                "num_filtered_segments": len(significant), "num_final_points": 0}

    df = pd.DataFrame(sampled)
    df['x'] = df['geometry'].apply(lambda p: p.x)
    df['y'] = df['geometry'].apply(lambda p: p.y)

    # keep the most extreme point in each group
    grouped_points = []
    for i in range(0, len(df), grouping_size):
        group = df.iloc[i:i + grouping_size]
        if len(group) > 0:
            j = group['change_m'].abs().idxmax()
            g = group.loc[j]
            grouped_points.append({'geometry': g['geometry'], 'change_m': g['change_m'],
                                   'x': g['x'], 'y': g['y']})

    max_abs = max(abs(p['change_m']) for p in grouped_points) if grouped_points else 0
    return {"buffer_points": grouped_points, "max_abs_change": max_abs,
            "num_original_segments": len(change_data_gdf),
            "num_filtered_segments": len(significant), "num_final_points": len(grouped_points)}


def prepare_map_specific_data_with_buffers(imbie_gdf, earlier_boundary_gdf, change_data_gdf):
    """Clip IMBIE regions to the earlier boundary and build buffer data."""
    earlier_boundary_gdf = earlier_boundary_gdf.copy()
    earlier_boundary_gdf['geometry'] = earlier_boundary_gdf.geometry.simplify(100)
    earlier_boundary_gdf['geometry'] = earlier_boundary_gdf.geometry.buffer(0.01)

    # union with fallbacks because shapely can be finicky
    boundary_union = None
    try:
        boundary_union = earlier_boundary_gdf.geometry.union_all()
    except:
        try:
            from shapely.ops import unary_union
            boundary_union = unary_union(earlier_boundary_gdf.geometry.values)
        except:
            if len(earlier_boundary_gdf) == 1:
                boundary_union = earlier_boundary_gdf.geometry.iloc[0]
            else:
                from shapely.ops import unary_union
                geoms = [g for g in earlier_boundary_gdf.geometry if g is not None and g.is_valid]
                if geoms:
                    boundary_union = geoms[0]
                    for g in geoms[1:]:
                        try:
                            boundary_union = boundary_union.union(g)
                        except:
                            pass

    if boundary_union is None:
        raise ValueError("Could not create boundary union")

    imbie_clipped = imbie_gdf.clip(boundary_union)
    buffer_data = process_coastline_data_for_buffers(
        change_data_gdf, min_threshold_km=BUFFER_STYLE_PARAMS["min_change_threshold_km"],
        sampling_interval_m=BUFFER_STYLE_PARAMS["sampling_interval_m"],
        grouping_size=BUFFER_STYLE_PARAMS["grouping_size"])

    if buffer_data["max_abs_change"] > 0:
        max_abs_change = buffer_data["max_abs_change"]
    else:
        max_abs_change = max(abs(change_data_gdf['change_m'].min()),
                            abs(change_data_gdf['change_m'].max()))

    norm = Normalize(vmin=-max_abs_change, vmax=max_abs_change)
    return {"imbie_clipped": imbie_clipped, "buffer_data": buffer_data,
            "normalization": norm, "max_abs_change_for_scaling": max_abs_change}


def load_hillshade_basemap(hillshade_path, target_crs="EPSG:3413"):
    """Load hillshade raster, mask out nodata values."""
    try:
        with rasterio.open(hillshade_path) as src:
            hillshade_array = src.read(1)
            hillshade_filtered = hillshade_array.astype(float)
            hillshade_filtered[(hillshade_array < 10) | (hillshade_array > 256)] = np.nan
            print(f"  Hillshade filtered: values 10-256 kept, others masked out")

            if src.crs.to_string() != target_crs:
                print(f"  Hillshade CRS: {src.crs}, reprojecting to {target_crs}...")

            return {'array': hillshade_filtered, 'transform': src.transform, 'crs': src.crs}
    except Exception as e:
        print(f"Warning: Could not load hillshade basemap: {e}")
        return None


def load_glacier_info_with_progress(csv_path, target_crs="EPSG:3413"):
    """Load important glacier names/locations for map labels."""
    try:
        df = pd.read_csv(csv_path)
        important = [
            "Upernavik Isstr\u00f8m", "Rink Gletscher", "Steenstrup Gletsjer", "Alison Glacier",
            "Harald Moltke Br\u00e6", "Jakobshavn Isbr\u00e6", "Kangiata Nunaata Sermia", "Helheim Gletsjer",
            "Kangerlussuaq Gletsjer", "Zachariae Isstr\u00f8m", "Academy Gletsjer", "Steensby Gletsjer",
            "Petermann Gletsjer", "Humboldt Gletsjer", "Eqalorutsit Killiit", "Midg\u00e5rd Gletsjer",
            "Storstr\u00f8mmen"
        ]
        name_cols = ['Official_n', 'GrnlndcNam', 'AltName']
        mask = pd.Series([False] * len(df))
        for c in name_cols:
            if c in df.columns:
                mask |= df[c].isin(important)
        sel = df[mask].copy()
        if len(sel) == 0 or 'POINT_X' not in sel.columns or 'POINT_Y' not in sel.columns:
            return None

        sel['display_name'] = sel.apply(
            lambda r: next((r[c] for c in name_cols
                           if c in sel.columns and pd.notna(r[c]) and r[c] in important),
                          r.get('Official_n', 'Unknown')), axis=1)

        gdf = gpd.GeoDataFrame(sel, geometry=gpd.points_from_xy(sel['POINT_X'], sel['POINT_Y']),
                               crs="EPSG:3413").to_crs(target_crs)
        return gdf
    except Exception:
        return None


# Plotting

def plot_ice_margin_map_with_buffers_and_labels(ax, map_data, style_params, buffer_params,
                                                common_defs, period_title_suffix,
                                                glacier_points=None, coastline_boundary=None,
                                                hillshade_data=None):
    """Draw the map: basins, hillshade, proportional circles, glacier dots."""
    ax.set_title(f"Greenland Ice Margin Change: {period_title_suffix}",
                 fontsize=style_params["title_fontsize"])
    imbie_clipped = map_data["imbie_clipped"]
    buffer_data = map_data["buffer_data"]
    norm = map_data["normalization"]
    cmap = common_defs["cmap"]
    region_colors = common_defs["region_colors"]

    if hillshade_data is not None:
        try:
            rioshow(hillshade_data['array'], transform=hillshade_data['transform'], ax=ax,
                    cmap=style_params["hillshade_cmap"], alpha=style_params["hillshade_alpha"], zorder=0)
        except Exception as e:
            print(f"Warning: Could not plot hillshade: {e}")

    # IMBIE basin fills, rasterised so the SVG doesn't balloon
    for _, region in imbie_clipped.iterrows():
        color = region_colors.get(region['SUBREGION1'], "#d1d2c6")
        if region.geometry.geom_type == 'Polygon':
            coll = ax.fill(*region.geometry.exterior.xy, color=color, edgecolor='white',
                           linewidth=1.0, alpha=style_params["imbie_region_alpha"])
            for c in coll:
                c.set_rasterized(style_params["rasterize_heavy"])
        else:
            for poly in region.geometry.geoms:
                coll = ax.fill(*poly.exterior.xy, color=color, edgecolor='white',
                               linewidth=1.0, alpha=style_params["imbie_region_alpha"])
                for c in coll:
                    c.set_rasterized(style_params["rasterize_heavy"])

    # proportional circles
    circles = []
    colors = []
    for p in buffer_data["buffer_points"]:
        r = abs(p['change_m']) * buffer_params["buffer_scale_factor"]
        circles.append(patches.Circle((p['geometry'].x, p['geometry'].y), r))
        colors.append(cmap(norm(p['change_m'])))
    if circles:
        pc = PatchCollection(circles, facecolor=colors, edgecolor=buffer_params["buffer_edge_color"],
                             linewidth=buffer_params["buffer_edge_width"],
                             alpha=buffer_params["buffer_alpha"], match_original=False)
        pc.set_rasterized(style_params["rasterize_heavy"])
        ax.add_collection(pc)

    if glacier_points is not None and len(glacier_points) > 0:
        ax.plot(glacier_points.geometry.x, glacier_points.geometry.y, 'o',
                markersize=style_params["glacier_marker_size"],
                color=style_params["glacier_marker_color"],
                markeredgecolor=style_params["glacier_marker_edgecolor"],
                markeredgewidth=style_params["glacier_marker_edgewidth"], zorder=10)

    ax.set_aspect('equal')
    ax.axis('off')


def prepare_barplot_specific_data_with_glacier_names(imbie_gdf, change_data_gdf, glacier_points=None):
    """Linearise the coastline and map each change segment to a position on it."""
    from shapely import force_2d
    from shapely.geometry import shape, mapping

    imbie_clean = imbie_gdf.copy()
    imbie_clean.geometry = imbie_clean.geometry.apply(lambda geom: shape(mapping(force_2d(geom))))

    boundary_geom = imbie_clean.geometry.iloc[0]
    for i in range(1, len(imbie_clean)):
        boundary_geom = boundary_geom.union(imbie_clean.geometry.iloc[i])

    boundary = boundary_geom.exterior
    coords = np.asarray(boundary.coords)
    north_idx = np.argmax(coords[:, 1])
    coords = np.roll(coords, -north_idx, axis=0)

    distances = np.zeros(len(coords))
    for i in range(1, len(coords)):
        distances[i] = distances[i - 1] + np.hypot(coords[i][0] - coords[i - 1][0],
                                                    coords[i][1] - coords[i - 1][1])
    total_length = distances[-1]

    glacier_positions = {}
    if glacier_points is not None and len(glacier_points) > 0:
        for _, g in glacier_points.iterrows():
            gx, gy = g.geometry.x, g.geometry.y
            name = g['display_name']
            dists = np.hypot(coords[:, 0] - gx, coords[:, 1] - gy)
            idx = np.argmin(dists)
            glacier_positions[name] = {'name': name, 'position': distances[idx],
                                       'normalized_position': distances[idx] / total_length,
                                       'coords': (gx, gy)}

    mapped = []
    for _, seg in change_data_gdf.iterrows():
        mid = seg.geometry.centroid
        dists = np.hypot(coords[:, 0] - mid.x, coords[:, 1] - mid.y)
        idx = np.argmin(dists)
        pt = Point(mid.x, mid.y)
        region_name = None
        for _, region in imbie_gdf.iterrows():
            if region.geometry.contains(pt):
                region_name = region['SUBREGION1']
                break
        if region_name:
            mapped.append({'coastline_position': distances[idx],
                           'normalized_position': distances[idx] / total_length * 100,
                           'change_m': seg['change_m'], 'region': region_name})

    mapped_df = pd.DataFrame(mapped).sort_values('normalized_position') if mapped else pd.DataFrame()
    return {"mapped_segments": mapped_df, "glacier_positions": glacier_positions,
            "total_coastline_length": total_length}


def plot_ice_margin_barplot_with_glacier_names(ax, barplot_data, style_params,
                                               common_defs, period_title_suffix):
    """Coastline bar-plot with glacier reference lines."""
    ax.set_title(f"Coastline Profile of Ice Margin Change: {period_title_suffix}",
                 fontsize=style_params["title_fontsize"])
    mapped_segments = barplot_data["mapped_segments"]
    glacier_positions = barplot_data["glacier_positions"]
    region_colors = common_defs["region_colors"]

    for y in style_params.get("extra_hlines", []):
        ax.axhline(y=y, zorder=0, **style_params.get("extra_hlines_style", {}))
    ax.axhline(y=style_params["hline_y"], color=style_params["hline_color"],
               linewidth=style_params["hline_linewidth"], alpha=style_params["hline_alpha"], zorder=0)

    if mapped_segments.empty:
        ax.text(0.5, 0.5, "No data for bar plot", ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        return

    for region_name in mapped_segments['region'].unique():
        rd = mapped_segments[mapped_segments['region'] == region_name]
        color = region_colors.get(region_name, "#d1d2c6")
        ax.bar(rd['normalized_position'], -rd['change_m'] / 1000.0,
               width=style_params["bar_width_normalized"], color=color,
               edgecolor=style_params["bar_edgecolor"], alpha=style_params["bar_alpha"],
               zorder=1, rasterized=True)

    for name, det in glacier_positions.items():
        x = det['normalized_position'] * 100.0
        ax.plot([x, x], [style_params["glacier_line_y_start"], style_params["glacier_line_y_end"]],
                color=style_params["glacier_line_color"], linewidth=style_params["glacier_line_width"],
                alpha=style_params["glacier_line_alpha"], linestyle='--', zorder=2)
        ax.text(x, style_params["glacier_label_y"], name,
                fontsize=style_params["glacier_text_fontsize"], ha='right', va='top',
                rotation=style_params["glacier_name_rotation"], rotation_mode='anchor')

    if style_params.get("use_fixed_y_limits", False):
        ax.set_ylim(style_params["y_axis_limits"])
    ax.set_xlabel('')
    ax.set_ylabel(style_params["ylabel"], fontsize=style_params["ylabel_fontsize"])
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', right=False, left=True, labelsize=12)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)


def create_custom_buffer_legend(ax, legend_params, cmap, norm):
    """Little legend of example circles in the corner of the map."""
    legend_x, legend_y = legend_params["legend_position"]
    circle_sizes_km = legend_params["legend_circle_sizes"]
    circle_sizes_m = np.array(circle_sizes_km) * 1000.0
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    legend_x_data = xlim[0] + legend_x * (xlim[1] - xlim[0])
    legend_y_data = ylim[0] + legend_y * (ylim[1] - ylim[0])
    y_spacing = (ylim[1] - ylim[0]) * 0.03

    for i, size_km in enumerate(circle_sizes_km):
        size_m = size_km * 1000.0
        r = abs(size_m) * legend_params["buffer_scale_factor"]
        cy = (legend_y_data + sum(abs(circle_sizes_m[:i + 1])) * legend_params["buffer_scale_factor"]
              + y_spacing * (i + 1))
        color = cmap(norm(size_m))
        circle = patches.Circle((legend_x_data, cy), r, facecolor=color, alpha=0.5,
                                edgecolor='black', linewidth=0.4)
        ax.add_patch(circle)
        ax.text(legend_x_data + (max(circle_sizes_m) * legend_params["buffer_scale_factor"] * 1.2),
                cy, f'{size_km} km', fontsize=11, va='center', ha='left')


def build_period_registry():
    """Load change results and earlier boundaries from disk."""
    registry = {}
    for period, filename in RESULT_FILES.items():
        result_fp = os.path.join(CHANGE_RESULTS_DIR, filename)
        boundary_fp = EARLIER_BOUNDARY_PATHS.get(period, None)

        if not os.path.exists(result_fp):
            print(f"[registry] Missing change result for {period}: {result_fp}")
            continue
        try:
            change_gdf = gpd.read_file(result_fp)
            if change_gdf.crs is None:
                print(f"[registry] Warning: no CRS in {result_fp}, assuming EPSG:32624.")
                change_gdf = change_gdf.set_crs("EPSG:32624", allow_override=True)
            change_gdf = change_gdf.to_crs(TARGET_CRS)
        except Exception as e:
            print(f"[registry] Could not read change result for {period}: {e}")
            continue

        if boundary_fp is None or not os.path.exists(boundary_fp):
            print(f"[registry] Missing earlier boundary for {period}: {boundary_fp}")
            continue
        try:
            boundary_gdf = gpd.read_file(boundary_fp)
            if boundary_gdf.crs is None:
                print(f"[registry] Warning: no CRS in boundary {boundary_fp}, assuming EPSG:32624.")
                boundary_gdf = boundary_gdf.set_crs("EPSG:32624", allow_override=True)
            boundary_gdf.geometry = boundary_gdf.geometry.simplify(100)
            boundary_gdf.geometry = boundary_gdf.geometry.buffer(0.01)
            boundary_gdf = boundary_gdf.to_crs(TARGET_CRS)
            if period == "2000_to_2022":
                try:
                    boundary_gdf = boundary_gdf.dissolve()
                except:
                    print(f"[registry] Warning: dissolve failed for {period}, keeping multiple features")
        except Exception as e:
            print(f"[registry] Could not read boundary for {period}: {e}")
            continue

        registry[period] = {"change_gdf": change_gdf, "earlier_boundary": boundary_gdf}
    return registry


# -------------------------------------------------------
if __name__ == "__main__":
    out_dir = os.path.join(BASE_DIR, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    color_csv_path = os.path.join(BASE_DIR, "outputs", "area_change_rates_with_uncertainty_and_colors_v3.csv")
    load_period_colors_from_csv(color_csv_path)

    imbie = gpd.read_file(IMBIE_BASE_PATH).to_crs(TARGET_CRS)

    print("Loading hillshade basemap...")
    hillshade_data = load_hillshade_basemap(HILLSHADE_PATH, TARGET_CRS)
    if hillshade_data:
        print("  Hillshade loaded successfully")

    glacier_points = load_glacier_info_with_progress(GLACIER_CSV_PATH, TARGET_CRS)

    registry = build_period_registry()
    if not registry:
        raise RuntimeError("No period registry built. Run calculate_retreat_with_flowlines.py first.")

    toggles = {
        "LIA_to_1980s": {"map": RUN_LIA_1980S_MAP, "bar": RUN_LIA_1980S_BAR},
        "1980s_to_2000": {"map": RUN_1980S_2000_MAP, "bar": RUN_1980S_2000_BAR},
        "2000_to_2022": {"map": RUN_2000_2022_MAP, "bar": RUN_2000_2022_BAR},
        "LIA_to_2022": {"map": RUN_LIA_2022_MAP, "bar": RUN_LIA_2022_BAR},
        "1980s_to_2022": {"map": RUN_1980S_2022_MAP, "bar": RUN_1980S_2022_BAR}
    }

    for key, run in toggles.items():
        if key not in registry:
            print(f"Skipping {key}: not in registry.")
            continue
        meta = PERIOD_META[key]
        title_suffix = meta["title_suffix"]
        prefix = meta["prefix"]
        change_gdf = registry[key]['change_gdf']
        earlier_boundary = registry[key]['earlier_boundary']
        if 'change_m' not in change_gdf.columns:
            print(f"Skipping {key}: 'change_m' column not found.")
            continue

        period_colors = get_period_colors(key)
        common_plot_definitions = {"region_colors": period_colors, "cmap": RETREAT_ADVANCE_CMAP}

        if run["map"]:
            print(f"Generating map for {key}...")
            map_data = prepare_map_specific_data_with_buffers(imbie, earlier_boundary, change_gdf)
            fig_map, ax_map = plt.subplots(figsize=MAP_STYLE_PARAMS["figsize"],
                                           facecolor=MAP_STYLE_PARAMS["facecolor"])
            coastline_geom = None
            try:
                if len(earlier_boundary) == 1:
                    coastline_geom = earlier_boundary.geometry.iloc[0]
                else:
                    coastline_geom = earlier_boundary.geometry.union_all()
            except:
                try:
                    from shapely.ops import unary_union
                    coastline_geom = unary_union(earlier_boundary.geometry.values)
                except:
                    pass

            plot_ice_margin_map_with_buffers_and_labels(
                ax_map, map_data, MAP_STYLE_PARAMS, BUFFER_STYLE_PARAMS,
                common_plot_definitions, title_suffix, glacier_points=glacier_points,
                coastline_boundary=coastline_geom, hillshade_data=hillshade_data)
            create_custom_buffer_legend(ax_map, BUFFER_STYLE_PARAMS, RETREAT_ADVANCE_CMAP,
                                        map_data["normalization"])
            plt.tight_layout()
            png_map = os.path.join(out_dir, f"map_buffers_{prefix}.png")
            svg_map = os.path.join(out_dir, f"map_buffers_{prefix}.svg")
            fig_map.savefig(png_map, dpi=MAP_STYLE_PARAMS["dpi"], bbox_inches='tight')
            fig_map.savefig(svg_map, dpi=MAP_STYLE_PARAMS["dpi"], bbox_inches='tight')
            plt.close(fig_map)
            print(f"Saved map: {png_map} and {svg_map}")

        if run["bar"]:
            print(f"Generating barplot for {key}...")
            bar_data = prepare_barplot_specific_data_with_glacier_names(imbie, change_gdf,
                                                                       glacier_points=glacier_points)
            fig_bar, ax_bar = plt.subplots(figsize=BAR_PLOT_STYLE_PARAMS["figsize"])
            plot_ice_margin_barplot_with_glacier_names(ax_bar, bar_data, BAR_PLOT_STYLE_PARAMS,
                                                      common_plot_definitions, title_suffix)
            plt.tight_layout()
            png_bar = os.path.join(out_dir, f"barplot_{prefix}.png")
            svg_bar = os.path.join(out_dir, f"barplot_{prefix}.svg")
            fig_bar.savefig(png_bar, dpi=BAR_PLOT_STYLE_PARAMS["dpi"], bbox_inches='tight')
            fig_bar.savefig(svg_bar, dpi=BAR_PLOT_STYLE_PARAMS["dpi"], bbox_inches='tight')
            plt.close(fig_bar)
            print(f"Saved barplot: {png_bar} and {svg_bar}")

    print("All visualizations completed!")
