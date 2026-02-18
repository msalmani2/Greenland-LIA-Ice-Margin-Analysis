"""
visualize_combined_barplot.py

Produces the combined bar-plot figure showing ice margin change along
Greenland's linearised perimeter for all four time periods stacked vertically
(Fig. 3 in the manuscript). The coastline is "unrolled" clockwise from the
northernmost point, each change segment is projected onto that linearised
perimeter, and the signed change (km) is plotted as coloured bars keyed to
IMBIE region. Dashed lines and rotated labels mark major outlet glaciers.

Used in:
    Salmani et al., "Nearly half of Greenland's post-Little Ice Age area loss
    occurred since 2000"
    - Fig. 3 (four-period coastline bar plots)

Inputs:  Change-result shapefiles, IMBIE basin shapefile, glacier name CSV
Outputs: combined_barplots_all_periods.png / .svg
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely import force_2d
from shapely.geometry import shape, mapping
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'

# SET THESE PATHS to where your data lives
BASE_DIR = "data/ice_margin_analysis"
IMBIE_BASE_PATH = os.path.join("data", "IMBIE_basins", "IMBIE basins - cleaned_extended.shp")
GLACIER_CSV_PATH = os.path.join("data", "glacier_names", "glacier_names_IDs.csv")
CHANGE_RESULTS_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "visualizations")

TARGET_CRS = "EPSG:3413"
COMPUTE_CRS = "EPSG:32624"

RESULT_FILES = {
    "LIA_to_1980s": "LIA_to_1980s.shp",
    "1980s_to_2000": "1980s_to_2000.shp",
    "2000_to_2022": "2000_to_2022.shp",
    "LIA_to_2022": "LIA_to_2022.shp",
}

PERIOD_META = {
    "LIA_to_1980s": {"title_suffix": "LIA to 1980s", "order": 0},
    "1980s_to_2000": {"title_suffix": "1980s to 2000", "order": 1},
    "2000_to_2022": {"title_suffix": "2000 to 2022", "order": 2},
    "LIA_to_2022": {"title_suffix": "LIA to 2022", "order": 3},
}

UNIFIED_REGION_COLORS = {
    'CE': "#55dde0", 'CW': "#326186", 'NE': "#1d2c35", 'NO': "#f8bd4f",
    'NW': "#e75a0d", 'SE': "#e75a0d", 'SW': "#5d2609"
}

COMBINED_PLOT_PARAMS = {
    "figsize": (18, 12), "bar_width_normalized": 0.4, "bar_alpha": 1.0,
    "bar_edgecolor": 'none',
    "hline_y": 0, "hline_color": 'black', "hline_linewidth": 0.8, "hline_alpha": 0.7,
    "extra_hlines": [10, -10, -20],
    "extra_hlines_style": {'color': '#888888', 'linestyle': '--', 'linewidth': 0.7, 'alpha': 0.9},
    "ylabel": 'Ice Margin Change (km)', "ylabel_fontsize": 18, "dpi": 300,
    "y_axis_limits": (-35, 20), "y_axis_limits_lia_to_2022": (-40, 10),
    "glacier_line_color": '#333333', "glacier_line_width": 0.6, "glacier_line_alpha": 0.4,
    "glacier_line_style": (0, (5, 10)),
    "glacier_text_fontsize": 13, "glacier_name_rotation": 45,
    "period_label_fontsize": 16, "period_label_weight": 'bold',
    "tick_labelsize": 14, "subplot_spacing": 0.1
}


def load_glacier_info(csv_path, target_crs="EPSG:3413"):
    """Grab the big-name glaciers from the CSV so we can label them."""
    try:
        df = pd.read_csv(csv_path)
        important_glaciers = [
            "Upernavik Isstr\u00f8m", "Rink Gletscher", "Steenstrup Gletsjer", "Alison Glacier",
            "Harald Moltke Br\u00e6", "Jakobshavn Isbr\u00e6", "Kangiata Nunaata Sermia", "Helheim Gletsjer",
            "Kangerlussuaq Gletsjer", "Zachariae Isstr\u00f8m", "Academy Gletsjer", "Steensby Gletsjer",
            "Petermann Gletsjer", "Humboldt Gletsjer", "Eqalorutsit Killiit", "Midg\u00e5rd Gletsjer",
            "Storstr\u00f8mmen"
        ]
        name_cols = ['Official_n', 'GrnlndcNam', 'AltName']
        mask = pd.Series([False] * len(df))
        for col in name_cols:
            if col in df.columns:
                mask |= df[col].isin(important_glaciers)
        selected = df[mask].copy()
        if len(selected) == 0 or 'POINT_X' not in selected.columns or 'POINT_Y' not in selected.columns:
            return None

        selected['display_name'] = selected.apply(
            lambda r: next((r[c] for c in name_cols
                           if c in selected.columns and pd.notna(r[c]) and r[c] in important_glaciers),
                          r.get('Official_n', 'Unknown')), axis=1)
        gdf = gpd.GeoDataFrame(selected, geometry=gpd.points_from_xy(selected['POINT_X'], selected['POINT_Y']),
                               crs="EPSG:3413").to_crs(target_crs)
        return gdf
    except Exception as e:
        print(f"Could not load glacier info: {e}")
        return None


def prepare_barplot_data(imbie_gdf, change_data_gdf, glacier_points=None):
    """Linearise the coastline clockwise from the north, project change segments onto it."""
    imbie_clean = imbie_gdf.copy()
    imbie_clean.geometry = imbie_clean.geometry.apply(lambda geom: shape(mapping(force_2d(geom))))

    # union basins into one polygon and grab exterior ring
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
        for _, glacier in glacier_points.iterrows():
            gx, gy = glacier.geometry.x, glacier.geometry.y
            name = glacier['display_name']
            dists_to_coords = np.hypot(coords[:, 0] - gx, coords[:, 1] - gy)
            closest_idx = np.argmin(dists_to_coords)
            glacier_positions[name] = {'name': name, 'position': distances[closest_idx],
                                       'normalized_position': distances[closest_idx] / total_length,
                                       'coords': (gx, gy)}

    mapped_segments = []
    for _, segment in change_data_gdf.iterrows():
        centroid = segment.geometry.centroid
        dists_to_coords = np.hypot(coords[:, 0] - centroid.x, coords[:, 1] - centroid.y)
        closest_idx = np.argmin(dists_to_coords)
        point = Point(centroid.x, centroid.y)
        region_name = None
        for _, region in imbie_gdf.iterrows():
            if region.geometry.contains(point):
                region_name = region['SUBREGION1']
                break
        if region_name:
            mapped_segments.append({'coastline_position': distances[closest_idx],
                                    'normalized_position': distances[closest_idx] / total_length * 100,
                                    'change_m': segment['change_m'], 'region': region_name})

    mapped_df = pd.DataFrame(mapped_segments).sort_values('normalized_position') if mapped_segments else pd.DataFrame()
    return {"mapped_segments": mapped_df, "glacier_positions": glacier_positions,
            "total_coastline_length": total_length}


def load_period_data():
    """Load change shapefiles for all periods and prepare barplot data."""
    imbie = gpd.read_file(IMBIE_BASE_PATH).to_crs(TARGET_CRS)
    glacier_points = load_glacier_info(GLACIER_CSV_PATH, TARGET_CRS)
    periods_data = {}

    for period_key, filename in RESULT_FILES.items():
        result_path = os.path.join(CHANGE_RESULTS_DIR, filename)
        if not os.path.exists(result_path):
            print(f"Missing file for {period_key}: {result_path}")
            continue
        try:
            change_gdf = gpd.read_file(result_path)
            if change_gdf.crs is None:
                change_gdf = change_gdf.set_crs(COMPUTE_CRS, allow_override=True)
            change_gdf = change_gdf.to_crs(TARGET_CRS)
            if 'change_m' not in change_gdf.columns:
                print(f"Missing 'change_m' column in {period_key}")
                continue
            print(f"Preparing barplot data for {period_key}...")
            barplot_data = prepare_barplot_data(imbie, change_gdf, glacier_points)
            periods_data[period_key] = {"barplot_data": barplot_data,
                                        "title_suffix": PERIOD_META[period_key]["title_suffix"],
                                        "order": PERIOD_META[period_key]["order"]}
        except Exception as e:
            print(f"Error loading {period_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return periods_data, glacier_points


def create_combined_barplot(periods_data, glacier_points, params):
    """Build the stacked subplot figure with all four periods."""
    sorted_periods = sorted(periods_data.items(), key=lambda x: PERIOD_META[x[0]]["order"])
    n_periods = len(sorted_periods)
    if n_periods == 0:
        raise ValueError("No valid period data available")

    fig, axes = plt.subplots(n_periods, 1, figsize=params["figsize"], sharex=True)
    if n_periods == 1:
        axes = [axes]

    first_period_data = list(periods_data.values())[0]
    glacier_positions = first_period_data["barplot_data"]["glacier_positions"]

    for idx, (period_key, period_info) in enumerate(sorted_periods):
        ax = axes[idx]
        mapped_segments = period_info["barplot_data"]["mapped_segments"]
        title_suffix = period_info["title_suffix"]

        for y in params.get("extra_hlines", []):
            ax.axhline(y=y, zorder=0, **params.get("extra_hlines_style", {}))
        ax.axhline(y=params["hline_y"], color=params["hline_color"],
                   linewidth=params["hline_linewidth"], alpha=params["hline_alpha"], zorder=1)

        if not mapped_segments.empty:
            for region_name in mapped_segments['region'].unique():
                region_data = mapped_segments[mapped_segments['region'] == region_name]
                color = UNIFIED_REGION_COLORS.get(region_name, "#d1d2c6")
                ax.bar(region_data['normalized_position'], -region_data['change_m'] / 1000.0,
                       width=params["bar_width_normalized"], color=color,
                       edgecolor=params["bar_edgecolor"], alpha=params["bar_alpha"],
                       zorder=2, rasterized=True)

        # LIA to 2022 needs wider y-limits since cumulative retreat is bigger
        if period_key == "LIA_to_2022":
            ax.set_ylim(params["y_axis_limits_lia_to_2022"])
        else:
            ax.set_ylim(params["y_axis_limits"])
        ax.tick_params(axis='y', which='both', right=False, left=True,
                       labelsize=params["tick_labelsize"])
        ax.text(1.02, 0.5, title_suffix, transform=ax.transAxes,
                fontsize=params["period_label_fontsize"], fontweight=params["period_label_weight"],
                va='center', ha='left', rotation=90)
        for spine_name in ['top', 'right', 'bottom']:
            ax.spines[spine_name].set_visible(False)
        if idx < n_periods - 1:
            ax.spines['bottom'].set_visible(False)

    middle_ax = axes[n_periods // 2]
    middle_ax.set_ylabel(params["ylabel"], fontsize=params["ylabel_fontsize"])

    # glacier lines go through all panels, names only at the bottom
    for name, details in glacier_positions.items():
        x_position = details['normalized_position'] * 100.0
        for ax in axes:
            ax.axvline(x=x_position, color=params["glacier_line_color"],
                       linewidth=params["glacier_line_width"], alpha=params["glacier_line_alpha"],
                       linestyle=params["glacier_line_style"], zorder=3)

    bottom_ax = axes[-1]
    glacier_label_y = params["y_axis_limits"][0] - 6
    for name, details in glacier_positions.items():
        x_position = details['normalized_position'] * 100.0
        bottom_ax.text(x_position, glacier_label_y, name, fontsize=params["glacier_text_fontsize"],
                       ha='right', va='top', rotation=params["glacier_name_rotation"],
                       rotation_mode='anchor')

    for ax in axes:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_xlabel('')

    plt.subplots_adjust(hspace=params["subplot_spacing"])
    plt.tight_layout()
    return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading period data...")
    periods_data, glacier_points = load_period_data()
    if not periods_data:
        print("No period data loaded. Exiting.")
        return
    print(f"Loaded {len(periods_data)} time periods")

    print("Creating combined barplot figure...")
    fig = create_combined_barplot(periods_data, glacier_points, COMBINED_PLOT_PARAMS)

    png_path = os.path.join(OUTPUT_DIR, "combined_barplots_all_periods.png")
    svg_path = os.path.join(OUTPUT_DIR, "combined_barplots_all_periods.svg")
    print("Saving figures...")
    fig.savefig(png_path, dpi=COMBINED_PLOT_PARAMS["dpi"], bbox_inches='tight')
    fig.savefig(svg_path, dpi=COMBINED_PLOT_PARAMS["dpi"], bbox_inches='tight')
    plt.close(fig)
    print(f"Combined barplot saved:")
    print(f"  PNG: {png_path}")
    print(f"  SVG: {svg_path}")
    print("Done!")


if __name__ == "__main__":
    main()
