"""
area_comparison_with_uncertainty.py

Computes basin-resolved ice-sheet area change and propagates observation-year
uncertainty into change-rate estimates via Monte Carlo sampling (10,000 trials).

The problem is that several of our ice masks don't have a single precise date:
the LIA trimline spans roughly 1850-1900, CCI covers 1999-2004, and PROMICE 1980s
years vary by basin. So we draw start/end years from those ranges and compute
the rate for each trial, giving us a proper distribution instead of pretending
we know the exact dates.

Used in:
    Salmani et al., "Nearly half of Greenland's post-Little Ice Age area loss
    occurred since 2000"
    - Supplementary Methods SM4 (Monte Carlo uncertainty quantification)
    - Table S1 (basin-resolved rates with 90% CIs)
    - Table S2 (total area change by basin and period)
    - Generates hex colour map used by the visualization scripts

Inputs:  Processed ice masks + IMBIE basin polygons
Outputs: area_change_rates_with_uncertainty_and_colors_v3.csv,
         Table_S1_area_change_rates_v3.csv, Table_S2_total_area_change_v3.csv
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# SET THESE PATHS to where your data lives
BASE_DIR = "data/ice_margin_analysis"
PROCESSED_MASKS_DIR = os.path.join(BASE_DIR, "processed_masks")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMBIE_PATH = os.path.join(BASE_DIR, "IMBIE_basins", "IMBIE basins - cleaned_extended.shp")


# Colour palette -- diverging, should be readable for colour-blind folks

def get_colorblind_friendly_palettes():
    return {
        'custom_diverging': {
            'retreat_colors': ["#5e241d", "#9e3a2e", "#c44536", "#d98a6d"],
            'advance_colors': ["#203130", "#21585a", "#197278", "#6f9b98"],
            'neutral_color': '#f9f69f',
            'name': 'Custom Diverging'
        }
    }


def create_colormap(palette_key='custom_diverging'):
    palettes = get_colorblind_friendly_palettes()
    palette = palettes[palette_key]
    colors = palette['retreat_colors'] + [palette['neutral_color']] + palette['advance_colors'][::-1]
    return LinearSegmentedColormap.from_list(palette_key, colors), palette['name']


def build_cmap_and_norm(fixed_scale_max=80, palette_key='custom_diverging'):
    cmap, _ = create_colormap(palette_key)
    norm = mcolors.Normalize(vmin=-fixed_scale_max, vmax=fixed_scale_max)
    return cmap, norm


def value_to_hex(val, cmap, norm):
    """Turn a numeric rate value into a hex colour string."""
    if pd.isna(val):
        return ''
    v = max(min(val, norm.vmax), norm.vmin)
    rgba = cmap(norm(v))
    return mcolors.to_hex(rgba, keep_alpha=False)


def add_rate_color_columns(out_df, period_keys, value_col_suffix='rate_mean',
                           palette_key='custom_diverging', fixed_scale_max=80,
                           color_col_suffix='color_hex'):
    """Append a hex-colour column for each period based on the rate value."""
    cmap, norm = build_cmap_and_norm(fixed_scale_max=fixed_scale_max, palette_key=palette_key)
    for key in period_keys:
        value_col = f"{key}_{value_col_suffix}"
        color_col = f"{key}_{color_col_suffix}"
        if value_col not in out_df.columns:
            raise KeyError(f"Column not found: {value_col}")
        out_df[color_col] = out_df[value_col].apply(lambda v: value_to_hex(v, cmap, norm))
    return out_df


# Data loading

def load_datasets():
    """Load the four processed masks and IMBIE basins, all reprojected to UTM 24N."""
    print("Loading datasets...")

    lia_path = os.path.join(PROCESSED_MASKS_DIR, "LIA_processed.shp")
    promice_1980s_path = os.path.join(PROCESSED_MASKS_DIR, "PROMICE_1980s_processed_v2.shp")
    cci_2000_path = os.path.join(PROCESSED_MASKS_DIR, "CCI_2000_processed.shp")
    promice_2022_path = os.path.join(PROCESSED_MASKS_DIR, "PROMICE_2022_processed.shp")

    print("  Loading and simplifying masks...")
    simplify_tolerance = 100  # metres, good enough for area calcs

    from shapely.ops import unary_union

    def load_and_process_mask(path, name):
        gdf = gpd.read_file(path).to_crs("EPSG:32624")
        gdf.geometry = gdf.geometry.simplify(simplify_tolerance)
        gdf.geometry = gdf.geometry.buffer(0.01)  # tiny buffer to fix topology issues
        print(f"    {name} loaded and simplified")

        if len(gdf) > 1:
            try:
                dissolved_geom = unary_union(gdf.geometry.values)
                gdf = gpd.GeoDataFrame([{'geometry': dissolved_geom}], crs=gdf.crs)
                print(f"    {name} dissolved ({len(gdf)} feature)")
            except Exception as e:
                print(f"    Warning: Could not dissolve {name}, keeping as multiple features: {e}")
        return gdf

    lia = load_and_process_mask(lia_path, "LIA")
    promice_1980s = load_and_process_mask(promice_1980s_path, "PROMICE 1980s")
    cci_2000 = load_and_process_mask(cci_2000_path, "CCI 2000")
    promice_2022 = load_and_process_mask(promice_2022_path, "PROMICE 2022")

    print("  Loading IMBIE basins...")
    imbie = gpd.read_file(IMBIE_PATH).to_crs("EPSG:32624")
    imbie.geometry = imbie.geometry.buffer(0)
    imbie.geometry = imbie.geometry.simplify(simplify_tolerance)

    return lia, promice_1980s, cci_2000, promice_2022, imbie


# Area change -- just geometry, no time uncertainty yet

def compute_change_areas(reference_mask, contemporary_mask, imbie):
    """
    Clip both masks to each IMBIE basin, compute areas in km2, take the
    difference. Returns a DataFrame with per-basin rows plus a Total row.
    """
    rows = []
    for _, basin in imbie.iterrows():
        region_name = basin['SUBREGION1']
        basin_gdf = gpd.GeoDataFrame([basin], geometry='geometry', crs=imbie.crs)

        ref_clip = gpd.overlay(reference_mask, basin_gdf, how='intersection')
        con_clip = gpd.overlay(contemporary_mask, basin_gdf, how='intersection')

        ref_area = ref_clip.area.sum() / 1e6
        con_area = con_clip.area.sum() / 1e6
        dA = con_area - ref_area
        dP = (dA / ref_area * 100.0) if ref_area > 0 else 0.0

        rows.append({
            'Region': region_name,
            'Reference_Area_km2': ref_area,
            'Contemporary_Area_km2': con_area,
            'Change_Area_km2': dA,
            'Change_Percentage': dP
        })

    df = pd.DataFrame(rows)

    total_ref = df['Reference_Area_km2'].sum()
    total_con = df['Contemporary_Area_km2'].sum()
    total_dA = df['Change_Area_km2'].sum()
    total_dP = (total_dA / total_ref * 100.0) if total_ref > 0 else 0.0

    df_total = pd.DataFrame([{
        'Region': 'Total',
        'Reference_Area_km2': total_ref,
        'Contemporary_Area_km2': total_con,
        'Change_Area_km2': total_dA,
        'Change_Percentage': total_dP
    }])

    return pd.concat([df, df_total], ignore_index=True)


# Monte Carlo time uncertainty

def get_nominal_year(year_setting, region=None):
    """Resolve a year setting to a single number.
    Accepts an int, a (min, max) tuple, or a dict keyed by region."""
    if isinstance(year_setting, dict):
        val = year_setting.get(region, None)
        if val is None:
            raise ValueError(f"No year setting for region {region}")
        return get_nominal_year(val, region)
    elif isinstance(year_setting, (int, float)):
        return float(year_setting)
    elif isinstance(year_setting, tuple) and len(year_setting) == 2:
        return 0.5 * (float(year_setting[0]) + float(year_setting[1]))
    else:
        raise ValueError(f"Unsupported year setting: {year_setting}")


def sample_years(year_setting, region, size, rng):
    """Draw year samples. Fixed years give a constant array, (min,max) gives uniform draws."""
    if isinstance(year_setting, dict):
        val = year_setting.get(region, None)
        if val is None:
            raise ValueError(f"No year setting for region {region}")
        return sample_years(val, region, size, rng)
    elif isinstance(year_setting, (int, float)):
        return np.full(size, float(year_setting))
    elif isinstance(year_setting, tuple) and len(year_setting) == 2:
        lo, hi = float(year_setting[0]), float(year_setting[1])
        if hi <= lo:
            raise ValueError(f"Invalid (min,max) year range: {year_setting}")
        return rng.uniform(lo, hi, size=size)
    else:
        raise ValueError(f"Unsupported year setting: {year_setting}")


def compute_rate_stats(change_df, start_year_setting, end_year_setting,
                       n_samples=10000, seed=42):
    """
    For each basin draw start/end years, compute rate = dA / dt, report mean
    and 5th/95th percentiles. Total is built by summing basin rates per trial
    so inter-basin timing correlations are preserved.
    """
    rng = np.random.default_rng(seed)

    basin_rows = change_df[change_df['Region'] != 'Total'].copy()
    stat_rows = []
    per_trial_total_rates = np.zeros(n_samples)

    for _, row in basin_rows.iterrows():
        region = row['Region']
        dA = row['Change_Area_km2']

        start_samples = sample_years(start_year_setting, region, n_samples, rng)
        end_samples = sample_years(end_year_setting, region, n_samples, rng)

        dt = end_samples - start_samples
        if np.any(dt <= 0):
            raise ValueError(f"Non-positive durations in region {region}. Check your year ranges.")

        rates = dA / dt  # km2/yr

        start_nom = get_nominal_year(start_year_setting, region)
        end_nom = get_nominal_year(end_year_setting, region)
        dt_nom = end_nom - start_nom
        if dt_nom <= 0:
            raise ValueError(f"Non-positive nominal duration for region {region}.")
        rate_nominal = dA / dt_nom

        stat_rows.append({
            'Region': region,
            'rate_nominal': rate_nominal,
            'rate_mean': rates.mean(),
            'rate_sd': rates.std(ddof=1),
            'rate_p05': np.percentile(rates, 5),
            'rate_p95': np.percentile(rates, 95)
        })

        per_trial_total_rates += rates

    total_nominal = sum(r['rate_nominal'] for r in stat_rows)

    total_stats = {
        'Region': 'Total',
        'rate_nominal': total_nominal,
        'rate_mean': per_trial_total_rates.mean(),
        'rate_sd': per_trial_total_rates.std(ddof=1),
        'rate_p05': np.percentile(per_trial_total_rates, 5),
        'rate_p95': np.percentile(per_trial_total_rates, 95)
    }

    return pd.DataFrame(stat_rows + [total_stats])


# Table formatters

def format_table_s1(period_results, period_keys):
    """Table S1: rates with 90% CIs, one column per period."""
    first_period_data = period_results[period_keys[0]]['change_and_stats']
    all_regions = sorted([r for r in first_period_data['Region'].unique() if r != 'Total']) + ['Total']

    table_rows = []
    for region in all_regions:
        row = {'Region': region}
        for key in period_keys:
            period_data = period_results[key]['change_and_stats']
            region_stats = period_data[period_data['Region'] == region]
            if len(region_stats) == 0:
                row[key] = 'N/A'
            else:
                rmean = region_stats['rate_mean'].values[0]
                rp05 = region_stats['rate_p05'].values[0]
                rp95 = region_stats['rate_p95'].values[0]
                row[key] = f"{rmean:.1f} [{rp05:.1f}, {rp95:.1f}]"
        table_rows.append(row)

    return pd.DataFrame(table_rows)


def format_table_s2(change_dfs, period_keys):
    """Table S2: total area change (km2) rounded to integers."""
    all_regions = sorted([r for r in change_dfs[period_keys[0]]['Region'].unique() if r != 'Total']) + ['Total']

    table_rows = []
    for region in all_regions:
        row = {'Region': region}
        for key in period_keys:
            change_df = change_dfs[key]
            region_data = change_df[change_df['Region'] == region]
            if len(region_data) == 0:
                row[key] = 0
            else:
                row[key] = round(region_data['Change_Area_km2'].values[0])
        table_rows.append(row)

    return pd.DataFrame(table_rows)


# -------------------------------------------------------
if __name__ == "__main__":
    print("=" * 20)
    print("GREENLAND ICE SHEET AREA CHANGE + RATE UNCERTAINTY")
    print("=" * 20)

    lia, promice_1980s, cci_2000, promice_2022, imbie = load_datasets()

    # Observation year definitions
    # LIA trimline: most mapped features date to 1850-1900 (Kjaer et al. 2022)
    lia_year_range = (1850, 1900)

    # PROMICE 1980s: basin-specific acquisition years
    promice_years = {'SW': 1985, 'CW': 1985, 'NW': 1985,
                     'CE': 1985, 'SE': 1981, 'NO': 1978, 'NE': 1978}

    # CCI 2000 composite spans 1999-2004
    cci_year_range = (1999, 2004)

    # PROMICE 2022, single fixed year
    promice_2022_year = 2022

    time_periods = {
        'LIA_to_1980s': {
            'display_name': 'LIA to PROMICE 1980s',
            'reference_mask': lia,
            'contemporary_mask': promice_1980s,
            'start_year_setting': lia_year_range,
            'end_year_setting': promice_years
        },
        '1980s_to_2000': {
            'display_name': 'PROMICE 1980s to CCI 2000',
            'reference_mask': promice_1980s,
            'contemporary_mask': cci_2000,
            'start_year_setting': promice_years,
            'end_year_setting': cci_year_range
        },
        '2000_to_2022': {
            'display_name': 'CCI 2000 to PROMICE 2022',
            'reference_mask': cci_2000,
            'contemporary_mask': promice_2022,
            'start_year_setting': cci_year_range,
            'end_year_setting': promice_2022_year
        },
        'LIA_to_2022': {
            'display_name': 'LIA to PROMICE 2022',
            'reference_mask': lia,
            'contemporary_mask': promice_2022,
            'start_year_setting': lia_year_range,
            'end_year_setting': promice_2022_year
        },
        '1980s_to_2022': {
            'display_name': 'PROMICE 1980s to PROMICE 2022',
            'reference_mask': promice_1980s,
            'contemporary_mask': promice_2022,
            'start_year_setting': promice_years,
            'end_year_setting': promice_2022_year
        }
    }

    N_SAMPLES = 10000
    RNG_SEED = 42

    period_results = {}
    change_dfs = {}

    for key, info in time_periods.items():
        print(f"\nProcessing period: {info['display_name']}")

        change_df = compute_change_areas(
            reference_mask=info['reference_mask'],
            contemporary_mask=info['contemporary_mask'],
            imbie=imbie
        )
        change_dfs[key] = change_df

        stats_df = compute_rate_stats(
            change_df=change_df[['Region', 'Change_Area_km2']],
            start_year_setting=info['start_year_setting'],
            end_year_setting=info['end_year_setting'],
            n_samples=N_SAMPLES,
            seed=RNG_SEED
        )

        merged = change_df.merge(stats_df, on='Region', how='left')
        period_results[key] = {
            'display_name': info['display_name'],
            'change_and_stats': merged
        }

        total_row = merged[merged['Region'] == 'Total'].iloc[0]
        total_dA = total_row['Change_Area_km2']
        if total_dA >= 0:
            print(f"  Total ice gain: {total_dA:.0f} km2")
        else:
            print(f"  Total ice loss: {abs(total_dA):.0f} km2")
        print(f"  Total rate (nominal): {total_row['rate_nominal']:.1f} km2/yr")
        print(f"  Total rate (mean +/- sd): {total_row['rate_mean']:.1f} +/- {total_row['rate_sd']:.1f} km2/yr")
        print(f"  Total rate (5-95%): {total_row['rate_p05']:.1f} to {total_row['rate_p95']:.1f} km2/yr")

    # wide output table
    period_keys = list(period_results.keys())
    all_regions = sorted(list(imbie['SUBREGION1'].unique())) + ['Total']
    out_df = pd.DataFrame({'Region': all_regions}).set_index('Region')

    for key, data in period_results.items():
        df = data['change_and_stats'].set_index('Region')
        out_df[f"{key}_area_km2"] = df['Change_Area_km2']
        out_df[f"{key}_rate_nominal"] = df['rate_nominal']
        out_df[f"{key}_rate_mean"] = df['rate_mean']
        out_df[f"{key}_rate_sd"] = df['rate_sd']
        out_df[f"{key}_rate_p05"] = df['rate_p05']
        out_df[f"{key}_rate_p95"] = df['rate_p95']

    # symmetric colour scale, matched to the map colourbar
    COLOR_SCALE_MAX = 80  # km2/yr

    out_df = add_rate_color_columns(
        out_df.reset_index(),
        period_keys=period_keys,
        value_col_suffix='rate_mean',
        palette_key='custom_diverging',
        fixed_scale_max=COLOR_SCALE_MAX,
        color_col_suffix='color_hex'
    )

    table_s1 = format_table_s1(period_results, period_keys)
    table_s2 = format_table_s2(change_dfs, period_keys)

    out_csv = os.path.join(OUTPUT_DIR, "area_change_rates_with_uncertainty_and_colors_v3.csv")
    out_df.to_csv(out_csv, index=False)

    table_s1_csv = os.path.join(OUTPUT_DIR, "Table_S1_area_change_rates_v3.csv")
    table_s1.to_csv(table_s1_csv, index=False)

    table_s2_csv = os.path.join(OUTPUT_DIR, "Table_S2_total_area_change_v3.csv")
    table_s2.to_csv(table_s2_csv, index=False)

    print("\n" + "=" * 60)
    print(f"CSV written: {out_csv}")
    print(f"Table S1 written: {table_s1_csv}")
    print(f"Table S2 written: {table_s2_csv}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Table S1. Basin-resolved area-change rates with 90% confidence intervals")
    print("=" * 60)
    print(table_s1.to_string(index=False))

    print("\n" + "=" * 60)
    print("Table S2. Total area change by basin and period")
    print("=" * 60)
    print(table_s2.to_string(index=False))
