"""
area_comparison_with_uncertainty.py

Area of the four ice masks in each IMBIE basin, the change between them, and the Monte Carlo
uncertainty of the changes and rates (SM4, Tables S1, S2, S6 and the headline numbers).
"""

import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely import area, intersection, make_valid, union_all
from shapely.geometry import Polygon

warnings.filterwarnings("ignore")

# SET THESE PATHS to where your data lives
DATA_DIR = "data"
LIA_DIR = os.path.join(DATA_DIR, "GRISHM")             # Zenodo deposit
MASK_DIR = os.path.join(DATA_DIR, "Ice_Masks")        # Zenodo deposit
INPUT_DIR = os.path.join(DATA_DIR, "Analysis_Inputs")  # Zenodo deposit
MASK_FILES = {
    "LIA": os.path.join(LIA_DIR, "GRISHM_nunataks_filled.shp"),
    "1980s": os.path.join(MASK_DIR, "Greenland_1980s_ice_mask_PROMICE_modified.shp"),
    "2000s": os.path.join(MASK_DIR, "Greenland_2000s_ice_mask_CCI_modified.shp"),
    "2022": os.path.join(MASK_DIR, "Greenland_2022_ice_mask_PROMICE_modified.shp"),
}
IMBIE_PATH = os.path.join(INPUT_DIR, "Greenland_Basins_PS_v1.4.2_extended.gpkg")
FEATURES_PATH = os.path.join(INPUT_DIR, "mc_uncertainty_features.gpkg")
OUTPUT_DIR = os.path.join("outputs", "area_change")

CRS = "EPSG:3413"
MASKS = ["LIA", "1980s", "2000s", "2022"]
BASIN_ORDER = ["NO", "NE", "CE", "SE", "SW", "CW", "NW"]
PERIODS = [("LIA", "1980s"), ("1980s", "2000s"), ("2000s", "2022"), ("LIA", "2022")]

N_TRIALS = 10000
SEED = 42

# mean chance that a feature of each class counts as a mask error (SM4), from the share of features we
# kept when reviewing that kind. 0.5 for the nunatak classes, the review there was too selective for a rate.
# RING/REV/FWD are nunataks and embayments closed in some masks and open in others, *_TIE split the masks 2 to 2
P_CENTRAL = {"MONO_LT4": 0.50, "MONO_LIA_LT50": 0.60, "NEST_UNDECIDED": 0.55,
             "ICECAP_UNDECIDED": 0.70, "RING_C": 0.50, "RING_TIE": 0.50,
             "REV_RING": 0.50, "REV_TIE": 0.50, "FWD_RING": 0.50, "FWD_TIE": 0.50}
TIE_CLASSES = ["RING_TIE", "REV_TIE", "FWD_TIE"]
CLASS_SD = 0.15

# observation dates, the first LIA window is the central case (the other two are for Table S6)
LIA_WINDOWS = [(1850, 1900), (1850, 1920), (1800, 1920)]
PROMICE_1980S_YEARS = {"SW": 1985, "CW": 1985, "NW": 1985, "CE": 1985, "SE": 1981, "NO": 1978, "NE": 1978}
CCI_YEARS = (1999, 2004)
PROMICE_2022_YEAR = 2022


def polygons(geom):
    # make_valid can give nested collections, so walk down to the polygons
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [p for g in getattr(geom, "geoms", []) for p in polygons(g)]


def load_filled_mask(path):
    # fill every part, then union, so a part sitting in another part's hole isn't counted twice.
    # has to happen before the basin clip, otherwise the basin boundary cuts rings open
    gdf = gpd.read_file(path).to_crs(CRS)
    filled = [Polygon(p.exterior) for g in gdf.geometry if g is not None and not g.is_empty
              for p in polygons(g if g.is_valid else make_valid(g))]
    return union_all(filled)


def basin_areas():
    basins = gpd.read_file(IMBIE_PATH).to_crs(CRS).dissolve(by="SUBREGION1").reset_index()
    geoms = [make_valid(g).buffer(0) for g in basins.geometry]
    out = {}
    for name in MASKS:
        mask = load_filled_mask(MASK_FILES[name])
        out[name] = {b: area(intersection(mask, g)) / 1e6 for b, g in zip(basins.SUBREGION1, geoms)}
        out[name]["All basins"] = area(mask) / 1e6
        print(f"  {name}: {out[name]['All basins']:,.1f} km2")
    return pd.DataFrame(out)[MASKS]


def beta_from(mean, sd):
    k = mean * (1 - mean) / sd ** 2 - 1
    return mean * k, (1 - mean) * k


def sample_years(setting, basin, size, rng):
    # dict = fixed per basin, tuple = uniform window, number = fixed
    if isinstance(setting, dict):
        return np.full(size, float(setting[basin]))
    if isinstance(setting, tuple):
        return rng.uniform(float(setting[0]), float(setting[1]), size=size)
    return np.full(size, float(setting))


def run_monte_carlo(features, areas, lia_years):
    rng = np.random.default_rng(SEED)
    basins = sorted(features.Basin.unique())
    col = features.Basin.map({b: i for i, b in enumerate(basins)}).values
    dA = {m: features[f"dA_{m}"].values for m in MASKS}
    is_tie = features.Class.isin(TIE_CLASSES).values
    classes = {c: (features.Class == c).values for c in features.Class.unique()}

    # one probability per class and trial, so errors inside a class are correlated
    err = {m: np.zeros((N_TRIALS, len(basins))) for m in MASKS}
    for t in range(N_TRIALS):
        drawn = np.zeros(len(features))
        for c, in_class in classes.items():
            p = rng.beta(*beta_from(P_CENTRAL[c], CLASS_SD))
            drawn[in_class] = rng.random(int(in_class.sum())) < p
        # 2-to-2 features: one of the two pairs is the wrong one, at random
        pair_a = rng.random(len(features)) < 0.5
        for m in MASKS:
            v = dA[m] * drawn
            v = np.where(is_tie & pair_a, np.maximum(v, 0), np.where(is_tie, np.minimum(v, 0), v))
            np.add.at(err[m][t], col, v)

    # signed change and rate, loss is negative
    years = {"LIA": lia_years, "1980s": PROMICE_1980S_YEARS, "2000s": CCI_YEARS, "2022": PROMICE_2022_YEAR}
    change, rate = {}, {}
    for start, end in PERIODS:
        key = f"{start}_to_{end}"
        change[key] = np.zeros((N_TRIALS, len(basins)))
        rate[key] = np.zeros((N_TRIALS, len(basins)))
        for j, b in enumerate(basins):
            d = (areas.at[b, end] + err[end][:, j]) - (areas.at[b, start] + err[start][:, j])
            t0 = sample_years(years[start], b, N_TRIALS, rng)
            t1 = sample_years(years[end], b, N_TRIALS, rng)
            change[key][:, j] = d
            rate[key][:, j] = d / (t1 - t0)
    return basins, change, rate


def interval(v):
    return v.mean(), np.percentile(v, 5), np.percentile(v, 95)


def make_table_s1(basins, change, rate):
    # totals are summed inside each trial, so their intervals aren't the sums of the basin intervals
    rows = []
    for key in change:
        for j, b in enumerate(basins + ["All basins"]):
            a = change[key].sum(axis=1) if b == "All basins" else change[key][:, j]
            r = rate[key].sum(axis=1) if b == "All basins" else rate[key][:, j]
            am, a05, a95 = interval(a)
            rm, r05, r95 = interval(r)
            rows.append(dict(period=key, basin=b, area_change_km2=am, area_p05=a05, area_p95=a95,
                             rate_km2_yr=rm, rate_p05=r05, rate_p95=r95))
    out = pd.DataFrame(rows)
    out["basin"] = pd.Categorical(out.basin, BASIN_ORDER + ["All basins"], ordered=True)
    out["period"] = pd.Categorical(out.period, list(change), ordered=True)
    return out.sort_values(["period", "basin"]).reset_index(drop=True)


def headline(change, rate):
    total = {k: v.sum(axis=1) for k, v in change.items()}
    total_rate = {k: v.sum(axis=1) for k, v in rate.items()}
    out = {f"area change {k}": v for k, v in total.items()}
    out.update({f"rate {k}": v for k, v in total_rate.items()})
    out["post-2000 share (%)"] = 100 * np.abs(total["2000s_to_2022"]) / np.abs(total["LIA_to_2022"])
    out["acceleration"] = total_rate["2000s_to_2022"] / total_rate["LIA_to_1980s"]
    # P(slower): 1980s to 2000s lost ice slower than LIA to 1980s
    p_slower = np.mean(-total_rate["1980s_to_2000s"] < -total_rate["LIA_to_1980s"])
    p_loss = np.mean(total["1980s_to_2000s"] < 0)
    return out, p_slower, p_loss


# -------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Hole-filled mask areas")
    areas = basin_areas().round(2).loc[BASIN_ORDER + ["All basins"]]
    areas.to_csv(os.path.join(OUTPUT_DIR, "basin_mask_areas.csv"), index_label="basin")

    # as mapped, later minus earlier (Table S2)
    s2 = pd.DataFrame({f"{a}_to_{b}": areas[b] - areas[a] for a, b in PERIODS}, index=areas.index)
    s2.round(1).to_csv(os.path.join(OUTPUT_DIR, "Table_S2_area_change_as_mapped.csv"), index_label="basin")
    total = s2.loc["All basins"]
    share_as_mapped = 100 * total["2000s_to_2022"] / total["LIA_to_2022"]
    print(s2.round(1).to_string())
    print(f"post-2000 share, as mapped: {share_as_mapped:.2f} %")

    features = pyogrio.read_dataframe(FEATURES_PATH, read_geometry=False)
    assert set(features.Basin) == set(BASIN_ORDER)  # a basin with no features would drop out of the totals
    print(f"\nMonte Carlo, {N_TRIALS:,} trials, {len(features):,} features")
    s6 = []
    for lia_years in LIA_WINDOWS:
        label = f"{lia_years[0]}-{lia_years[1]}"
        basins, change, rate = run_monte_carlo(features, areas, lia_years)
        head, p_slower, p_loss = headline(change, rate)
        if lia_years == LIA_WINDOWS[0]:
            central = (head, p_slower, p_loss)
            s1 = make_table_s1(basins, change, rate)
            s1.round(3).to_csv(os.path.join(OUTPUT_DIR, "Table_S1_area_change_monte_carlo.csv"), index=False)
        for k in ["rate LIA_to_1980s", "rate 1980s_to_2000s", "rate 2000s_to_2022", "rate LIA_to_2022",
                  "acceleration", "post-2000 share (%)"]:
            m, lo, hi = interval(head[k])
            s6.append(dict(quantity=k, lia_window=label, mean=m, p05=lo, p95=hi))
        s6.append(dict(quantity="P(slower)", lia_window=label, mean=p_slower, p05=np.nan, p95=np.nan))
        print(f"  LIA {label}: share {np.nanmean(head['post-2000 share (%)']):.1f} %, acceleration "
              f"{np.mean(head['acceleration']):.2f}, P(slower) {p_slower:.3f}, P(loss 1980s-2000s) {p_loss:.3f}")
    pd.DataFrame(s6).round(3).to_csv(os.path.join(OUTPUT_DIR, "Table_S6_lia_window_sensitivity.csv"), index=False)

    # headline numbers: as mapped next to the central MC run
    head, p_slower, p_loss = central
    rows = []
    for k, v in head.items():
        mapped = np.nan
        if k.startswith("area change "):
            mapped = total[k.replace("area change ", "")]
        elif k.startswith("post-2000"):
            mapped = share_as_mapped
        rows.append(dict(quantity=k, as_mapped=mapped, mc_mean=np.nanmean(v),
                         mc_p05=np.nanpercentile(v, 5), mc_p95=np.nanpercentile(v, 95)))
    rows.append(dict(quantity="P(slower)", mc_mean=p_slower))
    rows.append(dict(quantity="P(1980s to 2000s is a loss)", mc_mean=p_loss))
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUTPUT_DIR, "headline_numbers.csv"), index=False)

    print("\nTable S1, all basins:")
    print(s1[s1.basin == "All basins"].round(1).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
