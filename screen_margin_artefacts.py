"""
screen_margin_artefacts.py

Flags margin-change segments that are mapping artefacts rather than real margin change (SM2), so Figs. 2 and 3
can leave them out. Run after calculate_retreat_with_flowlines.py.
"""

import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString
from shapely.ops import substring

warnings.filterwarnings("ignore")

# SET THESE PATHS to where your data lives
DATA_DIR = "data"
INPUT_DIR = os.path.join(DATA_DIR, "Analysis_Inputs")  # Zenodo deposit
FLOWLINES_PATH = os.path.join(INPUT_DIR, "glacier_flowlines.gpkg")
IMBIE_PATH = os.path.join(INPUT_DIR, "Greenland_Basins_PS_v1.4.2_extended.gpkg")
BASE_DIR = os.path.join("outputs", "ice_margin_analysis")
OUTPUT_DIR = os.path.join(BASE_DIR, "artefact_screen")

CRS = "EPSG:32624"
SIMPLIFY = 100.0  # same as calculate_retreat_with_flowlines.py

PERIODS = [
    ("LIA_to_1980s", "LIA", "PROMICE_1980s"),
    ("1980s_to_2000s", "PROMICE_1980s", "CCI_2000s"),
    ("2000s_to_2022", "CCI_2000s", "PROMICE_2022"),
    ("1980s_to_2022", "PROMICE_1980s", "PROMICE_2022"),
    ("LIA_to_2022", "LIA", "PROMICE_2022"),
]
BASIN_ORDER = ["NO", "NE", "CE", "SE", "SW", "CW", "NW"]

MIN_TEST_KM = 1.0       # only rebuild the measurement path where |change| >= 1 km
RULE_KM = 2.0           # A3 and A4 only act where |change| >= 2 km
ADV_FRAC_MIN = 0.50     # A3: an advance path should run mostly over gained ice
PATH_FRAC_MIN = 0.25    # A4: a flowline reach should run at least 25 % over changed ice
ORPHAN_OVERLAP = 0.05   # A1: start body with < 5 % of its area inside the end mask


def load_mask(name):
    # same buffer(0) + simplify as the change step, so the body index matches segment_id
    g = gpd.read_file(os.path.join(BASE_DIR, "processed_masks", f"{name}_processed.shp")).to_crs(CRS)
    g["geometry"] = g.geometry.buffer(0)
    g["geometry"] = g.geometry.simplify(SIMPLIFY, preserve_topology=True)
    parts = []
    for geom in g.geometry:
        if geom is not None:
            parts += list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    return gpd.GeoSeries(parts, crs=CRS), g.geometry.union_all()


def frac_inside(lines, region, step=50.0):
    # share of each line inside region, every 50 m. skip the first and last 50 m, both ends sit on a margin
    out = np.ones(len(lines))
    for i, ln in enumerate(lines):
        L = ln.length
        if L <= 3 * step:
            continue
        d = np.arange(step, L - step + 1e-6, step)
        xy = shapely.get_coordinates(shapely.line_interpolate_point(ln, d))
        out[i] = shapely.contains_xy(region, xy[:, 0], xy[:, 1]).mean()
    return out


def screen_period(name, start, end, flowlines, basins):
    seg = gpd.read_file(os.path.join(BASE_DIR, "change", f"{name}.shp")).to_crs(CRS).reset_index(drop=True)
    sb, s_union = load_mask(start)
    eb, e_union = load_mask(end)
    shapely.prepare(e_union)

    # segment_id is the index of the start body the segment sits on
    seg["start_body"] = seg["segment_id"].astype(int)
    seg["start_area_km2"] = sb.area.values[seg.start_body] / 1e6

    # A1: how much of each start body the end mask still covers
    etree = shapely.STRtree(eb.values)
    overlap = {}
    for b in np.unique(seg.start_body):
        poly = sb.iloc[b]
        overlap[b] = sum(poly.intersection(eb.iloc[j]).area
                         for j in etree.query(poly, predicate="intersects")) / 1e6
    seg["overlap_frac"] = seg.start_body.map(overlap) / seg.start_area_km2
    seg["orphan_body"] = seg.overlap_frac < ORPHAN_OVERLAP

    # nearest end margin from both segment ends (the first pass of the change step).
    # rings are chopped into short pieces, the nearest search is much faster this way
    pieces = []
    for poly in eb:
        c = np.asarray(poly.exterior.coords)
        pieces += [LineString(c[i:i + 17]) for i in range(0, len(c) - 1, 16)]
    pieces = np.array(pieces, dtype=object)
    ptree = shapely.STRtree(pieces)
    res = {}
    for tag, k in (("a", 0), ("b", -1)):
        pts = shapely.points([ls.coords[k] for ls in seg.geometry])
        (qi, ti), dist = ptree.query_nearest(pts, return_distance=True, all_matches=False)
        order = np.argsort(qi)
        ti, dist = ti[order], dist[order]
        inside = shapely.contains(e_union, pts)
        res[tag] = dict(signed=np.where(inside, dist, -dist), inside=inside,
                        path=shapely.shortest_line(pts, pieces[ti]))
    nearest_m = (res["a"]["signed"] + res["b"]["signed"]) / 2.0
    # if the value isn't the nearest distance, the flowline step replaced it
    seg["flowline_derived"] = ~np.isclose(seg.change_m.values, nearest_m, atol=1.0)

    lost = s_union.difference(e_union)
    gain = e_union.difference(s_union)
    shapely.prepare(lost)
    shapely.prepare(gain)

    # share of the measurement path over lost ice (retreat) or gained ice (advance)
    ftree = shapely.STRtree(flowlines.geometry.values)
    absc = seg.change_m.abs().values / 1000.0
    path_frac = np.full(len(seg), np.nan)
    flowline_id = np.full(len(seg), -1)
    for i in np.where(absc >= MIN_TEST_KM)[0]:
        c = seg.change_m.iat[i]
        region = lost if c < 0 else gain
        if not seg.flowline_derived.iat[i]:
            paths = [res["a"]["path"][i], res["b"]["path"][i]]
            w = np.array([p.length for p in paths])
            path_frac[i] = (frac_inside(paths, region) * w).sum() / w.sum()
            continue
        # flowline value: rebuild the reach between the two margin crossings
        sgeom = seg.geometry.iat[i]
        ctr = sgeom.centroid
        best = None
        for j in ftree.query(sgeom, predicate="intersects"):
            fl = flowlines.geometry.iat[j]
            inter = sgeom.intersection(fl)
            if inter.is_empty:
                continue
            if inter.geom_type == "MultiPoint":
                inter = min(inter.geoms, key=ctr.distance)
            elif inter.geom_type != "Point":
                inter = inter.interpolate(inter.project(ctr))
            s0 = fl.project(inter)
            s1 = s0 + c
            if s1 < -1 or s1 > fl.length + 1:
                continue
            # keep the flowline whose other end lands closest to the end margin
            err = float(ptree.query_nearest(fl.interpolate(s1), return_distance=True)[1][0])
            if best is None or err < best[0]:
                best = (err, j, min(s0, s1), max(s0, s1))
        if best is not None:
            _, j, a, b = best
            path_frac[i] = frac_inside([substring(flowlines.geometry.iat[j], a, b)], region)[0]
            flowline_id[i] = j
    seg["path_frac"] = path_frac
    seg["flowline_id"] = flowline_id

    # flowline says advance but both ends are outside the end mask, or the other way round
    both_out = (~res["a"]["inside"]) & (~res["b"]["inside"])
    both_in = res["a"]["inside"] & res["b"]["inside"]
    seg["sign_conflict"] = seg.flowline_derived & (
        ((seg.change_m > 0) & both_out) | ((seg.change_m < 0) & both_in))

    mid = seg.geometry.interpolate(0.5, normalized=True)
    ll = mid.to_crs("EPSG:4326")
    seg["lat"] = ll.y.round(4)
    seg["lon"] = ll.x.round(4)
    bj = gpd.sjoin_nearest(gpd.GeoDataFrame(geometry=mid, crs=CRS).to_crs(basins.crs),
                           basins[["SUBREGION1", "geometry"]], how="left")
    seg["basin"] = bj[~bj.index.duplicated()]["SUBREGION1"].values  # ties give duplicate rows
    seg["period"] = name
    seg["seg_index"] = seg.index.values
    seg["length_km"] = seg.geometry.length / 1000.0
    seg["change_km"] = seg.change_m / 1000.0
    return seg


def apply_rules(seg):
    # why each segment is dropped, '' = kept
    reasons = []
    for c, pf, fl, orphan, conflict in zip(seg.change_km, seg.path_frac, seg.flowline_derived,
                                           seg.orphan_body, seg.sign_conflict):
        r = ["A1_orphan_body"] if orphan else []
        if abs(c) >= RULE_KM:
            # start mask leaves an inlet or nunatak open, the end mask covers it
            if c > 0 and not fl and np.isfinite(pf) and pf < ADV_FRAC_MIN:
                r.append("A3_advance_through_persistent_ice")
            if fl and (conflict or (np.isfinite(pf) and pf < PATH_FRAC_MIN)):
                r.append("A4_flowline_mismatch")
        reasons.append(";".join(r))
    return reasons


def summarise(seg):
    # length-weighted mean change
    wmean = lambda d: float(np.average(d.change_km, weights=d.length_km)) if len(d) else np.nan
    rows = []
    for name, *_ in PERIODS:
        s = seg[seg.period == name]
        for basin in BASIN_ORDER + ["ALL"]:
            d = s if basin == "ALL" else s[s.basin == basin]
            x, keep = d[d.excluded], d[~d.excluded]
            rows.append(dict(
                period=name, basin=basin, n_segments=len(d), margin_km=d.length_km.sum(),
                n_excluded=len(x), km_excluded=x.length_km.sum(),
                pct_km_excluded=100 * x.length_km.sum() / d.length_km.sum() if len(d) else np.nan,
                n_A1=int(x.reason.str.contains("A1").sum()),
                n_A3=int(x.reason.str.contains("A3").sum()),
                n_A4=int(x.reason.str.contains("A4").sum()),
                mean_change_km_all=wmean(d), mean_change_km_after=wmean(keep),
                max_retreat_km_all=d.change_km.min(), max_retreat_km_after=keep.change_km.min(),
                max_advance_km_all=d.change_km.max(), max_advance_km_after=keep.change_km.max()))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    flowlines = gpd.read_file(FLOWLINES_PATH).to_crs(CRS).reset_index(drop=True)
    basins = gpd.read_file(IMBIE_PATH)

    allseg = []
    for name, start, end in PERIODS:
        seg = screen_period(name, start, end, flowlines, basins)
        seg["reason"] = apply_rules(seg)
        seg["excluded"] = seg.reason != ""
        x = seg[seg.excluded]
        print(f"{name}: {len(seg)} segments, {seg.flowline_derived.sum()} from flowlines, "
              f"{len(x)} excluded ({x.length_km.sum():.1f} km of margin)")
        allseg.append(pd.DataFrame(seg.drop(columns="geometry")))
    seg = pd.concat(allseg, ignore_index=True)

    cols = ["period", "seg_index", "segment_id", "reason", "lat", "lon", "basin", "change_km",
            "flowline_derived", "path_frac", "length_km", "start_area_km2", "overlap_frac"]
    out = seg.loc[seg.excluded, cols].round({"change_km": 3, "path_frac": 4, "length_km": 3,
                                             "start_area_km2": 3, "overlap_frac": 4})
    out.to_csv(os.path.join(OUTPUT_DIR, "artefact_segments.csv"), index=False)
    # flowline_id is -1 where no flowline reach matched or |change| < 1 km
    fl = seg.loc[seg.flowline_derived, ["period", "seg_index", "flowline_id", "change_km"]]
    fl.round({"change_km": 3}).to_csv(os.path.join(OUTPUT_DIR, "flowline_segments.csv"), index=False)
    summary = summarise(seg)
    summary.round(3).to_csv(os.path.join(OUTPUT_DIR, "artefact_summary_by_basin.csv"), index=False)

    print(summary[summary.basin == "ALL"][["period", "n_segments", "n_excluded", "km_excluded",
                                           "n_A1", "n_A3", "n_A4"]].round(1).to_string(index=False))
    print("Saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
