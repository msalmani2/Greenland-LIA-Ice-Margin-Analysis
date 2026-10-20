"""
calculate_retreat_with_flowlines.py

Margin change between the ice masks (SM2). Prepares the four masks once (processed_masks/), then writes one
shapefile of short margin segments per period (change/), change_m < 0 = retreat.
Nearest distance along the coast, distance along the hand-digitized flowlines in fjords.
Use --period LIA_to_1980s to redo one period.
"""

import argparse
import os

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import nearest_points
from tqdm import tqdm

# SET THESE PATHS to where your data lives
DATA_DIR = "data"
LIA_DIR = os.path.join(DATA_DIR, "GRISHM")             # Zenodo deposit
MASK_DIR = os.path.join(DATA_DIR, "Ice_Masks")        # Zenodo deposit
INPUT_DIR = os.path.join(DATA_DIR, "Analysis_Inputs")  # Zenodo deposit
MASK_FILES = {
    "LIA": os.path.join(LIA_DIR, "GRISHM_nunataks_filled.shp"),
    "PROMICE_1980s": os.path.join(MASK_DIR, "Greenland_1980s_ice_mask_PROMICE_modified.shp"),
    "CCI_2000s": os.path.join(MASK_DIR, "Greenland_2000s_ice_mask_CCI_modified.shp"),
    "PROMICE_2022": os.path.join(MASK_DIR, "Greenland_2022_ice_mask_PROMICE_modified.shp"),
}
FLOWLINES_PATH = os.path.join(INPUT_DIR, "glacier_flowlines.gpkg")
BASE_DIR = os.path.join("outputs", "ice_margin_analysis")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_masks")
OUTPUT_DIR = os.path.join(BASE_DIR, "change")

TARGET_CRS = "EPSG:32624"
PREP_SIMPLIFY = 100.0   # m

# older mask, newer mask
PERIODS = {
    "LIA_to_1980s": ("LIA", "PROMICE_1980s"),
    "1980s_to_2000s": ("PROMICE_1980s", "CCI_2000s"),
    "2000s_to_2022": ("CCI_2000s", "PROMICE_2022"),
    "LIA_to_2022": ("LIA", "PROMICE_2022"),
    "1980s_to_2022": ("PROMICE_1980s", "PROMICE_2022"),
}
PARAMS = dict(
    sample_distance=1000,            # m between sample points
    simplification_tolerance=100,
    keep_main_island=False,
    line_buffer_if_line=20.0,
    flowline_search_radius=2000.0,   # how far we look for a flowline
    use_flowlines=True,
)


def remove_holes(g):
    if g is None:
        return None
    if g.geom_type == "Polygon":
        return Polygon(g.exterior)
    if g.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in g.geoms])
    return g


def prepare_mask(input_path, output_path, simplify_tolerance=PREP_SIMPLIFY):
    gdf = gpd.read_file(input_path)
    if gdf.crs != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)
    # one row per polygon, the deposit stores each mask as one row
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    # fill holes first so the simplification never touches nunatak rings
    gdf["geometry"] = gdf.geometry.apply(remove_holes)
    gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notna()]

    # dissolve, disconnected parts stay as one MultiPolygon
    parts = gdf.explode(index_parts=False, ignore_index=True)
    parts["geometry"] = parts.geometry.buffer(0)
    parts = parts[parts.geometry.is_valid]
    union = parts.geometry.union_all()
    if not union.is_valid:
        union = union.buffer(0)
    gdf = gpd.GeoDataFrame(geometry=[union], crs=gdf.crs)
    gdf["geometry"] = gdf.geometry.buffer(0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf.to_file(output_path)
    print(f"{os.path.basename(input_path)}: {gdf.geometry.area.sum() / 1e6:.2f} km2")


def is_polygonal(gdf):
    return any(g.geom_type in ("Polygon", "MultiPolygon") for g in gdf.geometry if g is not None)


def outer_rings(g):
    if g is None:
        return []
    if g.geom_type == "Polygon":
        return [LineString(g.exterior.coords)]
    if g.geom_type == "MultiPolygon":
        return [LineString(p.exterior.coords) for p in g.geoms]
    return []


def line_parts(g):
    if g is None:
        return []
    if g.geom_type == "LineString":
        return [g]
    if g.geom_type == "MultiLineString":
        return list(g.geoms)
    return []


def keep_largest(gdf):
    # main ice sheet only
    parts = gdf.explode(index_parts=True, ignore_index=True)
    return gpd.GeoDataFrame(geometry=[parts.geometry.loc[parts.geometry.area.idxmax()]], crs=gdf.crs)


def crossing_points(segs, flowline, flowline_buffer):
    # segments that cross the flowline -> crossing point nearest the segment centre
    pts = {}
    for i, seg in enumerate(segs):
        if not seg.intersects(flowline_buffer):
            continue
        inter = seg.intersection(flowline)
        if inter.is_empty:
            continue
        c = seg.centroid
        if inter.geom_type == "Point":
            pts[i] = inter
        elif inter.geom_type == "MultiPoint":
            pts[i] = min(inter.geoms, key=lambda p: c.distance(p))
        elif inter.geom_type == "LineString":
            pts[i] = inter.interpolate(inter.project(c))
    return pts


def best_match(flowline, old_pos, new_pts):
    # nearest newer crossing along the flowline, forward (toward the terminus) preferred
    best, min_d = None, float("inf")
    for i, p in new_pts.items():
        d = flowline.project(p) - old_pos
        if (d > 0 or best is None) and abs(d) < min_d:
            best, min_d = i, abs(d)
    return best


def along_flow_change(flowline, old_pt, new_pt):
    # margin moved up-glacier (smaller position on the flowline) = retreat = negative
    old_pos, new_pos = flowline.project(old_pt), flowline.project(new_pt)
    dist = abs(new_pos - old_pos)
    return -dist if new_pos < old_pos else dist


def calculate_ice_margin_change_gdf(older_gdf, newer_gdf, sample_distance=500,
                                    simplification_tolerance=100, keep_main_island=False,
                                    fix_invalid=True, line_buffer_if_line=20.0,
                                    flowlines_gdf=None, flowline_search_radius=5000.0,
                                    use_flowlines=True):
    # signed change (m) of the older margin on segments between sample points.
    # nw_four_bar_panels_only.py imports this too (older margin as lines there).
    older = older_gdf.copy().to_crs(TARGET_CRS)
    newer = newer_gdf.copy().to_crs(TARGET_CRS)
    # don't fall back to straight-line distance without noticing
    if use_flowlines and (flowlines_gdf is None or len(flowlines_gdf) == 0):
        raise ValueError("use_flowlines=True but no flowlines given")

    if fix_invalid:   # buffer(0) fixes self-intersections
        older["geometry"] = older.geometry.buffer(0)
        newer["geometry"] = newer.geometry.buffer(0)
    if simplification_tolerance and simplification_tolerance > 0:
        older["geometry"] = older.geometry.simplify(simplification_tolerance, preserve_topology=True)
        newer["geometry"] = newer.geometry.simplify(simplification_tolerance, preserve_topology=True)

    older_poly = is_polygonal(older)
    if keep_main_island and older_poly:
        older = keep_largest(older)
    older_lines = [l for g in older.geometry for l in (outer_rings(g) if older_poly else line_parts(g))]

    if is_polygonal(newer):
        if keep_main_island:
            newer = keep_largest(newer)
        newer_lines = [l for g in newer.geometry for l in outer_rings(g)]
        newer_all = newer.geometry.union_all()   # for the inside/outside test
    else:
        # newer is lines, buffer them into thin polygons for the inside test
        newer_lines = [l for g in newer.geometry for l in line_parts(g)]
        newer_all = newer.geometry.buffer(line_buffer_if_line).union_all()

    # pass 1: sample the older margin, nearest distance to the newer margin
    pts, ids = [], []
    for i, line in enumerate(older_lines):
        d = 0.0
        while d < line.length:
            pts.append(line.interpolate(d))
            ids.append(i)
            d += sample_distance

    dists = []
    for pt in tqdm(pts, desc="nearest distance"):
        min_d = min(pt.distance(nearest_points(pt, line)[1]) for line in newer_lines)
        # point inside the newer mask = the ice advanced over it
        dists.append(min_d if newer_all.contains(pt) else -min_d)

    # neighbouring points on the same line -> segment, change = mean of the two ends
    segs, change, seg_ids = [], [], []
    for k in range(len(pts) - 1):
        if ids[k] == ids[k + 1]:
            segs.append(LineString([pts[k], pts[k + 1]]))
            change.append((dists[k] + dists[k + 1]) / 2.0)
            seg_ids.append(ids[k])

    # pass 2: where a flowline crosses both margins, use the distance along the flowline instead
    if use_flowlines:
        flowlines = flowlines_gdf.to_crs(TARGET_CRS)
        newer_segs = []
        for line in newer_lines:
            d, prev = 0.0, None
            while d < line.length:
                pt = line.interpolate(d)
                if prev is not None:
                    newer_segs.append(LineString([prev, pt]))
                prev = pt
                d += sample_distance
            if prev is not None and prev.distance(Point(line.coords[-1])) > sample_distance * 0.5:
                newer_segs.append(LineString([prev, Point(line.coords[-1])]))

        updated = set()
        for flowline in tqdm(flowlines.geometry, desc="flowlines"):
            if flowline is None or flowline.length == 0:
                continue
            buf = flowline.buffer(flowline_search_radius)
            old_pts = crossing_points(segs, flowline, buf)
            if not old_pts:
                continue
            new_pts = crossing_points(newer_segs, flowline, buf)
            if not new_pts:
                continue

            def update(i):
                if i in updated or i not in old_pts:
                    return False
                best = best_match(flowline, flowline.project(old_pts[i]), new_pts)
                if best is None:
                    return False
                change[i] = along_flow_change(flowline, old_pts[i], new_pts[best])
                updated.add(i)
                return True

            for i in old_pts:
                # also the 2 segments on each side of the same line, each needs its own crossing
                if update(i):
                    for j in (i - 1, i - 2, i + 1, i + 2):
                        if 0 <= j < len(segs) and seg_ids[j] == seg_ids[i]:
                            update(j)
        print(f"  flowlines used for {len(updated)} segments")

    out = gpd.GeoDataFrame({"geometry": segs, "change_m": change, "segment_id": seg_ids}, crs=older.crs)
    out["category"] = np.where(out["change_m"] <= 0, "retreat", "advance")   # zero counts as retreat
    return out


def run_period(name, flowlines_gdf):
    older_name, newer_name = PERIODS[name]
    older = gpd.read_file(os.path.join(PROCESSED_DIR, f"{older_name}_processed.shp"))
    newer = gpd.read_file(os.path.join(PROCESSED_DIR, f"{newer_name}_processed.shp"))
    print(f"\n{older_name} -> {newer_name}")
    result = calculate_ice_margin_change_gdf(older, newer, flowlines_gdf=flowlines_gdf, **PARAMS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result.to_file(os.path.join(OUTPUT_DIR, f"{name}.shp"))
    print(f"  {len(result)} segments, {(result['change_m'] < 0).sum()} retreat, "
          f"mean change {result['change_m'].mean() / 1000:.2f} km")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", choices=list(PERIODS), action="append",
                        help="run only this period (repeatable), needs the prepared masks from a full run")
    args = parser.parse_args()

    if not args.period:
        for name, path in MASK_FILES.items():
            prepare_mask(path, os.path.join(PROCESSED_DIR, f"{name}_processed.shp"))

    flowlines = gpd.read_file(FLOWLINES_PATH)
    for name in args.period or PERIODS:
        run_period(name, flowlines)
    print(f"\ndone, results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
