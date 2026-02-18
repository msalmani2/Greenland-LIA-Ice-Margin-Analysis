"""
calculate_retreat_with_flowlines.py

Computes ice margin retreat/advance between two ice mask epochs using a hybrid
distance algorithm. For most of the coastline the script measures the straight-
line distance from sample points on the older margin to the nearest point on the
newer margin. In narrow fjords, where that simple approach underestimates retreat,
manually digitized glacier flowlines are used instead so the distance follows the
curving geometry of the fjord.

Used in:
    Salmani et al., "Nearly half of Greenland's post-Little Ice Age area loss
    occurred since 2000"
    - Supplementary Methods SM2 (distance computation algorithm)
    - Feeds into Fig. 2 (circle-marker maps) and Fig. 3 (coastline bar plots)

Inputs:  Processed ice masks (LIA, PROMICE 1980s, CCI 2000, PROMICE 2022),
         glacier flowlines shapefile (optional but recommended for fjord accuracy)
Outputs: One shapefile per period with line segments carrying a signed change_m
         field (positive = retreat, negative = advance)
"""

import os
import sys
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon
from shapely.ops import nearest_points
from tqdm import tqdm


def calculate_ice_margin_change_gdf(older_gdf, newer_gdf, sample_distance=500,
                                    simplification_tolerance=100, keep_main_island=False,
                                    fix_invalid=True, line_buffer_if_line=20.0,
                                    flowlines_gdf=None, flowline_search_radius=5000.0,
                                    use_flowlines=True):
    """
    Compute signed retreat/advance between two ice-mask GeoDataFrames.

    Works in two passes:
      1) Standard nearest-distance for every sample point along the older margin.
      2) Where a flowline intersects both margins, replace the straight-line
         distance with the along-flowline distance (+/- 2 neighbouring segments too).

    Returns a GeoDataFrame of short line segments with 'change_m' (metres) and
    'category' (retreat / advance).
    """

    older = older_gdf.copy()
    newer = newer_gdf.copy()

    # reproject to UTM 24N so distances come out in metres
    older = older.to_crs("EPSG:32624")
    newer = newer.to_crs("EPSG:32624")

    if flowlines_gdf is not None:
        flowlines = flowlines_gdf.copy().to_crs("EPSG:32624")
        from shapely.strtree import STRtree
        flowline_tree = STRtree(flowlines.geometry.values)
        flowline_geoms = list(flowlines.geometry.values)
        print(f"  Loaded {len(flowlines)} flow lines for guided matching")
    else:
        flowlines = None
        flowline_tree = None
        flowline_geoms = None
        use_flowlines = False

    # buffer(0) fixes self-intersections and other geometry headaches
    if fix_invalid:
        try:
            older["geometry"] = older.geometry.buffer(0)
        except Exception:
            pass
        try:
            newer["geometry"] = newer.geometry.buffer(0)
        except Exception:
            pass

    if simplification_tolerance and simplification_tolerance > 0:
        older["geometry"] = older.geometry.simplify(simplification_tolerance, preserve_topology=True)
        newer["geometry"] = newer.geometry.simplify(simplification_tolerance, preserve_topology=True)

    # keep only the biggest polygon (main ice sheet)
    def _keep_main(gg):
        if gg.empty:
            return gg
        exploded = gg.explode(index_parts=True, ignore_index=True)
        exploded["__area"] = exploded.geometry.area
        main_geom = exploded.loc[exploded["__area"].idxmax(), "geometry"]
        return gpd.GeoDataFrame(geometry=[main_geom], crs=gg.crs)

    def _has_polygonal(gg):
        return any(g.geom_type in ("Polygon", "MultiPolygon") for g in gg.geometry if g is not None)

    def _has_lineal(gg):
        return any(g.geom_type in ("LineString", "MultiLineString") for g in gg.geometry if g is not None)

    ## older margin: extract boundary lines we can sample along
    older_is_polygonal = _has_polygonal(older)
    older_is_lineal = _has_lineal(older)

    if keep_main_island and older_is_polygonal:
        older = _keep_main(older)

    def polygon_boundaries(geom):
        if geom is None:
            return []
        if geom.geom_type == "Polygon":
            return [LineString(geom.exterior.coords)]
        elif geom.geom_type == "MultiPolygon":
            return [LineString(g.exterior.coords) for g in geom.geoms]
        return []

    def line_parts(geom):
        if geom is None:
            return []
        if geom.geom_type == "LineString":
            return [geom]
        elif geom.geom_type == "MultiLineString":
            return list(geom.geoms)
        return []

    older_boundaries = []
    for g in older.geometry:
        if g is None:
            continue
        if older_is_polygonal:
            older_boundaries.extend(polygon_boundaries(g))
        elif older_is_lineal:
            older_boundaries.extend(line_parts(g))

    if not older_boundaries:
        raise ValueError("Older dataset does not provide usable boundaries (polygons or lines).")

    ## newer margin: boundary lines + unified polygon for containment test
    newer_is_polygonal = _has_polygonal(newer)
    newer_is_lineal = _has_lineal(newer)

    if keep_main_island and newer_is_polygonal:
        newer = _keep_main(newer)

    if newer_is_polygonal:
        newer_boundaries = []
        for g in newer.geometry:
            newer_boundaries.extend(polygon_boundaries(g))
        # need a single polygon for the inside/outside check
        try:
            newer_unified = newer.geometry.union_all()
        except Exception:
            try:
                from shapely.ops import unary_union
                newer_unified = unary_union(list(newer.geometry.values))
            except Exception:
                print("    Warning: union failed, falling back to first geometry for containment check")
                newer_unified = newer.geometry.iloc[0] if len(newer) > 0 else None
                if newer_unified is None:
                    raise ValueError("No valid geometries in newer dataset")
    else:
        # newer is lines, so buffer them into thin polygons for containment
        newer_boundaries = []
        for g in newer.geometry:
            newer_boundaries.extend(line_parts(g))
        if not newer_boundaries:
            raise ValueError("Newer dataset has neither polygons nor lines.")
        newer_buffered = newer.copy()
        newer_buffered["geometry"] = newer_buffered.geometry.buffer(line_buffer_if_line)
        try:
            newer_unified = newer_buffered.geometry.union_all()
        except Exception:
            try:
                from shapely.ops import unary_union
                newer_unified = unary_union(list(newer_buffered.geometry.values))
            except Exception:
                print("    Warning: union failed, falling back to first geometry for containment check")
                newer_unified = newer_buffered.geometry.iloc[0] if len(newer_buffered) > 0 else None
                if newer_unified is None:
                    raise ValueError("No valid geometries in newer dataset")

    if not newer_boundaries:
        raise ValueError("Newer dataset does not provide usable boundaries (polygons or lines).")

    # PASS 1: sample along older boundary, get nearest distance to newer
    sampled_points = []
    segment_ids = []
    point_positions = []
    for segment_id, seg in enumerate(older_boundaries):
        d = 0.0
        pos = 0
        if seg.length == 0:
            continue
        while d < seg.length:
            sampled_points.append(seg.interpolate(d))
            segment_ids.append(segment_id)
            point_positions.append(pos)
            d += sample_distance
            pos += 1

    change_distances = []
    for pt in tqdm(sampled_points, desc="Computing distances (standard method)"):
        min_d = float("inf")
        for line in newer_boundaries:
            _, on_line = nearest_points(pt, line)
            dist = pt.distance(on_line)
            if dist < min_d:
                min_d = dist
        # if the point is inside the newer mask the ice advanced there
        signed_dist = -min_d if newer_unified.contains(pt) else min_d
        change_distances.append(signed_dist)

    points_gdf = gpd.GeoDataFrame({
        "geometry": sampled_points,
        "change_distance": change_distances,
        "segment_id": segment_ids,
        "position": point_positions,
    }, crs=older.crs)

    # stitch adjacent sample points into short line segments
    line_segments = []
    segment_distances = []
    segment_ids_for_lines = []
    point_indices_for_segments = []

    num_segments = max(segment_ids) + 1 if segment_ids else 0
    for seg_id in range(num_segments):
        sub = points_gdf[points_gdf.segment_id == seg_id].sort_values("position")
        if len(sub) < 2:
            continue
        for i in range(len(sub) - 1):
            p1 = sub.iloc[i].geometry
            p2 = sub.iloc[i + 1].geometry
            line_segments.append(LineString([p1, p2]))
            segment_distances.append((sub.iloc[i].change_distance + sub.iloc[i + 1].change_distance) / 2.0)
            segment_ids_for_lines.append(seg_id)
            point_indices_for_segments.append((sub.index[i], sub.index[i + 1]))

    line_gdf = gpd.GeoDataFrame({
        "geometry": line_segments,
        "change_m": segment_distances,
        "segment_id": segment_ids_for_lines,
    }, crs=older.crs)

    # PASS 2: where flowlines cross both margins, use along-flow distance instead
    if use_flowlines and flowlines is not None and len(flowlines) > 0:
        print(f"\n  Applying flow line-guided matching to overlapping segments...")

        newer_segments = []
        newer_segment_ids = []
        for seg_id, seg in enumerate(newer_boundaries):
            if seg.length == 0:
                continue
            d = 0.0
            prev_pt = None
            while d < seg.length:
                curr_pt = seg.interpolate(d)
                if prev_pt is not None:
                    newer_segments.append(LineString([prev_pt, curr_pt]))
                    newer_segment_ids.append(seg_id)
                prev_pt = curr_pt
                d += sample_distance
            if prev_pt is not None and prev_pt.distance(Point(seg.coords[-1])) > sample_distance * 0.5:
                newer_segments.append(LineString([prev_pt, Point(seg.coords[-1])]))
                newer_segment_ids.append(seg_id)

        overlap_buffer = flowline_search_radius
        updated_segment_indices = set()
        flowline_used_count = 0

        for flowline_idx, flowline in enumerate(tqdm(flowlines.geometry, desc="Processing flow lines")):
            if flowline is None or flowline.length == 0:
                continue

            flowline_buffer = flowline.buffer(overlap_buffer)

            # which older-boundary segments does this flowline overlap?
            overlapping_old_segment_indices = []
            old_intersection_points = {}

            for seg_idx, seg_geom in enumerate(line_segments):
                if seg_geom.intersects(flowline_buffer):
                    inter = seg_geom.intersection(flowline)
                    if inter.is_empty:
                        continue

                    if inter.geom_type == 'Point':
                        old_intersection_points[seg_idx] = inter
                        overlapping_old_segment_indices.append(seg_idx)
                    elif inter.geom_type == 'MultiPoint':
                        seg_center = seg_geom.centroid
                        closest_pt = min(inter.geoms, key=lambda p: seg_center.distance(p))
                        old_intersection_points[seg_idx] = closest_pt
                        overlapping_old_segment_indices.append(seg_idx)
                    elif inter.geom_type == 'LineString':
                        seg_center = seg_geom.centroid
                        closest_pt = inter.interpolate(inter.project(seg_center))
                        old_intersection_points[seg_idx] = closest_pt
                        overlapping_old_segment_indices.append(seg_idx)

            if not overlapping_old_segment_indices:
                continue

            # same deal for the newer boundary
            overlapping_new_segment_indices = []
            new_intersection_points = {}

            for seg_idx, seg_geom in enumerate(newer_segments):
                if seg_geom.intersects(flowline_buffer):
                    inter = seg_geom.intersection(flowline)
                    if inter.is_empty:
                        continue

                    if inter.geom_type == 'Point':
                        new_intersection_points[seg_idx] = inter
                        overlapping_new_segment_indices.append(seg_idx)
                    elif inter.geom_type == 'MultiPoint':
                        seg_center = seg_geom.centroid
                        closest_pt = min(inter.geoms, key=lambda p: seg_center.distance(p))
                        new_intersection_points[seg_idx] = closest_pt
                        overlapping_new_segment_indices.append(seg_idx)
                    elif inter.geom_type == 'LineString':
                        seg_center = seg_geom.centroid
                        closest_pt = inter.interpolate(inter.project(seg_center))
                        new_intersection_points[seg_idx] = closest_pt
                        overlapping_new_segment_indices.append(seg_idx)

            if not overlapping_new_segment_indices:
                continue

            # match each old segment to its best newer counterpart along the flowline
            for old_seg_idx in overlapping_old_segment_indices:
                if old_seg_idx in updated_segment_indices:
                    continue

                old_intersection_pt = old_intersection_points[old_seg_idx]
                old_pos_on_flowline = flowline.project(old_intersection_pt)

                best_new_seg_idx = None
                min_dist_along_flowline = float('inf')

                for new_seg_idx in overlapping_new_segment_indices:
                    new_intersection_pt = new_intersection_points[new_seg_idx]
                    new_pos_on_flowline = flowline.project(new_intersection_pt)

                    dist_along = new_pos_on_flowline - old_pos_on_flowline

                    # prefer forward (toward terminus) matches
                    if dist_along > 0:
                        if abs(dist_along) < min_dist_along_flowline:
                            min_dist_along_flowline = abs(dist_along)
                            best_new_seg_idx = new_seg_idx
                    elif best_new_seg_idx is None:
                        if abs(dist_along) < min_dist_along_flowline:
                            min_dist_along_flowline = abs(dist_along)
                            best_new_seg_idx = new_seg_idx

                if best_new_seg_idx is None:
                    continue

                new_intersection_pt = new_intersection_points[best_new_seg_idx]

                old_pos = flowline.project(old_intersection_pt)
                new_pos = flowline.project(new_intersection_pt)
                dist_along_flowline = abs(new_pos - old_pos)

                # boundary moved inland = retreat = positive
                if new_pos < old_pos:
                    signed_dist = dist_along_flowline
                else:
                    signed_dist = -dist_along_flowline

                segment_distances[old_seg_idx] = signed_dist
                updated_segment_indices.add(old_seg_idx)
                flowline_used_count += 1

                # also update 2 segments on either side of the match
                seg_id = segment_ids_for_lines[old_seg_idx]
                same_seg_id_indices = [i for i, sid in enumerate(segment_ids_for_lines) if sid == seg_id]
                seg_pos_in_list = same_seg_id_indices.index(old_seg_idx)

                def update_nearby_segment(nearby_old_idx, offset):
                    if nearby_old_idx in updated_segment_indices:
                        return False
                    if nearby_old_idx not in old_intersection_points:
                        return False

                    nearby_old_intersection_pt = old_intersection_points[nearby_old_idx]
                    nearby_old_pos = flowline.project(nearby_old_intersection_pt)

                    best_nearby_new_seg_idx = None
                    min_nearby_dist = float('inf')

                    for new_seg_idx in overlapping_new_segment_indices:
                        new_intersection_pt = new_intersection_points[new_seg_idx]
                        new_pos = flowline.project(new_intersection_pt)
                        dist_along = new_pos - nearby_old_pos

                        if dist_along > 0:
                            if abs(dist_along) < min_nearby_dist:
                                min_nearby_dist = abs(dist_along)
                                best_nearby_new_seg_idx = new_seg_idx
                        elif best_nearby_new_seg_idx is None:
                            if abs(dist_along) < min_nearby_dist:
                                min_nearby_dist = abs(dist_along)
                                best_nearby_new_seg_idx = new_seg_idx

                    if best_nearby_new_seg_idx is None:
                        return False

                    nearby_new_intersection_pt = new_intersection_points[best_nearby_new_seg_idx]
                    nearby_new_pos = flowline.project(nearby_new_intersection_pt)
                    nearby_dist_along_flowline = abs(nearby_new_pos - nearby_old_pos)

                    if nearby_new_pos < nearby_old_pos:
                        nearby_signed_dist = nearby_dist_along_flowline
                    else:
                        nearby_signed_dist = -nearby_dist_along_flowline

                    segment_distances[nearby_old_idx] = nearby_signed_dist
                    updated_segment_indices.add(nearby_old_idx)
                    return True

                for offset in range(1, 3):
                    if seg_pos_in_list - offset >= 0:
                        nearby_idx = same_seg_id_indices[seg_pos_in_list - offset]
                        if update_nearby_segment(nearby_idx, -offset):
                            flowline_used_count += 1

                for offset in range(1, 3):
                    if seg_pos_in_list + offset < len(same_seg_id_indices):
                        nearby_idx = same_seg_id_indices[seg_pos_in_list + offset]
                        if update_nearby_segment(nearby_idx, offset):
                            flowline_used_count += 1

        line_gdf["change_m"] = segment_distances

        print(f"  Flow line-guided matching updated {flowline_used_count} segments")

    line_gdf["category"] = np.where(line_gdf["change_m"] >= 0, "retreat", "advance")

    return line_gdf


# -------------------------------------------------------
if __name__ == "__main__":

    # SET THESE PATHS to where your data lives
    base_dir = "data/ice_margin_analysis"
    processed_dir = os.path.join(base_dir, "processed_masks")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Loading processed ice masks...")
    print("=" * 60)

    lia_path = os.path.join(processed_dir, "LIA_processed.shp")
    promice_1980s_path = os.path.join(processed_dir, "PROMICE_1980s_processed_v2.shp")
    cci_2000_path = os.path.join(processed_dir, "CCI_2000_processed.shp")
    promice_2022_path = os.path.join(processed_dir, "PROMICE_2022_processed.shp")

    missing_masks = [p for p in [lia_path, promice_1980s_path, cci_2000_path, promice_2022_path]
                     if not os.path.exists(p)]
    if missing_masks:
        print("\n" + "!" * 60)
        print("ERROR: Processed ice masks not found!")
        print("!" * 60)
        for p in missing_masks:
            print(f"  - {p}")
        print("\nRun preprocessing first (process_ice_masks.py).")
        sys.exit(1)

    print("\nLoading processed masks...")
    lia = gpd.read_file(lia_path)
    promice_1980s = gpd.read_file(promice_1980s_path)
    cci_2000 = gpd.read_file(cci_2000_path)
    promice_2022 = gpd.read_file(promice_2022_path)

    print(f"  LIA: {len(lia)} feature(s), area: {lia.geometry.area.sum()/1e6:.2f} km2")
    print(f"  PROMICE 1980s: {len(promice_1980s)} feature(s), area: {promice_1980s.geometry.area.sum()/1e6:.2f} km2")
    print(f"  CCI 2000: {len(cci_2000)} feature(s), area: {cci_2000.geometry.area.sum()/1e6:.2f} km2")
    print(f"  PROMICE 2022: {len(promice_2022)} feature(s), area: {promice_2022.geometry.area.sum()/1e6:.2f} km2")

    # flowlines (optional but really helps in fjords)
    flowlines_path = os.path.join(base_dir, "flowlines", "glacier_flowlines_for_margin_change.shp")

    if os.path.exists(flowlines_path):
        flowlines_gdf = gpd.read_file(flowlines_path)
        print(f"\nLoaded {len(flowlines_gdf)} flow lines from: {flowlines_path}")
    else:
        flowlines_gdf = None
        print("\nWarning: Flow lines not found. Using standard nearest distance method.")
        print(f"  Expected path: {flowlines_path}")

    sample_distance = 1000          # metres between sample points
    simplify_tol = 100              # light simplification, masks are already clean
    keep_main = False
    line_buffer_if_line = 20.0
    flowline_search_radius = 2000.0  # how far we look for a nearby flowline
    use_flowlines = flowlines_gdf is not None

    print("\n" + "=" * 60)
    print("Starting ice margin change calculations...")
    print("=" * 60)

    comparisons = [
        {
            "name": "LIA_to_1980s",
            "older": lia,
            "newer": promice_1980s,
            "older_name": "LIA",
            "newer_name": "PROMICE 1980s"
        },
        {
            "name": "1980s_to_2000",
            "older": promice_1980s,
            "newer": cci_2000,
            "older_name": "PROMICE 1980s",
            "newer_name": "CCI 2000"
        },
        {
            "name": "1980s_to_2022",
            "older": promice_1980s,
            "newer": promice_2022,
            "older_name": "PROMICE 1980s",
            "newer_name": "PROMICE 2022"
        }
    ]

    for comp in comparisons:
        print(f"\n{'=' * 60}")
        print(f"Processing: {comp['older_name']} -> {comp['newer_name']}")
        print(f"{'=' * 60}")

        result = calculate_ice_margin_change_gdf(
            comp["older"], comp["newer"],
            sample_distance=sample_distance,
            simplification_tolerance=simplify_tol,
            keep_main_island=keep_main,
            line_buffer_if_line=line_buffer_if_line,
            flowlines_gdf=flowlines_gdf,
            flowline_search_radius=flowline_search_radius,
            use_flowlines=use_flowlines
        )

        output_file = os.path.join(output_dir, f"{comp['name']}.shp")
        result.to_file(output_file)

        retreat = result[result['change_m'] > 0]
        advance = result[result['change_m'] < 0]
        print(f"  Saved: {output_file}")
        print(f"  Total segments: {len(result)}")
        print(f"  Retreat segments: {len(retreat)} (max: {retreat['change_m'].max()/1000:.2f} km)")
        print(f"  Advance segments: {len(advance)} (max: {abs(advance['change_m'].min())/1000:.2f} km)")
        print(f"  Mean change: {result['change_m'].mean()/1000:.2f} km")

    print("\n" + "=" * 60)
    print("All comparisons completed successfully!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
