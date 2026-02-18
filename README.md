# Greenland LIA Ice Margin Analysis — Code

Code for computing and visualizing ice-sheet area change and margin retreat since the Little Ice Age (LIA) across Greenland. Accompanies:

> Salmani et al., "Nearly half of Greenland's post-Little Ice Age area loss occurred since 2000"

## What each script does

| Script | What it does | Paper figure/table |
|---|---|---|
| `calculate_retreat_with_flowlines.py` | Measures retreat/advance between ice mask pairs. Uses nearest-distance along the coast, switches to along-flowline distance in fjords where straight-line underestimates retreat. | SM2, feeds Figs. 2–3 |
| `area_comparison_with_uncertainty.py` | Computes basin-level area change and propagates observation-year uncertainty into rates via Monte Carlo (10k trials). | SM4, Tables S1–S2 |
| `visualize_retreat.py` | Generates the Greenland-wide maps with proportional circles showing retreat magnitude per coastal segment. | Fig. 2 |
| `visualize_combined_barplot.py` | Linearises Greenland's coastline and plots per-segment ice margin change as stacked bar plots for four time periods. | Fig. 3 |
| `nw_four_bar_panels_only.py` | Same idea as the combined barplot but zoomed into Northwest Greenland at higher resolution (500 m samples, 5 km bins). | Fig. 4, Table S5 |

## Setup

```
pip install geopandas matplotlib numpy pandas shapely tqdm rasterio
```

## Data paths

Each script has a section near the top marked `# SET THESE PATHS` where you point it at your local data. The expected directory layout looks roughly like:

```
data/
  ice_margin_analysis/
    processed_masks/       # LIA, PROMICE 1980s, CCI 2000, PROMICE 2022 shapefiles
    outputs/               # change result shapefiles (produced by calculate_retreat_with_flowlines.py)
    flowlines/             # glacier flowline shapefile (optional)
    visualizations/        # output maps and plots land here
  IMBIE_basins/            # IMBIE basin shapefile
  glacier_names/           # glacier name CSV and/or shapefile
  basemap/                 # hillshade raster (optional, cosmetic)
  NorthWest_Ice_masks/     # NW-specific masks for Fig. 4
```

You don't need all of these for every script — `visualize_retreat.py` needs the most, `area_comparison_with_uncertainty.py` just needs masks + IMBIE basins.

## Run order

1. **`calculate_retreat_with_flowlines.py`** — produces the change shapefiles in `outputs/`
2. **`area_comparison_with_uncertainty.py`** — produces Tables S1/S2 CSVs and the colour-mapping CSV
3. **`visualize_retreat.py`** — reads outputs from (1) and (2) to make the maps
4. **`visualize_combined_barplot.py`** — reads outputs from (1) to make the coastline bar plots
5. **`nw_four_bar_panels_only.py`** — standalone, reads NW masks directly

Steps 3 and 4 can run in any order. Step 5 is independent.

## Citation

If you use this code in your work, please cite our paper:

> Salmani et al., "Nearly half of Greenland's post-Little Ice Age area loss occurred since 2000" — *citation to be updated upon publication*
