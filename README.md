# Greenland LIA Ice Margin Analysis: code

Code for the area change, its uncertainty and the ice margin change around the Greenland Ice Sheet between the Little Ice Age (LIA) maximum and 2022. Accompanies:

> Salmani et al., "Nearly half of Greenland's post-Little Ice Age area loss occurred since 2000"

## Scripts

| Script | What it does | Paper |
|---|---|---|
| `area_comparison_with_uncertainty.py` | Mask areas per IMBIE basin, area change and its Monte Carlo uncertainty | SM4, Tables S1, S2, S6 |
| `calculate_retreat_with_flowlines.py` | Margin change between each pair of masks, along flowlines in fjords | SM2 |
| `screen_margin_artefacts.py` | Flags margin-change segments that are mapping artefacts | SM2 |
| `visualize_retreat.py` | Maps of margin change and basin area change rates | Fig. 2 |
| `visualize_combined_barplot.py` | Margin change along the whole LIA margin as bars | Fig. 3 |
| `nw_four_bar_panels_only.py` | Northwest Greenland margin change at higher resolution | Fig. 4 |

`fig2_layout.json` holds the panel and label positions of Fig. 2. Keep all the files in one folder, Fig. 3 and Fig. 4 import from `visualize_retreat.py` and `calculate_retreat_with_flowlines.py`.

## Setup

Python 3.12:

```
pip install geopandas==1.1.1 shapely==2.1.2 pyproj==3.7.2 pyogrio==0.11.1 numpy==2.2.6 pandas==2.2.2 scipy==1.16.2 matplotlib==3.8.4 rasterio==1.3.10 tqdm==4.66.4 cmcrameri==1.10
```

## Data

Each script has a `# SET THESE PATHS` section near the top. By default they read from `data/` and write everything under `outputs/`, so run them from this folder. Copy the `GRISHM`, `Ice_Masks` and `Analysis_Inputs` folders of our Zenodo deposit into `data/`, and add the public datasets below:

```
data/
  GRISHM/
    GRISHM_nunataks_filled.shp     # LIA maximum, nunataks filled
  Ice_Masks/                       # our modified masks
    Greenland_1980s_ice_mask_PROMICE_modified.shp
    Greenland_2000s_ice_mask_CCI_modified.shp
    Greenland_2022_ice_mask_PROMICE_modified.shp
  Analysis_Inputs/
    mc_uncertainty_features.gpkg   # features of the Monte Carlo analysis
    glacier_flowlines.gpkg         # the 211 hand-digitized flowlines
    Greenland_Basins_PS_v1.4.2_extended.gpkg   # IMBIE basins, 7 regions, extended seaward
  glacier_names/                   # GreenlandGlacierNames_GGNv01_WGS84.shp
  PROMICE_NW/                      # PROMICE_NW_1985.shp, GIMP_NW_1999_2002.shp (Fig. 4)
  basemap/                         # Greenland_Hillshade_ArcticDEM.tif (Fig. 2 background)
```

| Data | Source |
|---|---|
| GRISHM, modified ice masks and analysis inputs | https://doi.org/10.5281/zenodo.19007901 |
| IMBIE basins (our extended version is in Analysis_Inputs) | https://doi.org/10.7280/D1WT11 |
| Greenland glacier names (Bjørk et al. 2015) | https://doi.org/10.6084/m9.figshare.1449148.v2 |
| PROMICE_NW and GIMP_NW masks (Kjeldsen et al.) | https://doi.org/10.22008/FK2/RTSYDD |
| ArcticDEM mosaic (for the hillshade) | https://doi.org/10.7910/DVN/3VDC4W |

## Run order

1. `area_comparison_with_uncertainty.py`: basin areas, Tables S1, S2, S6 and the headline numbers (about 3 minutes)
2. `calculate_retreat_with_flowlines.py`: prepared masks and margin-change segments for five periods (about 12 minutes, `--period` redoes one period)
3. `screen_margin_artefacts.py`: the artefact list (about 1 minute)
4. `visualize_retreat.py`, `visualize_combined_barplot.py` and `nw_four_bar_panels_only.py`: Figs. 2, 3 and 4, each with its source data (2 to 3 minutes each)

Fig. 4 only needs the data, not steps 1 to 3.

## Citation

If you use this code, please cite our paper:

> Salmani et al., "Nearly half of Greenland's post-Little Ice Age area loss occurred since 2000" (citation to be updated upon publication)

## License

MIT, see `LICENSE`.
