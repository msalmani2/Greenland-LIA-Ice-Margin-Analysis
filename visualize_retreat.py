"""
visualize_retreat.py

Fig. 2: four maps of ice margin change (circles) and basin area change rates (fills).
Run after area_comparison_with_uncertainty.py, calculate_retreat_with_flowlines.py and screen_margin_artefacts.py.
"""

import json
import math
import os

import cmcrameri.cm as cmc
import geopandas as gpd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely
from matplotlib.collections import PatchCollection
from matplotlib.colors import FuncNorm, ListedColormap, to_hex, to_rgb
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.prepared import prep

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['mathtext.default'] = 'regular'
mpl.rcParams['axes.unicode_minus'] = True

# SET THESE PATHS to where your data lives
DATA_DIR = "data"
INPUT_DIR = os.path.join(DATA_DIR, "Analysis_Inputs")  # Zenodo deposit
IMBIE_PATH = os.path.join(INPUT_DIR, "Greenland_Basins_PS_v1.4.2_extended.gpkg")
HILLSHADE_PATH = os.path.join(DATA_DIR, "basemap", "Greenland_Hillshade_ArcticDEM.tif")
GLACIER_NAMES_PATH = os.path.join(DATA_DIR, "glacier_names", "GreenlandGlacierNames_GGNv01_WGS84.shp")
BASE_DIR = os.path.join("outputs", "ice_margin_analysis")
ARTEFACT_CSV = os.path.join(BASE_DIR, "artefact_screen", "artefact_segments.csv")
MC_TABLE = os.path.join("outputs", "area_change", "Table_S1_area_change_monte_carlo.csv")
LAYOUT_JSON = "fig2_layout.json"
OUTPUT_DIR = os.path.join("outputs", "figures")
STEM = "Fig2_maps"

PERIODS = ["LIA_to_1980s", "1980s_to_2000s", "2000s_to_2022", "LIA_to_2022"]
TITLE = {"LIA_to_1980s": "a  LIA to 1980s", "1980s_to_2000s": "b  1980s to 2000s",
         "2000s_to_2022": "c  2000s to 2022", "LIA_to_2022": "d  LIA to 2022"}
# basins of each map are clipped to the start-epoch mask
EARLIER = {"LIA_to_1980s": "LIA_processed.shp", "1980s_to_2000s": "PROMICE_1980s_processed.shp",
           "2000s_to_2022": "CCI_2000s_processed.shp", "LIA_to_2022": "LIA_processed.shp"}
TARGET = "EPSG:3413"
T_3413_LL = Transformer.from_crs(3413, 4326, always_xy=True)

# circles: segments with |change| >= 0.5 km, sampled every 2 km, groups of 10, radius 4x the change
BUF = dict(min_km=0.5, interval_m=2000, group=10, scale=4, alpha=0.9)

# circle colours: lajolla for retreat, oslo for advance, one range for all maps,
# 0 to 20 km takes 75 % of each half of the bar so the small changes stay readable
LAJ_LO, LAJ_HI = 0.30, 0.92
OSLO_LO, OSLO_HI = 0.35, 0.72
BREAK_KM = 20.0
INNER_FRAC = 0.75
MAJOR_TICKS_INNER = [-20, -10, 0, 10, 20]      # plus -R and +R
MINOR_TICKS = [-40, -30, -15, -5, 5, 15, 30, 40]

# basin fills: roma, same idea with 0 to 50 km2/yr taking 75 %
BASIN_CMAP = cmc.roma
BASIN_BREAK = 50.0
BASIN_INNER_FRAC = 0.75
BASIN_MAJOR_INNER = [-50, -20, 0, 20, 50]   # plus -R and +R
BASIN_MINOR = [-75, -40, -30, -10, 10, 30, 40, 75]

# figure size (mm) and fonts (pt, at 180 mm)
FIG_W = 180.0
MARGIN = dict(l=0.8, r=0.8, t=0.8, b=0.8)
PT = 25.4 / 72.0                                 # mm per pt
FS = dict(period=5.2, letter=5.6, basin_name=4.0, basin_rate=4.0, basin_unit=3.0, glacier=4.0,
          glacier_familiar=3.3, totals_title=4.4, totals_value=4.1, legend_title=4.2, legend=4.0, legend_small=3.6)
GLACIER_LINESPACING = 1.05
FAMILIAR_GAP_MM = 0.12
UNIT = r"km$^{2}$ yr$^{-1}$"
TICK_LEN, BREAK_TICK_LEN = 1.3, 2.6        # pt

# official GGN name, familiar name in brackets where it differs (GGN misspells Academy, has none for Alison)
GLACIER_TEXT = {
    "GGN0235": "Sermeq Kujalleq\n(Jakobshavn Isbræ)",
    "GGN0452": "Sermeq\n(Upernavik Isstrøm)",
    "GGN0670": "Zachariae\nIsstrøm",
    "GGN0163": "Helheim Gletsjer",
    "GGN0679": "Petermann Gletsjer",
    "GGN0567": "Rink Gletsjer",
    "GGN0693": "Steensby Gletsjer",
    "GGN0599": "Storstrømmen",
    "GGN0218": "Kangerlussuaq\nGletsjer",
    "GGN0094": "Kangiata Nunaata\nSermia",
    "GGN0033": "Eqalorutsit Killiit Sermiat",
    "GGN0671": "Sermersuaq\n(Humboldt Gletsjer)",
    "GGN0542": "Sermersuaq\n(Steenstrup Gletsjer)",
    "GGN0581": "Ullip Sermia\n(Harald Moltke Bræ)",
    "GGN0168": "Nigertiip Apusiia\n(Midgård Gletsjer)",
    "GGN0698": "Academy Gletsjer",
    "GGN0531": "Nunatakassaap Sermia\n(Alison Gletsjer)",
}
MARKER = dict(ms=2.0, mew=0.25)                  # pt
LABEL_PAD = 0.12                                 # mm around a label in the overlap tests
LABEL_SEP = 0.45                                 # extra mm between two glacier labels
LARGE_KM, MID_KM, SMALL_KM = 10.0, 5.0, 2.0      # circle classes for the label tests
LARGE_CLEAR = 0.2                                # mm kept between a glacier label and a circle >= 10 km
BASIN_LABEL_EXTRA = {(3, "SE"): (-1.0, 0.0)}     # manual nudge (dx, dy) in mm, keyed (map, basin)


def sgn(v, dec):
    # true minus sign, plus for gains, no sign on a rounded zero
    s = f"{abs(v):,.{dec}f}"
    if float(s.replace(',', '')) == 0:
        return s
    return ("−" if v < 0 else "+" if v > 0 else "") + s


class TwoSlope:
    # |change| (km) onto the circle ramp: 0..brk -> 0..frac, brk..R -> frac..1
    def __init__(self, R, brk=BREAK_KM, frac=INNER_FRAC):
        self.R, self.brk, self.frac = float(R), float(brk), float(frac)
        self.norm = FuncNorm((self.fwd, self.inv), vmin=-self.R, vmax=self.R)
        u = (np.arange(2048) + 0.5) / 2048       # position on the bar, 0 = -R, 1 = +R
        self.cmap = ListedColormap(self.direct_rgba(self.inv(2 * u - 1))[:, :3], name="margin_change_twoslope")

    def t(self, v):
        a = np.abs(np.asarray(v, float))
        return np.clip(np.where(a <= self.brk, self.frac * a / self.brk,
                                self.frac + (1 - self.frac) * (a - self.brk) / (self.R - self.brk)), 0, 1)

    def tinv(self, t):
        t = np.clip(np.abs(np.asarray(t, float)), 0, 1)
        return np.where(t <= self.frac, t * self.brk / self.frac,
                        self.brk + (t - self.frac) * (self.R - self.brk) / (1 - self.frac))

    def fwd(self, v):
        v = np.asarray(v, float)
        return np.sign(v) * self.t(v)

    def inv(self, u):
        u = np.asarray(u, float)
        return np.sign(u) * self.tinv(u)

    def direct_rgba(self, v):
        v = np.atleast_1d(np.asarray(v, float))
        t = self.t(v)
        out = np.empty(v.shape + (4,))
        neg = v <= 0
        out[neg] = cmc.lajolla(LAJ_HI - (LAJ_HI - LAJ_LO) * t[neg])
        out[~neg] = cmc.oslo(OSLO_HI - (OSLO_HI - OSLO_LO) * t[~neg])
        return out

    def rgba(self, v_km):
        # circle colour = colour bar colour
        return self.cmap(self.norm(np.atleast_1d(np.asarray(v_km, float))))


class BasinScale:
    # the bar is the plain colour map, only the tick positions show the stretch
    def __init__(self, cmap, R, brk=BASIN_BREAK, frac=BASIN_INNER_FRAC):
        self.cmap, self.R, self.brk, self.frac = cmap, float(R), float(brk), float(frac)

    def t(self, a):
        a = np.clip(np.abs(np.asarray(a, float)), 0, self.R)
        return np.where(a <= self.brk, self.frac * a / self.brk,
                        self.frac + (1 - self.frac) * (a - self.brk) / (self.R - self.brk))

    def fwd(self, v):
        v = np.asarray(v, float)
        return 0.5 + 0.5 * np.sign(v) * self.t(v)

    def colour(self, v):
        return self.cmap(self.fwd(v))


def make_circles(g):
    # sample every 2 km, groups of ten along the margin, each group drawn with its largest change
    sig = g[np.abs(g['change_m']) >= BUF["min_km"] * 1000.0]
    rows = []
    for idx, seg in sig.iterrows():
        geom, val = seg.geometry, seg['change_m']
        L = geom.length
        if L > BUF["interval_m"]:
            n = int(L / BUF["interval_m"])
            for i in range(n + 1):
                p = geom.interpolate(min(i * BUF["interval_m"], L))
                rows.append((idx, p.x, p.y, val))
        else:
            p = geom.centroid
            rows.append((idx, p.x, p.y, val))
    s = pd.DataFrame(rows, columns=["pos", "x", "y", "change_m"])
    s["gid"] = np.arange(len(s)) // BUF["group"]
    j = s.assign(a=s.change_m.abs()).groupby("gid", sort=True)["a"].idxmax()
    c = s.loc[j.values, ["gid", "pos", "x", "y", "change_m"]].reset_index(drop=True)
    c["n_members"] = s.groupby("gid").size().reindex(c.gid).values
    return c


def load_basins(period):
    bnd = gpd.read_file(os.path.join(BASE_DIR, "processed_masks", EARLIER[period]))
    bnd.geometry = bnd.geometry.simplify(100).buffer(0.01)
    bnd = bnd.to_crs(TARGET)
    if period == "2000s_to_2022":
        bnd = bnd.dissolve()
    imbie = gpd.read_file(IMBIE_PATH).to_crs(TARGET)
    return imbie.clip(bnd.geometry.simplify(100).buffer(0.01).union_all())


def load_hillshade(bounds, decimate):
    l, r, b, t = bounds
    with rasterio.open(HILLSHADE_PATH) as src:
        w = int(round((r - l) / src.res[0] / decimate))
        h = int(round((t - b) / src.res[1] / decimate))
        arr = src.read(1, window=from_bounds(l, b, r, t, src.transform), out_shape=(h, w),
                       resampling=Resampling.average, boundless=True, fill_value=0).astype(float)
    arr[(arr < 10) | (arr > 256)] = np.nan
    return arr, (l, r, b, t)


_MEASURE = {}


def measure(s, **kw):
    # text extent around its anchor in mm, y down: (x0, x1, y_top, y_bottom)
    key = (s, tuple(sorted(kw.items())))
    if key not in _MEASURE:
        if "fig" not in _MEASURE:
            fig = plt.figure(figsize=(6, 6), dpi=720)
            _MEASURE["fig"] = (fig, fig.canvas.get_renderer())
        fig, ren = _MEASURE["fig"]
        t = fig.text(0.5, 0.5, s, **kw)
        bb = t.get_window_extent(ren)
        ax_, ay_ = fig.transFigure.transform((0.5, 0.5))
        t.remove()
        k = 25.4 / fig.dpi
        _MEASURE[key] = ((bb.x0 - ax_) * k, (bb.x1 - ax_) * k, -(bb.y1 - ay_) * k, -(bb.y0 - ay_) * k)
    return _MEASURE[key]


def glacier_block(text, ha, va):
    # glacier label line by line; a line in brackets (familiar name) is smaller and hangs just under the one above
    lines = text.split("\n")
    fss = [FS["glacier_familiar"] if s.startswith("(") else FS["glacier"] for s in lines]
    ext = [measure(s, fontsize=f, style='italic', ha='left', va='baseline') for s, f in zip(lines, fss)]
    base = [0.0]
    for i in range(1, len(lines)):
        if lines[i].startswith("("):
            base.append(base[-1] + ext[i - 1][3] + FAMILIAR_GAP_MM - ext[i][2])
        else:
            kw = dict(fontsize=fss[i - 1], style='italic', ha='left', va='top', linespacing=GLACIER_LINESPACING)
            base.append(base[-1] + measure("Ag\nAg", **kw)[3] - measure("Ag", **kw)[3])
    top = base[0] + ext[0][2]
    bot = base[-1] + ext[-1][3]
    H = bot - top
    W = max(e[1] - e[0] for e in ext)
    shift = {"top": -top, "bottom": -bot, "center": -(top + bot) / 2, "baseline": 0.0}[va]
    x0, x1 = {"left": (0.0, W), "right": (-W, 0.0), "center": (-W / 2, W / 2)}[ha]
    return [(s, f, b + shift) for s, f, b in zip(lines, fss, base)], (x0, x1, top + shift, top + shift + H)


def rate_parts(val, cx, cy, lines):
    # basin rate (bold) and unit (small), on one line or two
    cap = FS["basin_rate"] * PT * 0.716
    if lines == 1:
        ev = measure(val, fontsize=FS["basin_rate"], fontweight='bold', ha='left', va='baseline')
        eu = measure(UNIT, fontsize=FS["basin_unit"], ha='left', va='baseline')
        wv, wu, gap = ev[1] - ev[0], eu[1] - eu[0], 0.35
        x_v = cx - (wv + gap + wu) / 2
        yb = cy + cap / 2
        parts = [(val, x_v, yb, 'left', FS["basin_rate"], 'bold', ev),
                 (UNIT, x_v + wv + gap, yb, 'left', FS["basin_unit"], 'normal', eu)]
    else:
        ev = measure(val, fontsize=FS["basin_rate"], fontweight='bold', ha='center', va='baseline')
        eu = measure(UNIT, fontsize=FS["basin_unit"], ha='center', va='baseline')
        dl = FS["basin_rate"] * PT * 1.1
        yb1 = cy - dl / 2 + cap / 2
        parts = [(val, cx, yb1, 'center', FS["basin_rate"], 'bold', ev),
                 (UNIT, cx, yb1 + dl, 'center', FS["basin_unit"], 'normal', eu)]
    xs0 = [p[1] + p[6][0] for p in parts]
    xs1 = [p[1] + p[6][1] for p in parts]
    ys0 = [p[2] + p[6][2] for p in parts]
    ys1 = [p[2] + p[6][3] for p in parts]
    return parts, (min(xs0), min(ys0), max(xs1), max(ys1))


def name_rect(name, cx, cy):
    e = measure(name, fontsize=FS["basin_name"], ha='center', va='center')
    pad = 0.25 * FS["basin_name"] * PT
    return (cx + e[0] - pad, cy + e[2] - pad, cx + e[1] + pad, cy + e[3] + pad)


class Layout:
    # layout page (mm) -> figure (mm, y down); G is an extra gap that pushes map d and the legend right
    def __init__(self, lay, content, G):
        hb = lay["hillshade_bounds_3413"]
        self.L, self.R, self.B, self.T = hb["left"], hb["right"], hb["bottom"], hb["top"]
        P = lay["panels"]
        self.sc0 = P[0]["mm_per_m_x"]                            # one map scale for all four panels
        self.ax_orig = [(p["axes_box"]["x"], p["axes_box"]["y"]) for p in P]
        self.X0, self.Y0, self.Xr, self.Yb = content["X0"], content["Y0"], content["Xr"], content["Yb"]
        self.G = float(G)
        self.W = FIG_W
        self.s = (self.W - MARGIN["l"] - MARGIN["r"] - self.G) / (self.Xr - self.X0)
        self.H = MARGIN["t"] + self.s * (self.Yb - self.Y0) + MARGIN["b"]
        self.mpm = self.s * self.sc0                               # mm per map metre
        self.lay = lay

    def nx(self, ox, k):
        return MARGIN["l"] + self.s * (np.asarray(ox, float) - self.X0) + (self.G if k == 3 else 0.0)

    def ny(self, oy):
        return MARGIN["t"] + self.s * (np.asarray(oy, float) - self.Y0)

    def to_mm(self, k, X, Y):
        ox = self.ax_orig[k][0] + (np.asarray(X, float) - self.L) * self.sc0
        oy = self.ax_orig[k][1] + (self.T - np.asarray(Y, float)) * self.sc0
        return self.nx(ox, k), self.ny(oy)

    def to_map(self, k, px, py):
        ox = (np.asarray(px, float) - MARGIN["l"] - (self.G if k == 3 else 0.0)) / self.s + self.X0
        oy = (np.asarray(py, float) - MARGIN["t"]) / self.s + self.Y0
        return self.L + (ox - self.ax_orig[k][0]) / self.sc0, self.T - (oy - self.ax_orig[k][1]) / self.sc0

    def axes_rect(self, k):
        x, y = self.to_mm(k, self.L, self.T)
        return float(x), float(y), (self.R - self.L) * self.mpm, (self.T - self.B) * self.mpm

    def period_centre(self, k):
        b = self.lay["panels"][k]["period_label"]["bbox"]
        ref = self.lay["panels"][1]["period_label"]["bbox"]           # panels b-d share one baseline
        return float(self.nx(b["x"] + b["w"] / 2, k)), float(self.ny(ref["y"] + ref["h"] / 2))

    def totals_rect(self, k):
        b = self.lay["totals_boxes"][k]["box"]
        return float(self.nx(b["x"], k)), float(self.ny(b["y"])), b["w"] * self.s, b["h"] * self.s

    def legend_rect(self):
        x0 = float(self.nx(self.lay["legend"]["size_box"]["x"], 3))
        x1 = self.W - MARGIN["r"]
        y0 = float(self.ny(self.lay["panels"][1]["period_label"]["bbox"]["y"]))
        y1 = float(self.ny(self.lay["content"]["totals_bottom"]))
        return x0, y0, x1 - x0, y1 - y0


class LandMask:
    # counts land (hillshade) pixels under a rectangle, from a summed-area table
    def __init__(self, arr, ext):
        valid = ~np.isnan(arr)
        self.I = np.zeros((valid.shape[0] + 1, valid.shape[1] + 1))
        self.I[1:, 1:] = valid.cumsum(0).cumsum(1)
        self.l, self.r, self.b, self.t = ext
        self.nr, self.nc = valid.shape
        self.pw = (self.r - self.l) / self.nc
        self.ph = (self.t - self.b) / self.nr

    def count(self, X0, X1, Y0, Y1):
        c0 = int(np.clip(np.floor((min(X0, X1) - self.l) / self.pw), 0, self.nc))
        c1 = int(np.clip(np.ceil((max(X0, X1) - self.l) / self.pw), 0, self.nc))
        r0 = int(np.clip(np.floor((self.t - max(Y0, Y1)) / self.ph), 0, self.nr))
        r1 = int(np.clip(np.ceil((self.t - min(Y0, Y1)) / self.ph), 0, self.nr))
        return self.I[r1, c1] - self.I[r0, c1] - self.I[r1, c0] + self.I[r0, c0]

    def area_mm2(self, L, k, rect):
        X0, Y0 = L.to_map(k, rect[0], rect[1])
        X1, Y1 = L.to_map(k, rect[2], rect[3])
        return self.count(X0, X1, Y0, Y1) * (self.pw * L.mpm) * (self.ph * L.mpm)


def rects_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def circle_pen(rect, cx, cy, r):
    # how far each circle reaches into a rectangle
    nx_ = np.clip(cx, rect[0], rect[2])
    ny_ = np.clip(cy, rect[1], rect[3])
    return np.maximum(r - np.hypot(cx - nx_, cy - ny_), 0.0)


def build_obstacles(L, data, glaciers):
    # everything a glacier label on map d has to avoid, in mm
    ob = {}
    for key, per in (("c4", "LIA_to_2022"), ("c3", "2000s_to_2022")):
        c = data["circles"][per]
        k = PERIODS.index(per)
        px, py = L.to_mm(k, c.x.values, c.y.values)
        ob[key] = dict(x=px, y=py, r=np.abs(c.change_m.values) * BUF["scale"] * L.mpm,
                       km=np.abs(c.change_m.values) / 1000.0)
    # glacier points on map d and on map c, the neighbour a map-d label can reach
    ob["markers"] = np.array([L.to_mm(k, g["X"], g["Y"]) for k in (3, 2) for g in glaciers])
    # basin labels, period label, totals box and legend of map d
    hard = []
    for b in data["labelpos"][3]:
        nr, rr, _, _ = basin_label_geom(L, data, 3, "LIA_to_2022", b, data["basin_offsets"].get((3, b), (0.0, 0.0)))
        hard += [nr, rr]
    x, y = L.period_centre(3)
    e = measure(TITLE["LIA_to_2022"], fontsize=FS["period"], fontweight='bold', ha='center', va='center')
    hard.append((x + e[0], y + e[2], x + e[1], y + e[3]))
    x, y, w, h = L.totals_rect(3)
    hard.append((x, y, x + w, y + h))
    x, y, w, h = L.legend_rect()
    hard.append((x, y, x + w, y + h))
    ob["hard"] = hard
    return ob


def label_rect(g, dx, dy):
    e = g["ext"]
    ax_, ay_ = g["anchor"]
    return (ax_ + dx + e[0] - LABEL_PAD, ay_ + dy + e[2] - LABEL_PAD, ax_ + dx + e[1] + LABEL_PAD, ay_ + dy + e[3] + LABEL_PAD)


def evaluate(L, data, ob, g, i, dx, dy, placed):
    # cost of a glacier label at (dx, dy): 1000 per hard conflict, plus soft overlaps and distance
    rect = label_rect(g, dx, dy)
    hard = 0
    if rect[0] < 0 or rect[1] < 0 or rect[2] > L.W or rect[3] > L.H:
        hard += 1
    sep = (rect[0] - LABEL_SEP, rect[1] - LABEL_SEP, rect[2] + LABEL_SEP, rect[3] + LABEL_SEP)
    for j, r2 in placed.items():
        if j != i and rects_overlap(sep, r2):
            hard += 1
    mk = ob["markers"]
    mr = MARKER["ms"] * PT / 2 + 0.05
    for j in range(len(mk)):
        if circle_pen(rect, mk[j, 0], mk[j, 1], mr) > 0:
            hard += 1
    for r2 in ob["hard"]:
        if rects_overlap(rect, r2):
            hard += 1
    c4 = ob["c4"]
    near = (np.abs(c4["x"] - (rect[0] + rect[2]) / 2) < 15) & (np.abs(c4["y"] - (rect[1] + rect[3]) / 2) < 15)
    pen4 = circle_pen(rect, c4["x"][near], c4["y"][near], c4["r"][near])
    km4 = c4["km"][near]
    pen4c = circle_pen(rect, c4["x"][near], c4["y"][near], c4["r"][near] + LARGE_CLEAR)
    hard += int(((pen4c > 0.0) & (km4 >= LARGE_KM)).sum())
    c3 = ob["c3"]
    near3 = (np.abs(c3["x"] - (rect[0] + rect[2]) / 2) < 15) & (np.abs(c3["y"] - (rect[1] + rect[3]) / 2) < 15)
    pen3 = circle_pen(rect, c3["x"][near3], c3["y"][near3], c3["r"][near3])
    hard += int(((pen3 > 0.0) & (c3["km"][near3] >= SMALL_KM)).sum())
    if data["land"].area_mm2(L, 2, rect) > 0.02:
        hard += 1
    land4 = data["land"].area_mm2(L, 3, rect)
    mid = float(pen4[(km4 >= MID_KM) & (km4 < LARGE_KM)].sum())
    small = float(pen4[(km4 >= SMALL_KM) & (km4 < MID_KM)].sum())
    tiny = float(pen4[km4 < SMALL_KM].sum())
    soft = 20 * mid + 5 * small + 1 * tiny + 0.4 * land4 + 10 * math.hypot(dx, dy)
    return 1000 * hard + soft, dict(hard=hard, pen_mid=mid)


def solve_labels(L, data, glaciers, ob, passes=5):
    # move each glacier label to its cheapest spot on a 0.2 mm grid, worst first, until nothing moves
    dxs = np.round(np.arange(-2.0, 2.001, 0.2), 3)
    dys = np.round(np.arange(-3.2, 3.201, 0.2), 3)
    off = {i: (0.0, 0.0) for i in range(len(glaciers))}
    placed = {i: label_rect(g, 0, 0) for i, g in enumerate(glaciers)}
    for _ in range(passes):
        order = sorted(range(len(glaciers)), key=lambda i: -evaluate(L, data, ob, glaciers[i], i, *off[i], placed)[0])
        changed = False
        for i in order:
            g = glaciers[i]
            best = (evaluate(L, data, ob, g, i, *off[i], placed)[0], off[i])
            for dx in dxs:
                for dy in dys:
                    c = evaluate(L, data, ob, g, i, dx, dy, placed)[0]
                    if c < best[0] - 1e-9:
                        best = (c, (float(dx), float(dy)))
            if best[1] != off[i]:
                changed = True
                off[i] = best[1]
                placed[i] = label_rect(g, *off[i])
        if not changed:
            break
    res = []
    for i, g in enumerate(glaciers):
        c, info = evaluate(L, data, ob, g, i, *off[i], placed)
        res.append(dict(dx=off[i][0], dy=off[i][1], cost=c, **info))
    return res


def basin_label_geom(L, data, k, per, b, off=(0.0, 0.0)):
    # name box, rate box and rate text of one basin label, shifted by off (mm)
    v = data["labelpos"][k][b]
    dx, dy = off
    cx, cy = L.to_mm(k, *v["name_box_centre_map"])
    rx, ry = L.to_mm(k, *v["rate_centre_map"])
    parts, rr = rate_parts(sgn(data["panels"][per]["rate"][b], 1), rx + dx, ry + dy, v["rate_lines"])
    return name_rect(b, cx + dx, cy + dy), rr, parts, (cx + dx, cy + dy)


def place_basin_labels(L, data, maps=(0, 1, 2, 3)):
    # a basin label stays put unless it sits on a circle >= 5 km; then the smallest move that clears it
    # and keeps it inside the basin
    offs = {}
    grid = np.round(np.arange(-4.4, 4.401, 0.2), 3)
    gmr = MARKER["ms"] * PT / 2 + 0.15
    for k, per in enumerate(PERIODS):
        if k not in maps:
            continue
        gmk = np.array([L.to_mm(k, g["X"], g["Y"]) for g in data["glaciers"]])
        c = data["circles"][per]
        px, py = L.to_mm(k, c.x.values, c.y.values)
        r = np.abs(c.change_m.values) * BUF["scale"] * L.mpm
        km = np.abs(c.change_m.values) / 1000.0
        bas = data["panels"][per]["basins"]
        for b in data["labelpos"][k]:
            geom = prep(shapely.union_all(bas[bas.SUBREGION1 == b].geometry.values))

            def cost(dx, dy):
                nr, rr, _, _ = basin_label_geom(L, data, k, per, b, (dx, dy))
                big, small = 0.0, 0.0
                inside = True
                for rect in (nr, rr):
                    pen = circle_pen(rect, px, py, r)
                    big += 50 * pen[km >= LARGE_KM].sum() + 20 * pen[(km >= MID_KM) & (km < LARGE_KM)].sum()
                    small += 3 * pen[(km >= SMALL_KM) & (km < MID_KM)].sum()
                    X0, Y0 = L.to_map(k, rect[0], rect[1])
                    X1, Y1 = L.to_map(k, rect[2], rect[3])
                    inside &= geom.contains(shapely.box(min(X0, X1), min(Y0, Y1), max(X0, X1), max(Y0, Y1)))
                    big += 200 * circle_pen(rect, gmk[:, 0], gmk[:, 1], gmr).sum()
                return big + small + (0 if inside else 1000) + 10 * math.hypot(dx, dy), big
            c0, big0 = cost(0.0, 0.0)
            off = (0.0, 0.0)
            if big0 > 0:
                best = (c0, off)
                for dx in grid:
                    for dy in grid:
                        cc = cost(float(dx), float(dy))[0]
                        if cc < best[0] - 1e-9:
                            best = (cc, (float(dx), float(dy)))
                off = best[1]
            extra = BASIN_LABEL_EXTRA.get((k, b))
            if extra:
                off = (round(off[0] + extra[0], 3), round(off[1] + extra[1], 3))
            offs[(k, b)] = off
    return offs


def draw_map(ax, hill, basins, fills, circ, scale, L):
    arr, ext = hill
    ax.imshow(arr, extent=ext, cmap='gray', alpha=0.5, zorder=0, interpolation='bilinear', rasterized=True)
    for _, region in basins.iterrows():
        color = fills.get(region['SUBREGION1'], "#d1d2c6")
        polys = [region.geometry] if region.geometry.geom_type == 'Polygon' else list(region.geometry.geoms)
        for poly in polys:
            for c in ax.fill(*poly.exterior.xy, color=color, edgecolor='white', linewidth=0.3):
                c.set_rasterized(True)
    cols = scale.rgba(circ.change_m.values / 1000.0)
    pc = PatchCollection([patches.Circle((x, y), abs(v) * BUF["scale"]) for x, y, v in
                          zip(circ.x.values, circ.y.values, circ.change_m.values)],
                         facecolor=cols, edgecolor='black', linewidth=0.20, alpha=BUF["alpha"],
                         match_original=False, zorder=5)
    pc.set_rasterized(True)
    pc.set_clip_on(False)
    ax.add_collection(pc)
    ax.set_xlim(L.L, L.R)
    ax.set_ylim(L.B, L.T)
    ax.axis('off')


def add_axes_mm(fig, L, x, y, w, h, **kw):
    return fig.add_axes([x / L.W, 1 - (y + h) / L.H, w / L.W, h / L.H], **kw)


def colour_bar(fig, ov, L, cx_box, top, bot, widest_tick, words, cmap):
    # vertical bar with the two words on its left and tick labels on its right, all centred in the box
    bw = 1.9
    side = FS["legend_small"] * PT * 1.15
    tl = measure(widest_tick, fontsize=FS["legend"], ha='left', va='center')
    blk = side + 0.35 + bw + (BREAK_TICK_LEN + 0.7) * PT + (tl[1] - tl[0])
    bx = cx_box - blk / 2 + side + 0.35
    mid = (top + bot) / 2
    for s_, yy in ((words[0], (top + mid) / 2), (words[1], (mid + bot) / 2)):
        ov.text(bx - 0.35, yy, s_, fontsize=FS["legend_small"], rotation=90, ha='right', va='center', zorder=8)
    cax = add_axes_mm(fig, L, bx, top, bw, bot - top)
    cax.set_zorder(11)
    u = (np.arange(1024) + 0.5) / 1024
    cax.imshow(u[:, None], aspect='auto', cmap=cmap, extent=(0, 1, 1, 0), interpolation='nearest', vmin=0, vmax=1)
    cax.set_xlim(0, 1)
    cax.set_ylim(1, 0)                                   # retreat / loss at the top
    return cax


def style_colour_bar(cax, positions, labels, minor, brk_pos):
    cax.set_yticks(positions)
    cax.set_yticklabels(labels)
    cax.set_yticks(minor, minor=True)
    cax.yaxis.tick_right()
    cax.tick_params(axis='y', which='major', labelsize=FS["legend"], length=TICK_LEN, width=0.3,
                    pad=BREAK_TICK_LEN - TICK_LEN + 0.7)
    cax.tick_params(axis='y', which='minor', length=0.8, width=0.25)
    cax.set_xticks([])
    for sp in cax.spines.values():
        sp.set_linewidth(0.3)
    # longer tick at the break of the two slopes
    for tk, is_break in zip(cax.yaxis.get_major_ticks(), brk_pos):
        if is_break:
            tk.tick2line.set_markersize(BREAK_TICK_LEN)
            tk.tick2line.set_markeredgewidth(0.45)


def draw_size_legend(fig, ov, L, x, y, w, h, scale):
    # box 1: grey size circles, then the circle colour bar
    cx_box = x + w / 2
    ov.text(cx_box, y + 0.9, "Margin change\n(km)", fontsize=FS["legend_title"], fontweight='bold', ha='center',
            va='top', linespacing=1.05, zorder=8)
    sizes = [5, 10, 20, 40]
    rr = [s_ * 1000 * BUF["scale"] * L.mpm for s_ in sizes]
    lab_w = [measure(str(s_), fontsize=FS["legend"], ha='left', va='center') for s_ in sizes]
    lab_gap = 0.45
    block_w = 2 * rr[-1] + lab_gap + max(e[1] - e[0] for e in lab_w)
    ccx = cx_box - block_w / 2 + rr[-1]
    cy = y + 4.9
    for s_, r_ in zip(sizes, rr):
        cy += r_
        ov.add_patch(patches.Circle((ccx, cy), r_, fc='#c8c8c8', ec='black', lw=0.24, alpha=BUF["alpha"], zorder=8))
        ov.text(ccx + r_ + lab_gap, cy, str(s_), fontsize=FS["legend"], ha='left', va='center', zorder=8)
        cy += r_ + 0.75
    cax = colour_bar(fig, ov, L, cx_box, cy + 1.4, y + h - 1.6, "+52.2", ("retreat", "advance"), scale.cmap)
    R = scale.R
    major = [-R] + MAJOR_TICKS_INNER + [R]
    style_colour_bar(cax, [float(scale.norm(v)) for v in major],
                     [sgn(v, 1) if abs(v) == R else sgn(v, 0) for v in major],
                     [float(scale.norm(v)) for v in MINOR_TICKS], [abs(v) == BREAK_KM for v in major])


def draw_fill_bar(fig, ov, L, x, y, w, h, sc):
    # box 2: basin fill colour bar
    cx_box = x + w / 2
    ov.text(cx_box, y + 0.9, "Area change rate\n" + r"(km$^{2}$ yr$^{-1}$)", fontsize=FS["legend_title"],
            fontweight='bold', ha='center', va='top', linespacing=1.05, zorder=8)
    cax = colour_bar(fig, ov, L, cx_box, y + 5.4, y + h - 1.6, "−100", ("loss", "gain"), sc.cmap)
    major = [-sc.R] + BASIN_MAJOR_INNER + [sc.R]
    style_colour_bar(cax, [float(sc.fwd(v)) for v in major], [sgn(v, 0) for v in major],
                     [float(sc.fwd(v)) for v in BASIN_MINOR if abs(v) < sc.R], [abs(v) == sc.brk for v in major])


def totals_lines(t, fs_value):
    return [("Total area change", FS["totals_title"], 'bold'),
            (f"{sgn(t['area'], 0)} [{sgn(t['area_lo'], 0)}, {sgn(t['area_hi'], 0)}] km$^{{2}}$", fs_value, 'normal'),
            ("Total area change rate", FS["totals_title"], 'bold'),
            (f"{sgn(t['rate'], 1)} [{sgn(t['rate_lo'], 1)}, {sgn(t['rate_hi'], 1)}] " + UNIT, fs_value, 'normal')]


def build_figure(L, data, placement):
    scale = data["scale"]
    fig = plt.figure(figsize=(L.W / 25.4, L.H / 25.4), facecolor='white')
    for k, per in enumerate(PERIODS):
        ax = add_axes_mm(fig, L, *L.axes_rect(k))
        draw_map(ax, data["hill"], data["panels"][per]["basins"], data["panels"][per]["fills"],
                 data["circles"][per], scale, L)
    ov = fig.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, L.W)
    ov.set_ylim(L.H, 0)
    ov.axis('off')
    ov.set_zorder(10)
    # period labels, basin labels, totals
    for k, per in enumerate(PERIODS):
        x, y = L.period_centre(k)
        ov.text(x, y, TITLE[per], fontsize=FS["period"], fontweight='bold', ha='center', va='center')
        fills = data["panels"][per]["fills"]
        for b in data["labelpos"][k]:
            _, _, parts, (cx, cy) = basin_label_geom(L, data, k, per, b, data["basin_offsets"].get((k, b), (0.0, 0.0)))
            ov.text(cx, cy, b, fontsize=FS["basin_name"], ha='center', va='center', zorder=8,
                    bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.45', fc='white', ec='black', lw=0.15, alpha=0.65))
            # white text on dark fills
            r_, g_, b_ = to_rgb(fills[b])
            tc = 'white' if 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_ < 0.42 else 'black'
            halo = dict(boxstyle='round,pad=0.12', fc=fills[b], ec='none', alpha=0.6)
            for (s, xx, yy, ha, fs, fw, _e) in parts:
                ov.text(xx, yy, s, fontsize=fs, fontweight=fw, ha=ha, va='baseline', color=tc, zorder=8, bbox=halo)
        x, y, w, h = L.totals_rect(k)
        ov.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.3",
                                            fc='#f2f2f2', ec='none', zorder=6))
        tx = x + data["totals_text_dx"] * L.s
        for i, (s_, fs_, fw_) in enumerate(totals_lines(data["panels"][per]["tot"], data["fs_totals_value"])):
            ov.text(tx, y + h * (0.155 + 0.232 * i), s_, fontsize=fs_, fontweight=fw_, ha='left', va='center', zorder=7)
    # legend column
    lx, ly, lw, lh = L.legend_rect()
    gap = data["legend_gap"]
    h1 = data["legend_h1"]
    h2 = lh - h1 - gap
    for (yy, hh) in ((ly, h1), (ly + h1 + gap, h2)):
        ov.add_patch(patches.FancyBboxPatch((lx, yy), lw, hh, boxstyle="round,pad=0,rounding_size=1.0",
                                            fc='white', ec='black', lw=0.35, zorder=6))
    draw_size_legend(fig, ov, L, lx, ly, lw, h1, scale)
    draw_fill_bar(fig, ov, L, lx, ly + h1 + gap, lw, h2, data["basin_scale"])
    # glacier points on all four maps, names on map d only
    gls = data["glaciers"]
    for k in range(len(PERIODS)):
        mk = np.array([L.to_mm(k, g["X"], g["Y"]) for g in gls])
        ov.plot(mk[:, 0], mk[:, 1], 'o', ms=MARKER["ms"], mfc='black', mec='white', mew=MARKER["mew"], zorder=12)
    for g, p in zip(gls, placement):
        ax_, ay_ = g["anchor"]
        for s_, fs_, yb in g["lines"]:
            ov.text(ax_ + p["dx"], ay_ + p["dy"] + yb, s_, fontsize=fs_, style='italic', ha=g["ha"], va='baseline',
                    zorder=12)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"{STEM}.{ext}"), dpi=600, facecolor='white')
    plt.close(fig)


def write_source_data(data, R):
    for per in PERIODS:
        c = data["circles"][per]
        lon, lat = T_3413_LL.transform(c.x.values, c.y.values)
        v = c.change_m.values / 1e3
        # basin under each circle: nearest clipped basin of this map (circles sit on the margin)
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(c.x.values, c.y.values), crs=TARGET)
        jb = gpd.sjoin_nearest(pts, data["panels"][per]["basins"][["SUBREGION1", "geometry"]], how="left")
        jb = jb[~jb.index.duplicated()]
        pd.DataFrame(dict(
            circle=np.arange(1, len(c) + 1), x_epsg3413_m=c.x.values.round(1), y_epsg3413_m=c.y.values.round(1),
            lon=lon.round(5), lat=lat.round(5), basin=jb["SUBREGION1"].values, margin_change_km=v.round(3),
            circle_radius_map_km=(np.abs(c.change_m.values) * BUF["scale"] / 1e3).round(2),
            n_segments_in_group=c.n_members.values, segment_row_index=c.pos.values.astype(int),
            colour_scale_limit_km=R,
            colour_hex=["#%02x%02x%02x" % tuple(int(round(255 * ch)) for ch in r[:3]) for r in data["scale"].rgba(v)])
        ).to_csv(os.path.join(OUTPUT_DIR, f"{STEM}_{per}_circles_source_data.csv"), index=False)
        P = data["panels"][per]
        rate, sub = P["rate"], P["sub"]
        pd.DataFrame([dict(basin=b, area_change_rate_km2_per_yr=round(rate[b], 3),
                           p05_km2_per_yr=round(float(sub.loc[b, "rate_p05"]), 3),
                           p95_km2_per_yr=round(float(sub.loc[b, "rate_p95"]), 3),
                           area_change_km2=round(float(sub.loc[b, "area_change_km2"]), 1),
                           area_change_p05_km2=round(float(sub.loc[b, "area_p05"]), 1),
                           area_change_p95_km2=round(float(sub.loc[b, "area_p95"]), 1),
                           fill_hex=P["fills"][b]) for b in sorted(rate)]
                     ).to_csv(os.path.join(OUTPUT_DIR, f"{STEM}_{per}_basin_rates_source_data.csv"), index=False)
    pd.DataFrame([dict(period=per, **{k_: round(v_, 1) for k_, v_ in data["panels"][per]["tot"].items()}) for per in PERIODS]
                 ).rename(columns=dict(area="area_change_km2", area_lo="area_change_p05_km2", area_hi="area_change_p95_km2",
                                       rate="rate_km2_per_yr", rate_lo="rate_p05_km2_per_yr", rate_hi="rate_p95_km2_per_yr")
                          ).to_csv(os.path.join(OUTPUT_DIR, f"{STEM}_totals_source_data.csv"), index=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LAYOUT_JSON, encoding="utf-8") as f:
        lay = json.load(f)
    artefacts = pd.read_csv(ARTEFACT_CSV)
    mc = pd.read_csv(MC_TABLE)

    print("Building circles...")
    circles = {}
    for per in PERIODS:
        g = gpd.read_file(os.path.join(BASE_DIR, "change", f"{per}.shp")).to_crs(TARGET).reset_index(drop=True)
        drop = artefacts.loc[artefacts.period == per, "seg_index"]      # artefact segments out before grouping
        circles[per] = make_circles(g.loc[g.index.difference(drop)])
        print(f"  {per}: {len(circles[per])} circles")
    # one colour range for all four maps; the 1 m tolerance stops float noise from raising R a step
    max_abs = max(float(np.abs(c.change_m.values).max()) / 1000 for c in circles.values())
    R = math.ceil(10 * (max_abs - 0.001)) / 10
    scale = TwoSlope(R)
    # basin rates and totals from the Monte Carlo table (negative = loss), one fill range for all maps too
    panels = {}
    for per in PERIODS:
        sub = mc[(mc.period == per) & (mc.basin != "All basins")].set_index("basin")
        t = mc[(mc.period == per) & (mc.basin == "All basins")].iloc[0]
        panels[per] = dict(sub=sub, rate={b: float(sub.loc[b, "rate_km2_yr"]) for b in sub.index},
                           tot=dict(area=t.area_change_km2, area_lo=t.area_p05, area_hi=t.area_p95,
                                    rate=t.rate_km2_yr, rate_lo=t.rate_p05, rate_hi=t.rate_p95))
    all_rates = np.array([v for per in PERIODS for v in panels[per]["rate"].values()])
    RB = 10.0 * math.ceil(np.abs(all_rates).max() / 10.0)
    basin_scale = BasinScale(BASIN_CMAP, RB)
    print(f"  colour ranges: circles +/-{R:.1f} km, basin fills +/-{RB:.0f} km2/yr")

    print("Loading basins and hillshade...")
    for per in PERIODS:
        P = panels[per]
        P["basins"] = load_basins(per)
        P["fills"] = {b: to_hex(basin_scale.colour(v)) for b, v in P["rate"].items()}
    hb = lay["hillshade_bounds_3413"]
    hill = load_hillshade((hb["left"], hb["right"], hb["bottom"], hb["top"]), decimate=12)

    ggn = gpd.read_file(GLACIER_NAMES_PATH)
    ggn = ggn[ggn.ID.isin(list(GLACIER_TEXT))].to_crs(3413).set_index("ID")
    glaciers = []
    for gl in lay["glacier_labels"]:
        i = gl["ggn"]
        X, Y = float(ggn.loc[i].geometry.x), float(ggn.loc[i].geometry.y)
        # label keeps its layout offset from the glacier point
        anchor_map = [gl["anchor_map_3413"][0] + X - gl["marker_map_3413"][0],
                      gl["anchor_map_3413"][1] + Y - gl["marker_map_3413"][1]]
        lines, ext = glacier_block(GLACIER_TEXT[i], gl["ha"], gl["va"])
        glaciers.append(dict(X=X, Y=Y, ha=gl["ha"], anchor_map=anchor_map, lines=lines, ext=ext))

    # content extent on the layout page: westmost land or circle of map a to the legend
    sc0 = lay["panels"][0]["mm_per_m_x"]
    arr = hill[0]
    rows_, cols_ = np.where(~np.isnan(arr))
    Xl = hb["left"] + (cols_.min() + 0.5) * (hb["right"] - hb["left"]) / arr.shape[1]
    c0 = circles["LIA_to_1980s"]
    Xmin = min(Xl, float((c0.x - c0.change_m.abs() * BUF["scale"]).min()))
    ymax_c = max(float((c.y + c.change_m.abs() * BUF["scale"]).max()) for c in circles.values())
    Yt_land = hb["top"] - (rows_.min() + 0.5) * (hb["top"] - hb["bottom"]) / arr.shape[0]
    ax0 = lay["panels"][0]["axes_box"]
    top_bcd = lay["panels"][1]["period_label"]["bbox"]["y"]
    content = dict(X0=ax0["x"] + (Xmin - hb["left"]) * sc0,
                   Y0=min(top_bcd, ax0["y"] + (hb["top"] - max(ymax_c, Yt_land)) * sc0),
                   Xr=lay["content"]["legend_right"], Yb=lay["content"]["totals_bottom"])

    data = dict(circles=circles, panels=panels, hill=hill, land=LandMask(*hill), labelpos=lay["basin_labels"],
                scale=scale, basin_scale=basin_scale, glaciers=glaciers)

    # gap G before map d: the smallest G where the glacier labels have no conflicts
    print("Placing labels...")
    trials = []
    for G in np.arange(0.0, 10.01, 0.5):
        L = Layout(lay, content, G)
        data["basin_offsets"] = place_basin_labels(L, data, maps=(3,))
        for g in glaciers:
            g["anchor"] = tuple(float(v) for v in L.to_mm(3, *g["anchor_map"]))
        res = solve_labels(L, data, glaciers, build_obstacles(L, data, glaciers))
        n_hard = sum(r["hard"] for r in res)
        max_d = max(math.hypot(r["dx"], r["dy"]) for r in res)
        mid = sum(r["pen_mid"] for r in res)
        trials.append((G, L, res, n_hard, max_d, mid))
        # four clean steps in a row, a bigger G won't help
        if len(trials) >= 4 and all(t[3] == 0 for t in trials[-4:]):
            break
    ok = [t for t in trials if t[3] == 0 and t[5] < 0.05]
    if ok:
        dmin = min(t[4] for t in ok)
        G, L, placement = next(t for t in ok if t[4] <= dmin + 0.25)[:3]
    else:
        G, L, placement = min(trials, key=lambda t: (t[3], t[4]))[:3]
        print("  Warning: no G without label conflicts, took the one with the fewest")
    data["basin_offsets"] = place_basin_labels(L, data)
    for g in glaciers:
        g["anchor"] = tuple(float(v) for v in L.to_mm(3, *g["anchor_map"]))
    print(f"  G = {G:.1f} mm; figure {L.W:.1f} x {L.H:.1f} mm")

    # legend boxes, and the totals font shrunk until the widest value line fits
    data["legend_gap"] = lay["legend"]["gap_between_boxes"] * L.s
    lx, ly, lw, lh = L.legend_rect()
    data["legend_h1"] = round(0.64 * (lh - data["legend_gap"]), 2)
    tb = lay["totals_boxes"][0]
    data["totals_text_dx"] = tb["text_bbox"]["x"] - tb["box"]["x"]
    fsv = FS["totals_value"]
    tl = [totals_lines(panels[p]["tot"], fsv) for p in PERIODS]
    widest = max([t[1][0] for t in tl] + [t[3][0] for t in tl],
                 key=lambda s_: measure(s_, fontsize=fsv, ha='left', va='center')[1])
    room = tb["box"]["w"] * L.s - 2 * data["totals_text_dx"] * L.s * 0.6
    while measure(widest, fontsize=fsv, ha='left', va='center')[1] > room and fsv > 3.4:
        fsv = round(fsv - 0.1, 1)
    data["fs_totals_value"] = fsv

    print("Rendering...")
    build_figure(L, data, placement)
    write_source_data(data, R)
    print(f"Saved {STEM}.png / .svg / .pdf and the source data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
