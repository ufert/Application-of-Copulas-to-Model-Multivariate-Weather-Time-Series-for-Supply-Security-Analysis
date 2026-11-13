# Thesis: the 734 German land points of the 0.25 deg grid (land and country masks from Natural Earth) and their cells;
# the table 'points' of its output is read by conversion.py, pecd.py and point1_night_ev.py. It also computes a
# per-point wind map, which point3_compare.py compares with the potential and risk areas of point3_value_map.py
# Run: cd analysis && python risk_map_025.py   (needs data/grid025_wind_2000-2024.npy; cartopy downloads Natural Earth once)
# Writes: data/risk_map_025.pkl, data/risk_map_025_points.csv
"""German land points of the 0.25 deg grid and a per-point wind map, onshore.
Land and German masks from Natural Earth; every 0.25 deg point of the study box is classified as land or sea and as
German or not, and assigned to its cell of the 4 x 4 partition (the table "points" of the output).
Wind map: national fleet = uniform capacity on every German land point; national lull = the 5 % of hours with the
lowest centred 72-h mean fleet CF. Per point: mean CF, CF during national lulls, retention (lull CF / mean CF),
correlation with the national fleet, own multi-day lulls per year, own longest lull, hours with CF < 5 %.
Power curve: hub 100 m, shear 0.25, cut-in 3 / rated 12 / cut-out 25 m/s.
"""
import pickle

import cartopy.io.shapereader as shpreader
import numpy as np
import pandas as pd
import shapely
from shapely.ops import unary_union
import paths  # repository folders: data, figures, external raw data (see paths.py)

D = paths.DATA
LAT_EDGES, LON_EDGES = [55, 53, 51, 49, 47], [5, 7.75, 10.5, 13.25, 16]
CURVES = {"base": (100, 12.0),        # hub m, rated m/s
          "modern": (135, 10.5)}      # low specific power, taller hub - robustness only
LOW_SHARE = 0.05                      # national low-wind hours = the fleet's lowest 5%


# 10 m conversion with a fixed shear (alpha 0.25, hub 100 m, rated 12 m/s)
def cf(v10, hub=100, rated=12.0):
    v = v10 * np.float32((hub / 10) ** 0.25)
    out = np.zeros_like(v)
    r = (v >= 3) & (v < rated)
    out[r] = (v[r] ** 3 - 27) / (rated ** 3 - 27)
    out[(v >= rated) & (v < 25)] = 1.0
    return out


# start and length of the runs where mask is true
def runs(mask):
    x = mask.astype(np.int8)
    d = np.diff(np.concatenate([[0], x, [0]]))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    return st, en


# data and masks
W = np.load(f"{D}/grid025_wind_2000-2024.npy", mmap_mode="r")
meta = pickle.load(open(f"{D}/grid025_meta.pkl", "rb"))
lat, lon, idx = meta["lat"], meta["lon"], meta["time"]
ny = idx.year.nunique()
LON, LAT = np.meshgrid(lon, lat)

land_geom = unary_union(list(shpreader.Reader(shpreader.natural_earth("50m", "physical", "land")).geometries()))
deu_geom = next(r.geometry for r in shpreader.Reader(
    shpreader.natural_earth("50m", "cultural", "admin_0_countries")).records()
    if r.attributes["ADM0_A3"] == "DEU")
off = np.arange(-0.125 + 0.0125, 0.125, 0.025)                 # 10 x 10 sub-samples per 0.25 deg box
oy, ox = np.meshgrid(off, off, indexing="ij")
SX = LON[..., None, None] + ox
SY = LAT[..., None, None] + oy
land_share = shapely.contains_xy(land_geom, SX, SY).mean(axis=(2, 3))
deu_share = shapely.contains_xy(deu_geom, SX, SY).mean(axis=(2, 3))
is_land = land_share >= 0.5
is_deu = is_land & (deu_share >= 0.5)
print(f"points: {is_land.size} in box, {is_land.sum()} land, {is_deu.sum()} German land")

# capacity factors and metrics, per curve
pts = np.argwhere(is_land)                                      # (n, 2) row/col
de_cols = is_deu[pts[:, 0], pts[:, 1]]
night = np.isin(idx.hour, [20, 21, 22, 23, 0, 1, 2, 3, 4, 5])


# capacity factors of all points with the 10 m conversion, in blocks of 200 points
def build_cf(hub, rated):
    CF = np.empty((W.shape[0], len(pts)), dtype=np.float32)
    for k0 in range(0, len(pts), 200):
        blk = pts[k0:k0 + 200]
        CF[:, k0:k0 + 200] = cf(np.asarray(W[:, blk[:, 0], blk[:, 1]], dtype=np.float32), hub, rated)
    return CF


def low_hours(fleet, share, window=72):
    """hours whose centred `window`-h mean fleet CF is in the lowest `share`"""
    sm = pd.Series(fleet).rolling(window, center=True, min_periods=1).mean().to_numpy() if window > 1 else fleet
    return sm <= np.quantile(sm, share)


# per-point metrics of the wind map: mean CF, CF during national lulls, retention and synchrony
def metrics(CF, fleet, low, full=True):
    mean_cf = CF.mean(axis=0, dtype=np.float64)
    lull_cf = CF[low].mean(axis=0, dtype=np.float64)
    out = {"mean CF": mean_cf, "lull CF": lull_cf, "retention": lull_cf / mean_cf}
    if not full:
        return out
    fz = (fleet - fleet.mean()) / fleet.std()
    corr, low5, own72, longest = (np.empty(len(pts)) for _ in range(4))
    for k in range(len(pts)):
        x = CF[:, k].astype(np.float64)
        corr[k] = ((x - x.mean()) / x.std() * fz).mean()
        low5[k] = (x < 0.05).sum() / ny
        s_, e_ = runs(x < 0.10)
        own72[k] = ((e_ - s_) >= 72).sum() / ny
        longest[k] = (e_ - s_).max() / 24
    out.update({"corr fleet": corr, "CF<5% h/yr": low5, "own lulls>=72h /yr": own72,
                "longest own lull d": longest})
    return out


CF = build_cf(*CURVES["base"])
fleet = CF[:, de_cols].mean(axis=1, dtype=np.float64)
low = low_hours(fleet, LOW_SHARE)
st, en = runs(fleet < 0.10)
L = en - st
print(f"national fleet (uniform on German land, base curve): mean CF {fleet.mean():.3f}, "
      f"median {np.median(fleet):.3f}; 'CF<10% for >=72 h' would cover {L[L >= 72].sum() / len(fleet):.0%} of hours")
for name, m_ in (("SUSTAINED (72-h mean, lowest 5%)", low), ("hourly lowest 5%", low_hours(fleet, LOW_SHARE, 1))):
    st, en = runs(m_)
    mon = pd.Series(idx.month[m_]).value_counts(normalize=True).reindex(range(1, 13), fill_value=0)
    print(f"  {name:34} {m_.sum() / ny:.0f} h/yr, fleet CF in them {fleet[m_].mean():.4f}; "
          f"{len(st) / ny:.1f} episodes/yr, median {np.median(en - st):.0f} h; night share {night[m_].mean():.0%} "
          f"(base {night.mean():.0%}); Jun-Sep {mon.loc[6:9].sum():.0%}, Nov-Feb {mon.loc[[11, 12, 1, 2]].sum():.0%}")

M = metrics(CF, fleet, low)

# robustness of the spatial pattern: ranks under other definitions
rob = {}
rob["hourly lowest 5%"] = metrics(CF, fleet, low_hours(fleet, LOW_SHARE, 1), full=False)
rob["24-h mean, lowest 5%"] = metrics(CF, fleet, low_hours(fleet, LOW_SHARE, 24), full=False)
for share in (0.01, 0.10):
    rob[f"72-h mean, lowest {share:.0%}"] = metrics(CF, fleet, low_hours(fleet, share), full=False)
rob["night hours only"] = metrics(CF, fleet, low & night, full=False)
north = de_cols & (lat[pts[:, 0]] >= 52)
fleet_n = CF[:, north].mean(axis=1, dtype=np.float64)
rob["north-only fleet (>=52N)"] = metrics(CF, fleet_n, low_hours(fleet_n, LOW_SHARE), full=False)
del CF
CFm = build_cf(*CURVES["modern"])
fleet_m = CFm[:, de_cols].mean(axis=1, dtype=np.float64)
rob["modern curve"] = metrics(CFm, fleet_m, low_hours(fleet_m, LOW_SHARE), full=False)
print(f"  modern curve (hub 135 m, rated 10.5): national fleet mean CF {fleet_m.mean():.3f}")
del CFm
print("\nROBUSTNESS: Spearman rank correlation with the base map (German land points)")
for k, v in rob.items():
    print(f"  {k:20} lull CF {pd.Series(v['lull CF'][de_cols]).corr(pd.Series(M['lull CF'][de_cols]), 'spearman'):.3f}"
          f"   retention {pd.Series(v['retention'][de_cols]).corr(pd.Series(M['retention'][de_cols]), 'spearman'):.3f}")

pv = meta["pv_cf_by_year"].mean(axis=0)[pts[:, 0], pts[:, 1]]


# cell r{row}c{col} of the 4 x 4 partition that contains a grid point (row 0 = north, column 0 = west)
def cell_of(r, c):
    rr = next(i for i in range(4) if LAT_EDGES[i + 1] <= lat[r] <= LAT_EDGES[i])
    cc = next(j for j in range(4) if LON_EDGES[j] <= lon[c] <= LON_EDGES[j + 1])
    return f"r{rr}c{cc}"


P = pd.DataFrame({"lat": lat[pts[:, 0]], "lon": lon[pts[:, 1]], "germany": de_cols,
                  "cell": [cell_of(r, c) for r, c in pts],
                  **M, "PV CF": pv})

# designation (German land points, quintiles of lull CF)
G = P[P.germany]
q_lo, q_hi = G["lull CF"].quantile([0.2, 0.8])
P["class"] = np.select([P["lull CF"] >= q_hi, P["lull CF"] <= q_lo], ["potential", "risk"], "neither")
G = P[P.germany]
lg = np.log(G[["mean CF", "retention", "lull CF"]])
share_res = lg["mean CF"].var() / lg["lull CF"].var()
share_ret = lg["retention"].var() / lg["lull CF"].var()
cov2 = 2 * lg["mean CF"].cov(lg["retention"]) / lg["lull CF"].var()

pd.set_option("display.width", 220)
print(f"\nGERMAN LAND POINTS (n={len(G)}): distribution of each metric")
print(G.drop(columns=["lat", "lon", "germany", "cell", "class"]).describe(percentiles=[.1, .5, .9]).round(3).to_string())
sp = G.drop(columns=["lat", "lon", "germany", "cell", "class", "PV CF"]).corr("spearman")
print("\nSpearman rank correlations between metrics (German land points):")
print(sp.round(2).to_string())
print(f"\nlull CF = mean CF x retention. Spatial variance of log lull CF: resource {share_res:.0%}, "
      f"diversification {share_ret:.0%}, covariance term {cov2:+.0%}")
print(f"\ndesignation: potential = top 20% lull CF (>= {q_hi:.3f}), risk = bottom 20% (<= {q_lo:.3f})")
for c_ in ("potential", "risk"):
    g_ = G[G["class"] == c_]
    print(f"  {c_:10} n={len(g_)}  mean CF {g_['mean CF'].mean():.3f}  lull CF {g_['lull CF'].mean():.3f}  "
          f"retention {g_['retention'].mean():.2f}  corr {g_['corr fleet'].mean():.2f}  "
          f"own lulls>=72h {g_['own lulls>=72h /yr'].mean():.1f}/yr  "
          f"cells: {g_['cell'].value_counts().head(5).to_dict()}")

print("\nPER 4x4 CELL, land points only (Germany and neighbours), point metrics averaged:")
agg = P.groupby("cell").agg(n=("mean CF", "size"), de=("germany", "mean"),
                            mean_cf=("mean CF", "mean"), lull=("lull CF", "mean"),
                            retention=("retention", "mean"), corr=("corr fleet", "mean"),
                            own72=("own lulls>=72h /yr", "mean"), longest=("longest own lull d", "mean"),
                            pv=("PV CF", "mean"))
print(agg.round(3).to_string())

top = G.nlargest(6, "lull CF")[["lat", "lon", "cell", "mean CF", "lull CF", "retention", "corr fleet"]]
bot = G.nsmallest(6, "lull CF")[["lat", "lon", "cell", "mean CF", "lull CF", "retention", "corr fleet"]]
print("\nhighest lull CF (German land):\n" + top.round(3).to_string())
print("\nlowest lull CF (German land):\n" + bot.round(3).to_string())

with open(f"{D}/risk_map_025.pkl", "wb") as f:
    pickle.dump({"points": P, "lat": lat, "lon": lon, "land_share": land_share, "deu_share": deu_share,
                 "fleet": fleet, "low": low, "quintiles": (q_lo, q_hi), "robustness": rob,
                 "decomposition": (share_res, share_ret, cov2)}, f)
P.to_csv(f"{D}/risk_map_025_points.csv", index=False)
print(f"\nSaved {D}/risk_map_025.pkl and risk_map_025_points.csv")
