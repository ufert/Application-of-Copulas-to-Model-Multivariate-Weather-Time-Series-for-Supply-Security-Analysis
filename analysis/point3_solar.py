# Thesis: Results, Potential and Risk Areas (photovoltaics)
# Run: cd analysis && python point3_solar.py   (~4 min)
# Writes: data/point3_solar.pkl
"""Potential and risk areas for photovoltaics and for wind and PV combined, with the stress windows and rules of
point3_value_map.py: per year the worst 72-h window of the 2030 residual load (with offshore, no EVs; series of
point1_night_ev.py), 75 years 1950-2024; firmness of a site = its CF in the stress hours / that of an average MW of the
same technology over the 734 German land points (ratio of sums over years, 90 % bootstrap interval over years);
potential: firmness >= 1.10 and interval above 1; risk: <= 0.90 and interval below 1.
  wind      onshore, the point-level 100 m conversion of point2_aggregation.py (as in point3_value_map.py)
  PV        PV formula of conversion.py per point (extract_solar025.py: GHI and temperature at the 734 points)
  combined  a site hosting the 2030 capacity mix, 115 GW wind : 215 GW PV, per MW of that mix
Writes data/point3_solar.pkl.
"""
import pickle
import time

import numpy as np
import pandas as pd

import conversion as cv

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)
D = cv.D
YEARS = np.arange(1950, 2025)
# conversions: PV of conversion.py and the point-level 100 m wind conversion of point 2 (one class per point)
C = cv.calibrate_16cells()
PVC = C["pv"]
PC = cv.point_classes()
NP = len(PC["class"])
P2 = pickle.load(open(f"{D}/point2_aggregation.pkl", "rb"))
ONEHOT = np.stack([(PC["class"] == k).astype(float) for k in cv.CLASSES], axis=1)
CONV_ON = cv.Wind100Conversion(cv.CLASSES, ONEHOT, np.full(NP, 1 / NP), P2["sigma_pt"], P2["kappa"])
W_MIX = {"wind": 115 / 330, "pv": 215 / 330}
rng = np.random.default_rng(55)

# stress windows (worst 72 h of the 2030 residual load per year), calendar order
# stress windows: the worst 72-hour window of the 2030 residual load in each observed year
S1 = pickle.load(open(f"{D}/point1_night_ev.pkl", "rb"))["series"]["real 1950-2024"]
order = [(list(range(2000, 2025)) + list(range(1950, 2000))).index(y) for y in YEARS]
RL = (S1["demand"] - 215 * S1["pv"] - 30 * S1["offshore"] - 115 * S1["onshore 16 cells"])[order].astype(np.float64)
MASK = np.zeros(RL.shape, dtype=bool)
for i, r in enumerate(RL):
    c = np.concatenate([[0.0], np.cumsum(r)])
    j = int(np.argmax(c[72:] - c[:-72]))
    MASK[i, j:j + 72] = True

# 100 m wind of the 0.25 deg box and irradiance and temperature of the 734 German land points
meta = pickle.load(open(f"{D}/grid025_meta.pkl", "rb"))
BOX = {"2000-2024": (np.load(f"{D}/grid025_ws100_2000-2024.npy", mmap_mode="r"), meta["time"]),
       "1950-1999": (np.load(f"{D}/grid025_ws100_1950-1999.npy", mmap_mode="r"), pickle.load(open(f"{D}/grid025_meta_1950-1999.pkl", "rb"))["time"])}
SM = pickle.load(open(f"{D}/grid025de_meta_1950-2024.pkl", "rb"))["time"]
GHI = np.load(f"{D}/grid025de_ghi_1950-2024.npy", mmap_mode="r")
TK = np.load(f"{D}/grid025de_t2m_1950-2024.npy", mmap_mode="r")

# per year and site: mean capacity factor in the stress window (S) and over the whole year (M)
S = {k: np.empty((len(YEARS), NP)) for k in ("wind", "pv", "combined")}
M = {k: np.empty((len(YEARS), NP)) for k in ("wind", "pv", "combined")}
for i, y in enumerate(YEARS):
    arr, t = BOX["2000-2024" if y >= 2000 else "1950-1999"]
    rows = np.where((t.year == y) & ~((t.month == 2) & (t.day == 29)))[0]
    w100 = np.asarray(arr[rows[0]:rows[-1] + 1], dtype=np.float32)[np.isin(np.arange(rows[0], rows[-1] + 1), rows)][:, PC["ri"], PC["ci"]]
    lm, lh = cv.local_time(t[rows])
    cw = CONV_ON.regional(w100, lm, lh).astype(np.float64)
    srows = np.where((SM.year == y) & ~((SM.month == 2) & (SM.day == 29)))[0]
    g = np.asarray(GHI[srows], dtype=np.float32)
    tk = np.asarray(TK[srows], dtype=np.float32)
    cp = np.roll(cv.pv_cells(g, tk) * PVC.pr, PVC.lag, axis=0) * PVC.ratio[lm - 1, lh][:, None]
    for k, x in (("wind", cw), ("pv", cp), ("combined", W_MIX["wind"] * cw + W_MIX["pv"] * cp)):
        S[k][i] = x[MASK[i]].mean(axis=0)
        M[k][i] = x.mean(axis=0)
    if y % 10 == 0:
        log(f"{y} done")

# firmness of every site relative to an average megawatt, 90 % bootstrap interval over the years, designation
SITES = PC["points"][["lat", "lon", "cell"]].copy()
SUM = {}
for k in S:
    f = S[k].mean(axis=1)
    r = S[k].sum(axis=0) / f.sum()
    b = rng.integers(0, len(YEARS), (1000, len(YEARS)))
    rb = S[k][b].sum(axis=1) / f[b].sum(axis=1)[:, None]
    lo, hi = np.quantile(rb, 0.05, axis=0), np.quantile(rb, 0.95, axis=0)
    cls = np.where((r >= 1.10) & (lo > 1), "potential", np.where((r <= 0.90) & (hi < 1), "risk", "average"))
    SITES[f"r {k}"], SITES[f"class {k}"], SITES[f"mean CF {k}"] = r, cls, M[k].mean(axis=0)
    SITES[f"stress CF {k}"] = S[k].mean(axis=0)
    SUM[k] = {"fleet CF in stress": f.mean(), "fleet annual CF": M[k].mean(), "kept": f.mean() / M[k].mean(),
              "r p10": np.quantile(r, 0.1), "r median": np.median(r), "r p90": np.quantile(r, 0.9),
              "potential": int((cls == "potential").sum()), "risk": int((cls == "risk").sum()), "average": int((cls == "average").sum()),
              "corr(r, mean CF)": np.corrcoef(r, M[k].mean(axis=0))[0, 1]}
    for per, sel in (("1950-1999", YEARS < 2000), ("2000-2024", YEARS >= 2000)):
        SITES[f"r {k} {per}"] = S[k][sel].sum(axis=0) / S[k][sel].mean(axis=1).sum()
    SUM[k]["period rank corr"] = SITES[f"r {k} 1950-1999"].rank().corr(SITES[f"r {k} 2000-2024"].rank())
SUM = pd.DataFrame(SUM).T
pd.set_option("display.width", 220)
print("\nFIRMNESS IN THE WORST 72-H RESIDUAL-LOAD WINDOWS (734 German land points, 1950-2024)")
print(SUM.round(3).to_string())
chk = pickle.load(open(f"{D}/point3_value_map.pkl", "rb"))["sites"]
chk = chk[chk.type == "onshore"]["r 72 h"].to_numpy()
print(f"wind check against point3_value_map.py: max |diff| {np.abs(chk - SITES['r wind'].to_numpy()).max():.2e}")
for lat0, lat1, lab in ((53.5, 55.1, "north > 53.5N"), (50.5, 52.5, "centre 50.5-52.5N"), (47, 48.5, "south < 48.5N")):
    m = (SITES.lat >= lat0) & (SITES.lat <= lat1)
    print(f"{lab}: median r wind {SITES.loc[m, 'r wind'].median():.2f}, PV {SITES.loc[m, 'r pv'].median():.2f}, combined {SITES.loc[m, 'r combined'].median():.2f}")
for lon0, lon1, lab in ((5, 9, "west < 9E"), (12, 16, "east > 12E")):
    m = (SITES.lon >= lon0) & (SITES.lon <= lon1)
    print(f"{lab}: median r wind {SITES.loc[m, 'r wind'].median():.2f}, PV {SITES.loc[m, 'r pv'].median():.2f}, combined {SITES.loc[m, 'r combined'].median():.2f}")
# how related the wind and PV maps are (rank correlation of the firmness)
print("rank correlation wind vs PV firmness:", round(SITES["r wind"].rank().corr(SITES["r pv"].rank()), 3))
with open(f"{D}/point3_solar.pkl", "wb") as f:
    pickle.dump({"sites": SITES, "summary": SUM, "mix": W_MIX}, f)
log("saved point3_solar.pkl")
