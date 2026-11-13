# Thesis: Results, From Weather to Feed-in (photovoltaics)
# Run: cd analysis && python point2_solar.py   (~2 min)
# Writes: data/point2_solar.pkl
"""Error of averaging the weather before converting it into PV feed-in.
truth (B)  734 German land points at 0.25 deg (extract_solar025.py), 2000-2024, 29 February dropped, equal capacity per
           point; PV conversion of conversion.py: CF = GHI/1000 (1 - 0.004 (T_cell - 25)), T_cell = T + 0.03125 GHI,
           x performance ratio, shifted by -1 h (ERA5 radiation is hour-ending), x orientation factor per (month, local
           hour); also with inverter clipping (DC/AC 1.3)
A          per region: mean irradiance and mean temperature of its points converted; national = capacity-weighted sum;
           regions Germany, 4 quadrants, 2 deg blocks, 16 cells, 1 deg, 0.5 deg
metrics    energy; daylight hours with CF < 5 %; dark days (daily PV energy < 30 % of the mean of the calendar month);
           worst 72-h and 7-day mean CF in winter; 99th percentile of the hourly CF; 2030 residual load (series of
           point1_night_ev.py, 215 GW PV)
Writes data/point2_solar.pkl.
"""
import pickle

import numpy as np
import pandas as pd

import conversion as cv

D = cv.D
# PV conversion of conversion.py; irradiance and temperature of the 734 German land points, 2000-2024
C = cv.calibrate_16cells()
PVC = C["pv"]
meta = pickle.load(open(f"{D}/grid025de_meta_1950-2024.pkl", "rb"))
idx = meta["time"]
rows = np.where((idx.year >= 2000) & ~((idx.month == 2) & (idx.day == 29)))[0]
TIME = idx[rows]
GHI = np.asarray(np.load(f"{D}/grid025de_ghi_1950-2024.npy", mmap_mode="r")[rows], dtype=np.float32)
TK = np.asarray(np.load(f"{D}/grid025de_t2m_1950-2024.npy", mmap_mode="r")[rows], dtype=np.float32)
LMON, LHR = cv.local_time(TIME)
NP = GHI.shape[1]
RATIO = PVC.ratio[LMON - 1, LHR]
PC = cv.point_classes()
ri, ci = PC["ri"], PC["ci"]
rr = np.array([int(c[1]) for c in PC["points"]["cell"]])
cc = np.array([int(c[3]) for c in PC["points"]["cell"]])
# regions of increasing size over which the weather is averaged before the conversion (Germany ... 0.5 deg)
LEVELS = {"Germany": np.zeros(NP, int), "4 quadrants": (rr // 2) * 2 + cc // 2, "2 deg blocks": (ri // 8) * 100 + ci // 8,
          "16 cells": rr * 4 + cc, "1 deg blocks": (ri // 4) * 100 + ci // 4, "0.5 deg blocks": (ri // 2) * 100 + ci // 2}
SEAS_DJF = np.isin(LMON, [12, 1, 2])


def convert(ghi, tk, clip):
    """(T, k) irradiance and temperature -> PV CF (T, k): module, performance ratio, -1 h shift, orientation; optional clipping"""
    dc = cv.pv_cells(ghi, tk) * PVC.pr
    dc = np.roll(dc, PVC.lag, axis=0) * RATIO[:, None]
    return np.minimum(dc, 1 / 1.3) if clip else dc


# lowest w-hour moving mean in winter (DJF), averaged over the 25 years
def roll_min(x, w):
    out = []
    for y in range(25):
        u = x[y * 8760:(y + 1) * 8760]
        m = SEAS_DJF[y * 8760:(y + 1) * 8760]
        c = np.concatenate([[0.0], np.cumsum(u)])
        r = (c[w:] - c[:-w]) / w
        out.append(r[m[:len(r)]].min())
    return np.mean(out)


def metrics(cf, ref):
    """sun: daylight slots (climatological truth CF > 0.02 per month and local hour), set per clipping variant below"""
    d = cf.reshape(-1, 24).mean(axis=1)
    mon = TIME[::24].month.to_numpy()
    md = pd.Series(d).groupby(mon).transform("mean").to_numpy()
    sun = cf_sun_mask
    return {"mean CF": cf.mean(), "daylight h/yr CF<5%": ((cf < 0.05) & sun).sum() / 25, "dark days/yr": (d < 0.3 * md).sum() / 25,
            "worst 72 h winter CF": roll_min(cf, 72), "worst 7 d winter CF": roll_min(cf, 168), "p99 CF": np.quantile(cf, 0.99),
            "corr with truth": np.corrcoef(cf, ref)[0, 1]}


# reference: every point converted on its own; then each level converted from averaged weather
ROWS, SER = {}, {}
for clip in (False, True):
    lab = "with clipping" if clip else "no clipping"
    truth = convert(GHI, TK, clip).mean(axis=1)
    cf_sun_mask = (pd.Series(truth).groupby([LMON, LHR]).transform("mean").to_numpy() > 0.02)
    ROWS[(lab, "points (truth)")] = metrics(truth, truth) | {"regions": NP}
    SER[(lab, "points (truth)")] = truth.astype(np.float32)
    for lev, ids in LEVELS.items():
        regs = np.unique(ids)
        g = np.stack([GHI[:, ids == r].mean(axis=1) for r in regs], axis=1)
        t = np.stack([TK[:, ids == r].mean(axis=1) for r in regs], axis=1)
        w = np.array([(ids == r).mean() for r in regs])
        cf = (convert(g, t, clip) * w).sum(axis=1)
        ROWS[(lab, lev)] = metrics(cf, truth) | {"regions": len(regs)}
        SER[(lab, lev)] = cf.astype(np.float32)
T = pd.DataFrame(ROWS).T
REL = {}
for lab in ("no clipping", "with clipping"):
    b = T.loc[(lab, "points (truth)")]
    for lev in LEVELS:
        r = T.loc[(lab, lev)]
        REL[(lab, lev)] = {"energy %": 100 * (r["mean CF"] / b["mean CF"] - 1), "daylight low hours x": r["daylight h/yr CF<5%"] / b["daylight h/yr CF<5%"],
                           "dark days x": r["dark days/yr"] / b["dark days/yr"], "worst 72 h winter %": 100 * (r["worst 72 h winter CF"] / b["worst 72 h winter CF"] - 1),
                           "p99 %": 100 * (r["p99 CF"] / b["p99 CF"] - 1)}
REL = pd.DataFrame(REL).T
# 2030 residual load with 215 GW PV from the truth or from the Germany-averaged weather
S1 = pickle.load(open(f"{D}/point1_night_ev.pkl", "rb"))["series real 2000-2024"]
base = (S1["demand"] - 115 * S1["onshore 16 cells"] - 30 * S1["offshore"]).astype(np.float64).reshape(-1)
SYS = {}
for lev in ("points (truth)", "Germany", "16 cells"):
    rl = (base - 215 * SER[("no clipping", lev)]).reshape(25, 8760)
    c = np.concatenate([np.zeros((25, 1)), np.cumsum(rl, axis=1)], axis=1)
    SYS[lev] = {"peak RL (GW)": rl.max(axis=1).mean(), "worst 72 h RL (GW)": ((c[:, 72:] - c[:, :-72]) / 72).max(axis=1).mean()}
SYS = pd.DataFrame(SYS).T
# mechanism: national PV CF by national-mean irradiance, truth vs Germany-averaged
gn = GHI.mean(axis=1)
bins = np.arange(0, 1001, 50)
bi = np.digitize(np.roll(gn, PVC.lag), bins)
GAP = pd.DataFrame({"national GHI": [bins[i - 1] + 25 for i in np.unique(bi)], "share": [(bi == i).mean() for i in np.unique(bi)],
                    "truth CF": [SER[("no clipping", "points (truth)")][bi == i].mean() for i in np.unique(bi)],
                    "Germany CF": [SER[("no clipping", "Germany")][bi == i].mean() for i in np.unique(bi)]})
pd.set_option("display.width", 220)
print("PV: ABSOLUTE (2000-2024)")
print(T.round(4).to_string())
print("\nPV: RELATIVE TO FULL RESOLUTION")
print(REL.round(3).to_string())
print("\n2030 residual load with 215 GW PV")
print(SYS.round(2).to_string())
with open(f"{D}/point2_solar.pkl", "wb") as f:
    pickle.dump({"table": T, "relative": REL, "system": SYS, "gap": GAP}, f)
print("saved point2_solar.pkl")
