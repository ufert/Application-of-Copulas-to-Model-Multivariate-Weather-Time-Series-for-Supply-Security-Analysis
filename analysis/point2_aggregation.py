# Thesis: Case Study, conversion parameters (wind scale, spread, turbine classes, table of the conversion)
# and Results, From Weather to Feed-in (wind: aggregation before and after the conversion)
# Run: cd analysis && python point2_aggregation.py   (~25 min)
# Writes: data/point2_aggregation.pkl
"""Error of averaging the weather before converting it into wind feed-in: "average weather x average power curve" against
"aggregate of (regional weather x regional power curve)", with the 100 m conversion of conversion.py.

truth (B)  734 German land points at 0.25 deg, ERA5 100 m wind (PECD 4.2, scaled to the Global Wind Atlas) 2000-2024,
           29 February dropped. Each point gets a turbine class by its mean 10 m wind (terciles;
           conversion.point_classes): strong-wind (specific power 400 W/m2, hub 100 m), medium (300 W/m2, 120 m),
           low-wind (220 W/m2, 140 m); rated speed from the specific power (Cp 0.45, 1.225 kg/m3), cut-in 3, cut-out
           25 m/s, cubic ramp; equal capacity per point. Hub wind = kappa(month, local hour) * v100 * (hub/100)^(1/7),
           fleet curve with a spread sigma_pt (turbines and sites within a 0.25 deg box); kappa per slot such that the
           mean CF of the fleet per slot follows observed German feed-in (OPSD 2015-2019, 0.207 overall), sigma_pt by
           the quantile match (conversion.calibrate_wind100 with the 734 points as regions)
ladder     regions: Germany, 4 quadrants, 2 deg blocks, the 16 cells, 1 deg, 0.5 deg, points (= B)
A          per region: mean 100 m wind of its points x the average curve of the region (capacity-weighted class mix;
           same kappa and sigma_pt as B); national = capacity-weighted sum of the regions
A_sigma    as A, with the curve spread widened by the spread of the point winds around the region mean:
           sigma = sqrt(sigma_pt^2 + SD of the pooled log ratios v_point / v_region of the region)
single     all points in the medium class, same kappa and sigma_pt: the averaging effect without a class mix
adopted    the 16-cell conversion of conversion.py: 16 cell-mean 100 m winds (inclusive cell edges), class mix per
           cell, spread calibrated at the cell level against OPSD
metrics    mean CF; hours per year below 5 % / 10 % of capacity; lulls >= 24 h / 72 h (CF < 10 %) and the longest per
           year; 99th percentile of the hourly ramps; hourly correlation with B; against OPSD 2015-2019: correlation,
           hours below 5 % (observed scaled per year to the mean), lulls in the observed feed-in
system     2030 system of point1_night_ev.py (observed 2000-2024): demand, PV, 30 GW offshore; 115 GW onshore from
           each series. Peak residual load, worst 72-h and 7-day mean, hours above the 99th percentile of B
PV         national-mean irradiance and temperature x PV model against the aggregate of the 16 cells, with and without
           inverter clipping (DC/AC 1.3)
Writes data/point2_aggregation.pkl.
"""
import pickle
import time

import numpy as np
import pandas as pd

import conversion as cv

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)
D = cv.D
CELLS = cv.CELLS
CLASSES = cv.CLASSES

# data
meta = pickle.load(open(f"{D}/grid025_meta.pkl", "rb"))
PC = cv.point_classes()
G, ri, ci, CLS = PC["points"], PC["ri"], PC["ci"], PC["class"]
TIME = meta["time"]
keep = ~((TIME.month == 2) & (TIME.day == 29))
V = np.asarray(np.load(f"{D}/grid025_ws100_2000-2024.npy", mmap_mode="r")[:, ri, ci], dtype=np.float32)[keep]
TIME = TIME[keep]
YR = TIME.year.to_numpy()
NP, NT = V.shape[1], V.shape[0]
LMON, LHR = cv.local_time(TIME)
CALW = (YR >= 2015) & (YR <= 2019)
OBS = cv.opsd(TIME, "wind_onshore")
log(f"{NP} German land points, {NT} hours, mean 100 m wind {V.mean():.2f} m/s; classes "
    + ", ".join(f"{k} {(CLS == k).sum()}" for k in CLASSES))


# shares of the turbine classes among the points idx
def shares_of(cls, idx=slice(None)):
    return np.array([(cls[idx] == k).mean() for k in CLASSES])


ONEHOT = {"mix": np.stack([(CLS == k).astype(float) for k in CLASSES], axis=1),
          "single": np.tile(np.array([0.0, 1.0, 0.0]), (NP, 1))}
CLASS_OF = {"mix": CLS, "single": np.full(NP, "medium")}


def chunked(conv, v, lm=LMON, lh=LHR):
    """national CF of conv on (T, regions) wind, year by year (memory)"""
    out = np.empty(v.shape[0], dtype=np.float64)
    for y in np.unique(YR):
        r = np.where(YR == y)[0]
        out[r] = conv.cf(v[r], lm[r], lh[r])
    return out


# calibration of the full-resolution truth
TRUTH, DIAG = cv.calibrate_wind100(V[CALW], LMON[CALW], LHR[CALW], YR[CALW], OBS[CALW], CLASSES, ONEHOT["mix"],
                                   np.full(NP, 1 / NP), 0.207, sigmas=np.arange(0, 0.41, 0.04))
SIG_PT, KAPPA = TRUTH.sigma, TRUTH.kappa_mh
log(f"truth calibrated: sigma_pt {SIG_PT:.2f}, kappa {KAPPA.min():.3f}-{KAPPA.max():.3f} (night {KAPPA[:, :6].mean():.3f}, "
    f"afternoon {KAPPA[:, 13:17].mean():.3f}); 2015-2019: r {DIAG['corr']:.3f}, h/yr <5% {DIAG['h/yr <5% model']:.0f} vs "
    f"{DIAG['h/yr <5% observed']:.0f} observed")
CONV_PT = {v: cv.Wind100Conversion(CLASSES, ONEHOT[v], np.full(NP, 1 / NP), SIG_PT, KAPPA) for v in ONEHOT}

# regions
rr = np.array([int(c[1]) for c in G["cell"]])
cc = np.array([int(c[3]) for c in G["cell"]])
LEVELS = {"Germany": np.zeros(NP, int), "4 quadrants": (rr // 2) * 2 + cc // 2, "2 deg blocks": (ri // 8) * 100 + ci // 8,
          "16 cells": rr * 4 + cc, "1 deg blocks": (ri // 4) * 100 + ci // 4, "0.5 deg blocks": (ri // 2) * 100 + ci // 2}


def aggregate(level_ids, cls, with_sigma):
    """national CF from regional mean wind x regional average curve (optionally variance-aware)"""
    regs = np.unique(level_ids)
    vreg = np.stack([V[:, level_ids == r].mean(axis=1) for r in regs], axis=1)
    sh = np.stack([shares_of(cls, level_ids == r) for r in regs])
    w = np.array([(level_ids == r).mean() for r in regs])
    if not with_sigma:
        return chunked(cv.Wind100Conversion(CLASSES, sh, w, SIG_PT, KAPPA), vreg), 0.0
    out = np.zeros(NT, dtype=np.float64)
    sigmas = []
    for j, r in enumerate(regs):
        idx = np.where(level_ids == r)[0]
        s_in = float(np.log(np.maximum(V[:, idx], 0.1) / np.maximum(vreg[:, j], 0.1)[:, None]).std()) if len(idx) > 1 else 0.0
        sigmas.append(s_in)
        conv = cv.Wind100Conversion(CLASSES, sh[j:j + 1], w[j:j + 1], np.sqrt(SIG_PT ** 2 + s_in ** 2), KAPPA)
        out += chunked(conv, vreg[:, [j]])
    return out, float(np.median(sigmas))


# lengths of the runs below thr (lulls)
def runs(b, thr=0.10):
    x = np.diff(np.concatenate([[0], (b < thr).astype(np.int8), [0]]))
    return np.where(x == -1)[0] - np.where(x == 1)[0]


# statistics of an hourly capacity factor series of 25 years against the reference: energy, low hours, lulls
def metrics(cf, ref):
    y = cf.reshape(25, 8760)
    L = [runs(r) for r in y]
    return {"mean CF": cf.mean(), "h/yr CF<5%": (cf < 0.05).sum() / 25, "h/yr CF<10%": (cf < 0.10).sum() / 25,
            "lulls>=24h /yr": np.mean([(l >= 24).sum() for l in L]), "lulls>=72h /yr": np.mean([(l >= 72).sum() for l in L]),
            "longest lull h": np.mean([l.max() if len(l) else 0 for l in L]),
            "ramp p99": np.quantile(np.abs(np.diff(y, axis=1)), 0.99), "corr with truth": np.corrcoef(cf, ref)[0, 1],
            "q01": np.quantile(cf, 0.01), "q50": np.quantile(cf, 0.5), "q99": np.quantile(cf, 0.99)}


CONVNAME = "100 m"
ROWS, SERIES_CF, B = {}, {}, {}
for v in ONEHOT:
    B[v] = chunked(CONV_PT[v], V)
    ROWS[(CONVNAME, v, "points (truth)", "B")] = metrics(B[v], B[v]) | {"regions": NP}
    SERIES_CF[(CONVNAME, v, "points (truth)", "B")] = B[v].astype(np.float32)
    for lev, ids in LEVELS.items():
        for lab, ws in (("A", False), ("A_sigma", True)):
            if lab == "A_sigma" and v == "single":
                continue
            cf, sig = aggregate(ids, CLASS_OF[v], ws)
            ROWS[(CONVNAME, v, lev, lab)] = metrics(cf, B[v]) | {"regions": len(np.unique(ids)), "median sigma": sig}
            SERIES_CF[(CONVNAME, v, lev, lab)] = cf.astype(np.float32)
    log(f"fleet '{v}': truth mean CF {B[v].mean():.3f}; all levels done")
C16 = cv.calibrate_16cells()
ADOPT = chunked(C16["onshore"], cv.wind100_cells(TIME))
ROWS[(CONVNAME, "mix", "16 cells", "adopted (conversion.py)")] = metrics(ADOPT, B["mix"]) | {"regions": 16, "median sigma": C16["onshore"].sigma}
SERIES_CF[(CONVNAME, "mix", "16 cells", "adopted (conversion.py)")] = ADOPT.astype(np.float32)
T = pd.DataFrame(ROWS).T

# against observed German feed-in 2015-2019
yc = YR[CALW]
obs_c = OBS[CALW]
ok = ~np.isnan(obs_c)
obs_filled = pd.Series(obs_c).interpolate(limit_area="inside").to_numpy()


def at_level(c):
    """observed CF rescaled per year to the series' yearly mean (the OPSD capacity series lags new plants)"""
    o = obs_c.copy()
    of = obs_filled.copy()
    for y in np.unique(yc):
        m = yc == y
        o[m] *= c[m].mean() / np.nanmean(obs_c[m])
        of[m] *= c[m].mean() / np.nanmean(obs_filled[m])
    return o, of


# modelled against observed (OPSD 2015-2019): correlation, low-output hours and long lulls
def vs_obs(cf):
    c = cf[CALW]
    o, of = at_level(c)
    Lc = [runs(c[yc == y]) for y in np.unique(yc)]
    Lo = [runs(of[yc == y]) for y in np.unique(yc)]
    return {"obs corr": np.corrcoef(c[ok], o[ok])[0, 1], "obs h/yr CF<5% model": (c[ok] < 0.05).sum() / 5,
            "obs h/yr CF<5% observed": (o[ok] < 0.05).sum() / 5,
            "obs lulls>=72h model/observed": sum((l >= 72).sum() for l in Lc) / sum((l >= 72).sum() for l in Lo),
            "obs lulls>=48h model/observed": sum((l >= 48).sum() for l in Lc) / sum((l >= 48).sum() for l in Lo)}


OBSV = {k: vs_obs(s) for k, s in SERIES_CF.items() if k[1] == "mix"}
T = T.join(pd.DataFrame(OBSV).T)

# low output by local hour: model vs observed (2015-2019)
LOCAL_H = LHR[CALW]


# share of hours below 5 % and mean capacity factor by local hour, model and observed
def low_by_hour(cf):
    c = cf[CALW]
    o, _ = at_level(c)
    return pd.DataFrame({"model": [(c[ok & (LOCAL_H == h)] < 0.05).mean() for h in range(24)],
                         "observed": [(o[ok & (LOCAL_H == h)] < 0.05).mean() for h in range(24)],
                         "model mean CF": [c[ok & (LOCAL_H == h)].mean() for h in range(24)],
                         "observed mean CF": [o[ok & (LOCAL_H == h)].mean() for h in range(24)]})


DIURNAL = {k: low_by_hour(SERIES_CF[k]) for k in [(CONVNAME, "mix", "points (truth)", "B"), (CONVNAME, "mix", "16 cells", "A"),
                                                  (CONVNAME, "mix", "Germany", "A"), (CONVNAME, "mix", "16 cells", "adopted (conversion.py)")]}

# system: 2030 residual load (point 1, real 2000-2024)
S1 = pickle.load(open(f"{D}/point1_night_ev.pkl", "rb"))["series real 2000-2024"]
base = S1["demand"].astype(np.float64) - 215 * S1["pv"] - 30 * S1["offshore"]


# highest w-hour moving mean of each row (each year)
def roll_max(x, w):
    c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
    return ((c[:, w:] - c[:, :-w]) / w).max(axis=1)


THETA = np.quantile(base - 115 * B["mix"].reshape(25, 8760), 0.99)
SYS = {}
for k, sr in SERIES_CF.items():
    if k[1] != "mix":
        continue
    rl = base - 115 * sr.reshape(25, 8760)
    SYS[k] = {"peak RL (GW)": rl.max(axis=1).mean(), "worst 72 h mean RL (GW)": roll_max(rl, 72).mean(),
              "worst 7 d mean RL (GW)": roll_max(rl, 168).mean(), "h/yr RL > truth 99th pct": (rl > THETA).sum() / 25}
T = T.join(pd.DataFrame(SYS).T)

# where the gap comes from (Germany, mixed fleet)
vn = V.mean(axis=1)
bins = np.arange(0, 16.5, 0.5)
bi = np.digitize(vn, bins)
A_de = SERIES_CF[(CONVNAME, "mix", "Germany", "A")]
GAP = pd.DataFrame({"national 100 m wind": [bins[i - 1] + 0.25 for i in np.unique(bi)],
                    "share of hours": [(bi == i).mean() for i in np.unique(bi)],
                    "truth CF": [B["mix"][bi == i].mean() for i in np.unique(bi)],
                    "A CF": [A_de[bi == i].mean() for i in np.unique(bi)]})

# PV: national mean irradiance vs aggregate of the 16 cells
Wc = G.groupby("cell").size().reindex(CELLS).fillna(0).to_numpy(float)
Wc /= Wc.sum()
g = {v: pd.read_pickle(f"{D}/grid4x4_{v}_2000-2024.pkl")[[f"{c}_{v}" for c in CELLS]].to_numpy()[keep] for v in ("solar", "temp")}
ghi = np.clip(g["solar"], 0, None)


# PV capacity factor with the NOCT model, optionally clipped at the inverter (DC/AC ratio 1.3)
def pv(ghi_, tk, clip):
    dc = np.clip(ghi_ / 1000 * (1 - 0.004 * (tk - 273.15 + 0.03125 * ghi_ - 25)), 0, None)
    return np.minimum(dc, 1 / 1.3) if clip else dc


PV = {}
for clip in (False, True):
    b_ = (pv(ghi, g["temp"], clip) * Wc).sum(axis=1)
    a_ = pv((ghi * Wc).sum(axis=1), (g["temp"] * Wc).sum(axis=1), clip)
    PV["with clipping" if clip else "no clipping"] = {"aggregate of cells": b_.mean(), "national mean weather": a_.mean(),
                                                      "gap %": 100 * (a_.mean() / b_.mean() - 1),
                                                      "p99 cells": np.quantile(b_, 0.99), "p99 national": np.quantile(a_, 0.99)}

# report
pd.set_option("display.width", 280)
pd.set_option("display.max_columns", 40)
cols = ["regions", "mean CF", "h/yr CF<5%", "h/yr CF<10%", "lulls>=72h /yr", "longest lull h", "ramp p99", "corr with truth",
        "median sigma", "obs corr", "obs h/yr CF<5% model", "obs h/yr CF<5% observed", "obs lulls>=72h model/observed",
        "peak RL (GW)", "worst 72 h mean RL (GW)", "h/yr RL > truth 99th pct"]
print(f"\n=== 100 m CONVERSION (sigma_pt {SIG_PT:.2f}) ===")
print("MIXED FLEET (turbine class by local wind), 2000-2024: average weather x average curve (A), variance-aware (A_sigma), truth (B)")
print(T.loc[(CONVNAME, "mix"), [c for c in cols if c in T.columns]].round(3).to_string())
print("SINGLE TURBINE CLASS (medium everywhere): the averaging effect alone")
print(T.loc[(CONVNAME, "single"), ["regions", "mean CF", "h/yr CF<5%", "h/yr CF<10%", "lulls>=72h /yr", "longest lull h", "ramp p99",
                                   "corr with truth"]].round(3).to_string())
REL = T.loc[(CONVNAME, "mix")].copy()
b = REL.loc[("points (truth)", "B")]
REL = pd.DataFrame({"energy vs truth %": 100 * (REL["mean CF"] / b["mean CF"] - 1),
                    "hours <5% vs truth": REL["h/yr CF<5%"] / b["h/yr CF<5%"],
                    "lulls >=72 h vs truth": REL["lulls>=72h /yr"] / b["lulls>=72h /yr"],
                    "ramp p99 vs truth": REL["ramp p99"] / b["ramp p99"],
                    "peak RL - truth (GW)": REL["peak RL (GW)"] - b["peak RL (GW)"],
                    "worst 72 h - truth (GW)": REL["worst 72 h mean RL (GW)"] - b["worst 72 h mean RL (GW)"],
                    "critical h vs truth": REL["h/yr RL > truth 99th pct"] / b["h/yr RL > truth 99th pct"]})
print("\nRELATIVE TO THE TRUTH (mixed fleet)")
print(REL.round(3).to_string())
print("\nWHERE THE GAP COMES FROM (Germany as one region, mixed fleet), by national-mean 100 m wind")
print(GAP.round(3).to_string(index=False))
print("\nPV: national-mean weather vs aggregate of the 16 cells")
print(pd.DataFrame(PV).T.round(4).to_string())
print("\nLOW OUTPUT (CF < 5%) BY LOCAL HOUR, 2015-2019, share of hours: model vs observed (level-matched per year)")
for k, d in DIURNAL.items():
    print(f"  {k[2][:8] + ' ' + k[3][:8]:18} model   " + " ".join(f"{100 * x:3.0f}" for x in d["model"]))
    print(f"  {'':18} observed" + " ".join(f"{100 * x:3.0f}" for x in d["observed"]))
print(f"\n2030 critical threshold (truth 99th percentile) {THETA:.1f} GW")
with open(f"{D}/point2_aggregation.pkl", "wb") as f:
    pickle.dump({"conversion": "100 m (log 4.Q)", "table": T, "relative": REL, "gap_by_wind": GAP, "pv": PV, "diurnal": DIURNAL,
                 "sigma_pt": SIG_PT, "kappa": KAPPA, "truth diag": {k: v for k, v in DIAG.items() if not isinstance(v, pd.DataFrame)},
                 "classes": CLS, "points": G, "theta": THETA,
                 "series": {k: v for k, v in SERIES_CF.items() if k[1] == "mix" and k[2] in ("points (truth)", "Germany", "16 cells", "0.5 deg blocks")}}, f)
log(f"saved {D}/point2_aggregation.pkl")
