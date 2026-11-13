# Thesis: Case Study, Power System 2030 (capacities, demand model, residual load, critical threshold)
# and Results, Critical Hours
# Run: cd analysis && python point1_night_ev.py   (also imported by other scripts for the 2030 system)
# Writes: data/point1_*.pkl
"""The 2030 power system: demand model, electric-vehicle charging, residual load, critical hours and Dunkelflaute hours.

supply   0.5 deg units of the unified model (units05.py): wind from the ERA5 100 m wind per unit (observed years and the
         model, which simulates 100 m wind directly); per unit a mix of three turbine classes (by the site wind of its
         0.25 deg points), fleet curve with the point-level spread and the wind scale per (month, local hour) of
         point2_aggregation.py; offshore: the German offshore points (calibration of point3_value_map.py); national =
         weighted by points; PV formula of conversion.py (timing, performance ratio, orientation)
demand   least squares on the ENTSO-E load 2015-2019 (OPSD): month x day type x local hour, regional holidays x hour,
         Christmas x hour, heating and cooling degree hours of a smoothed (EWMA, 24 h half-life) national temperature,
         year levels (2019 used). Stochastic part: whole residual days resampled by month and day type, the same draws
         for every EV scenario of a year
EVs      15 million cars, 13,000 km per year, 0.18 kWh/km at >= 18 C, +1.2 % per K below; 70 % home, 15 % work
         (workdays), 15 % public charging; 50 % of the sessions at 11 kW (1.5 h), 50 % at 3.7 kW (4.25 h). Home
         charging: uncontrolled (start on arrival), timer (all start at 22:00), staggered (start 21:00-03:00), smart
         (valley filling of the residual load 18:00-07:00 with perfect foresight)
system   2030: onshore 115 GW, offshore 30 GW, PV 215 GW; variant without offshore
weather  observed 2000-2024 (25 years), observed 1950-2024 (75 years, 1950-1999 adjusted to the 2000-2024 climate),
         unified model (300 years); all on the 2025 calendar (UTC), local time Europe/Berlin
Per year: peak residual load (RL) and its local hour; critical hours = RL above the 99th percentile of the RL without
EVs in 2000-2024, by time of day; worst 72-h and 7-day mean RL; share of the RL energy at night; Dunkelflaute hours
under three definitions (wind + PV CF < 10 %, all hours or daylight only; feed-in < 25 % of the load) with their night
share and events >= 24 h.
Option --pv-lag=0: no PV timing correction (output tag _lag0). Importing the module gives the demand and EV models and
the saved threshold THETA without running the analysis.
Writes data/point1_night_ev{TAG}.pkl, including the 2030 system series of every weather set.
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import lfilter

import conversion as cv
import units05 as u05
import paths  # repository folders: data, figures, external raw data (see paths.py)

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)
D = paths.DATA
CELLS = [f"r{r}c{c}" for r in range(4) for c in range(4)]
OFF = CELLS.index("r0c0")
CAP = {"on": 115.0, "off": 30.0, "pv": 215.0}
TARGET = pd.date_range("2025-01-01", periods=8760, freq="h")
LOC = TARGET.tz_localize("UTC").tz_convert("Europe/Berlin")
LH = LOC.hour.to_numpy()
LMON = LOC.month.to_numpy()
NIGHT = np.isin(LH, [22, 23, 0, 1, 2, 3, 4, 5])
EVENING = np.isin(LH, [17, 18, 19, 20, 21])
rng = np.random.default_rng(20261013)
P = pickle.load(open(f"{D}/risk_map_025.pkl", "rb"))["points"]
W = P[P.germany].groupby("cell").size().reindex(CELLS).fillna(0).to_numpy(float)
W /= W.sum()


# 1. data for calibration and demand (Germany 2015-2019)
op = pd.read_pickle(f"{D}/opsd_germany_60min.pkl")
CAL = pd.date_range("2015-01-01", "2019-12-31 23:00", freq="h")
_tm = pickle.load(open(f"{D}/grid025de_meta_1950-2024.pkl", "rb"))["time"]
TNAT_CAL = np.asarray(np.load(f"{D}/grid025de_t2m_1950-2024.npy", mmap_mode="r")[_tm.get_indexer(CAL)], dtype=np.float64).mean(axis=1) - 273.15
# conversion of conversion.py: 100 m wind, turbine classes per cell, wind scale per (month, local hour),
# calibrated spread, offshore via cell r0c0, PV with timing, performance ratio and orientation correction
PV_LAG = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--pv-lag=")), None)    # sensitivity: --pv-lag=0
CV = cv.calibrate_16cells(pv_lag=PV_LAG)
LAG = CV["pv"].lag
u05.PVC = CV["pv"]                                               # PV timing (and the --pv-lag sensitivity)
TAG = "" if PV_LAG is None else f"_lag{LAG}"
CALIB = {k: {kk: vv for kk, vv in d.items() if not isinstance(vv, pd.DataFrame)} for k, d in CV["diag"].items()} | {
    "onshore sigma": CV["onshore"].sigma, "offshore sigma": CV["offshore"].sigma, "onshore kappa": CV["onshore"].kappa_mh,
    "offshore kappa": CV["offshore"].kappa_mh, "PV ratio": CV["pv"].ratio, "diag frames": CV["diag"]}
d_on, d_off, d_pv = CV["diag"]["onshore"], CV["diag"]["offshore"], CV["diag"]["pv"]
log(f"conversion (100 m): onshore sigma {CV['onshore'].sigma:.2f}, wind scale night {CV['onshore'].kappa_mh[:, :6].mean():.3f} / afternoon "
    f"{CV['onshore'].kappa_mh[:, 13:17].mean():.3f}, r {d_on['corr']:.3f}, h/yr < 5% {d_on['h/yr <5% model']:.0f} vs {d_on['h/yr <5% observed']:.0f}; "
    f"offshore sigma {CV['offshore'].sigma:.2f}, r {d_off['corr']:.3f}; PV lag {LAG:+d} h, PR {d_pv['PR']:.3f}, r {d_pv['corr']:.3f}")


# 2. demand model
def easter(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    return pd.Timestamp(y, (h + l_ - 7 * m + 114) // 31, (h + l_ - 7 * m + 114) % 31 + 1)


# German public holidays of year y: nationwide and regional ones
def holidays(y):
    e = easter(y)
    full = {pd.Timestamp(y, 1, 1), e - pd.Timedelta(days=2), e + pd.Timedelta(days=1), pd.Timestamp(y, 5, 1),
            e + pd.Timedelta(days=39), e + pd.Timedelta(days=50), pd.Timestamp(y, 10, 3), pd.Timestamp(y, 12, 25),
            pd.Timestamp(y, 12, 26)} | ({pd.Timestamp(2017, 10, 31)} if y == 2017 else set())
    partial = {pd.Timestamp(y, 1, 6), e + pd.Timedelta(days=60), pd.Timestamp(y, 11, 1)}      # BW/BY/NRW...
    return full, partial


# calendar of local time: month, day type (workday, Saturday, Sunday or holiday) and hour
def calendar(utc_index):
    loc = utc_index.tz_localize("UTC").tz_convert("Europe/Berlin")
    date = loc.tz_localize(None).normalize()
    full, part = set(), set()
    for y in sorted(set(date.year)):
        f, p = holidays(y)
        full |= f
        part |= p
    isfull, ispart = date.isin(list(full)), date.isin(list(part))
    dow = loc.dayofweek.to_numpy()
    daytype = np.where(isfull | (dow == 6), 2, np.where(dow == 5, 1, 0))
    xmas = ((loc.month == 12) & (loc.day >= 24)) | ((loc.month == 1) & (loc.day == 1))
    return {"hour": loc.hour.to_numpy(), "month": loc.month.to_numpy(), "daytype": daytype,
            "partial": np.asarray(ispart & (daytype == 0)), "xmas": np.asarray(xmas), "date": date}


# design matrix of the calendar terms of the demand model: month x day type x hour plus holiday terms
def cal_design(c):
    n = len(c["hour"])
    X = np.zeros((n, 864 + 48))
    X[np.arange(n), ((c["month"] - 1) * 3 + c["daytype"]) * 24 + c["hour"]] = 1
    X[np.where(c["partial"])[0], 864 + c["hour"][c["partial"]]] = 1
    X[np.where(c["xmas"])[0], 888 + c["hour"][c["xmas"]]] = 1
    return X


ALPHA = 1 - 0.5 ** (1 / 24)


def smooth(t):
    """EWMA along the last axis, half-life 24 h, started at the first value"""
    t = np.atleast_2d(t)
    y, _ = lfilter([ALPHA], [1, -(1 - ALPHA)], t, axis=1, zi=(1 - ALPHA) * t[:, :1])
    return y


# heating and cooling degrees: below 15 C and above 22 C
def degree(ts):
    return np.maximum(15 - ts, 0), np.maximum(ts - 22, 0)


load = op.loc[CAL, "DE_load_actual_entsoe_transparency"].interpolate(limit=3) / 1000
tnat_cal = TNAT_CAL                                                   # mean over the German land points
ccal = calendar(CAL)
hdd, cdd = degree(smooth(tnat_cal)[0])
yrs = CAL.year.to_numpy()
Xd = np.hstack([cal_design(ccal), hdd[:, None], cdd[:, None], np.stack([(yrs == y) * 1.0 for y in range(2016, 2020)], axis=1)])
ok = load.notna().to_numpy()
beta, *_ = np.linalg.lstsq(Xd[ok], load.to_numpy()[ok], rcond=None)
fit = Xd @ beta
resid = load.to_numpy() - fit
r2 = 1 - np.nanvar(resid) / np.nanvar(load.to_numpy())
B_H, B_C, LEVEL2019 = beta[912], beta[913], beta[-1]
log(f"demand model: R2 {r2:.3f}, residual SD {np.nanstd(resid):.2f} GW, heating {B_H:.3f} GW/K, cooling {B_C:.3f} GW/K")
# residual days (UTC days) by month and day type of the local date at noon
rday = pd.DataFrame(resid.reshape(-1, 24))
rkey = [(d.month, ccal["daytype"][i * 24 + 12]) for i, d in enumerate(CAL[::24])]
POOL = {}
for i, k in enumerate(rkey):
    if not np.isnan(rday.iloc[i]).any():
        POOL.setdefault(k, []).append(i)
RDAYS = rday.to_numpy()
CT = calendar(TARGET)
BASE_T = cal_design(CT) @ beta[:912] + LEVEL2019
TKEY = [(d.month, CT["daytype"][i * 24 + 12]) for i, d in enumerate(TARGET[::24])]


def demand(tnat, seed):
    """tnat (units, 8760) C -> demand (units, 8760) GW, 2019 level, residual days resampled"""
    g = np.random.default_rng(seed)
    h, c = degree(smooth(tnat))
    pick = np.stack([g.choice(POOL[k], size=tnat.shape[0]) for k in TKEY], axis=1)        # (units, 365)
    return BASE_T[None] + B_H * h + B_C * c + RDAYS[pick].reshape(tnat.shape[0], -1)


# 3. electric vehicles
N_EV, KM, KWH = 15e6, 13000 / 365, 0.18
MID = pd.date_range("2025-01-01", periods=366, freq="D", tz="Europe/Berlin").tz_convert("UTC").tz_localize(None)
POS = ((MID - TARGET[0]) / pd.Timedelta(hours=1)).to_numpy().astype(int)                   # UTC index of local midnights
LDATE = pd.date_range("2025-01-01", periods=365, freq="D")
cl = calendar(pd.DatetimeIndex(LDATE + pd.Timedelta(hours=11)))                            # local noon -> day type
WORKDAY = (cl["daytype"] == 0)
WORKDAY_H = CT["daytype"] == 0                                                           # per UTC hour of 2025
KMDAY = KM * np.where(WORKDAY, 1.05, 0.89) / np.mean(np.where(WORKDAY, 1.05, 0.89))
Q = np.arange(48 * 4) / 4 + 0.125                                                          # 15-min bins over 48 h


# normal density on a grid, truncated to [lo, hi) and normalized (EV arrival and departure times)
def pdf_normal(mu, sd, lo, hi):
    p = np.exp(-0.5 * ((Q - mu) / sd) ** 2) * ((Q >= lo) & (Q < hi))
    return p / p.sum()


def kernel(start_pdf):
    """energy share per hour after local midnight (48 h) for a start-time density; 50% of sessions at
    11 kW (1.5 h), 50% at 3.7 kW (4.25 h)"""
    k = 0.5 * np.convolve(start_pdf, np.ones(6) / 6)[:192] + 0.5 * np.convolve(start_pdf, np.ones(17) / 17)[:192]
    return k.reshape(48, 4).sum(axis=1)


K = {"home uncontrolled work": kernel(0.75 * pdf_normal(17.5, 1.5, 6, 30) + 0.25 * pdf_normal(13, 3, 6, 30)),
     "home uncontrolled free": kernel(pdf_normal(15.5, 3.5, 6, 30)),
     "home timer": kernel((np.abs(Q - 22.125) < 0.01) * 1.0),
     "home staggered": kernel(((Q >= 21) & (Q < 27)) / ((Q >= 21) & (Q < 27)).sum()),
     "work": kernel(pdf_normal(8.5, 1.0, 6, 12)),
     "public": kernel(pdf_normal(13, 3.5, 6, 22))}


def spread(E, Kmat):
    """E (units, 365) GWh per local day, Kmat (365, 48) -> load (units, 8760) GW"""
    out = np.zeros((E.shape[0], 8760))
    for tau in range(48):
        idx = POS[:365] + tau
        ok = (idx >= 0) & (idx < 8760)
        out[:, idx[ok]] += E[:, ok] * Kmat[ok, tau][None]
    return out


def ev_energy(tnat):
    """GWh per local day: (home, work, public), colder days need more energy"""
    td = np.stack([tnat[:, max(POS[d], 0):POS[d + 1]].mean(axis=1) for d in range(365)], axis=1)
    e = N_EV * KMDAY[None] * KWH * (1 + 0.012 * np.maximum(18 - td, 0)) / 1e6
    return 0.70 * e, np.where(WORKDAY, 0.15, 0.0) * e, np.where(WORKDAY, 0.15, 0.30) * e


def smart(Eh, base, cap=0.3 * N_EV * 11e-6):
    """valley filling 18:00-07:00 local: per night the charge c_t = min(cap, max(0, L - base_t)) with
    sum c_t = E, i.e. the schedule that minimises the night's maximum (perfect foresight)"""
    idx = POS[:365, None] + np.arange(18, 31)[None]                                         # (365, 13)
    valid = idx < 8760
    b = np.where(valid[None], base[:, np.minimum(idx, 8759)], np.inf)                       # (units, 365, 13)
    lo = np.min(b, axis=2) - 1
    hi = np.max(np.where(np.isfinite(b), b, -np.inf), axis=2) + Eh                         # enough for any night
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        f = np.clip(mid[..., None] - b, 0, cap).sum(axis=2)
        lo, hi = np.where(f < Eh, mid, lo), np.where(f < Eh, hi, mid)
    c = np.clip(hi[..., None] - b, 0, cap)
    out = np.zeros_like(base)
    for k in range(13):
        ok = valid[:, k]
        out[:, idx[ok, k]] += c[:, ok, k]
    return out


# 4. weather sets
MON, HR = TARGET.month.to_numpy(), TARGET.hour.to_numpy()


try:                                                                  # importers (point 5) get the saved threshold
    THETA = pickle.load(open(f"{D}/point1_night_ev{TAG}.pkl", "rb"))["theta"]
except FileNotFoundError:
    THETA = {}
if __name__ == "__main__":                                            # the analysis (~14 min); not run on import
    _R = u05.real_units(adjust=True)                                      # 1950-2024, 1950-99 adjusted to 2000-24
    IN = {v: _R[v][50:] for v in ("wind", "temp", "solar")}
    OUT = {v: _R[v][:50] for v in ("wind", "temp", "solar")}
    SETS = {"real 2000-2024": IN, "real 1950-2024": {v: np.concatenate([IN[v], OUT[v]]) for v in IN},
            "model": dict(u05.model_units(), temp_in_c=True)}
    del OUT, _R
    DAYLIGHT = np.roll((IN["solar"] * u05.W_LAND).sum(axis=2).mean(axis=0) > 1, LAG)                    # climatological daylight
    log("weather sets loaded (0.5 deg units): " + ", ".join(f"{k} {v['wind'].shape[0]} yr" for k, v in SETS.items()))


    # 5. system and metrics
    def system(X):
        """national temperature (C), onshore, offshore and PV CF (units, 8760) from the 0.5 deg units"""
        F = u05.feedin({k: X[k] for k in ("wind", "temp", "solar")}, temp_in_c=X.get("temp_in_c", False))
        return F["tnat"].astype(np.float64), F["on"].astype(np.float64), F["off"].astype(np.float64), F["pv"].astype(np.float64)


    # lengths of the runs where mask is true
    def runs(mask):
        x = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
        return np.where(x == -1)[0] - np.where(x == 1)[0]


    # highest w-hour moving mean of each row (each year)
    def roll_max(x, w):
        c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
        return ((c[:, w:] - c[:, :-w]) / w).max(axis=1)


    # per-year system metrics: peak residual load, critical hours above theta, Dunkelflaute hours and their timing
    def metrics(rl, re, ld, re_cf, theta):
        crit = rl > theta
        df_cf, df_cov = re_cf < 0.10, re / ld < 0.25
        out = {"peak RL (GW)": rl.max(axis=1), "peak hour local": LH[rl.argmax(axis=1)],
               "critical h/yr": crit.sum(axis=1),
               "critical: evening share": (crit & EVENING).sum(axis=1) / np.maximum(crit.sum(axis=1), 1),
               "critical: night share": (crit & NIGHT).sum(axis=1) / np.maximum(crit.sum(axis=1), 1),
               "worst 72 h mean RL (GW)": roll_max(rl, 72), "worst 7 d mean RL (GW)": roll_max(rl, 168),
               "RL energy at night (share)": (np.maximum(rl, 0) * NIGHT).sum(axis=1) / np.maximum(rl, 0).sum(axis=1)}
        # alternative conventions: top 1 % of the hours of each year, night 20-06 h, feed-in < 20 %
        n2006 = np.isin(LH, [20, 21, 22, 23, 0, 1, 2, 3, 4, 5])
        top = rl >= np.quantile(rl, 0.99, axis=1, keepdims=True)
        out["cmp: top-1% share 20-06"] = (top & n2006).sum(axis=1) / top.sum(axis=1)
        out["cmp: RE<20% of load h/yr"] = (re / ld < 0.20).sum(axis=1)
        out["cmp: RE<20% share 20-06"] = ((re / ld < 0.20) & n2006).sum(axis=1) / np.maximum((re / ld < 0.20).sum(axis=1), 1)
        for lab, m in (("DF CF<10% all hours", df_cf), ("DF CF<10% daylight", df_cf & DAYLIGHT[None]), ("DF RE<25% of load", df_cov)):
            out[f"{lab}: h/yr"] = m.sum(axis=1)
            out[f"{lab}: night share"] = (m & NIGHT).sum(axis=1) / np.maximum(m.sum(axis=1), 1)
            out[f"{lab}: events>=24h"] = np.array([(runs(r) >= 24).sum() for r in m])
        return pd.DataFrame(out)


    EV_SCEN = ["no EV", "uncontrolled", "timer 22:00", "staggered night", "smart (valley filling)"]
    RES, HOURLY, THETA, SERIES_ALL = {}, {}, {}, {}
    for sup in ("with offshore", "without offshore"):
        for sname, X in SETS.items():
            tnat, on, off, pv = system(X)
            re = CAP["on"] * on + CAP["pv"] * pv + (CAP["off"] * off if sup == "with offshore" else 0)
            re_cf = re / (CAP["on"] + CAP["pv"] + (CAP["off"] if sup == "with offshore" else 0))
            dem = demand(tnat, seed={"real 2000-2024": 1, "real 1950-2024": 2, "model": 3}[sname])
            eh, ew, ep = ev_energy(tnat)
            work_pub = spread(ew, np.tile(K["work"], (365, 1))) + spread(ep, np.tile(K["public"], (365, 1)))
            Kh = np.where(WORKDAY[:, None], K["home uncontrolled work"][None], K["home uncontrolled free"][None])
            evs = {"no EV": 0.0, "uncontrolled": spread(eh, Kh) + work_pub,
                   "timer 22:00": spread(eh, np.tile(K["home timer"], (365, 1))) + work_pub,
                   "staggered night": spread(eh, np.tile(K["home staggered"], (365, 1))) + work_pub}
            evs["smart (valley filling)"] = smart(eh, dem + work_pub - re) + work_pub
            if sname == "real 2000-2024":
                THETA[sup] = np.quantile(dem - re, 0.99)
                if sup == "with offshore":                                                  # for point 2
                    SERIES = {"demand": dem.astype(np.float32), "pv": pv.astype(np.float32), "offshore": off.astype(np.float32),
                              "onshore 16 cells": on.astype(np.float32), "tnat": tnat.astype(np.float32)}
            if sup == "with offshore":                                                      # for point 3: every weather set
                SERIES_ALL[sname] = {"demand": dem.astype(np.float32), "pv": pv.astype(np.float32), "offshore": off.astype(np.float32),
                                     "onshore 16 cells": on.astype(np.float32), "tnat": tnat.astype(np.float32),
                                     "ev uncontrolled": evs["uncontrolled"].astype(np.float32),
                                     "ev smart": evs["smart (valley filling)"].astype(np.float32)}
            for scen in EV_SCEN:
                ld = dem + evs[scen]
                rl = ld - re
                RES[(sup, sname, scen)] = metrics(rl, re, ld, re_cf, THETA[sup])
                crit = rl > THETA[sup]
                HOURLY[(sup, sname, scen)] = np.bincount(np.broadcast_to(LH, crit.shape)[crit], minlength=24) / max(crit.sum(), 1)
            ld0 = dem
            for lab, m in (("DF CF<10% all hours", re_cf < 0.10), ("DF CF<10% daylight", (re_cf < 0.10) & DAYLIGHT[None]),
                           ("DF RE<25% of load", re / ld0 < 0.25)):
                HOURLY[(sup, sname, lab)] = np.bincount(np.broadcast_to(LH, m.shape)[m], minlength=24) / max(m.sum(), 1)
            for scen in EV_SCEN[1:]:
                ev = evs[scen] if not np.isscalar(evs[scen]) else np.zeros_like(dem)
                win = np.isin(LMON, [12, 1, 2])[None] & WORKDAY_H[None]
                HOURLY[(sup, sname, f"EV load {scen}")] = np.array([ev[:, win[0] & (LH == h)].mean() for h in range(24)])
            HOURLY[(sup, sname, "EV load profile, uncontrolled")] = np.bincount(LH, weights=(evs["uncontrolled"]).mean(axis=0), minlength=24) / np.bincount(LH, minlength=24)
            log(f"{sup}, {sname}: {X['wind'].shape[0]} years, EV energy {(eh + ew + ep).sum(axis=1).mean() / 1000:.1f} TWh/yr, "
                f"demand {dem.sum(axis=1).mean() / 1000:.0f} TWh/yr, RE {re.sum(axis=1).mean() / 1000:.0f} TWh/yr")
            del tnat, on, off, pv, re, dem, evs, work_pub

    # 6. report
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    COLS = ["peak RL (GW)", "critical h/yr", "critical: evening share", "critical: night share", "worst 72 h mean RL (GW)",
            "worst 7 d mean RL (GW)", "RL energy at night (share)"]
    for sup in ("with offshore", "without offshore"):
        print(f"\n=== 2030 system {sup}; critical = RL > {THETA[sup]:.1f} GW (no-EV 99th percentile, 2000-2024) ===")
        T = pd.DataFrame({(sname, scen): RES[(sup, sname, scen)][COLS].mean() for sname in SETS for scen in EV_SCEN}).T
        print(T.round(3).to_string())
    print("\n=== Dunkelflaute definitions (2030 with offshore, no EV / uncontrolled): hours per year, night share, events >= 24 h ===")
    DFC = [c for c in RES[("with offshore", "real 2000-2024", "no EV")].columns if c.startswith("DF ")]
    print(pd.DataFrame({(s, e): RES[("with offshore", s, e)][DFC].mean() for s in SETS for e in ("no EV", "uncontrolled")}).round(3).to_string())
    print("\n=== model vs real 1950-2024, 2030 with offshore: z = gap / combined SE ===")
    for scen in ("no EV", "uncontrolled", "staggered night", "smart (valley filling)"):
        r, m = RES[("with offshore", "real 1950-2024", scen)], RES[("with offshore", "model", scen)]
        print(f"  {scen:24} " + "  ".join(
            f"{c.split(' (')[0]} {m[c].mean():.2f} vs {r[c].mean():.2f} (z {(m[c].mean() - r[c].mean()) / np.sqrt(r[c].var() / len(r) + m[c].var() / len(m)):+.1f})"
            for c in ("peak RL (GW)", "worst 72 h mean RL (GW)", "critical: night share", "critical: evening share")))
    print("\n=== local hour of critical hours, real 2000-2024 with offshore (% per hour) ===")
    for scen in EV_SCEN:
        print(f"  {scen:24} " + " ".join(f"{100 * x:3.0f}" for x in HOURLY[("with offshore", "real 2000-2024", scen)]))
    if __name__ == "__main__":                                                     # importable (point 5.3) without rewriting the output
        with open(f"{D}/point1_night_ev{TAG}.pkl", "wb") as f:
            pickle.dump({"res": RES, "hourly": HOURLY, "theta": THETA, "calibration": CALIB, "demand_r2": r2, "series real 2000-2024": SERIES,
                         "series": SERIES_ALL,
                         "heating GW/K": B_H, "kernels": K}, f)
        log(f"saved {D}/point1_night_ev{TAG}.pkl")
