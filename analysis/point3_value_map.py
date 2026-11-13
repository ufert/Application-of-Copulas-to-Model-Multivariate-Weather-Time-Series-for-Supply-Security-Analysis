# Thesis: Results, Potential and Risk Areas (wind: firmness of 734 onshore and 87 offshore sites)
# Run: cd analysis && python point3_value_map.py   (~3 min, ~4 GB memory)
# Writes: data/point3_value_map.pkl
"""Potential and risk areas for wind: where does an additional MW of wind contribute most (potential area) or least (risk
area) to the security of supply of the 2030 system?

system     2030 with offshore (series of point1_night_ev.py): demand without EVs (sensitivity: smart and uncontrolled
           charging), 215 GW PV, 30 GW offshore, 115 GW onshore; residual load RL = demand - feed-in
stress     per year: "peak" = the 88 hours with the highest RL (top 1 %), "72 h" = the worst 72-hour mean RL window,
           "7 d" = the worst 168-hour window (main definition: 72 h)
sites      734 German land points at 0.25 deg (turbine class by site wind; point-level 100 m conversion of
           point2_aggregation.py: wind scale per (month, local hour), spread 0.36) and 87 offshore points of the German
           offshore zones inside the study box (PECD mask >= 0.5; offshore class 350 W/m2 at 120 m; one wind scale and a
           spread fitted to OPSD offshore feed-in 2015-2019 at point level); 75 observed years 1950-2024, ERA5 100 m wind
value      s_x,y = CF of site x averaged over the stress hours of year y; f_y = the same for an average MW of a uniform
           onshore fleet (mean over the 734 points). Firmness r_x = sum_y s_x,y / sum_y f_y: r > 1 means that a MW at x
           produces more than an average MW when the system is short. Also the resource m_x (mean CF) and the kept
           share s_x / m_x
designation potential area: r >= 1.10 and the 90 % bootstrap interval over years above 1; risk area: r <= 0.90 and the
           interval below 1; otherwise average
robustness stress definitions; 1950-1999 against 2000-2024; EV charging; the geography of the fleet that defines the
           stress (16-cell fleet, uniform points, resource-weighted, strong-wind sites only)
system effect  move 10 % of the onshore capacity (11.5 GW) from risk to potential areas, or from the poorest to the
           richest resource sites; designated on one period and evaluated on the other (out of sample)
model      the 203 land units of the unified model: firmness in the 75 observed years and in the 300 simulated years,
           each with its own residual load
Writes data/point3_value_map.pkl (about 4 GB of memory needed).
"""
import pickle
import time

import numpy as np
import pandas as pd

import conversion as cv
import pecd

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)
D = cv.D
CLASSES = cv.CLASSES
YEARS = np.arange(1950, 2025)
P1_YEARS = list(range(2000, 2025)) + list(range(1950, 2000))          # order of the real 1950-2024 set in point 1
P1_ORDER = [P1_YEARS.index(y) for y in YEARS]
DEFS = {"peak": None, "72 h": 72, "7 d": 168}
CAP = {"on": 115.0, "off": 30.0, "pv": 215.0}
rng = np.random.default_rng(3)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

# sites
meta = pickle.load(open(f"{D}/grid025_meta.pkl", "rb"))
LAT, LON = meta["lat"], meta["lon"]
PC = cv.point_classes()
ri_on, ci_on, CLS = PC["ri"], PC["ci"], PC["class"]
offm = pecd.zone_weights("offshore")["full"].sum(axis=0)
landm = pecd.zone_weights("onshore")["full"].sum(axis=0)
ri_off, ci_off = np.where((offm >= 0.5) & (landm < 0.1))
NON, NOFF = len(ri_on), len(ri_off)
SITES = pd.DataFrame({"lat": np.r_[LAT[ri_on], LAT[ri_off]], "lon": np.r_[LON[ci_on], LON[ci_off]],
                      "type": ["onshore"] * NON + ["offshore"] * NOFF, "class": list(CLS) + ["offshore"] * NOFF,
                      "cell": list(PC["points"]["cell"]) + [None] * NOFF})
log(f"{NON} onshore and {NOFF} offshore sites")

# wind 1950-2024 at the sites, per year (Feb 29 dropped)
BOX = {"2000-2024": (np.load(f"{D}/grid025_ws100_2000-2024.npy", mmap_mode="r"), meta["time"]),
       "1950-1999": (np.load(f"{D}/grid025_ws100_1950-1999.npy", mmap_mode="r"),
                     pickle.load(open(f"{D}/grid025_meta_1950-1999.pkl", "rb"))["time"])}


# 0.25 deg 100 m wind of all points in one year (29 February dropped)
def year_wind(y):
    arr, t = BOX["2000-2024" if y >= 2000 else "1950-1999"]
    rows = np.where((t.year == y) & ~((t.month == 2) & (t.day == 29)))[0]
    x = np.asarray(arr[rows[0]:rows[-1] + 1], dtype=np.float32)
    x = x[np.isin(np.arange(rows[0], rows[-1] + 1), rows)]
    lm, lh = cv.local_time(t[rows])
    return x[:, ri_on, ci_on], x[:, ri_off, ci_off], lm, lh


# conversions: onshore = point 2's full-resolution calibration; offshore calibrated here at point level
P2 = pickle.load(open(f"{D}/point2_aggregation.pkl", "rb"))
ONEHOT = np.stack([(CLS == k).astype(float) for k in CLASSES], axis=1)
CONV_ON = cv.Wind100Conversion(CLASSES, ONEHOT, np.full(NON, 1 / NON), P2["sigma_pt"], P2["kappa"])
cal = [year_wind(y) for y in range(2015, 2020)]
v_off = np.concatenate([c[1] for c in cal])
lm_c, lh_c = np.concatenate([c[2] for c in cal]), np.concatenate([c[3] for c in cal])
yrs_c = np.repeat(np.arange(2015, 2020), 8760)
idx_c = pd.DatetimeIndex(np.concatenate([BOX["2000-2024"][1][(BOX["2000-2024"][1].year == y) & ~((BOX["2000-2024"][1].month == 2)
                                                                                                  & (BOX["2000-2024"][1].day == 29))] for y in range(2015, 2020)]))
CONV_OFF, DOFF = cv.calibrate_wind100(v_off, lm_c, lh_c, yrs_c, cv.opsd(idx_c, "wind_offshore"), cv.OFFSHORE, np.ones((NOFF, 1)),
                                      np.full(NOFF, 1 / NOFF), 0.40, diurnal=False)
log(f"offshore sites calibrated: sigma {CONV_OFF.sigma:.2f}, scale {CONV_OFF.kappa_mh[0, 0]:.3f}, r {DOFF['corr']:.3f} with OPSD offshore; "
    f"onshore sites: sigma {P2['sigma_pt']:.2f}, scale {P2['kappa'].min():.3f}-{P2['kappa'].max():.3f}")
del cal, v_off

CF = np.empty((len(YEARS), 8760, NON + NOFF), dtype=np.float16)
for i, y in enumerate(YEARS):
    von, voff, lm, lh = year_wind(y)
    CF[i, :, :NON] = CONV_ON.regional(von, lm, lh)
    CF[i, :, NON:] = CONV_OFF.regional(voff, lm, lh)
log(f"site capacity factors 1950-2024 done: onshore mean {CF[:, :, :NON].mean(dtype=np.float64):.3f}, "
    f"offshore {CF[:, :, NON:].mean(dtype=np.float64):.3f}")
M = CF.mean(axis=1, dtype=np.float64)                                   # (years, sites) annual mean CF
UNIF = CF[:, :, :NON].mean(axis=2, dtype=np.float64)                    # (years, 8760) uniform onshore fleet

# system: 2030 residual load per real year
S1 = pickle.load(open(f"{D}/point1_night_ev.pkl", "rb"))["series"]
THETA1 = pickle.load(open(f"{D}/point1_night_ev.pkl", "rb"))["theta"]["with offshore"]       # critical threshold of point 1


# system series of point 1 (demand, feed-in, residual load) for one scenario, optionally reordered
def sysser(name, order=None):
    s = S1[name]
    pick = (lambda a: a[order]) if order is not None else (lambda a: a)
    return {k: pick(v).astype(np.float64) for k, v in s.items()}


R = sysser("real 1950-2024", P1_ORDER)
BASE = R["demand"] - CAP["pv"] * R["pv"] - CAP["off"] * R["offshore"]          # demand minus PV and offshore
RL = {"16-cell fleet (base)": BASE - CAP["on"] * R["onshore 16 cells"]}
RL["smart charging"] = RL["16-cell fleet (base)"] + R["ev smart"]
RL["uncontrolled charging"] = RL["16-cell fleet (base)"] + R["ev uncontrolled"]
mbar = M[:, :NON].mean(axis=0)
w_res = mbar ** 2 / (mbar ** 2).sum()
w_strong = (CLS == "strong-wind") / (CLS == "strong-wind").sum()
RL["uniform points"] = BASE - CAP["on"] * UNIF
RL["resource-weighted fleet"] = BASE - CAP["on"] * np.einsum("yhs,s->yh", CF[:, :, :NON], w_res, dtype=np.float64)
RL["strong-wind sites only"] = BASE - CAP["on"] * np.einsum("yhs,s->yh", CF[:, :, :NON], w_strong, dtype=np.float64)
log("2030 residual load per year: " + ", ".join(f"{k} peak {v.max(axis=1).mean():.1f} GW" for k, v in RL.items()))


def stress_mask(rl, d):
    """(years, 8760) boolean: stress hours of each year for definition d"""
    out = np.zeros(rl.shape, dtype=bool)
    for i, r in enumerate(rl):
        if DEFS[d] is None:
            out[i, np.argsort(r)[-88:]] = True
        else:
            w = DEFS[d]
            c = np.concatenate([[0.0], np.cumsum(r)])
            j = int(np.argmax(c[w:] - c[:-w]))
            out[i, j:j + w] = True
    return out


def stress_cf(mask):
    """s (years, sites): mean CF of every site in the stress hours; f (years,): the uniform onshore fleet"""
    s = np.stack([CF[i][mask[i]].mean(axis=0, dtype=np.float64) for i in range(len(YEARS))])
    return s, s[:, :NON].mean(axis=1)


# firmness of every site (Definition of firmness) with a bootstrap interval over the years
def firmness(s, f, years_sel=None, boot=1000):
    sel = np.arange(len(YEARS)) if years_sel is None else np.where(years_sel)[0]
    r = s[sel].sum(axis=0) / f[sel].sum()
    b = rng.integers(0, len(sel), (boot, len(sel)))
    rb = s[sel][b].sum(axis=1) / f[sel][b].sum(axis=1)[:, None]
    return r, np.quantile(rb, 0.05, axis=0), np.quantile(rb, 0.95, axis=0)


# potential area if firmness >= 1.10 and the interval lies above one, risk area if <= 0.90 and below one
def designate(r, lo, hi):
    return np.where((r >= 1.10) & (lo > 1), "potential", np.where((r <= 0.90) & (hi < 1), "risk", "average"))


STATS = {}
for rlname in RL:
    for d in DEFS:
        if rlname != "16-cell fleet (base)" and d != "72 h":
            continue
        mask = stress_mask(RL[rlname], d)
        s, f = stress_cf(mask)
        STATS[(rlname, d)] = {"s": s, "f": f, "mask": mask}
log("stress statistics done")

# main table: per site
B = STATS[("16-cell fleet (base)", "72 h")]
for d in DEFS:
    st = STATS[("16-cell fleet (base)", d)]
    r, lo, hi = firmness(st["s"], st["f"])
    SITES[f"r {d}"], SITES[f"r {d} lo"], SITES[f"r {d} hi"] = r, lo, hi
    SITES[f"class {d}"] = designate(r, lo, hi)
    SITES[f"kept {d}"] = st["s"].mean(axis=0) / M.mean(axis=0)
SITES["mean CF"] = M.mean(axis=0)
SITES["stress CF 72 h"] = B["s"].mean(axis=0)
for per, sel in (("1950-1999", YEARS < 2000), ("2000-2024", YEARS >= 2000)):
    r, lo, hi = firmness(B["s"], B["f"], sel)
    SITES[f"r 72 h {per}"], SITES[f"class 72 h {per}"] = r, designate(r, lo, hi)
for rlname in RL:
    if rlname == "16-cell fleet (base)":
        continue
    st = STATS[(rlname, "72 h")]
    r, lo, hi = firmness(st["s"], st["f"])
    SITES[f"r 72 h | {rlname}"], SITES[f"class 72 h | {rlname}"] = r, designate(r, lo, hi)

on = SITES["type"] == "onshore"
print("\nFLEET IN STRESS (uniform onshore fleet, mean CF in the stress hours / annual mean):")
for d in DEFS:
    st = STATS[("16-cell fleet (base)", d)]
    print(f"  {d:5}: fleet CF {st['f'].mean():.3f} vs annual {M[:, :NON].mean():.3f} (kept {st['f'].mean() / M[:, :NON].mean():.2f}); "
          f"offshore sites {st['s'][:, NON:].mean():.3f} (kept {st['s'][:, NON:].mean() / M[:, NON:].mean():.2f})")
print("\nFIRMNESS r (CF in stress / CF of an average onshore MW in stress), onshore sites: distribution and designations")
rows = {}
for d in DEFS:
    r = SITES.loc[on, f"r {d}"]
    rows[d] = {"min": r.min(), "p10": r.quantile(0.1), "median": r.median(), "p90": r.quantile(0.9), "max": r.max(),
               "potential": int((SITES.loc[on, f"class {d}"] == "potential").sum()), "risk": int((SITES.loc[on, f"class {d}"] == "risk").sum()),
               "offshore median r": SITES.loc[~on, f"r {d}"].median(),
               "corr(r, mean CF)": np.corrcoef(r, SITES.loc[on, "mean CF"])[0, 1],
               "corr(r, kept)": np.corrcoef(r, SITES.loc[on, f"kept {d}"])[0, 1]}
print(pd.DataFrame(rows).T.round(3).to_string())

# the stress definition of risk_map_025.py on these data (for point3_compare.py): wind-only lulls, the 5 % of hours
# (2000-2024, pooled) with the lowest centred 72-h mean CF of a uniform onshore fleet, in every season
sel00 = YEARS >= 2000
flat = UNIF[sel00].ravel()
c72 = pd.Series(flat).rolling(72, center=True, min_periods=36).mean().to_numpy()
lullmask = c72 <= np.quantile(c72, 0.05)
cf00 = CF[sel00].reshape(-1, NON + NOFF)
SITES["wind-lull CF (earlier definition)"] = cf00[lullmask].mean(axis=0, dtype=np.float64)
SITES["wind-lull r (earlier definition)"] = SITES["wind-lull CF (earlier definition)"] / cf00[lullmask][:, :NON].mean(dtype=np.float64)
LULL_MONTHS = np.bincount(np.tile(pd.date_range("2025-01-01", periods=8760, freq="h").month.to_numpy(), int(sel00.sum()))[lullmask], minlength=13)[1:]
print("\nEARLIER STRESS DEFINITION on the fresh data (wind-only lulls, lowest 5% of the 72-h mean fleet CF, 2000-2024):")
print("  months of the lull hours: " + ", ".join(f"{m}: {n}" for m, n in zip(range(1, 13), LULL_MONTHS)))
print(f"  Spearman with the fresh firmness (72 h): {SITES.loc[SITES.type == 'onshore', 'wind-lull r (earlier definition)'].rank().corr(SITES.loc[SITES.type == 'onshore', 'r 72 h'].rank()):.3f}")
del cf00

# robustness
def agree(a, b):
    m = on.to_numpy()
    ca, cb = SITES.loc[m, a], SITES.loc[m, b]
    return {"spearman r": SITES.loc[m, a.replace("class", "r")].rank().corr(SITES.loc[m, b.replace("class", "r")].rank()),
            "same class": (ca == cb).mean(), "potential kept": ((ca == "potential") & (cb == "potential")).sum() / max((ca == "potential").sum(), 1),
            "risk kept": ((ca == "risk") & (cb == "risk")).sum() / max((ca == "risk").sum(), 1),
            "opposite": (((ca == "potential") & (cb == "risk")) | ((ca == "risk") & (cb == "potential"))).mean()}


ROB = {"peak vs 72 h": agree("class 72 h", "class peak"), "7 d vs 72 h": agree("class 72 h", "class 7 d"),
       "1950-1999 vs 2000-2024": agree("class 72 h 2000-2024", "class 72 h 1950-1999")}
for rlname in RL:
    if rlname != "16-cell fleet (base)":
        ROB[f"stress from: {rlname}"] = agree("class 72 h", f"class 72 h | {rlname}")
ROB = pd.DataFrame(ROB).T
print("\nROBUSTNESS of the designation (onshore sites; base = 72 h, 75 years, 16-cell fleet, no EVs)")
print(ROB.round(3).to_string())

# system effect: relocate 10% of the onshore capacity, out of sample
def rl_with(w, sel):
    fleet = np.einsum("yhs,s->yh", CF[sel][:, :, :NON], w, dtype=np.float64)
    return BASE[sel] - CAP["on"] * fleet, fleet


# highest w-hour moving mean of each row (each year)
def roll_max(x, w):
    c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
    return ((c[:, w:] - c[:, :-w]) / w).max(axis=1)


RELOC = {}
for fit_p, ev_p in (("2000-2024", "1950-1999"), ("1950-1999", "2000-2024")):
    sel_fit = (YEARS >= 2000) if fit_p == "2000-2024" else (YEARS < 2000)
    sel_ev = ~sel_fit
    r, lo, hi = firmness(B["s"][:, :NON], B["f"], sel_fit)
    cls = designate(r, lo, hi)
    w0 = np.full(NON, 1 / NON)
    rl0, f0 = rl_with(w0, sel_ev)
    n_move = 0.10
    variants = {}
    if (cls == "potential").any() and (cls == "risk").any():
        variants["risk -> potential areas"] = (cls == "risk", cls == "potential")
    k = max(int((cls == "risk").sum()), int(0.2 * NON))
    order_r = np.argsort(r)
    variants["lowest -> highest firmness (same number of sites)"] = (np.isin(np.arange(NON), order_r[:k]), np.isin(np.arange(NON), order_r[-k:]))
    mfit = M[sel_fit, :NON].mean(axis=0)
    order_m = np.argsort(mfit)
    variants["poorest -> richest resource (energy view)"] = (np.isin(np.arange(NON), order_m[:k]), np.isin(np.arange(NON), order_m[-k:]))
    for vname, (src, dst) in variants.items():
        w = w0.copy()
        take = n_move * w[src] / w[src].sum()
        w[src] -= take
        w[dst] += n_move * np.ones(dst.sum()) / dst.sum()
        rl1, f1 = rl_with(w, sel_ev)
        p0, p1 = roll_max(rl0, 72), roll_max(rl1, 72)
        RELOC[(f"designated {fit_p}, evaluated {ev_p}", vname)] = {
            "sites from / to": f"{int(src.sum())} / {int(dst.sum())}",
            "energy (mean CF) %": 100 * (f1.mean() / f0.mean() - 1),
            "peak RL (GW)": (rl1.max(axis=1) - rl0.max(axis=1)).mean(),
            "worst 72 h (GW)": (p1 - p0).mean(), "1-in-10-yr worst 72 h (GW)": np.quantile(p1, 0.9) - np.quantile(p0, 0.9),
            "worst 7 d (GW)": (roll_max(rl1, 168) - roll_max(rl0, 168)).mean(),
            "RL energy above 57 GW (TWh/yr)": (np.maximum(rl1 - THETA1, 0).sum(axis=1) - np.maximum(rl0 - THETA1, 0).sum(axis=1)).mean() / 1000}
RELOC = pd.DataFrame(RELOC).T
print("\nSYSTEM EFFECT: move 10% of the 2030 onshore capacity (11.5 GW); designated on one period, evaluated on the other (differences)")
print(RELOC.round(3).to_string())

# the weather model at its 0.5 deg units
import units05 as u05  # noqa: E402

cell_real = u05.feedin(u05.real_units(adjust=False), units_cf=True)["units"]          # (75, 8760, 203) onshore CF
cell_mod = u05.feedin(u05.model_units(), temp_in_c=True, units_cf=True)["units"]       # (300, 8760, 203)
RM = sysser("model")
RL_MOD = RM["demand"] - CAP["pv"] * RM["pv"] - CAP["off"] * RM["offshore"] - CAP["on"] * RM["onshore 16 cells"]
wc = pd.Series(u05.W_LAND, index=[f"u{x}" for x in u05.LAND])
CELLS_T = {}
for lab, cells, rl in (("real (75 yr)", cell_real, RL["16-cell fleet (base)"]), ("model (300 yr)", cell_mod, RL_MOD)):
    for d in DEFS:
        mask = stress_mask(rl, d)
        s = np.stack([cells[i][mask[i]].mean(axis=0) for i in range(len(cells))])
        f = s @ wc.to_numpy()
        b = rng.integers(0, len(s), (1000, len(s)))
        rb = s[b].sum(axis=1) / f[b].sum(axis=1)[:, None]
        r = s.sum(axis=0) / f.sum()
        CELLS_T[(lab, d)] = pd.DataFrame({"r": r, "lo": np.quantile(rb, 0.05, axis=0), "hi": np.quantile(rb, 0.95, axis=0),
                                          "share of years with s < 0.8 f": (s < 0.8 * f[:, None]).mean(axis=0)}, index=wc.index)
del cell_real, cell_mod
cmp_rows = {}
for d in DEFS:
    a, b_ = CELLS_T[("real (75 yr)", d)], CELLS_T[("model (300 yr)", d)]
    sel = (wc > 0).to_numpy()
    cmp_rows[d] = {"corr r (cells)": np.corrcoef(a["r"][sel], b_["r"][sel])[0, 1], "max |diff r|": (a["r"] - b_["r"])[sel].abs().max(),
                   "model inside real 90% interval": ((b_["r"] >= a["lo"]) & (b_["r"] <= a["hi"]))[sel].mean(),
                   "range r real": f"{a['r'][sel].min():.2f}-{a['r'][sel].max():.2f}", "range r model": f"{b_['r'][sel].min():.2f}-{b_['r'][sel].max():.2f}"}
print("\nWEATHER MODEL at its 0.5 deg units: firmness r_k real (75 yr) vs model (300 yr), 100 m wind, each with its own RL")
print(pd.DataFrame(cmp_rows).T.to_string())
print(pd.concat({k[0]: v for k, v in CELLS_T.items() if k[1] == "72 h"}, axis=1).round(3).to_string())

with open(f"{D}/point3_value_map.pkl", "wb") as f:
    pickle.dump({"sites": SITES, "robustness": ROB, "relocation": RELOC, "cells": CELLS_T, "cells cmp": pd.DataFrame(cmp_rows).T,
                 "fleet stress": {d: (STATS[("16-cell fleet (base)", d)]["f"], STATS[("16-cell fleet (base)", d)]["s"][:, NON:].mean(axis=1)) for d in DEFS},
                 "years": YEARS, "earlier-definition lull months": LULL_MONTHS, "offshore conversion": {"sigma": CONV_OFF.sigma, "kappa": CONV_OFF.kappa_mh[0, 0], "r": DOFF["corr"]},
                 "stress months": {d: np.bincount(np.tile(pd.date_range("2025-01-01", periods=8760, freq="h").month.to_numpy(), (75, 1))[STATS[("16-cell fleet (base)", d)]["mask"]], minlength=13)[1:] for d in DEFS}}, f)
log(f"saved {D}/point3_value_map.pkl")
