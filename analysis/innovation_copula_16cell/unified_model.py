# Thesis: Case Study, Innovation Copula; the unified model on the 16 cells, on which the Gaussian and Student-t
# innovation copulas were compared with the empirical one
# Run: cd analysis/innovation_copula_16cell && python unified_model.py   (~3 min)
# Writes: data/unified_model_16cell.pkl, data/sim_unified_16cell.pkl
"""Unified model on the 16 cells: wind, temperature and irradiance jointly, with the spatial and cross-variable
dependence in the dynamics.

Per variable (16 cells): transformation (wind Box-Cox lambda 0.25, irradiance log(GHI + 1), temperature as is), linear
trend and smoothed (month, day, hour) climatology removed, EOF decomposition of the anomalies.

  modes        k leading EOF scores per variable (wind 2, temperature 2, irradiance 3) -> one seasonal VAR across
               all 7 scores; its cross-lags carry the spatial coherence and the cross-variable coupling
  remainder    seasonal AR per cell and variable (48 series)
  innovations  daily block bootstrap of the joint residual vector (7 mode and 48 remainder residuals): each simulated
               day takes the residuals of one observed day within +-10 days of the same date in the same season.
               This keeps night residuals at night (the point mass of irradiance at zero), the hour-of-day pattern of
               the residual variance and the persistence of the residuals within a day
  year offset  one random training year per trajectory, shared by all 48 series

Variants, all with the same sampled innovation days:
  UNIFIED     one VAR across all modes
  SEPARATE    one VAR per variable's modes (no cross-variable lags)
  AR-ONLY     AR per cell on all 48 anomaly series, no modes (dependence only through the innovations)
  OLD         the 16-region simulation of simulate_4_16_regions.py (AR + Student-t copula per variable, variables
              simulated independently)
"""
import os
import sys

# the shared modules (paths, spatial_common, ...) live one folder up, in analysis/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import sys
import time

import numpy as np
import pandas as pd

import spatial_common as sc
from simulate_4_16_regions import (SEASONS, build_smoothed_climatology_ns, fit_linear_trend_ns, get_season,
                                   map_climatology_ns, trend_value_ns)
import paths  # repository folders: data, figures, external raw data (see paths.py)

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
VARS = ["wind", "temp", "solar"]
KMODES = {"wind": 2, "temp": 2, "solar": 3}
MAXLAG = 72
GRID = [1, 2, 3, 6, 12, 24, 36, 48, 72]
WINDOW = 10
BURN = sc.BURN_DAYS  # 61 warm-up days: each trajectory first runs through 1 Nov - 31 Dec (those coefficients
                     # and innovation pools) and that part is discarded, so January starts from the model's own
                     # state instead of the real 31 Dec 2024 shared by all trajectories
N = 30 if "--smoke" in sys.argv else sc.N_TRAJ
INNOV = "hour" if "--innov=hour" in sys.argv else "day"   # main: whole real days; hour = sensitivity run
SLOW = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--slow=")), 0)  # two-scale window, days (0 = off)
TAILS = next((a.split("=")[1] for a in sys.argv if a.startswith("--tails=")), "gpd")   # temperature tails, final step
assert TAILS in ("gpd", "empirical"), "--tails=gpd (default) or --tails=empirical (quantiles stop at the record)"
TAG = ((f"_slow{SLOW}" if SLOW else "") + ("_hour" if INNOV == "hour" else "") + ("_raw" if "--no-margins" in sys.argv else "")
       + ("_ecdf" if TAILS == "empirical" else "") + ("_smoke" if "--smoke" in sys.argv else ""))
FWD = {"wind": sc.wind_fwd, "temp": lambda x: x, "solar": lambda x: np.log1p(np.clip(x, 0, None))}
INV = {"wind": sc.wind_inv, "temp": lambda y: y, "solar": lambda y: np.clip(np.expm1(y), 0, None)}
FILES = {v: f"{sc.D}/grid4x4_{v}_2000-2024.pkl" for v in VARS}
T = len(sc.TARGET)


# preprocessing
def preprocess_var(df, fwd):
    cols = list(df.columns)
    y = pd.DataFrame(fwd(df.to_numpy(dtype=float)), index=df.index, columns=cols)
    tr = fit_linear_trend_ns(y, cols, window_days=15)
    det = y - trend_value_ns(y.index, tr, cols)
    clim = build_smoothed_climatology_ns(det, cols, window_days=15)
    anom = y - (map_climatology_ns(y.index, clim, cols) + trend_value_ns(y.index, tr, cols))
    return {"anom": anom, "offsets": anom.groupby(anom.index.year).mean(),
            "sim_mean": map_climatology_ns(sc.TARGET, clim, cols).to_numpy()}


real = {v: pd.read_pickle(FILES[v])[[f"{c}_{v}" for c in sc.CELLS]] for v in VARS}
real["solar"] = real["solar"].clip(lower=0)
idx = real["wind"].index
PRE = {v: preprocess_var(real[v], FWD[v]) for v in VARS}
log("preprocessed 3 x 16 series")
if SLOW:
    # TWO-SCALE: slow = centred SLOW-day running mean of the anomalies (solar: over daylight hours only,
    # applied only in daylight so nights stay zero); fast = anomaly - slow. The fast part is what the EOF /
    # VAR / AR machinery below is fitted to; the slow part is resampled as whole real years in simulation.
    Wh = SLOW * 24
    ghi_clim = real["solar"].groupby([real["solar"].index.month, real["solar"].index.hour]).transform("mean")
    DAYMASK = (ghi_clim > 5).astype(float)
    DAYMASK.columns = PRE["solar"]["anom"].columns
    leap = (idx.month == 2) & (idx.day == 29)
    for v in VARS:
        a = PRE[v]["anom"]
        if v == "solar":
            num = (a * DAYMASK).rolling(Wh, center=True, min_periods=Wh // 2).sum()
            den = DAYMASK.rolling(Wh, center=True, min_periods=Wh // 2).sum()
            slow = (num / den.where(den > 0)).fillna(0.0) * DAYMASK
        else:
            slow = a.rolling(Wh, center=True, min_periods=Wh // 2).mean()
        PRE[v]["anom"] = a - slow
        sl = slow[~leap]
        PRE[v]["slow_years"] = {y: sl.loc[str(y)].to_numpy() for y in sorted(set(idx.year))}
        log(f"two-scale {v}: slow part carries {np.nanvar(slow.to_numpy()) / np.var(a.to_numpy()):.1%} of anomaly "
            f"variance (window {SLOW} d)")


# leading k EOF modes of the anomalies: loadings, scores and the remainder
def eof(anom, k):
    X = anom.to_numpy(dtype=float)
    mu = X.mean(axis=0)
    ev, evec = np.linalg.eigh(np.cov((X - mu).T))
    o = np.argsort(ev)[::-1]
    E = evec[:, o[:k]]
    S = (X - mu) @ E
    return E, mu, S, X - mu - S @ E.T, ev[o] / ev.sum()


EOFS = {v: eof(PRE[v]["anom"], KMODES[v]) for v in VARS}
for v in VARS:
    log(f"EOF {v}: {KMODES[v]} modes explain {EOFS[v][4][:KMODES[v]].sum():.1%}")
scores = pd.DataFrame(np.hstack([EOFS[v][2] for v in VARS]), index=idx,
                      columns=[f"{v}_pc{i + 1}" for v in VARS for i in range(KMODES[v])])
remain = pd.DataFrame(np.hstack([EOFS[v][3] for v in VARS]), index=idx,
                      columns=[f"{v}_{c}" for v in VARS for c in sc.CELLS])
anom_all = pd.concat([PRE[v]["anom"] for v in VARS], axis=1)


# fitting
def blockdiag(blocks):
    ds = [B.shape[1] for B in blocks]
    ps = [(B.shape[0] - 1) // d for B, d in zip(blocks, ds)]
    Dst, p = sum(ds), max(ps)
    out = np.zeros((1 + Dst * p, Dst))
    off = 0
    for B, d, pi in zip(blocks, ds, ps):
        out[0, off:off + d] = B[0]
        for lag in range(1, pi + 1):
            out[1 + (lag - 1) * Dst + off:1 + (lag - 1) * Dst + off + d, off:off + d] = B[1 + (lag - 1) * d:1 + lag * d]
        off += d
    return out, p


def residual_frame(parts):
    """parts: list of (fits, n_cols) with fits[s]['pool'] and an index -> one time-indexed frame"""
    frames = []
    for s in SEASONS:
        cols = []
        for fits in parts:
            ix = fits[s]["index"] if "index" in fits[s] else fits[s]["pool_df"].index
            cols.append(pd.DataFrame(fits[s]["pool"], index=ix))
        frames.append(pd.concat(cols, axis=1, join="inner"))
    out = pd.concat(frames).sort_index()
    out.columns = range(out.shape[1])
    return out


ar_rem = sc.fit_ar(remain, maxlag=MAXLAG)
log(f"remainder ARs fitted (48 series), median order {np.median([o for s in SEASONS for o in ar_rem[s]['orders']]):.0f}")
var_joint = sc.fit_var(scores, GRID)
log("joint mode VAR orders " + str({s: var_joint[s]["p"] for s in SEASONS})
    + "; spectral radius " + str({s: round(sc.spectral_radius(var_joint[s]["B"], scores.shape[1]), 4) for s in SEASONS}))
var_sep = {}
off = 0
for v in VARS:
    var_sep[v] = sc.fit_var(scores.iloc[:, off:off + KMODES[v]], GRID)
    off += KMODES[v]
log("separate mode VAR orders " + str({v: [var_sep[v][s]["p"] for s in SEASONS] for v in VARS}))
ar_all = sc.fit_ar(anom_all, maxlag=MAXLAG)
log("AR-only fitted (48 series)")

MODELS = {}
MODELS["UNIFIED"] = {"B": {s: blockdiag([var_joint[s]["B"], ar_rem[s]["B"]]) for s in SEASONS},
                     "res": residual_frame([var_joint, ar_rem]),
                     "hist": np.hstack([scores.to_numpy(), remain.to_numpy()]), "modes": True}
MODELS["SEPARATE"] = {"B": {s: blockdiag([var_sep[v][s]["B"] for v in VARS] + [ar_rem[s]["B"]]) for s in SEASONS},
                      "res": residual_frame([var_sep[v] for v in VARS] + [ar_rem]),
                      "hist": np.hstack([scores.to_numpy(), remain.to_numpy()]), "modes": True}
MODELS["AR-ONLY"] = {"B": {s: (ar_all[s]["B"], ar_all[s]["p"]) for s in SEASONS},
                     "res": residual_frame([ar_all]), "hist": anom_all.to_numpy(), "modes": False}
for k, m in MODELS.items():
    log(f"{k}: state dim {m['hist'].shape[1]}, residual rows {len(m['res'])}, lag order "
        f"{max(p for B, p in m['B'].values())}")
assert MODELS["UNIFIED"]["res"].index.equals(MODELS["AR-ONLY"]["res"].index), "variants must share residual days"


# daily block bootstrap (common random days for all variants)
def day_blocks(res):
    r = res[~((res.index.month == 2) & (res.index.day == 29))]
    day = r.index.normalize()
    cnt = pd.Series(1, index=day).groupby(level=0).size()
    full = cnt.index[cnt == 24]
    sub = r[day.isin(full)]
    arr = sub.to_numpy(dtype=np.float32).reshape(len(full), 24, r.shape[1])
    doy = np.array([pd.Timestamp(2025, d.month, d.day).dayofyear for d in full])
    seas = np.array([get_season(d.month) for d in full])
    return arr, doy, seas


_, POOL_DOY, POOL_SEAS = day_blocks(MODELS["UNIFIED"]["res"])
tdays = pd.date_range("2025-01-01", periods=365, freq="D")
CANDS = []
for d in tdays:
    dist = np.abs(POOL_DOY - d.dayofyear)
    dist = np.minimum(dist, 365 - dist)
    CANDS.append(np.where((dist <= WINDOW) & (POOL_SEAS == get_season(d.month)))[0])
rng = np.random.default_rng(20261006)
if INNOV == "day":                                                         # whole real days
    DRAW = np.repeat(np.stack([rng.choice(c, size=N) for c in CANDS], axis=1)[:, :, None], 24, axis=2)
else:                                                                      # independent hours, same hour of day
    DRAW = np.stack([rng.choice(c, size=(N, 24)) for c in CANDS], axis=1)  # (N, 365, 24) pool-day index
YEARS = rng.choice(PRE["wind"]["offsets"].index.to_numpy(), size=N)


def burn_draws(gen, n):
    """(n, BURN, 24) pool-day indices for the warm-up: the last BURN target days (Nov-Dec)"""
    c = CANDS[-BURN:]
    if INNOV == "day":
        return np.repeat(np.stack([gen.choice(x, size=n) for x in c], axis=1)[:, :, None], 24, axis=2)
    return np.stack([gen.choice(x, size=(n, 24)) for x in c], axis=1)


BURN_DRAW = burn_draws(rng, N)                     # drawn after DRAW and YEARS, which stay unchanged
SIM_SEAS = np.concatenate([np.repeat([get_season(d.month) for d in tdays[-BURN:]], 24), sc.SEASON_LAB])
TB = 24 * BURN
log(f"innovations '{INNOV}': {len(POOL_DOY)} complete real days; candidates per target day "
    f"{min(len(c) for c in CANDS)}-{max(len(c) for c in CANDS)}")


def simulate_model(m, draw=None, years=None, scale=1.0, E=None, burn_draw=None):
    """scale != 1 multiplies the simulated anomalies (= scaling every innovation, since the
    dynamics are linear), used for the variance inflation of the fit assessment.
    The first BURN days (Nov-Dec) are a warm-up and discarded. Innovations: whole real days,
    `burn_draw` (N, BURN, 24) for the warm-up and `draw` (N, 365, 24) for the target year
    (defaults BURN_DRAW, DRAW; a custom draw needs its own burn_draw, see burn_draws), or
    E (N, TB + 8760, D) generated elsewhere (parametric copula variants), warm-up included."""
    years = YEARS if years is None else years
    if E is None:
        if draw is None:
            draw, burn_draw = DRAW, BURN_DRAW
        assert burn_draw is not None and burn_draw.shape[0] == draw.shape[0], "a custom draw needs its burn_draw"
        full = np.concatenate([burn_draw, draw], axis=1)
        arr, _, _ = day_blocks(m["res"])
        E = arr[full, np.arange(24)[None, None, :]].reshape(full.shape[0], TB + T, arr.shape[2])
    assert E.shape[1] == TB + T, "E must include the warm-up"
    N = E.shape[0]
    hist = m["hist"]
    pmax = max(p for B, p in m["B"].values())
    D = hist.shape[1]
    sim = np.empty((N, pmax + TB + T, D))
    sim[:, :pmax] = hist[-pmax:][None]
    for t in range(TB + T):
        B, p = m["B"][SIM_SEAS[t]]
        i = t + pmax
        sim[:, i] = B[0] + sim[:, i - p:i][:, ::-1].reshape(N, p * D) @ B[1:] + E[:, t]
    A = sim[:, pmax + TB:]
    out, off_pc, off_rem = {}, 0, sum(KMODES.values()) if m["modes"] else 0
    for j, v in enumerate(VARS):
        if m["modes"]:
            Ev, mu, _, _, _ = EOFS[v]
            an = A[..., off_pc:off_pc + KMODES[v]] @ Ev.T + A[..., off_rem + 16 * j:off_rem + 16 * (j + 1)] + mu
            off_pc += KMODES[v]
        else:
            an = A[..., 16 * j:16 * (j + 1)]
        if SLOW:
            low = np.stack([PRE[v]["slow_years"][y] for y in years])                      # (N, 8760, 16)
        else:
            low = PRE[v]["offsets"].loc[years].to_numpy()[:, None, :]
        level = scale * an + PRE[v]["sim_mean"][None] + low
        out[v] = INV[v](level).astype(np.float32)
    return out


# final step: margins from the data (Sklar)
MARGINS = "--no-margins" not in sys.argv
# Temperature tails: beyond the outer 2% of each (cell, month, hour) slot
# the margins follow a generalized Pareto tail fitted to 2000-2024, so single-hour extremes can go beyond the
# record; --tails=empirical keeps the empirical quantiles throughout (they stop at the observed extremes).
# Wind and solar stay empirical: below cut-in the CF is already 0, cell means never reach cut-out, solar is bounded.
GPD_TEMP = sc.fit_gpd_tails(real["temp"].to_numpy(dtype=np.float64), real["temp"].index) if TAILS == "gpd" else None
if GPD_TEMP is not None:
    log("GPD temperature tails, shape xi per season: " + ", ".join(f"{s} {t} {x:+.3f}" for (s, t), x in GPD_TEMP["xi"].items()))


def margin_map(X):
    """Copula-consistent last step: keep the simulated ranks (dependence across cells,
    variables and time) and impose each cell's real hourly distribution per (month, hour)
    from the training years 2000-2024, with GPD tails for temperature (GPD_TEMP). Ranks are
    taken within the simulated ensemble, so apply to ensembles of >= ~300 years (9,000
    values per slot)."""
    if not MARGINS:
        return X
    out = {}
    for v in VARS:
        out[v] = sc.margfix(X[v].astype(np.float64), real[v].to_numpy(dtype=np.float64), real[v].index,
                            tails=GPD_TEMP if v == "temp" else None).astype(np.float32)
    return out


# evaluation
if __name__ == "__main__":
    CAL = sc.TARGET
    MON, HR = CAL.month.to_numpy(), CAL.hour.to_numpy()
    SEAS = sc.SEASON_LAB
    DOY = CAL.dayofyear.to_numpy()
    W_WIND, W_PV = 63 / 162, 99 / 162          # installed onshore wind vs PV in Germany, end 2024 (GW)


    # PV capacity factor from GHI and temperature (K) with the NOCT cell-temperature model
    def pv_cf(ghi, tk):
        tc = tk - 273.15 + 0.03125 * ghi
        return np.clip(ghi / 1000 * (1 - 0.004 * (tc - 25)), 0, 1)


    # drop 29 February and reshape an hourly frame to (years, 8760, cells)
    def to_units(df):
        df = df[~((df.index.month == 2) & (df.index.day == 29))]
        return df.to_numpy(dtype=np.float32).reshape(-1, T, df.shape[1])      # (25, 8760, 16)


    REAL_U = {v: to_units(real[v]) for v in VARS}
    # wind feed-in (conversion.py): observed data from the ERA5 100 m wind, simulated 10 m
    # wind quantile-mapped to 100 m per (cell, local month, local hour); German-land-weighted onshore fleet
    import conversion as cv
    CONV = cv.calibrate_16cells()
    LMON, LHR = cv.local_time(CAL)
    REAL_U["ws100"] = to_units(pd.DataFrame(cv.wind100_cells(real["wind"].index), index=real["wind"].index))


    # fleet onshore capacity factor: model 10 m wind mapped to 100 m first, then converted
    def fleet_wind(X):
        v100 = X["ws100"] if "ws100" in X else CONV["map"](X["wind"], LMON, LHR)
        return CONV["onshore"].cf(v100, LMON, LHR)


    dark_slots = (REAL_U["solar"].max(axis=0) < 1e-6)                          # (8760, 16) always dark in real
    day_slots = REAL_U["solar"].mean(axis=(0, 2)) > 20                          # daylight hours for wind-solar coupling


    def anomalies_vs_own_clim(x):
        """x: (units, 8760) -> anomalies relative to the (month, hour) mean over ALL units"""
        out = np.empty_like(x)
        for m in range(1, 13):
            for h in range(24):
                sl = (MON == m) & (HR == h)
                out[:, sl] = x[:, sl] - x[:, sl].mean()
        return out


    # winter and summer correlation of two hourly series (optionally on a subset of hours)
    def season_corr(a, b, mask=None):
        res = {}
        for s in ("winter", "summer"):
            sl = (SEAS == s) if mask is None else (SEAS == s) & mask
            res[s] = np.corrcoef(a[:, sl].ravel(), b[:, sl].ravel())[0, 1]
        return res


    def metrics(X):
        """X: dict var -> (units, 8760, 16) physical. Returns (summary dict, per-unit DataFrame)."""
        U = X["wind"].shape[0]
        cfw = fleet_wind(X)                                                    # onshore fleet CF (units, 8760)
        cfp = pv_cf(X["solar"], X["temp"]).mean(axis=2)
        comb = W_WIND * cfw + W_PV * cfp
        tnat = X["temp"].mean(axis=2)
        per = []
        for u in range(U):
            rw, rc = sc.lulls(cfw[u]), sc.lulls(comb[u])
            per.append({"wind CF<5% h": (cfw[u] < 0.05).sum(), "wind lulls>=72h": (rw >= 72).sum(),
                        "wind longest lull": rw.max() if len(rw) else 0,
                        "RE CF<5% h": (comb[u] < 0.05).sum(), "RE lulls>=24h": (rc >= 24).sum(),
                        "RE worst 72h mean %": 100 * pd.Series(comb[u]).rolling(72).mean().min(),
                        "RE worst 7d mean %": 100 * pd.Series(comb[u]).rolling(168).mean().min(),
                        "cold RE-lull h": ((comb[u] < 0.10) & (tnat[u] < 273.15)).sum()})
        per = pd.DataFrame(per)
        nat = {v: X[v].mean(axis=2) for v in VARS}
        an = {v: anomalies_vs_own_clim(nat[v]) for v in VARS}
        out = {"mean wind CF": cfw.mean(), "mean PV CF": cfp.mean()}
        wt = season_corr(an["wind"], an["temp"])
        ws = season_corr(an["wind"], an["solar"], day_slots)
        out.update({"wind-temp winter": wt["winter"], "wind-temp summer": wt["summer"],
                    "wind-solar winter": ws["winter"], "wind-solar summer": ws["summer"]})
        cellwt = []
        for c in range(16):
            aw, at = anomalies_vs_own_clim(X["wind"][..., c]), anomalies_vs_own_clim(X["temp"][..., c])
            cellwt.append(season_corr(aw, at))
        out["cell wind-temp winter"] = np.mean([d["winter"] for d in cellwt])
        out["cell wind-temp summer"] = np.mean([d["summer"] for d in cellwt])
        for v in VARS:
            dm = an[v].reshape(U, 365, 24).mean(axis=2)
            out[f"{v} daily lag1"] = np.mean([np.corrcoef(d[:-1], d[1:])[0, 1] for d in dm])
            take = range(0, U, max(1, U // 10))
            coh = []
            for u in take:
                df = pd.DataFrame(X[v][u], index=CAL)
                z = (df - df.groupby([CAL.month, CAL.hour]).transform("mean")).to_numpy()
                C = np.corrcoef(z.T)
                coh.append(C[~np.eye(16, dtype=bool)].mean())
            out[f"{v} x-cell corr"] = np.mean(coh)
        out["night GHI>1 share"] = (X["solar"][:, dark_slots] > 1).mean()
        return out, per, {v: nat[v].reshape(U, 365, 24).mean(axis=2) for v in VARS}


    # coverage of the 5-95 % band of the simulated daily values, per observed year
    def coverage(sim_daily, real_daily):
        lo, hi = np.percentile(sim_daily, 5, axis=0), np.percentile(sim_daily, 95, axis=0)
        return ((real_daily >= lo) & (real_daily <= hi)).mean(axis=1)            # per real year


    summ, per_unit, daily = {}, {}, {}
    summ["REAL"], per_unit["REAL"], daily["REAL"] = metrics(REAL_U)
    log("real metrics done")
    for k, m in MODELS.items():
        X = margin_map(simulate_model(m))
        summ[k], per_unit[k], daily[k] = metrics(X)
        if k == "UNIFIED" and N == sc.N_TRAJ:
            with open(f"{sc.D}/sim_unified_16cell{TAG}.pkl", "wb") as f:
                pickle.dump({"phys": {v: X[v].astype(np.float32 if v == "temp" else np.float16) for v in VARS},
                             "target_index": CAL,
                             "cells": sc.CELLS, "kmodes": KMODES}, f)
        del X
        log(f"{k}: simulated + evaluated")
    old = {v: np.stack([pickle.load(open(f"{sc.D}/sim_16region_{v}.pkl", "rb"))["phys"][c][:N] for c in sc.CELLS], axis=2)
           for v in VARS}
    summ["OLD"], per_unit["OLD"], daily["OLD"] = metrics(margin_map(old))
    del old
    log("OLD evaluated")

    for k in summ:
        if k == "REAL":
            continue
        for v in VARS:
            summ[k][f"{v} daily coverage"] = coverage(daily[k][v], daily["REAL"][v]).mean()

    pd.set_option("display.width", 220)
    order = ["REAL", "UNIFIED", "SEPARATE", "AR-ONLY", "OLD"]
    S = pd.DataFrame(summ).T.loc[order]
    print("\n" + "=" * 110)
    print("UNIFIED MODEL, 16 cells x {wind, temp, solar} - correlations, coherence, coverage, persistence")
    print("=" * 110)
    print(S.T.round(3).to_string())
    print("\nPER-YEAR STATISTICS: mean (gap to REAL in combined SE)")
    R = per_unit["REAL"]
    rows = []
    for st in R.columns:
        row = {"statistic": st, "REAL": f"{R[st].mean():.1f} (SE {R[st].std(ddof=1) / np.sqrt(len(R)):.1f})"}
        for k in order[1:]:
            p = per_unit[k][st]
            se = np.sqrt(R[st].var(ddof=1) / len(R) + p.var(ddof=1) / len(p))
            row[k] = f"{p.mean():.1f} ({(p.mean() - R[st].mean()) / se:+.1f})"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("statistic").to_string())
    print("\nRE = capacity-weighted wind + PV output (63 GW onshore wind, 99 GW PV), all hours incl. night;"
          " cold RE-lull = RE CF < 10% while national temperature < 0 C")

    with open(f"{sc.D}/unified_model_16cell{TAG}.pkl", "wb") as f:
        pickle.dump({"summary": S, "per_unit": per_unit, "kmodes": KMODES,
                     "var_orders": {s: var_joint[s]["p"] for s in SEASONS}, "gpd_temp": GPD_TEMP}, f)
    log(f"saved unified_model_16cell{TAG}.pkl")
