# Thesis: Results, Model Validation, Fit Assessment (coverage, CRPS and variogram scores; variance inflation)
# Run: cd analysis && python fit_assessment.py [--rebuild]   (three model builds and ensembles, ~21 min)
# Needs: data/sim_16region_*.pkl (simulate_4_16_regions.py) for the variant with independent variables
# Writes: data/fit_assessment.pkl, data/fit_assessment_ens05.pkl
"""Fit assessment of the weather model: coverage, rank histograms and proper scores of the simulated ensembles against the
observed days.

Observations: the 75 observed years (1950-1999 adjusted to the 2000-2024 climate; units05.real_units), daily means,
national and on the 16 cells of the 4 x 4 partition (point-weighted aggregates of the 0.5 deg units; wind at 100 m).
Every model gives a 300-member ensemble per calendar day.

Measures
  coverage      share of observed days inside the 5-95 % band of the ensemble (nominal 0.90)
  rank hist.    position of the observed value in the ensemble; flat = calibrated, dome = too wide, U = too narrow
  CRPS          proper score of the daily mean of each variable (fair ensemble estimator)
  energy score  proper multivariate score of the 16-cell vector
  variogram sc. proper score of the dependence between components, p = 0.5: the 16 cells of each variable (spatial)
                and the standardized national wind, temperature and irradiance (coupling)

Models (ensembles cached in data/fit_assessment_ens05.pkl; --rebuild recomputes them)
  inflation  the unified model with all anomalies scaled by k (every innovation scaled, the dynamics being linear),
             without the final margin map, k = 0.8 ... 2.0
  ladder     OLD, AR-ONLY, SEPARATE, UNIFIED, all with mapped margins (k = 1); UNIFIED = the 300 years of
             unified_model_05.py; OLD = the 16-region simulation of simulate_4_16_regions.py (AR + Student-t copula per
             variable, variables independent), mapped onto the margins of the 16-cell aggregates so that only its
             dependence structure is scored
  real       each observed year scored against the other 74 years (the observed climate used as a generator)
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist

import spatial_common as sc
import units05 as u05

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)

D = u05.D
VARS = ["wind", "temp", "solar"]
KS = [0.8, 1.0, 1.25, 1.5, 2.0]
MON, HR = sc.TARGET.month.to_numpy(), sc.TARGET.hour.to_numpy()
ENS_FILE = f"{D}/fit_assessment_ens05.pkl"


# observations: 75 real years of daily means
def daily(X):
    """16-cell hourly (n, 8760, 16) -> daily means (n, 365, 16)"""
    return {v: np.asarray(X[v], dtype=np.float32).reshape(X[v].shape[0], 365, 24, 16).mean(axis=2) for v in VARS}


def daily16(X, temp_in_c=False, chunk=50):
    """0.5 deg unit series (n, 8760, U) -> 16-cell daily means (n, 365, 16), temperature in K"""
    n = X["wind"].shape[0]
    out = {v: np.empty((n, 365, 16), np.float32) for v in VARS}
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        for v in VARS:
            x = np.asarray(X[v][a:b, :, :u05.NL], dtype=np.float32)
            if v == "temp" and temp_in_c:
                x = x + 273.15
            out[v][a:b] = u05.to_cells(x).reshape(b - a, 365, 24, 16).mean(axis=2)
    return out


OBS = daily16(u05.real_units(adjust=True))
# standardisation for the cross-variable score: real climatological SD of national daily means per day
NAT_SD = {v: pd.Series(OBS[v].mean(axis=2).std(axis=0)).rolling(31, center=True, min_periods=1).mean().to_numpy()
          for v in VARS}
NAT_MU = {v: OBS[v].mean(axis=2).mean(axis=0) for v in VARS}
log("observations: 75 years of 16-cell daily means")


# scores (fair ensemble estimators)
def crps(ens, obs):
    """ens (N, 365), obs (M, 365) -> mean fair CRPS over days and observations"""
    N = ens.shape[0]
    out = []
    for d in range(ens.shape[1]):
        x = np.sort(ens[:, d])
        spread = 2 * np.sum((2 * np.arange(1, N + 1) - N - 1) * x) / (N * (N - 1))
        out.append(np.abs(x[:, None] - obs[None, :, d]).mean(axis=0) - 0.5 * spread)
    return float(np.mean(out))


# share of observed values inside the 5-95 % band of the ensemble
def coverage(ens, obs):
    lo, hi = np.percentile(ens, 5, axis=0), np.percentile(ens, 95, axis=0)
    return float(((obs >= lo) & (obs <= hi)).mean())


# rank histogram: where the observed value falls within the ensemble
def rank_hist(ens, obs, bins=10):
    r = (ens[:, None, :] < obs[None, :, :]).mean(axis=0).ravel()
    return np.histogram(r, bins=bins, range=(0, 1))[0] / r.size


def energy(ens, obs):
    """ens (N, 365, k), obs (M, 365, k) -> mean fair energy score"""
    N = ens.shape[0]
    out = []
    for d in range(ens.shape[1]):
        e, o = ens[:, d], obs[:, d]
        out.append(cdist(e, o).mean(axis=0) - pdist(e).sum() / (N * (N - 1)))
    return float(np.mean(out))


IU, JU = np.triu_indices(16, 1)


# variogram score of order p across the cells of each variable (Definition 36)
def variogram(ens, obs, p=0.5):
    out = []
    for d in range(ens.shape[1]):
        e, o = ens[:, d], obs[:, d]
        m = (np.abs(e[:, IU] - e[:, JU]) ** p).mean(axis=0)
        out.append(((np.abs(o[:, IU] - o[:, JU]) ** p - m[None]) ** 2).sum(axis=1))
    return float(np.mean(out))


def variogram3(ens, obs, p=0.5):
    """standardized national (wind, temp, solar) vector; ens/obs dicts of (n, 365)"""
    z = lambda X: np.stack([(X[v] - NAT_MU[v]) / NAT_SD[v] for v in VARS], axis=2)
    e, o = z(ens), z(obs)
    pairs = [(0, 1), (0, 2), (1, 2)]
    tot = 0.0
    for i, j in pairs:
        m = (np.abs(e[..., i] - e[..., j]) ** p).mean(axis=0)
        tot += ((np.abs(o[..., i] - o[..., j]) ** p - m[None]) ** 2).mean()
    return float(tot)


# all scores of one ensemble against the observed days: coverage, CRPS, variogram (spatial and coupling)
def score_all(Ed, Od):
    nat_e = {v: Ed[v].mean(axis=2) for v in VARS}
    nat_o = {v: Od[v].mean(axis=2) for v in VARS}
    r = {}
    for v in VARS:
        r[f"coverage {v}"] = coverage(nat_e[v], nat_o[v])
        r[f"CRPS {v}"] = crps(nat_e[v], nat_o[v])
        rh = rank_hist(nat_e[v], nat_o[v])
        r[f"central 40% share {v}"] = rh[3:7].sum()        # 0.40 = calibrated; more = too wide; less = too narrow
        r[f"rank hist {v}"] = rh
        r[f"energy {v}"] = energy(Ed[v], Od[v])
        r[f"variogram {v}"] = variogram(Ed[v], Od[v])
    r["variogram coupling"] = variogram3(nat_e, nat_o)
    return r


# models
def ensembles():
    """daily 16-cell ensembles (300 members) of every model -> ENS_FILE"""
    import unified_model_05 as um05
    E = {}
    M = um05.build("unified")
    for k in KS:
        mm, _ = um05.simulate(M, 300, seed=um05.SEED, scale=k, margins=False)
        E[f"inflation k={k}"] = daily16(mm, temp_in_c=True)
        um05.remove(mm)
        log(f"inflation k={k} simulated")
    E["ladder UNIFIED"] = daily16(u05.model_units(), temp_in_c=True)
    idx, REAL = M["idx"], M["REAL"]
    real16 = {v: u05.to_cells(REAL[v][:, :u05.NL]) for v in VARS}           # 2000-2024, temperature in C
    del M
    for name, variant in (("SEPARATE", "separate"), ("AR-ONLY", "ar-only")):
        Mv = um05.build(variant)
        mm, _ = um05.simulate(Mv, 300, seed=um05.SEED)
        E[f"ladder {name}"] = daily16(mm, temp_in_c=True)
        um05.remove(mm)
        del Mv
        log(f"ladder {name} simulated")
    old = {v: np.stack([pickle.load(open(f"{D}/sim_16region_{v}.pkl", "rb"))["phys"][c] for c in sc.CELLS],
                       axis=2).astype(np.float32) for v in VARS}
    tails16 = sc.fit_gpd_tails(real16["temp"].astype(np.float64), idx)
    for v in VARS:
        um05.margin_map(old[v], real16[v], idx, tails16 if v == "temp" else None)
    old["temp"] += 273.15
    E["ladder OLD"] = daily(old)
    with open(ENS_FILE, "wb") as f:
        pickle.dump(E, f)
    log(f"saved {ENS_FILE}")
    return E


if __name__ == "__main__":
    E = ensembles() if ("--rebuild" in sys.argv or not os.path.exists(ENS_FILE)) else pickle.load(open(ENS_FILE, "rb"))
    RES = {}
    for name, X in E.items():
        RES[name] = score_all(X, OBS)
        log(f"{name} scored")

    # real climate as a generator: leave-one-year-out, 74 members
    loo = []
    for y in range(OBS["wind"].shape[0]):
        keep = np.arange(OBS["wind"].shape[0]) != y
        loo.append(score_all({v: OBS[v][keep] for v in VARS}, {v: OBS[v][[y]] for v in VARS}))
    RES["real climate (leave one year out)"] = {k: (np.mean([d[k] for d in loo], axis=0)) for k in loo[0]}
    log("real-climate reference scored")

    with open(f"{D}/fit_assessment.pkl", "wb") as f:
        pickle.dump(RES, f)
    cols = [c for c in RES["inflation k=1.0"] if not c.startswith("rank hist")]
    T = pd.DataFrame({k: {c: v[c] for c in cols} for k, v in RES.items()}).T
    pd.set_option("display.width", 260)
    print(T.round(4).to_string())
    log(f"saved {D}/fit_assessment.pkl")
