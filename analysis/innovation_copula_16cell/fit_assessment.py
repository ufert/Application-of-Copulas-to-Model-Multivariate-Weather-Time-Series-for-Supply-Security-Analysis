# Thesis: Case Study, Innovation Copula; the 16-cell version of the fit assessment, whose ensembles the comparison uses
# Run: cd analysis/innovation_copula_16cell && python fit_assessment.py   (after unified_model.py)
# Writes: data/fit_assessment_16cell.pkl
"""Fit assessment of the unified model on the 16 cells (unified_model.py): coverage, rank histograms and proper scores of
the simulated ensembles against the observed days. Provides the ensembles and scores used by parametric_copula.py.

Observations: the 75 observed years (1950-1999 adjusted to the 2000-2024 climate), daily means, national and per cell.
Every model gives a 300-member ensemble per calendar day.

Measures
  coverage      share of observed days inside the 5-95 % band of the ensemble (nominal 0.90)
  rank hist.    position of the observed value in the ensemble; flat = calibrated, dome = too wide, U = too narrow
  CRPS          proper score of the daily mean of each variable (fair ensemble estimator)
  energy score  proper multivariate score of the 16-cell vector
  variogram sc. proper score of the dependence between components, p = 0.5: the 16 cells of each variable (spatial)
                and the standardized national wind, temperature and irradiance (coupling)

Models
  inflation  the unified model with all anomalies scaled by k, without the final margin map, k = 0.8 ... 2.0
  ladder     OLD, AR-ONLY, SEPARATE, UNIFIED, all with mapped margins (k = 1)
  real       each observed year scored against the other 74 years
"""
import os
import sys

# the shared modules (paths, spatial_common, ...) live one folder up, in analysis/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist

import spatial_common as sc
import paths  # repository folders: data, figures, external raw data (see paths.py)

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
import unified_model as um  # noqa: E402

D = sc.D
VARS = ["wind", "temp", "solar"]
KS = [0.8, 1.0, 1.25, 1.5, 2.0]
MON, HR = sc.TARGET.month.to_numpy(), sc.TARGET.hour.to_numpy()
log("model fitted")


# observations: 75 real years of daily means
def units(period, v):
    g = pd.read_pickle(f"{D}/grid4x4_{v}_{period}.pkl")[[f"{c}_{v}" for c in sc.CELLS]]
    g = g[~((g.index.month == 2) & (g.index.day == 29))]
    x = g.to_numpy(dtype=np.float32).reshape(-1, 8760, 16).copy()
    return np.clip(x, 0, None) if v == "solar" else x


IN = {v: units("2000-2024", v) for v in VARS}
OUT = {v: units("1950-1999", v) for v in VARS}
for m in range(1, 13):
    sl = MON == m
    OUT["temp"][:, sl] += (IN["temp"][:, sl].mean(axis=(0, 1)) - OUT["temp"][:, sl].mean(axis=(0, 1)))[None, None, :]
    for h in range(24):
        sh = sl & (HR == h)
        mi, mo = IN["solar"][:, sh].mean(axis=(0, 1)), OUT["solar"][:, sh].mean(axis=(0, 1))
        OUT["solar"][:, sh] *= np.where(mo > 1, mi / np.maximum(mo, 1e-6), 1.0)[None, None, :]


# daily means of hourly fields: (years, 365, 16 cells)
def daily(X):
    return {v: X[v].reshape(X[v].shape[0], 365, 24, 16).mean(axis=2) for v in VARS}      # (n, 365, 16)


OBS = daily({v: np.concatenate([IN[v], OUT[v]]) for v in VARS})
del IN, OUT
# standardisation for the cross-variable score: real climatological SD of national daily means per day
NAT_SD = {v: pd.Series(OBS[v].mean(axis=2).std(axis=0)).rolling(31, center=True, min_periods=1).mean().to_numpy()
          for v in VARS}
NAT_MU = {v: OBS[v].mean(axis=2).mean(axis=0) for v in VARS}


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


# variogram score of order p across the 16 cells of each variable
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


# all scores of one ensemble against the observed days
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
RES = {}
for k in KS:
    X = um.simulate_model(um.MODELS["UNIFIED"], scale=k)
    RES[f"inflation k={k}"] = score_all(daily(X), OBS)
    del X
    log(f"inflation k={k} scored")
for name in ("UNIFIED", "SEPARATE", "AR-ONLY"):
    X = um.margin_map(um.simulate_model(um.MODELS[name]))
    RES[f"ladder {name}"] = score_all(daily(X), OBS)
    del X
    log(f"ladder {name} scored")
old = {v: np.stack([pickle.load(open(f"{D}/sim_16region_{v}.pkl", "rb"))["phys"][c] for c in sc.CELLS],
                   axis=2).astype(np.float32) for v in VARS}
RES["ladder OLD"] = score_all(daily(um.margin_map(old)), OBS)
del old
log("ladder OLD scored")

# real climate as a generator: leave-one-year-out, 74 members
loo = []
for y in range(OBS["wind"].shape[0]):
    keep = np.arange(OBS["wind"].shape[0]) != y
    loo.append(score_all({v: OBS[v][keep] for v in VARS}, {v: OBS[v][[y]] for v in VARS}))
RES["real climate (leave one year out)"] = {k: (np.mean([d[k] for d in loo], axis=0)) for k in loo[0]}
log("real-climate reference scored")

with open(f"{D}/fit_assessment_16cell.pkl", "wb") as f:
    pickle.dump(RES, f)
cols = [c for c in RES["inflation k=1.0"] if not c.startswith("rank hist")]
T = pd.DataFrame({k: {c: v[c] for c in cols} for k, v in RES.items()}).T
pd.set_option("display.width", 260)
print(T.round(4).to_string())
log(f"saved {D}/fit_assessment_16cell.pkl")
