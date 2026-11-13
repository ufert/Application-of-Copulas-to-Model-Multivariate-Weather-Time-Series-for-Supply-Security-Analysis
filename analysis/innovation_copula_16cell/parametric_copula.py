# Thesis: Case Study, Innovation Copula (Gaussian and Student-t copula on the whole-day innovation vector)
# Run: cd analysis/innovation_copula_16cell && python parametric_copula.py [--smoke]   (~6 min, after fit_assessment.py)
# Writes: data/sim_unified_16cell_{gcop,tcop}.pkl and the paired proper scores
"""Parametric copulas for the innovations of the unified model (16 cells), compared with the empirical copula of whole
observed days.

The unified model draws the innovations of each simulated day as the 24-h joint residual vector of one observed day
within +-10 days (7 mode and 48 remainder residuals): the empirical copula of the day vector (24 h x 55 = 1,320
dimensions). Here this copula is replaced by a parametric one and everything else is kept:
  margins    the distribution of each innovation per (target day, hour, component) is the same +-10-day pool of
             observed residuals (empirical quantile function, linear interpolation between order statistics)
  dynamics   the fitted unified model (EOF modes -> one seasonal VAR; per-cell ARs)
  offsets    the same random training year per trajectory as the run of unified_model.py (paired comparison)
  last step  margin_map (observed per-(cell, month, hour) distributions)

Copulas, one per season on the 1,320-dimensional day vector (pseudo-observations = ranks of the residuals of each
observed day within its own +-10-day pool):
  GAUSS   Gaussian copula, R = correlation of the normal scores
  T       Student-t copula, R = correlation of the t scores, nu by profile likelihood; one shared mixing variable per
          day, i.e. a common scale for the whole day

Run: python parametric_copula.py [--smoke]
Writes data/sim_unified_16cell_{gcop,tcop}{_smoke}.pkl (300 years, margin-mapped) and data/parametric_copula{_smoke}.pkl
(fits, profile of nu, paired proper scores).
"""
import os
import sys

# the shared modules (paths, spatial_common, ...) live one folder up, in analysis/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

import numpy as np
from scipy import stats
from scipy.special import gammaln
import paths  # repository folders: data, figures, external raw data (see paths.py)

exec(open("fit_assessment_sig.py").read().split("S = {}")[0])    # um (fitted), OBS, daily(), per_year(), log()

SMOKE = "_smoke" if um.N != sc.N_TRAJ else ""
NU_GRID = [4, 6, 8, 12, 16, 24, 32, 48, 64]
H = 24
rng = np.random.default_rng(20261011)

# residual days, pools, pseudo-observations
m = um.MODELS["UNIFIED"]
ARR, DOY, SEAS = um.day_blocks(m["res"])                     # (8822, 24, 55)
DIM = H * ARR.shape[2]
FLAT = ARR.reshape(len(ARR), DIM)
POOLS = [np.sort(FLAT[c], axis=0) for c in um.CANDS]          # per target day: sorted (n_c, 1320)
U = np.empty_like(FLAT, dtype=np.float64)
for i, c in enumerate(um.CANDS):                              # rank of each real day within its own pool
    mem = np.where(DOY == i + 1)[0]
    if len(mem):
        U[mem] = ((POOLS[i][None] < FLAT[mem][:, None]).sum(axis=1) + 0.5) / len(c)
log(f"pseudo-observations: {FLAT.shape[0]} days x {DIM} dims")


# log density of the multivariate Student-t distribution (correlation R, degrees of freedom nu)
def t_logpdf_mv(X, R_inv, logdet, nu):
    q = ((X @ R_inv) * X).sum(axis=1)
    d = X.shape[1]
    return (gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log(nu * np.pi) - logdet / 2
            - (nu + d) / 2 * np.log1p(q / nu))


# symmetric sample correlation matrix of the columns of X
def corr_of(X):
    C = np.corrcoef(X.T)
    return (C + C.T) / 2


def chol_logdet(R):
    """Cholesky factor and log-determinant (np.linalg.slogdet raises spurious FP warnings with
    macOS Accelerate at this size; the value is identical)"""
    L = np.linalg.cholesky(R)
    return L, 2 * np.log(np.diag(L)).sum()


# fit per season
FIT = {}
for s in um.SEASONS:
    Us = U[SEAS == s]
    Z = stats.norm.ppf(Us)
    Rg = corr_of(Z)
    Lg, logdet = chol_logdet(Rg)
    Rinv = np.linalg.inv(Rg)
    ll_g = float(np.sum(-logdet / 2 - 0.5 * ((Z @ (Rinv - np.eye(DIM))) * Z).sum(axis=1)))
    prof = {"gauss": ll_g}
    for nu in NU_GRID:
        X = stats.t.ppf(Us, nu)
        Rt = corr_of(X)
        _, ld = chol_logdet(Rt)
        prof[nu] = float(t_logpdf_mv(X, np.linalg.inv(Rt), ld, nu).sum() - stats.t.logpdf(X, nu).sum())
    nu_hat = max(NU_GRID, key=lambda k: prof[k])
    Rt = corr_of(stats.t.ppf(Us, nu_hat))
    off = ~np.eye(DIM, dtype=bool)
    FIT[s] = {"n": len(Us), "R_gauss": Rg, "R_t": Rt, "nu": nu_hat, "profile": prof,
              "L_gauss": Lg, "L_t": np.linalg.cholesky(Rt),
              "mean_abs_r": float(np.abs(Rg[off]).mean()), "min_eig": float(np.linalg.eigvalsh(Rg).min())}
    log(f"{s}: n={len(Us)} days, mean |r| {FIT[s]['mean_abs_r']:.3f}, min eigenvalue {FIT[s]['min_eig']:.3f}; "
        f"loglik gauss {ll_g:,.0f}; t: " + ", ".join(f"nu={k} {prof[k] - ll_g:+,.0f}" for k in NU_GRID)
        + f" -> nu = {nu_hat}")


# simulation
TDAY_SEAS = np.array([um.get_season(d.month) for d in um.tdays])


def sample_day(family, s, d, n):
    """n draws of the 1,320-dim innovation vector for target day d (0-based day of the 2025
    calendar, season s): copula draw -> quantile functions of that day's +-10-day pool"""
    L = FIT[s]["L_gauss" if family == "gauss" else "L_t"]
    Zs = rng.standard_normal((n, DIM)) @ L.T
    if family == "gauss":
        Us = stats.norm.cdf(Zs)
    else:
        nu = FIT[s]["nu"]
        Us = stats.t.cdf(Zs / np.sqrt(rng.chisquare(nu, size=(n, 1)) / nu), nu)
    P = POOLS[d]
    k = len(P)
    pos = np.clip(Us * k - 0.5, 0, k - 1)
    lo = np.minimum(pos.astype(int), k - 2)
    w = pos - lo
    cols = np.arange(DIM)[None]
    return P[lo, cols] * (1 - w) + P[lo + 1, cols] * w


def innovations(family):
    """(N, 24 * (BURN + 365), 55) innovations: the warm-up days (Nov-Dec pools) then the target year"""
    days = list(range(365 - um.BURN, 365)) + list(range(365))
    E = np.empty((um.N, len(days), DIM), dtype=np.float32)
    for j, d in enumerate(days):
        E[:, j] = sample_day(family, TDAY_SEAS[d], d, um.N)
    return E.reshape(um.N, len(days) * H, -1)


if __name__ == "__main__":
    SIMS = {}
    for fam, tag in (("gauss", "_gcop"), ("t", "_tcop")):
        E = innovations(fam)
        X = um.margin_map(um.simulate_model(m, E=E))
        del E
        with open(f"{D}/sim_unified_16cell{tag}{SMOKE}.pkl", "wb") as f:
            pickle.dump({"phys": {v: X[v].astype(np.float32 if v == "temp" else np.float16) for v in VARS},
                         "target_index": sc.TARGET,
                         "cells": sc.CELLS, "kmodes": um.KMODES, "copula": fam}, f)
        SIMS[tag] = per_year(daily(X), OBS)
        del X
        log(f"{fam} copula: simulated, saved, scored")
    X = um.margin_map(um.simulate_model(m))                       # empirical copula (whole observed days), same offset years
    SIMS["_emp"] = per_year(daily(X), OBS)
    del X
    log("empirical copula scored")

    # paired proper scores (75 real years)
    print(f"\n{'paired score difference vs empirical copula':44}" + "".join(f"{s:>22}" for s in SIMS["_emp"]))
    for tag in ("_gcop", "_tcop"):
        line = f"  {tag:42}"
        for k in SIMS["_emp"]:
            dlt = SIMS[tag][k] - SIMS["_emp"][k]
            t = dlt.mean() / (dlt.std(ddof=1) / np.sqrt(len(dlt)))
            rel = dlt.mean() / SIMS["_emp"][k].mean()
            line += f"{f'{rel:+.2%} (t={t:+.1f})':>22}"
        print(line)
    print("(t > 2: parametric copula significantly worse, t < -2: significantly better)")

    with open(f"{D}/parametric_copula{SMOKE}.pkl", "wb") as f:
        pickle.dump({"fit": {s: {k: v for k, v in FIT[s].items() if not k.startswith(("R_", "L_"))} for s in FIT},
                     "scores": SIMS, "nu_grid": NU_GRID}, f)
    log(f"saved {D}/parametric_copula{SMOKE}.pkl")
