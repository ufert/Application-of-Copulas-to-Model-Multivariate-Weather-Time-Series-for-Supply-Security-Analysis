# Thesis: shared helpers of the regional and unified models: preprocessing, seasonal AR and VAR with BIC,
# EOF modes, simulation, GPD tails of temperature and the evaluation statistics
# Module: imported by the model scripts
"""Shared functions of the spatial-dependence tests (4 quadrants, 16 cells).

Preprocessing per region: wind in Box-Cox space (lambda = 0.25), linear trend and 15-day-smoothed (month, day, hour)
climatology removed; each trajectory adds the mean anomaly of one randomly drawn training year (interannual
variability). Simulated calendar: 2025, 300 trajectories.

Two dynamics, both seasonal and block-masked (lags never cross a season or year gap), orders by BIC:
  AR   one AR(p <= 168) per region              -> diagonal coefficient blocks
  VAR  one VAR(p) across all regions, p from a lag grid -> full blocks
Both are stored as B (1 + d*p, d): row 0 constant, row 1 + (lag-1)*d + i the coefficient of region i at that lag, so
one simulator serves both.

Two innovation schemes, both bootstraps of the fitted residuals of the same season:
  ind    the residual of each region drawn independently (no contemporaneous spatial dependence)
  joint  whole residual rows drawn (the empirical copula of the residuals)
Also the generalized Pareto tails of temperature and the evaluation statistics.
"""
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats
import paths  # repository folders: data, figures, external raw data (see paths.py)

sys.path.insert(0, paths.CODE)
sys.path.insert(0, paths.ANALYSIS)
from vectorized_simulation import WIND_LAMBDA, wind_fwd, wind_inv  # noqa: E402
from simulate_4_16_regions import (SEASONS, build_smoothed_climatology_ns, fit_ar_model_seasonal_one,  # noqa: E402
                                   fit_linear_trend_ns, get_season, map_climatology_ns, season_block_ids,
                                   trend_value_ns)

D = paths.DATA
N_TRAJ = 300
TARGET = pd.date_range("2025-01-01", periods=8760, freq="h")
SEASON_LAB = np.array([get_season(m) for m in TARGET.month])
BURN_DAYS = 61     # warm-up before the target year (1 Nov - 31 Dec, their coefficients and pools), discarded,
                   # so trajectories do not all start from the real end-of-2024 state
BURN_SEAS = np.array([get_season(m) for m in pd.date_range("2024-11-01", periods=24 * BURN_DAYS, freq="h").month])
SIM_SEAS = np.concatenate([BURN_SEAS, SEASON_LAB])
SCALE = (100 / 10) ** 0.25
QUADRANTS = {"Q_NW": [f"r{r}c{c}" for r in [0, 1] for c in [0, 1]],
             "Q_NE": [f"r{r}c{c}" for r in [0, 1] for c in [2, 3]],
             "Q_SW": [f"r{r}c{c}" for r in [2, 3] for c in [0, 1]],
             "Q_SE": [f"r{r}c{c}" for r in [2, 3] for c in [2, 3]]}
CELLS = [f"r{r}c{c}" for r in range(4) for c in range(4)]


# data and preprocessing
def load_regions(groups):
    g = pd.read_pickle(f"{D}/grid4x4_wind_2000-2024.pkl")
    return pd.DataFrame({rn: g[[f"{c}_wind" for c in cs]].mean(axis=1) for rn, cs in groups.items()})


# transform, detrend and remove the smoothed climatology; anomalies and the pieces to undo it
def preprocess(df):
    cols = list(df.columns)
    y = pd.DataFrame(wind_fwd(df.to_numpy()), index=df.index, columns=cols)
    trend = fit_linear_trend_ns(y, cols, window_days=15)
    det = y - trend_value_ns(y.index, trend, cols)
    clim = build_smoothed_climatology_ns(det, cols, window_days=15)
    anom = y - (map_climatology_ns(y.index, clim, cols) + trend_value_ns(y.index, trend, cols))
    return {"anom": anom, "offsets": anom.groupby(anom.index.year).mean(),
            "sim_mean": map_climatology_ns(TARGET, clim, cols).to_numpy()}


# dynamics
def fit_ar(anom, maxlag=168):
    cols, d = list(anom.columns), anom.shape[1]
    fits = {}
    for s in SEASONS:
        bid = season_block_ids(anom.index, s)
        parts = [fit_ar_model_seasonal_one(y=anom[c], block_id=bid, col=c, maxlag=maxlag) for c in cols]
        p = max(max(f["lags"], default=0) for f in parts)
        B = np.zeros((1 + d * p, d))
        for i, (c, f) in enumerate(zip(cols, parts)):
            B[0, i] = f["params"]["const"]
            for lag in f["lags"]:
                B[1 + (lag - 1) * d + i, i] = f["params"][f"{c}.L{lag}"]
        pool = pd.concat([f["resid"].rename(c) for c, f in zip(cols, parts)], axis=1).dropna()
        fits[s] = {"B": B, "p": p, "pool": pool.to_numpy(), "pool_df": pool,
                   "orders": [len(f["lags"]) for f in parts]}
    return fits


# lagged design matrix of a seasonal fit (rows with all lags inside the same season block)
def _design(Y, bid, p):
    n = len(Y)
    cols = []
    for lag in range(1, p + 1):
        sv = np.roll(Y, lag, axis=0)
        sb = np.roll(bid, lag)
        ok = (bid == sb) & (bid != -1) & (np.arange(n) >= lag)
        cols.append(np.where(ok[:, None], sv, np.nan))
    X = np.column_stack([np.ones(n)] + cols)
    valid = (bid != -1) & ~np.isnan(X).any(axis=1)
    return X[valid], Y[valid], valid


def fit_var(anom, grid):
    """Seasonal VAR, BIC over `grid` on a common sample (rows with max(grid)
    lags of in-block history), nested models solved from one QR."""
    Y, d, pmax = anom.to_numpy(dtype=float), anom.shape[1], max(grid)
    fits = {}
    for s in SEASONS:
        X, y, valid = _design(Y, season_block_ids(anom.index, s), pmax)
        n = len(y)
        Q, R = np.linalg.qr(X)
        C = Q.T @ y
        yty = y.T @ y
        bics = {}
        for p in grid:
            k = 1 + d * p
            if n < 10 * k:
                continue
            S = (yty - C[:k].T @ C[:k]) / n
            bics[p] = n * np.linalg.slogdet(S)[1] + (d * d * p + d) * np.log(n)
        p = min(bics, key=bics.get)
        k = 1 + d * p
        B = np.linalg.solve(R[:k, :k], C[:k])
        fits[s] = {"B": B, "p": p, "pool": y - X[:, :k] @ B, "bics": bics, "index": anom.index[valid]}
        del Q
    return fits


def fit_eof(anom, k, grid, maxlag=168):
    """State = [k leading PC scores | 16-cell remainder field]. PC scores
    follow a seasonal VAR (coherent large-scale dynamics), the remainder a
    per-cell AR; innovations are bootstrapped as joint rows of both, so the
    contemporaneous dependence between and within them is kept. Cost of
    the VAR part does not grow with the number of cells."""
    assert max(grid) == maxlag, "common valid rows need the same maximum lag"
    X = anom.to_numpy(dtype=float)
    d = X.shape[1]
    mu = X.mean(axis=0)
    ev, evec = np.linalg.eigh(np.cov((X - mu).T))
    order = np.argsort(ev)[::-1]
    E = evec[:, order[:k]]
    Sc = pd.DataFrame((X - mu) @ E, index=anom.index, columns=[f"pc{i + 1}" for i in range(k)])
    Rf = pd.DataFrame(X - mu - Sc.to_numpy() @ E.T, index=anom.index, columns=anom.columns)
    vf, af = fit_var(Sc, grid), fit_ar(Rf, maxlag=maxlag)
    Dst = k + d
    fits = {}
    for s in SEASONS:
        pv_ = pd.DataFrame(vf[s]["pool"], index=vf[s]["index"])
        pa_ = af[s]["pool_df"]
        common = pv_.index.intersection(pa_.index)
        p = max(vf[s]["p"], af[s]["p"])
        B = np.zeros((1 + Dst * p, Dst))
        B[0, :k], B[0, k:] = vf[s]["B"][0], af[s]["B"][0]
        for lag in range(1, vf[s]["p"] + 1):
            B[1 + (lag - 1) * Dst:1 + (lag - 1) * Dst + k, :k] = vf[s]["B"][1 + (lag - 1) * k:1 + lag * k]
        for lag in range(1, af[s]["p"] + 1):
            B[1 + (lag - 1) * Dst + k:1 + lag * Dst, k:] = af[s]["B"][1 + (lag - 1) * d:1 + lag * d]
        fits[s] = {"B": B, "p": p, "pool": np.hstack([pv_.loc[common].to_numpy(), pa_.loc[common].to_numpy()]),
                   "var_p": vf[s]["p"], "n_common": len(common), "n_var": len(pv_), "n_ar": len(pa_)}
    explained = ev[order] / ev.sum()
    project = lambda A: A[..., :k] @ E.T + A[..., k:] + mu
    return fits, np.hstack([Sc.to_numpy(), Rf.to_numpy()]), project, explained


# spectral radius of the companion matrix of a VAR (below one: stable)
def spectral_radius(B, d):
    p = (B.shape[0] - 1) // d
    A = B[1:].reshape(p, d, d).transpose(0, 2, 1)          # A[k][i, j]: effect of region j at lag k+1 on region i
    comp = np.zeros((d * p, d * p))
    comp[:d] = np.concatenate(list(A), axis=1)
    comp[d:, :-d] = np.eye(d * (p - 1))
    return float(np.abs(np.linalg.eigvals(comp)).max())


def longrun_corr(B, pool):
    """Correlation of the process spectrum at frequency zero,
    (I - A(1))^-1 Sigma (I - A(1))^-T - the coherence of slow (multi-day)
    variations implied by the fitted dynamics."""
    d = pool.shape[1]
    p = (B.shape[0] - 1) // d
    A1 = B[1:].reshape(p, d, d).sum(axis=0).T
    M = np.linalg.inv(np.eye(d) - A1)
    S0 = M @ np.cov(pool.T) @ M.T
    s = np.sqrt(np.diag(S0))
    C = S0 / np.outer(s, s)
    return C[~np.eye(d, dtype=bool)].mean()


# simulate fitted seasonal models with the chosen innovation scheme and undo the preprocessing
def simulate(pre, fits, innov, seed, hist=None, project=None):
    offsets, sim_mean = pre["offsets"], pre["sim_mean"]
    hist = pre["anom"].dropna().to_numpy() if hist is None else hist
    Dst, T = hist.shape[1], len(TARGET)
    pmax = max(f["p"] for f in fits.values())
    rng = np.random.default_rng(seed)
    yrs = rng.choice(offsets.index.to_numpy(), size=N_TRAJ)

    # draw innovations for every time step from the pool of its season
    def draw(labels):
        E = np.empty((N_TRAJ, len(labels), Dst))
        for s in SEASONS:
            m = labels == s
            if not m.any():
                continue
            pool = fits[s]["pool"]
            if innov == "joint":
                E[:, m] = pool[rng.integers(0, len(pool), size=(N_TRAJ, int(m.sum())))]
            else:
                for i in range(Dst):
                    E[:, m, i] = pool[rng.integers(0, len(pool), size=(N_TRAJ, int(m.sum()))), i]
        return E

    E = draw(SEASON_LAB)
    E = np.concatenate([draw(BURN_SEAS), E], axis=1)      # warm-up drawn after the target year, which stays unchanged
    TB = len(BURN_SEAS)
    sim = np.empty((N_TRAJ, pmax + TB + T, Dst))
    sim[:, :pmax] = hist[-pmax:][None]
    for t in range(TB + T):
        f = fits[SIM_SEAS[t]]
        i, p = t + pmax, f["p"]
        lagstack = sim[:, i - p:i][:, ::-1].reshape(N_TRAJ, p * Dst)
        sim[:, i] = f["B"][0] + lagstack @ f["B"][1:] + E[:, t]
    del E
    A = sim[:, pmax + TB:]
    if project is not None:
        A = project(A)
    return wind_inv(A + sim_mean[None] + offsets.loc[yrs].to_numpy()[:, None, :])   # (N, T, d) m/s


# evaluation
def cf(v10):
    v = np.asarray(v10, dtype=float) * SCALE
    out = np.zeros_like(v)
    r = (v >= 3) & (v < 12)
    out[r] = (v[r] ** 3 - 27) / (1728 - 27)
    out[(v >= 12) & (v < 25)] = 1.0
    return out


# lengths of the runs below thr (lulls)
def lulls(b, thr=0.10):
    x = (b < thr).astype(np.int8)
    dd = np.diff(np.concatenate([[0], x, [0]]))
    return np.where(dd == -1)[0] - np.where(dd == 1)[0]


def coherence(W, idx, block=1):
    """mean off-diagonal correlation of (month, hour) anomalies, after
    averaging over non-overlapping blocks of `block` hours"""
    df = pd.DataFrame(W, index=idx)
    an = (df - df.groupby([idx.month, idx.hour]).transform("mean")).to_numpy()
    if block > 1:
        n = len(an) // block * block
        an = an[:n].reshape(-1, block, an.shape[1]).mean(axis=1)
    C = np.corrcoef(an.T)
    return C[~np.eye(C.shape[0], dtype=bool)].mean()


def summarize(name, W, idx_list, n_years):
    """W: list of (T_j, d) arrays (one per year or trajectory) with calendars idx_list"""
    B = [cf(w).mean(axis=1) for w in W]
    runs = np.concatenate([lulls(b) for b in B])
    allB = np.concatenate(B)
    take = range(0, len(W), max(1, len(W) // 12))
    row = {"variant": name}
    for blk, lab in ((1, "corr 1h"), (24, "corr 24h"), (72, "corr 72h")):
        row[lab] = np.mean([coherence(W[j], idx_list[j], blk) for j in take]) if len(W) > 1 else coherence(W[0], idx_list[0], blk)
    row.update({"mean CF": allB.mean(), "CF<5% h/yr": (allB < 0.05).sum() / n_years,
                "lulls>=24h /yr": (runs >= 24).sum() / n_years, "lulls>=72h /yr": (runs >= 72).sum() / n_years,
                "longest h": int(runs.max())})
    return row


def _quantile(ref, u):
    """empirical quantile function of a sorted sample, as used by margfix"""
    return np.interp(u, np.linspace(0, 1, len(ref)), ref)


def fit_gpd_tails(real, real_idx, p=0.02):
    """Generalized Pareto tails for margfix. Per (cell, month, hour) slot the empirical
    quantiles at p and 1 - p are the thresholds and the slot IQR is the unit; the exceedances
    of all slots of a season are pooled per cell. Shape xi per (season, tail) common to all
    cells (regional shape, by profile likelihood), scale sigma per (cell, season, tail).
    Returns {"p", "xi": {(season, tail)}, "sigma": {(cell, season, tail)}, "n": {...}}."""
    rm, rh = real_idx.month.to_numpy(), real_idx.hour.to_numpy()
    nc = real.shape[1]
    exc = {(c, s, t): [] for c in range(nc) for s in SEASONS for t in ("lower", "upper")}
    for c in range(nc):
        for mo in range(1, 13):
            for h in range(24):
                ref = np.sort(real[(rm == mo) & (rh == h), c])
                iqr = np.subtract(*np.percentile(ref, [75, 25]))
                lo, hi = _quantile(ref, p), _quantile(ref, 1 - p)
                exc[(c, get_season(mo), "upper")].append((ref[ref > hi] - hi) / iqr)
                exc[(c, get_season(mo), "lower")].append((lo - ref[ref < lo]) / iqr)
    exc = {k: np.concatenate(v) for k, v in exc.items()}
    fit = {"p": p, "xi": {}, "sigma": {}, "n": {k: len(v) for k, v in exc.items()}}
    for s in SEASONS:
        for t in ("lower", "upper"):
            ys = [exc[(c, s, t)] for c in range(nc)]

            # negative profile log-likelihood of the GPD shape xi, the scale fitted per series
            def nll(xi):
                return -sum(stats.genpareto.logpdf(y, xi, scale=stats.genpareto.fit(y, f0=xi, floc=0)[2]).sum()
                            for y in ys)
            xi = optimize.minimize_scalar(nll, bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-3}).x
            fit["xi"][(s, t)] = xi
            for c, y in enumerate(ys):
                fit["sigma"][(c, s, t)] = stats.genpareto.fit(y, f0=xi, floc=0)[2]
    return fit


def margfix(W, real, real_idx, tails=None):
    """quantile-map each region of the sim onto the real distribution per
    (month, hour): monotone, so the sim's dependence (ranks) is untouched.
    tails (fit_gpd_tails): the outer p of each slot follows the fitted GPD instead of the
    empirical quantiles, so values can go beyond the observed range; continuous at the
    thresholds. Default: empirical throughout (bounded by the observed range)."""
    out = np.empty_like(W)
    rm, rh = real_idx.month.to_numpy(), real_idx.hour.to_numpy()
    sm, sh = TARGET.month.to_numpy(), TARGET.hour.to_numpy()
    for c in range(W.shape[2]):
        for mo in range(1, 13):
            for h in range(24):
                ss = (sm == mo) & (sh == h)
                ref = np.sort(real[(rm == mo) & (rh == h), c])
                blk = W[:, ss, c]
                u = (blk.reshape(-1).argsort().argsort() + 0.5) / blk.size
                x = _quantile(ref, u)
                if tails is not None:
                    p, s = tails["p"], get_season(mo)
                    iqr = np.subtract(*np.percentile(ref, [75, 25]))
                    up, lo = u > 1 - p, u < p
                    gp = lambda q, t: stats.genpareto.ppf(q, tails["xi"][(s, t)], scale=tails["sigma"][(c, s, t)])
                    x[up] = _quantile(ref, 1 - p) + iqr * gp((u[up] - (1 - p)) / p, "upper")
                    x[lo] = _quantile(ref, p) - iqr * gp((p - u[lo]) / p, "lower")
                out[:, ss, c] = x.reshape(blk.shape)
    return out


# observed statistics and a decoupled reference (years shuffled independently per cell, no spatial dependence)
def real_reference(df, n_shuffle=20, seed=4242):
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    years = sorted(df.index.year.unique())
    W = [df.loc[str(y)].to_numpy() for y in years]
    idx = [df.loc[str(y)].index for y in years]
    rows = [summarize("REAL", W, idx, len(years))]
    by_year = np.stack(W)
    rng = np.random.default_rng(seed)
    dec = []
    for _ in range(n_shuffle):
        perm = np.stack([rng.permutation(len(years)) for _ in range(df.shape[1])], axis=1)
        Wd = np.stack([by_year[perm[:, c], :, c] for c in range(df.shape[1])], axis=2)
        dec.append(summarize("REAL-DECOUPLED", list(Wd), idx, len(years)))
    dec = pd.DataFrame(dec)
    rows.append({"variant": "REAL-DECOUPLED", **dec.drop(columns="variant").mean().to_dict()})
    return rows, df


# statistics of a simulation before and after mapping it onto the observed margins
def evaluate(name, W, real_df):
    rows = [summarize(name + " (raw)", list(W), [TARGET] * len(W), len(W))]
    Wf = margfix(W, real_df.to_numpy(), real_df.index)
    rows.append(summarize(name, list(Wf), [TARGET] * len(Wf), len(Wf)))
    return rows


# recovered share of the spatial dependence: (model - decoupled) / (observed - decoupled)
def recovery_table(res, variants, cols=("corr 1h", "corr 24h", "corr 72h", "CF<5% h/yr",
                                        "lulls>=24h /yr", "lulls>=72h /yr", "longest h")):
    R, F = res.loc["REAL"], res.loc["REAL-DECOUPLED"]
    out = {}
    for v in variants:
        out[v] = {c: (res.loc[v, c] - F[c]) / (R[c] - F[c]) for c in cols}
    return pd.DataFrame(out).T
