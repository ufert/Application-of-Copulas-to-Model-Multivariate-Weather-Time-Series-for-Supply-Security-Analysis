# Thesis: Results, Dependence in the Extremes (dark and calm, dark two days running)
# Run: cd analysis && python point5_tails_solar.py
# Writes: data/point5_tails_solar.pkl
"""Dependence in the extremes for the pairs involving irradiance, with the measures and conventions of point5_tails.py
(uniforms by rank within calendar slots, observed ranks within 1950-1999 and 2000-2024 separately, chi(u) against a
Gaussian copula with the same normal-scores correlation, 90 % bootstrap interval over years).
pairs   dark-cold       national PV CF low and national temperature low, winter daylight hours
        dark persists   daily mean PV CF low today and tomorrow, winter days (uniforms by rank within the month)
        dark-calm       from point5_tails.py (winter daylight hours)
Writes data/point5_tails_solar.pkl.
"""
import pickle

import numpy as np
import pandas as pd
from scipy import stats

import point5_common as pc

U_GRID = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
B = 300
rng = np.random.default_rng(54)


# chi(u) of a Gaussian copula with correlation rho (benchmark)
def chi_gauss(rho, u):
    z = stats.norm.ppf(u)
    return stats.multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf([z, z]) / u


# counts of single and joint exceedances on the grid of levels u
def counts(U, V, mask):
    m = np.broadcast_to(mask, U.shape)
    ina = (U[..., None] <= U_GRID) & m[..., None]
    joint = ina & (V[..., None] <= U_GRID)
    z1, z2 = stats.norm.ppf(np.clip(U, 1e-6, 1 - 1e-6)) * m, stats.norm.ppf(np.clip(V, 1e-6, 1 - 1e-6)) * m
    return {"na": ina.sum(axis=1), "nj": joint.sum(axis=1), "n": m.sum(axis=1), "s1": z1.sum(axis=1), "s2": z2.sum(axis=1),
            "s11": (z1 ** 2).sum(axis=1), "s22": (z2 ** 2).sum(axis=1), "s12": (z1 * z2).sum(axis=1)}


# correlation and chi(u) from the pooled counts of the units rows
def estimate(c, rows=None):
    rows = np.arange(len(c["n"])) if rows is None else rows
    s = {k: v[rows].sum(axis=0) for k, v in c.items()}
    n = s["n"]
    cov = s["s12"] / n - s["s1"] / n * s["s2"] / n
    rho = cov / np.sqrt((s["s11"] / n - (s["s1"] / n) ** 2) * (s["s22"] / n - (s["s2"] / n) ** 2))
    return rho, s["nj"] / np.maximum(s["na"], 1)


# estimates with a 90 % bootstrap interval over the units (years) and the Gaussian benchmark
def analyse(c):
    rho, chi = estimate(c)
    bs = [estimate(c, rng.integers(0, len(c["n"]), len(c["n"]))) for _ in range(B)]
    chis = np.array([b[1] for b in bs])
    return {"rho": rho, "chi": chi, "chi lo": np.quantile(chis, 0.05, axis=0), "chi hi": np.quantile(chis, 0.95, axis=0),
            "chi_G": np.array([chi_gauss(rho, u) for u in U_GRID])}


def month_uniform(x, groups):
    """(units, 365) daily values -> uniforms by rank within each calendar month, per group of years"""
    months = pd.date_range("2025-01-01", periods=365, freq="D").month.to_numpy()
    groups = np.zeros(x.shape[0], int) if groups is None else groups
    u = np.empty(x.shape)
    for g in np.unique(groups):
        r = groups == g
        for m in range(1, 13):
            blk = x[r][:, months == m]
            u[np.ix_(np.where(r)[0], np.where(months == m)[0])] = (stats.rankdata(blk.ravel()) / (blk.size + 1)).reshape(blk.shape)
    return u


RES = {}
djf = pc.SEAS == "DJF"
djf_days = np.isin(pd.date_range("2025-01-01", periods=365, freq="D").month.to_numpy(), [12, 1, 2])
for name in ("real", "model"):
    X = pc.load_real() if name == "real" else pc.load_model()
    F = pc.feedin(X)
    del X
    g = pc.REAL_PERIOD if name == "real" else None
    day = F["pv"].mean(axis=0) > 0.05
    u_pv, u_t = pc.slot_uniform(F["pv"], g), pc.slot_uniform(F["tnat"], g)
    RES[(name, "dark-cold (DJF daylight)")] = analyse(counts(u_pv, u_t, djf & day))
    dpv = month_uniform(F["pv"].reshape(len(F["pv"]), 365, 24).mean(axis=2), g)
    RES[(name, "dark persists 1 day (DJF)")] = analyse(counts(dpv[:, :-1], dpv[:, 1:], djf_days[:-1] & djf_days[1:]))
T5 = pickle.load(open(f"{pc.D}/point5_tails.pkl", "rb"))["results"]
for name in ("real", "model"):
    RES[(name, "dark-calm (DJF daylight)")] = T5[(name, "dark-calm (DJF daylight)")]
i5, i1 = list(U_GRID).index(0.05), list(U_GRID).index(0.01)
rows = {}
for (name, p), r in RES.items():
    rows[(p, name)] = {"rho_N": r["rho"], "chi(0.2)": r["chi"][2], "chi(0.05)": r["chi"][i5], "90%": f"{r['chi lo'][i5]:.3f}-{r['chi hi'][i5]:.3f}",
                       "Gauss chi(0.05)": r["chi_G"][i5], "ratio 0.05": r["chi"][i5] / r["chi_G"][i5] if r["chi_G"][i5] > 0 else np.nan,
                       "chi(0.01)": r["chi"][i1], "Gauss chi(0.01)": r["chi_G"][i1]}
T = pd.DataFrame(rows).T.sort_index()
pd.set_option("display.width", 220)
print(T.to_string(float_format=lambda v: f"{v:.3f}"))
with open(f"{pc.D}/point5_tails_solar.pkl", "wb") as f:
    pickle.dump({"u": U_GRID, "results": RES, "table": T}, f)
print("saved point5_tails_solar.pkl")
