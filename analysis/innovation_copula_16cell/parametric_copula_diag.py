# Thesis: Case Study, Innovation Copula (dependence within a day of observed and parametric innovations)
# Run: cd analysis/innovation_copula_16cell && python parametric_copula_diag.py   (~4 min, after parametric_copula.py)
"""Dependence within a day of the observed innovations and of innovations drawn from the fitted Gaussian and Student-t
copulas (parametric_copula.py). For each observed residual day, one synthetic day is drawn from each copula through the
same +-10-day pool margins, so that the sets differ only in the copula. Compared per season:
  neighbour corr     same-hour Pearson correlation of the remainder residuals of neighbouring cells (irradiance:
                     daylight hours)
  daily-sum corr     the same for the daily sums over the daylight hours
  daily-sum var      Var(sum over the daylight hours) / sum of the hourly variances
  |e| clustering     lag-1 h correlation of the standardised absolute residuals
  chi_L / chi_U      lower / upper 10 % co-exceedance of neighbouring irradiance cells, rank scale
Run: python parametric_copula_diag.py
"""
import os
import sys

# the shared modules (paths, spatial_common, ...) live one folder up, in analysis/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

import numpy as np

import parametric_copula as pc
import paths  # repository folders: data, figures, external raw data (see paths.py)

K = sum(pc.um.KMODES.values())
CELLS = pc.sc.CELLS
REM = {v: K + 16 * j + np.arange(16) for j, v in enumerate(pc.VARS)}
SOLAR_MODES = np.arange(4, 7)
PAIRS = [(CELLS.index(f"r{r}c{c}"), CELLS.index(f"r{r + dr}c{c + dc}"))
         for r in range(4) for c in range(4) for dr, dc in ((0, 1), (1, 0)) if r + dr < 4 and c + dc < 4]
Q = 0.10


def synthetic(family, s):
    """one synthetic day per real day of season s, drawn for the real day's own pool"""
    sel = np.where(pc.SEAS == s)[0]
    out = np.empty((len(sel), pc.DIM), dtype=np.float32)
    for d in np.unique(pc.DOY[sel]):
        rows = np.where(pc.DOY[sel] == d)[0]
        out[rows] = pc.sample_day(family, s, d - 1, len(rows))
    return out.reshape(len(sel), pc.H, -1)


# pseudo-observations: ranks scaled to (0, 1) within each column
def ranks(x):
    return (x.argsort(axis=0).argsort(axis=0) + 0.5) / len(x)


def stats_of(X, day_h):
    """X (days, 24, 55) residual days; day_h: daylight hours for solar"""
    r = {}
    for v in pc.VARS:
        hrs = day_h if v == "solar" else np.arange(24)
        a = X[:, hrs][..., REM[v]]
        r[f"neighbour corr {v}"] = np.mean([np.corrcoef(a[..., i].ravel(), a[..., j].ravel())[0, 1] for i, j in PAIRS])
        s = a.sum(axis=1)
        r[f"daily-sum corr {v}"] = np.mean([np.corrcoef(s[:, i], s[:, j])[0, 1] for i, j in PAIRS])
        r[f"daily-sum var {v}"] = np.mean(s.var(axis=0) / a.var(axis=0).sum(axis=0))
    m = X[:, day_h][..., SOLAR_MODES]
    r["daily-sum var solar modes"] = np.mean(m.sum(axis=1).var(axis=0) / m.var(axis=0).sum(axis=0))
    z = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
    for v, comps in (("wind", REM["wind"]), ("solar", REM["solar"])):
        hrs = day_h if v == "solar" else np.arange(24)
        a, b = z[:, hrs[:-1]][..., comps], z[:, hrs[1:]][..., comps]
        r[f"|e| clustering {v}"] = np.mean([np.corrcoef(a[..., c].ravel(), b[..., c].ravel())[0, 1]
                                           for c in range(len(comps))])
    a = X[:, day_h][..., REM["solar"]]
    lo, hi = [], []
    for h in range(a.shape[1]):
        u = ranks(a[:, h])
        for i, j in PAIRS:
            lo.append(((u[:, i] < Q) & (u[:, j] < Q)).mean() / Q)
            hi.append(((u[:, i] > 1 - Q) & (u[:, j] > 1 - Q)).mean() / Q)
    r["chi_L solar neighbours (overcast)"] = np.mean(lo)
    r["chi_U solar neighbours (clear)"] = np.mean(hi)
    return r


RES = {}
for s in ("winter", "summer"):
    real = pc.FLAT[pc.SEAS == s].reshape(-1, pc.H, pc.FLAT.shape[1] // pc.H)
    sd = real[..., REM["solar"]].std(axis=(0, 2))
    day_h = np.where(sd > 0.5 * sd.max())[0]
    RES[(s, "real")] = stats_of(real, day_h)
    for fam in ("gauss", "t"):
        RES[(s, fam)] = stats_of(synthetic(fam, s), day_h)
    pc.log(f"{s}: daylight hours UTC {day_h.min()}-{day_h.max()}")

import pandas as pd  # noqa: E402

T = pd.DataFrame(RES)
pd.set_option("display.width", 200)
print(T.round(3).to_string())
with open(f"{pc.D}/parametric_copula_diag.pkl", "wb") as f:
    pickle.dump(T, f)
pc.log(f"saved {pc.D}/parametric_copula_diag.pkl")
