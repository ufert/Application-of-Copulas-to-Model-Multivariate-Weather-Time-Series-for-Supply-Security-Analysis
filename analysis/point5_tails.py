# Thesis: Results, Dependence in the Extremes (chi(u) of cold and calm, high demand and low feed-in)
# Run: cd analysis && python point5_tails.py   (after point5_model05.py)
# Writes: data/point5_tails.pkl
"""Dependence in the extremes: does the dependence between the variables become stronger in the extremes?

pairs     in feed-in terms:
            calm-calm     onshore CF of two cells, all 120 pairs (by distance), all hours
            cold-calm     national temperature low and national onshore CF low, winter (DJF)
            hot-calm      national temperature high and national onshore CF low, summer (JJA)
            dark-calm     national PV CF low and onshore CF low, winter daylight hours
            calm persists national onshore CF low now and 24 h / 72 h later, all hours
            demand-supply demand high and renewable feed-in (onshore + offshore + PV) low, winter (2030 system of
                          point1_night_ev.py)
margins   uniforms by rank within each (month, hour) slot (the calendar removed); observed ranks within 1950-1999 and
          2000-2024 separately (level trends removed)
measures  chi(u) = P(V <= u | U <= u) for u = 0.5 ... 0.01 (lower tails; upper tails mirrored); chi_bar(u) = 2 log u /
          log P(U <= u, V <= u) - 1; benchmark chi_G(u) of a Gaussian copula with the same normal-scores correlation;
          90 % intervals by bootstrap over years (300 resamples)
sets      observed 75 years (1950-2024); model: the per-year tail counts of the 3,000 simulated years from
          data/point5_model05.pkl (ranks within each batch of 300 years); --model=300 uses only the 300 years of
          unified_model_05.py
Writes data/point5_tails.pkl.
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

import point5_common as pc

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)
U_GRID = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
B = 300
rng = np.random.default_rng(51)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
DIST = pc.cell_distance_km()


# chi(u) of a Gaussian copula with correlation rho (benchmark)
def chi_gauss(rho, u):
    z = stats.norm.ppf(u)
    return stats.multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf([z, z]) / u


def per_year_counts(U, V, mask):
    """per year: n(U<=u), n(U<=u, V<=u) for the u grid, and the normal-score sums for rho_N"""
    m = np.broadcast_to(mask, U.shape)
    ina = (U[..., None] <= U_GRID) & m[..., None]
    joint = ina & (V[..., None] <= U_GRID)
    z1, z2 = stats.norm.ppf(np.clip(U, 1e-6, 1 - 1e-6)) * m, stats.norm.ppf(np.clip(V, 1e-6, 1 - 1e-6)) * m
    return {"na": ina.sum(axis=1), "nj": joint.sum(axis=1), "n": m.sum(axis=1), "s1": z1.sum(axis=1), "s2": z2.sum(axis=1),
            "s11": (z1 ** 2).sum(axis=1), "s22": (z2 ** 2).sum(axis=1), "s12": (z1 * z2).sum(axis=1)}


# correlation, chi(u) and chi-bar(u) from the pooled counts of the units rows
def estimate(c, rows=None):
    rows = np.arange(len(c["n"])) if rows is None else rows
    s = {k: v[rows].sum(axis=0) for k, v in c.items()}
    n = s["n"]
    cov = s["s12"] / n - s["s1"] / n * s["s2"] / n
    rho = cov / np.sqrt((s["s11"] / n - (s["s1"] / n) ** 2) * (s["s22"] / n - (s["s2"] / n) ** 2))
    chi = s["nj"] / np.maximum(s["na"], 1)
    pj = s["nj"] / n
    chibar = np.where(pj > 0, 2 * np.log(s["na"] / n) / np.log(np.maximum(pj, 1e-12)) - 1, np.nan)
    return rho, chi, chibar


# estimates with a 90 % bootstrap interval over the units (years) and the Gaussian benchmark
def analyse(c):
    rho, chi, chibar = estimate(c)
    nunits = len(c["n"])
    bs = [estimate(c, rng.integers(0, nunits, nunits)) for _ in range(B)]
    chis = np.array([b[1] for b in bs])
    rhos = np.array([b[0] for b in bs])
    return {"rho": rho, "rho lo": np.quantile(rhos, 0.05), "rho hi": np.quantile(rhos, 0.95), "chi": chi,
            "chi lo": np.quantile(chis, 0.05, axis=0), "chi hi": np.quantile(chis, 0.95, axis=0), "chibar": chibar,
            "chi_G": np.array([chi_gauss(rho, u) for u in U_GRID])}


MODEL300 = "--model=300" in sys.argv
SETS = {}
for name in (("real", "model") if MODEL300 else ("real",)):
    X = pc.load_real() if name == "real" else pc.load_model()
    F = pc.feedin(X)
    del X
    groups = pc.REAL_PERIOD if name == "real" else None
    SETS[name] = {"F": F, "groups": groups}
    log(f"{name}: feed-in done ({F['on'].shape[0]} years)")
SYS = pc.system_series()

djf, jja = pc.SEAS == "DJF", pc.SEAS == "JJA"
RES = {}
for name, S in SETS.items():
    F, g = S["F"], S["groups"]
    u_on = pc.slot_uniform(F["on"], g)
    u_t = pc.slot_uniform(F["tnat"], g)
    u_pv = pc.slot_uniform(F["pv"], g)
    day = F["pv"].mean(axis=0) > 0.05                                             # climatological daylight
    sysn = SYS[name]
    u_dem = pc.slot_uniform(sysn["demand"], g)
    u_re = pc.slot_uniform(sysn["re"], g)
    pairs = {"cold-calm (DJF)": (u_t, u_on, djf), "hot-calm (JJA)": (1 - u_t, u_on, jja),
             "dark-calm (DJF daylight)": (u_pv, u_on, djf & day),
             "calm persists 24 h": (u_on[:, :-24], u_on[:, 24:], np.ones(8760 - 24, bool)),
             "calm persists 72 h": (u_on[:, :-72], u_on[:, 72:], np.ones(8760 - 72, bool)),
             "high demand-low RE (DJF)": (1 - u_dem, u_re, djf)}
    for pname, (a, b, m) in pairs.items():
        RES[(name, pname)] = analyse(per_year_counts(a, b, m))
    uc = np.stack([pc.slot_uniform(F["cells"][..., k], g) for k in range(16)], axis=-1)
    for i in range(16):
        for j in range(i + 1, 16):
            RES[(name, f"cells {pc.CELLS[i]}-{pc.CELLS[j]}")] = analyse(per_year_counts(uc[..., i], uc[..., j], np.ones(8760, bool)))
    log(f"{name}: tail dependence done")
if not MODEL300:
    M3 = pickle.load(open(f"{pc.D}/point5_model05.pkl", "rb"))
    for pname, c in M3["tails counts"].items():
        RES[("model", pname)] = analyse(c)
    log(f"model: tail dependence of {M3['years']} simulated years from point5_model05.pkl")

# tables
iu = [list(U_GRID).index(u) for u in (0.2, 0.05, 0.01)]
rows = {}
for (name, pname), r in RES.items():
    if pname.startswith("cells"):
        continue
    rows[(pname, name)] = {"rho_N": r["rho"], "chi(0.2)": r["chi"][iu[0]], "chi(0.05)": r["chi"][iu[1]],
                           "chi(0.05) 90%": f"{r['chi lo'][iu[1]]:.2f}-{r['chi hi'][iu[1]]:.2f}", "chi(0.01)": r["chi"][iu[2]],
                           "Gauss chi(0.05)": r["chi_G"][iu[1]], "Gauss chi(0.01)": r["chi_G"][iu[2]],
                           "chi/Gauss at 0.2": r["chi"][iu[0]] / r["chi_G"][iu[0]], "chi/Gauss at 0.05": r["chi"][iu[1]] / r["chi_G"][iu[1]],
                           "chi/Gauss at 0.01": r["chi"][iu[2]] / r["chi_G"][iu[2]], "chi_bar(0.05)": r["chibar"][iu[1]],
                           "chi_bar(0.01)": r["chibar"][iu[2]]}
T = pd.DataFrame(rows).T
print(f"\nPAIRS: normal-scores correlation, chi(u) with the Gaussian benchmark of the same rho_N, chi_bar (real: 75 years; "
      f"model: {300 if MODEL300 else 3000})")
print(T.to_string(float_format=lambda v: f"{v:.3f}"))

# cell pairs by distance
cp = []
for (name, pname), r in RES.items():
    if not pname.startswith("cells"):
        continue
    a, b = pname.split()[1].split("-")
    d = DIST[pc.CELLS.index(a), pc.CELLS.index(b)]
    cp.append({"set": name, "pair": pname, "distance km": d, "rho_N": r["rho"], "chi(0.2)": r["chi"][iu[0]], "chi(0.05)": r["chi"][iu[1]],
               "chi(0.01)": r["chi"][iu[2]], "Gauss chi(0.05)": r["chi_G"][iu[1]], "Gauss chi(0.01)": r["chi_G"][iu[2]],
               "chi_bar(0.05)": r["chibar"][iu[1]], "chi lo 0.05": r["chi lo"][iu[1]], "chi hi 0.05": r["chi hi"][iu[1]]})
CP = pd.DataFrame(cp)
CP["band"] = pd.cut(CP["distance km"], [0, 250, 450, 700, 1000], labels=["< 250 km", "250-450 km", "450-700 km", "> 700 km"])
G = CP.groupby(["band", "set"], observed=True)[["rho_N", "chi(0.2)", "chi(0.05)", "Gauss chi(0.05)", "chi(0.01)", "Gauss chi(0.01)",
                                                 "chi_bar(0.05)"]].mean()
G["chi/Gauss at 0.05"] = G["chi(0.05)"] / G["Gauss chi(0.05)"]
G["chi/Gauss at 0.01"] = G["chi(0.01)"] / G["Gauss chi(0.01)"]
print("\nCALM-CALM between cells, by distance (mean over the pairs of a band)")
print(G.round(3).to_string())
real_cp = CP[CP.set == "real"].set_index("pair")
mod_cp = CP[CP.set == "model"].set_index("pair")
inside = ((mod_cp["chi(0.05)"] >= real_cp["chi lo 0.05"]) & (mod_cp["chi(0.05)"] <= real_cp["chi hi 0.05"])).mean()
print(f"cell pairs: model chi(0.05) inside the real 90% interval for {inside:.0%}; mean model/real chi(0.05) "
      f"{(mod_cp['chi(0.05)'] / real_cp['chi(0.05)']).mean():.2f}; mean model/real rho_N {(mod_cp['rho_N'] / real_cp['rho_N']).mean():.2f}")

# model vs real for the named pairs
MR = {}
for pname in dict.fromkeys(p for (_, p) in RES if not p.startswith("cells")):
    r, m = RES[("real", pname)], RES[("model", pname)]
    MR[pname] = {"rho_N real / model": f"{r['rho']:.2f} / {m['rho']:.2f}",
                 "chi(0.05) real [90%] / model": f"{r['chi'][iu[1]]:.2f} [{r['chi lo'][iu[1]]:.2f}-{r['chi hi'][iu[1]]:.2f}] / {m['chi'][iu[1]]:.2f}",
                 "model inside real interval (0.05)": bool(r["chi lo"][iu[1]] <= m["chi"][iu[1]] <= r["chi hi"][iu[1]]),
                 "chi(0.01) real / model": f"{r['chi'][iu[2]]:.2f} / {m['chi'][iu[2]]:.2f}",
                 "model/real chi(0.05)": m["chi"][iu[1]] / r["chi"][iu[1]]}
print("\nMODEL vs REAL")
print(pd.DataFrame(MR).T.to_string())
with open(f"{pc.D}/point5_tails.pkl", "wb") as f:
    pickle.dump({"u": U_GRID, "results": RES, "table": T, "cells": CP, "by distance": G, "model vs real": pd.DataFrame(MR).T}, f)
log(f"saved {pc.D}/point5_tails.pkl")
