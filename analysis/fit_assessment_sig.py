# Thesis: Results, Fit Assessment (paired tests: are the score differences significant?)
# Run: cd analysis && python fit_assessment_sig.py   (after fit_assessment.py)
# Writes: data/fit_assessment_sig.pkl
"""Significance of the score differences between models: scores per observed year (75 years), paired differences between
two models across the years (the same observations score both models), mean +- standard error and t = mean / SE.
Ensembles from data/fit_assessment_ens05.pkl (fit_assessment.py).
"""
import pickle

import numpy as np
from scipy.spatial.distance import cdist, pdist

exec(open("fit_assessment.py").read().split("# scores (fair ensemble estimators)")[0])


# CRPS per observed year (fair estimator over the ensemble members)
def crps_y(ens, obs):
    N = ens.shape[0]
    out = np.zeros(obs.shape[0])
    for d in range(ens.shape[1]):
        x = np.sort(ens[:, d])
        spread = 2 * np.sum((2 * np.arange(1, N + 1) - N - 1) * x) / (N * (N - 1))
        out += np.abs(x[:, None] - obs[None, :, d]).mean(axis=0) - 0.5 * spread
    return out / ens.shape[1]


IU, JU = np.triu_indices(16, 1)


# spatial variogram score per observed year
def vario_y(ens, obs, p=0.5):
    out = np.zeros(obs.shape[0])
    for d in range(ens.shape[1]):
        e, o = ens[:, d], obs[:, d]
        m = (np.abs(e[:, IU] - e[:, JU]) ** p).mean(axis=0)
        out += ((np.abs(o[:, IU] - o[:, JU]) ** p - m[None]) ** 2).sum(axis=1)
    return out / ens.shape[1]


# coupling variogram score per observed year, across standardized national wind, temperature and irradiance
def vario3_y(ens, obs, p=0.5):
    z = lambda X: np.stack([(X[v] - NAT_MU[v]) / NAT_SD[v] for v in VARS], axis=2)
    e, o = z(ens), z(obs)
    tot = np.zeros(o.shape[0])
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        m = (np.abs(e[..., i] - e[..., j]) ** p).mean(axis=0)
        tot += ((np.abs(o[..., i] - o[..., j]) ** p - m[None]) ** 2).mean(axis=1)
    return tot


# all scores per observed year, for the paired tests between models
def per_year(Ed, Od):
    ne = {v: Ed[v].mean(axis=2) for v in VARS}
    no = {v: Od[v].mean(axis=2) for v in VARS}
    r = {f"CRPS {v}": crps_y(ne[v], no[v]) for v in VARS}
    r.update({f"variogram {v}": vario_y(Ed[v], Od[v]) for v in VARS})
    r["variogram coupling"] = vario3_y(ne, no)
    return r


E = pickle.load(open(ENS_FILE, "rb"))
S = {f"k={k}": per_year(E[f"inflation k={k}"], OBS) for k in (1.0, 1.25, 1.5)}
for name in ("UNIFIED", "SEPARATE", "OLD"):
    S[name] = per_year(E[f"ladder {name}"], OBS)
log("models scored per year")
n = OBS["wind"].shape[0]
loo = [per_year({v: OBS[v][np.arange(n) != y] for v in VARS}, {v: OBS[v][[y]] for v in VARS}) for y in range(n)]
S["REAL"] = {k: np.array([d[k][0] for d in loo]) for k in loo[0]}
log("real-climate reference")

COMP = [("k=1.25", "k=1.0"), ("k=1.5", "k=1.0"), ("SEPARATE", "UNIFIED"), ("OLD", "UNIFIED"), ("UNIFIED", "REAL")]
print(f"\n{'paired difference (A - B), mean over 75 years':48}" + "".join(f"{s:>22}" for s in S["UNIFIED"]))
for a, b in COMP:
    line = f"  {a + ' - ' + b:46}"
    for s in S["UNIFIED"]:
        d = S[a][s] - S[b][s]
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        line += f"{f'{d.mean() / S[b][s].mean():+.1%} (t={t:+.1f})':>22}"
    print(line)
print("\n(relative difference to B; t > 2: A significantly worse, t < -2: A significantly better)")
with open(f"{D}/fit_assessment_sig.pkl", "wb") as f:
    pickle.dump(S, f)
log(f"saved {D}/fit_assessment_sig.pkl")
