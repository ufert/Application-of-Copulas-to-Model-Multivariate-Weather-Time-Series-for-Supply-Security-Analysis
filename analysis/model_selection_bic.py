# Thesis: Results, Model Validation, Model Selection and Parametrization (BIC and parameter counts)
# Run: cd analysis && python model_selection_bic.py   (three model builds, ~9 min)
# Writes: data/model_selection_bic.pkl; prints the table of the variants
"""Parametrization and BIC of the 0.5 deg model variants (unified, separate, ar-only).

For each variant, the one-step prediction errors of the 640 anomaly series (234 wind, 203 temperature, 203 irradiance
units) are reconstructed from the innovation pool of unified_model_05.build (mode innovations projected with the EOF
loadings plus remainder innovations; identical rows for all variants). Per season s:

    BIC_s = n_s log det(Sigma_s) + k_s log n_s,     Sigma_s = covariance of the 640-dimensional errors,

k_s = coefficients of the VAR and of the remainder ARs (intercepts included) of season s; the EOF loadings are counted
once with the log of all rows; the error covariance is the same for all variants and not counted. The innovations of
the generator are resampled empirically, so this Gaussian BIC judges the dynamics (conditional mean) only.
Also stores the BIC of the mode VAR for every order of the grid and the variance explained by the leading EOF modes.
Output: data/model_selection_bic.pkl and a printed summary.
"""
import pickle

import numpy as np

import unified_model_05 as um
from simulate_4_16_regions import SEASONS, season_block_ids

OUT = f"{um.D}/model_selection_bic.pkl"


def season_day_counts(idx):
    """number of full innovation days per season, in the order in which build() concatenates the pool"""
    T = len(idx)
    dayno = np.arange(T) // 24
    feb29 = np.asarray((idx.month == 2) & (idx.day == 29))
    counts = []
    for s in SEASONS:
        bid = season_block_ids(idx, s)
        ok = bid != -1
        for lag in range(1, um.MAXLAG + 1):
            ok &= (np.roll(bid, lag) == bid) & (np.arange(T) >= lag)
        tv = np.where(ok)[0]
        _, first, cnt = np.unique(dayno[tv], return_index=True, return_counts=True)
        counts.append(int(((cnt == 24) & ~feb29[tv[first]]).sum()))
    return counts


# build one variant and compute its parameter count and Gaussian BIC per season
def evaluate(variant):
    M = um.build(variant, gpd=False)
    idx, pool, NM, NS = M["idx"], M["pool"], M["NM"], M["NS"]
    counts = season_day_counts(idx)
    assert sum(counts) == len(pool), (counts, len(pool))
    sizes = {v: M["EOFS"][v][0].shape[0] for v in um.VARS}
    loadings = sum(sizes[v] * M["kmodes"][v] for v in um.VARS)
    res, start = {}, 0
    for s, nd in zip(SEASONS, counts):
        P = pool[start:start + nd].reshape(nd * 24, NM + NS).astype(np.float64)
        start += nd
        uS, uR = P[:, :NM], P[:, NM:]
        E = uR.copy()                                   # 640-dim one-step errors of the anomalies
        mo, ro = 0, 0
        for v in um.VARS:
            k, nv = M["kmodes"][v], sizes[v]
            if k:
                E[:, ro:ro + nv] += uS[:, mo:mo + k] @ M["EOFS"][v][0].T
            mo, ro = mo + k, ro + nv
        n = len(E)
        sign, logdet = np.linalg.slogdet(np.cov(E.T, bias=True))
        assert sign > 0
        k_var = int(np.count_nonzero(M["VAR"][s]["B"])) if NM else 0
        k_ar = int(sum(o + 1 for o in M["AR"][s]["orders"]))
        res[s] = {"n": n, "logdet": float(logdet), "k_var": k_var, "k_ar": k_ar,
                  "var_p": (M["VAR"][s]["p"] if NM else 0), "ar_median": float(np.median(M["AR"][s]["orders"])),
                  "bic_dyn": float(n * logdet + (k_var + k_ar) * np.log(n))}
        del P, E, uS, uR
    N = sum(r["n"] for r in res.values())
    d = NS
    out = {"variant": variant, "seasons": res, "loadings": loadings, "cov_params_per_season": d * (d + 1) // 2,
           "bic": sum(r["bic_dyn"] for r in res.values()) + loadings * np.log(N), "N": N,
           "k_dyn": sum(r["k_var"] + r["k_ar"] for r in res.values()) + loadings,
           "var_bics": ({s: M["VAR"][s].get("bics") for s in SEASONS} if variant == "unified" else None),
           "eof_expl": {v: np.cumsum(M["EOFS"][v][2][:8]).tolist() for v in um.VARS} if NM else None}
    um.log(f"[{variant}] BIC {out['bic']:.6g}, dynamic parameters {out['k_dyn']:,}, loadings {loadings:,}, "
           + ", ".join(f"{s}: p {r['var_p']} AR median {r['ar_median']:.0f} k_var {r['k_var']} k_ar {r['k_ar']}"
                       for s, r in res.items()))
    del M, pool
    return out


if __name__ == "__main__":
    R = {}
    for v in ("unified", "separate", "ar-only"):
        R[v] = evaluate(v)
        pickle.dump(R, open(OUT, "wb"))
    base = R["unified"]["bic"]
    print("\nvariant     dynamic params   BIC - BIC(unified)   per hour")
    for v, r in R.items():
        print(f"{v:10s} {r['k_dyn']:14,d} {r['bic'] - base:20,.0f} {(r['bic'] - base) / r['N']:10.4f}")
    print("\nBIC of the mode VAR (unified), relative to the minimum, per season:")
    for s, b in R["unified"]["var_bics"].items():
        m = min(b.values())
        print(f"  {s}: " + ", ".join(f"p={p}: {v - m:,.0f}" for p, v in sorted(b.items())))
    print("\ncumulative explained variance of the EOF modes 1..8:")
    for v, e in R["unified"]["eof_expl"].items():
        print(f"  {v:6s} " + " ".join(f"{x:.3f}" for x in e))
