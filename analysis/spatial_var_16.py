# Thesis: Case Study, Model Development: the same test on the 16 cells, with the EOF modes
# Run: cd analysis && python spatial_var_16.py   (needs data/grid4x4_*)
"""Spatial dependence on the 16 cells (wind): dynamics against a copula on the shocks, and the EOF modes.
  AR-JOINT      AR per cell + empirical residual copula
  VAR16-JOINT   full 16-dimensional seasonal VAR + residual copula (256 coefficients per lag)
  EOFk-JOINT    VAR on the k leading EOF modes + AR per cell on the remainder, joint residual bootstrap (k = 1..4);
                the VAR part has k dimensions whatever the number of cells
  OLD t-copula  the 16-region simulation of simulate_4_16_regions.py
Same evaluation as spatial_var_4q.py: share of the observed spatial dependence each variant recovers.
"""
import pickle
import time

import numpy as np
import pandas as pd

import spatial_common as sc

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
GRID16 = [1, 2, 3, 6, 12, 24, 36, 48, 72]
GRID_EOF = [1, 2, 3, 6, 12, 24, 36, 48, 72, 96, 120, 168]
d = 16

# observed wind of the 16 cells, its anomalies and the observed and decoupled reference
real = sc.load_regions({c: [c] for c in sc.CELLS})
pre = sc.preprocess(real)
rows, real_df = sc.real_reference(real)
log("preprocessed + real reference")


# simulate one variant, evaluate it against the record and log the result
def run(name, fits, seed, hist=None, project=None, keep=False):
    W = sc.simulate(pre, fits, "joint", seed, hist=hist, project=project)
    out = sc.evaluate(name, W, real_df)
    log(f"{name}: simulated + evaluated (min wind {W.min():.2f} m/s)")
    if keep:
        SAVE[name] = W.astype(np.float16)
    return out


# separate ARs per cell, coupled only through jointly resampled innovations (copula on the shocks)
SAVE = {}
ar = sc.fit_ar(pre["anom"])
log(f"AR fitted; median lag order {np.median([o for s in sc.SEASONS for o in ar[s]['orders']]):.0f}")
rows += run("AR-JOINT", ar, 202)

# a full VAR across the 16 cells (order by BIC); innovation correlation against the long-run coherence it implies
var = sc.fit_var(pre["anom"], GRID16)
mech = []
for s in sc.SEASONS:
    b = var[s]["bics"]
    log(f"VAR16 {s:7} p={var[s]['p']:>3} radius {sc.spectral_radius(var[s]['B'], d):.4f}  BIC rel.: "
        + " ".join(f"{p}:{b[p] - min(b.values()):+.0f}" for p in sorted(b)))
    for nm, f in (("AR", ar[s]), ("VAR16", var[s])):
        C = np.corrcoef(f["pool"].T)
        mech.append({"season": s, "model": nm, "innovation corr": C[~np.eye(d, dtype=bool)].mean(),
                     "long-run corr": sc.longrun_corr(f["B"], f["pool"])})
rows += run("VAR16-JOINT", var, 204, keep=True)
del var

# the scalable alternative: a VAR on the k leading EOF modes plus an AR per cell for the remainder (k = 1..4)
eof_info = {}
for k in (1, 2, 3, 4):
    fits, hist, project, expl = sc.fit_eof(pre["anom"], k, GRID_EOF)
    eof_info[k] = {"explained": expl[:k].sum(), "var_p": {s: fits[s]["var_p"] for s in sc.SEASONS},
                   "pool rows (common/var/ar)": {s: (fits[s]["n_common"], fits[s]["n_var"], fits[s]["n_ar"])
                                                for s in sc.SEASONS}}
    for s in sc.SEASONS:
        f = fits[s]
        C = np.corrcoef(f["pool"][:, k:].T)
        mech.append({"season": s, "model": f"EOF{k}", "innovation corr": C[~np.eye(d, dtype=bool)].mean(),
                     "long-run corr": np.nan})
    log(f"EOF{k}: modes explain {expl[:k].sum():.1%} of anomaly variance; PC-VAR orders {eof_info[k]['var_p']}")
    rows += run(f"EOF{k}-JOINT", fits, 210 + k, hist=hist, project=project, keep=True)
    del fits, hist

# the 16-region simulation (AR + Student-t copula, simulate_4_16_regions.py) for comparison
s_ = pickle.load(open(f"{sc.D}/sim_16region_wind.pkl", "rb"))
rows += sc.evaluate("OLD t-copula", np.stack([s_["phys"][r] for r in sc.CELLS], axis=2), real_df)
log("OLD t-copula evaluated")

# statistics of every variant and the share of the observed spatial dependence each one recovers
res = pd.DataFrame(rows).set_index("variant")
mech = pd.DataFrame(mech)
pd.set_option("display.width", 220)
print("\nEOF spectrum (share of anomaly variance, first 6 modes):", np.round(expl[:6], 3))
print("\nMECHANISM: mean innovation correlation vs long-run (frequency-zero) coherence implied by the fit")
print("(EOF rows: innovation corr of the remainder field only)")
print(mech.pivot(index="season", columns="model").round(3).to_string())
print("\n" + "=" * 120)
print("SIXTEEN CELLS - fleet of 16 cell power curves; lulls = fleet CF < 10%, all hours")
print("=" * 120)
print(res.round(3).to_string())
order = ["AR-JOINT", "OLD t-copula", "EOF1-JOINT", "EOF2-JOINT", "EOF3-JOINT", "EOF4-JOINT", "VAR16-JOINT"]
print("\nSHARE OF THE REAL DEPENDENCE EFFECT RECOVERED (marginals repaired; 0% = decoupled, 100% = real)")
print((sc.recovery_table(res, order) * 100).round(0).to_string())

with open(f"{sc.D}/spatial_var_16.pkl", "wb") as f:
    pickle.dump({"table": res, "mechanism": mech, "eof": eof_info, "eof_spectrum": expl}, f)
with open(f"{sc.D}/sim_16cell_dynamics_wind.pkl", "wb") as f:
    pickle.dump({"phys": SAVE, "target_index": sc.TARGET, "cells": sc.CELLS,
                 "note": "float16 m/s, (300, 8760, 16) per variant; lambda=0.25 Box-Cox space models"}, f)
log("saved spatial_var_16.pkl and sim_16cell_dynamics_wind.pkl")
