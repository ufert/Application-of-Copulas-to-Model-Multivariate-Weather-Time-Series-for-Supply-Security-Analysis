# Thesis: Case Study, Model Development: spatial dependence carried by the dynamics or by a copula of the shocks
# (4 quadrants)
# Run: cd analysis && python spatial_var_4q.py   (needs data/grid4x4_*)
"""Spatial dependence on 4 quadrants (wind): does it have to be carried by the dynamics (lags across regions), or can a
copula of the residuals carry it?

2 x 2 design, one code path (spatial_common.py):
                      innovations: ind            joint (empirical copula)
  AR per region       AR-IND (no dependence)      AR-JOINT
  spatial VAR         VAR-IND (lags only)         VAR-JOINT (lags + copula)
Plus the 4-region simulations with a Student-t copula and with an R-vine copula on AR residuals
(simulate_4_16_regions.py, simulate_4region_vine.py). All evaluated identically: power curve per region, fleet CF =
mean over the regions, margins mapped per (month, hour) so that only the dependence differs, and the share of the
observed dependence effect recovered (0 % = decoupled regions, 100 % = observed).
Mechanism: for separate ARs, the coherence of slow variations equals the innovation correlation; the cross-lags of a
VAR can raise it. Reported as the long-run (frequency-zero) correlation implied by each fit, against the observed
coherence of 72-h blocks.
"""
import pickle
import time

import numpy as np
import pandas as pd

import spatial_common as sc

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
GRID = [1, 2, 3, 6, 12, 24, 36, 48, 72, 96, 120, 168]

# observed wind of the four quadrants, 2000-2024, transformed into anomalies
real = sc.load_regions(sc.QUADRANTS)
pre = sc.preprocess(real)
log("preprocessed")

# the two dynamics: one seasonal AR per quadrant, and one seasonal VAR across the quadrants (order by BIC on GRID)
ar = sc.fit_ar(pre["anom"])
log(f"AR fitted; lag orders per season: { {s: ar[s]['orders'] for s in sc.SEASONS} }")
var = sc.fit_var(pre["anom"], GRID)
for s in sc.SEASONS:
    b = var[s]["bics"]
    log(f"VAR {s:7} p={var[s]['p']:>3}  spectral radius {sc.spectral_radius(var[s]['B'], 4):.4f}  "
        f"BIC rel. to best: " + " ".join(f"{p}:{b[p] - min(b.values()):+.0f}" for p in sorted(b)))

# mechanism: separate ARs cannot make slow variations more coherent than their shocks, the cross-lags of a VAR can
print("\nMECHANISM: innovation correlation vs long-run (frequency-zero) coherence implied by the fit")
mech = []
for s in sc.SEASONS:
    for nm, f in (("AR", ar[s]), ("VAR", var[s])):
        C = np.corrcoef(f["pool"].T)
        mech.append({"season": s, "model": nm, "innovation corr": C[~np.eye(4, dtype=bool)].mean(),
                     "long-run corr": sc.longrun_corr(f["B"], f["pool"])})
mech = pd.DataFrame(mech)
print(mech.pivot(index="season", columns="model").round(3).to_string())

# reference: the observed record and a decoupled record (years shuffled per quadrant, no spatial dependence)
rows, real_df = sc.real_reference(real)
log("real reference done")

# the 2 x 2 design: AR or VAR dynamics, with independent or jointly resampled innovations (empirical copula);
# then the simulations with a Student-t and an R-vine copula on the AR residuals
sims = {}
for nm, fits, innov, seed in (("AR-IND", ar, "ind", 101), ("AR-JOINT", ar, "joint", 102),
                              ("VAR-IND", var, "ind", 103), ("VAR-JOINT", var, "joint", 104)):
    sims[nm] = sc.simulate(pre, fits, innov, seed)
    rows += sc.evaluate(nm, sims[nm], real_df)
    log(f"{nm} simulated + evaluated; min wind {sims[nm].min():.2f} m/s")
for nm, fn in (("OLD t-copula", "sim_4region_wind.pkl"), ("OLD vine", "sim_4region_vine_wind.pkl")):
    s_ = pickle.load(open(f"{sc.D}/{fn}", "rb"))
    W = np.stack([s_["phys"][r] for r in sc.QUADRANTS], axis=2)
    rows += sc.evaluate(nm, W, real_df)
    log(f"{nm} evaluated")

# statistics of every variant and the share of the observed spatial dependence each one recovers
res = pd.DataFrame(rows).set_index("variant")
pd.set_option("display.width", 220)
print("\n" + "=" * 120)
print("FOUR QUADRANTS - fleet of 4 regional power curves; lulls = fleet CF < 10%, all hours")
print("=" * 120)
print(res.round(3).to_string())
order = ["AR-IND", "AR-JOINT", "OLD t-copula", "OLD vine", "VAR-IND", "VAR-JOINT"]
print("\nSHARE OF THE REAL DEPENDENCE EFFECT RECOVERED (marginals repaired; 0% = decoupled, 100% = real)")
print((sc.recovery_table(res, order) * 100).round(0).to_string())

with open(f"{sc.D}/spatial_var_4q.pkl", "wb") as f:
    pickle.dump({"table": res, "mechanism": mech, "var_orders": {s: var[s]["p"] for s in sc.SEASONS},
                 "ar_orders": {s: ar[s]["orders"] for s in sc.SEASONS},
                 "sims": {k: v.astype(np.float32) for k, v in sims.items()}}, f)
log(f"saved {sc.D}/spatial_var_4q.pkl")
