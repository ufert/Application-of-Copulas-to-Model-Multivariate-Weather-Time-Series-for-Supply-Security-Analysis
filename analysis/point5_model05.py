# Thesis: Results, the 3,000 simulated years behind the validation, the extremes and the convergence
# Run: cd analysis && python point5_model05.py [--smoke]   (10 batches of 300 years, ~1 h)
# Writes: data/point5_model05.pkl
"""3,000 simulated years of the unified model on 0.5 deg units for the validation, extreme-value and convergence analyses.

years      10 batches of 300: batch 0 = the simulation of unified_model_05.py (data/sim05_*), batches 1-9 =
           unified_model_05.simulate with seeds 9001-9009; each batch margin-mapped on its own
per batch  feed-in per unit (units05.feedin), aggregated to national values and the 16 cells; 2030 system without EVs
           (demand model of point1_night_ev.py, residual-day seed 500 + batch) and with staggered and smart charging;
             per year   every statistic of point5_properties.per_year() (16-cell aggregates, anomalies and rank
                        uniforms within each batch of 300 years)
             tails      the counts of point5_tails.per_year_counts() for every pair (6 named and 120 cell pairs)
             mc         the per-year system statistics of point5_mc.py (peak, worst 72 h and 7 d RL, hours above the
                        critical threshold, lulls, PV, EV scenarios, without offshore)
Writes data/point5_model05.pkl (read by point5_properties.py, point5_tails.py and point5_mc.py).
Option --smoke: 2 x 10 years, written to point5_model05_smoke.pkl.
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


# run the definitions of another script up to the marker and return them (shared statistics, no analysis)
def definitions(path, marker):
    src = open(path).read()
    assert marker in src, f"{path}: marker {marker!r} not found"
    ns = {"__name__": f"defs_{path}"}
    exec(src.split(marker)[0], ns)
    return ns


PP = definitions("point5_properties.py", "\nMODEL300 = ")                 # per_year
PT = definitions("point5_tails.py", "\nMODEL300 = ")                      # per_year_counts
pc = PP["pc"]
import point1_night_ev as p1  # noqa: E402  (demand and EV model)
import unified_model_05 as um05  # noqa: E402
import units05 as u05  # noqa: E402

THETA = p1.THETA["with offshore"]
log(f"point-1 system loaded; critical threshold {THETA:.2f} GW")
SMOKE = "--smoke" in sys.argv                                         # 2 batches of 10 years, _smoke output
N_TOTAL, BATCH = (20, 10) if SMOKE else (3000, 300)
djf, jja = pc.SEAS == "DJF", pc.SEAS == "JJA"


# per-year statistics of one batch of simulated years (national weather and the 2030 system)
def year_stats(F, dem):
    tnat, on, off, pv = (F[k].astype(np.float64) for k in ("tnat", "on", "off", "pv"))
    re = p1.CAP["on"] * on + p1.CAP["off"] * off + p1.CAP["pv"] * pv
    rl = dem - re
    eh, ew, ep = p1.ev_energy(tnat)
    work_pub = p1.spread(ew, np.tile(p1.K["work"], (365, 1))) + p1.spread(ep, np.tile(p1.K["public"], (365, 1)))
    stag = p1.spread(eh, np.tile(p1.K["home staggered"], (365, 1))) + work_pub
    smart = p1.smart(eh, dem + work_pub - re) + work_pub
    L = [pc.runs(u) for u in on]
    return pd.DataFrame({"peak RL (GW)": rl.max(axis=1), "worst 72 h RL (GW)": pc.roll_max(rl, 72),
                         "worst 7 d RL (GW)": pc.roll_max(rl, 168), "hours RL > 57 GW": (rl > THETA).sum(axis=1),
                         "onshore mean CF": on.mean(axis=1), "lulls >= 72 h": [(l >= 72).sum() for l in L],
                         "year with 7-day lull": [float((l >= 168).any()) for l in L],
                         "peak RL, staggered EV (GW)": (rl + stag).max(axis=1), "peak RL, smart EV (GW)": (rl + smart).max(axis=1),
                         "worst 72 h RL without offshore (GW)": pc.roll_max(rl + p1.CAP["off"] * off, 72),
                         "PV mean CF": pv.mean(axis=1), "worst 72 h PV CF": pc.roll_min(pv, 72),
                         "worst 72 h RE CF": pc.roll_min(re / (p1.CAP["on"] + p1.CAP["off"] + p1.CAP["pv"]), 72)})


M = None
PY_PARTS, TC_PARTS, MC_PARTS = {}, {}, []
for b in range(N_TOTAL // BATCH):
    if b == 0:
        mm = u05.model_units()
        if SMOKE:
            mm = {v: x[:BATCH] for v, x in mm.items()}
    else:
        if M is None:
            M = um05.build("unified")
        mm, _ = um05.simulate(M, BATCH, seed=9000 + b)
    F = u05.feedin(mm, temp_in_c=True)
    dem = p1.demand(F["tnat"].astype(np.float64), seed=500 + b).astype(np.float64)
    re = (pc.CAP["on"] * F["on"] + pc.CAP["off"] * F["off"] + pc.CAP["pv"] * F["pv"]).astype(np.float64)
    SYS = {"demand": dem, "re": re, "rl": dem - re}
    for k, v in PP["per_year"]({"temp": F["temp16"], "solar": F["solar16"]}, F, SYS, None).items():
        PY_PARTS.setdefault(k, []).append(v)
    MC_PARTS.append(year_stats(F, dem))
    u_on, u_t, u_pv = pc.slot_uniform(F["on"]), pc.slot_uniform(F["tnat"]), pc.slot_uniform(F["pv"])
    day = F["pv"].mean(axis=0) > 0.05
    u_dem, u_re = pc.slot_uniform(dem), pc.slot_uniform(re)
    pairs = {"cold-calm (DJF)": (u_t, u_on, djf), "hot-calm (JJA)": (1 - u_t, u_on, jja),
             "dark-calm (DJF daylight)": (u_pv, u_on, djf & day),
             "calm persists 24 h": (u_on[:, :-24], u_on[:, 24:], np.ones(8760 - 24, bool)),
             "calm persists 72 h": (u_on[:, :-72], u_on[:, 72:], np.ones(8760 - 72, bool)),
             "high demand-low RE (DJF)": (1 - u_dem, u_re, djf)}
    for pname, (a, c, m) in pairs.items():
        TC_PARTS.setdefault(pname, []).append(PT["per_year_counts"](a, c, m))
    uc = np.stack([pc.slot_uniform(F["cells"][..., k]) for k in range(16)], axis=-1)
    for i in range(16):
        for j in range(i + 1, 16):
            TC_PARTS.setdefault(f"cells {pc.CELLS[i]}-{pc.CELLS[j]}", []).append(
                PT["per_year_counts"](uc[..., i], uc[..., j], np.ones(8760, bool)))
    del F, uc, u_on, u_t, u_pv, u_dem, u_re, dem, re, SYS
    if b > 0:
        um05.remove(mm)
    log(f"batch {b + 1}/{N_TOTAL // BATCH} done")


# concatenate the per-batch values of one statistic
def pool_stat(parts):
    kind, flag = parts[0][0], parts[0][2]
    if kind == "ratio":
        return kind, (np.concatenate([p[1][0] for p in parts]), np.concatenate([p[1][1] for p in parts])), flag
    return kind, np.concatenate([p[1] for p in parts]), flag


OUT = {"per year": {k: pool_stat(v) for k, v in PY_PARTS.items()},
       "tails counts": {k: {f: np.concatenate([p[f] for p in parts]) for f in parts[0]} for k, parts in TC_PARTS.items()},
       "mc": pd.concat(MC_PARTS, ignore_index=True), "theta": THETA, "years": N_TOTAL, "batch": BATCH,
       "u grid": np.asarray(PT["U_GRID"])}
OUTF = f"{pc.D}/point5_model05{'_smoke' if SMOKE else ''}.pkl"
with open(OUTF, "wb") as f:
    pickle.dump(OUT, f)
log(f"saved {OUTF}: {len(OUT['per year'])} statistics, {len(OUT['tails counts'])} pairs, {N_TOTAL} years")
