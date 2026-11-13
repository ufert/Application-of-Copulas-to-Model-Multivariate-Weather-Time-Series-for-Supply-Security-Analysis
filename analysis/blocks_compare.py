# Thesis: Case Study, table of the unified model at four resolutions against the record and the 0.25 deg truth
# Run: cd analysis && python blocks_compare.py   (after unified_model_blocks.py for all four resolutions)
# Writes: data/blocks_compare.pkl; prints the table
"""Compares the unified model at four spatial resolutions (outputs of unified_model_blocks.py) with the observed record
converted at the same resolution and with the 0.25 deg reference, in which every point is converted on its own.
Metrics of the national onshore fleet: mean CF, hours below 5 % and 10 % of capacity, lulls (CF < 10 %) of at least
24 h and 72 h, longest lull, share of years with a 7-day lull, worst 72-h mean CF; hours with wind + PV below 10 % of
capacity; 2030 residual load without EVs (demand model of point1_night_ev.py): peak, worst 72-h mean, hours above 57 GW.
Writes data/blocks_compare.pkl. Option --smoke uses the short test runs.
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd

import conversion as cv
import point5_common as pc

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)
SM = "_smoke" if "--smoke" in sys.argv else ""
D = cv.D
import point1_night_ev as p1  # noqa: E402  (demand model)

RESS = [r for r in ("16", "1", "0.5", "0.25") if __import__("os").path.exists(f"{D}/blocks_{r}{SM}.pkl")]
B = {r: pickle.load(open(f"{D}/blocks_{r}{SM}.pkl", "rb")) for r in RESS}
log("loaded " + ", ".join(f"{r}: {len(B[r]['units land'])} land units, N = {B[r]['N']}" for r in RESS))

# 0.25 deg truth (observed, point by point)
P2 = pickle.load(open(f"{D}/point2_aggregation.pkl", "rb"))
truth_on = P2["series"][("100 m", "mix", "points (truth)", "B")].reshape(25, 8760).astype(np.float64)
meta = pickle.load(open(f"{D}/grid025_meta.pkl", "rb"))
idx = pd.DatetimeIndex(meta["time"])
keep = ~((idx.month == 2) & (idx.day == 29))
rlm, rlh = cv.local_time(idx[keep])
S3 = pickle.load(open(f"{D}/point3_value_map.pkl", "rb"))["sites"]
OFFP = S3[S3.type == "offshore"]
LATG, LONG = np.asarray(meta["lat"]), np.asarray(meta["lon"])
ori = np.array([int(np.argmin(np.abs(LATG - la))) for la in OFFP.lat])
oci = np.array([int(np.argmin(np.abs(LONG - lo))) for lo in OFFP.lon])
W100 = np.load(f"{D}/grid025_ws100_2000-2024.npy", mmap_mode="r")
wo = np.concatenate([np.asarray(W100[a:a + 20000], dtype=np.float32)[:, ori, oci] for a in range(0, len(idx), 20000)])[keep]
OFFC = pickle.load(open(f"{D}/point3_value_map.pkl", "rb"))["offshore conversion"]
conv_off = cv.Wind100Conversion(cv.OFFSHORE, np.ones((len(ori), 1)), np.full(len(ori), 1 / len(ori)), OFFC["sigma"],
                                np.full((12, 24), OFFC["kappa"]))
truth_off = conv_off.cf(wo, rlm, rlh).reshape(25, 8760).astype(np.float64)
SMT = pickle.load(open(f"{D}/grid025de_meta_1950-2024.pkl", "rb"))["time"]
rows = np.where(SMT.year >= 2000)[0][keep]
G = np.asarray(np.load(f"{D}/grid025de_ghi_1950-2024.npy", mmap_mode="r")[rows], dtype=np.float32)
TK = np.asarray(np.load(f"{D}/grid025de_t2m_1950-2024.npy", mmap_mode="r")[rows], dtype=np.float32)
PVC = cv.calibrate_16cells()["pv"]
base = cv.pv_cells(np.clip(G, 0, None), TK).mean(axis=1)
truth_pv = (np.roll(PVC.pr * base, PVC.lag) * PVC.ratio[rlm - 1, rlh]).reshape(25, 8760).astype(np.float64)
truth_t = (TK.mean(axis=1) - 273.15).reshape(25, 8760).astype(np.float64)
del G, TK, wo
log("0.25 deg truth built")

# unified model on 0.5 deg units and the observed record
Fm = pc.feedin(pc.load_model())
Fr = pc.feedin(pc.load_real())
r25 = pc.YEARS >= 2000
SETS = {"truth 0.25 deg (observed)": {"on": truth_on, "off": truth_off, "pv": truth_pv, "tnat": truth_t, "seed": 1}}
for r in RESS:
    SETS[f"observed {r}"] = {**{k: v.astype(np.float64) for k, v in B[r]["observed"].items()}, "seed": 1}
for r in RESS:
    SETS[f"model {r}"] = {**{k: v.astype(np.float64) for k, v in B[r]["model"].items()}, "seed": 3}
SETS["observed, unified model (0.5 deg units)"] = {"on": Fr["on"][r25], "off": Fr["off"][r25], "pv": Fr["pv"][r25], "tnat": Fr["tnat"][r25], "seed": 1}
SETS["model, unified model (0.5 deg units)"] = {"on": Fm["on"], "off": Fm["off"], "pv": Fm["pv"], "tnat": Fm["tnat"], "seed": 3}
del Fm, Fr


# system statistics of one resolution: onshore lulls and residual-load metrics of the 2030 system
def metrics(s):
    on = s["on"]
    L = [pc.runs(u) for u in on]
    re = p1.CAP["on"] * on + p1.CAP["off"] * s["off"] + p1.CAP["pv"] * s["pv"]
    dem = p1.demand(s["tnat"], seed=s["seed"])
    rl = dem - re
    return {"onshore mean CF": on.mean(), "h/yr onshore CF < 5%": (on < 0.05).sum(axis=1).mean(),
            "h/yr onshore CF < 10%": (on < 0.10).sum(axis=1).mean(), "lulls >= 24 h /yr": np.mean([(l >= 24).sum() for l in L]),
            "lulls >= 72 h /yr": np.mean([(l >= 72).sum() for l in L]), "longest lull (h)": np.mean([l.max() if len(l) else 0 for l in L]),
            "P(year with 7-day lull)": np.mean([float((l >= 168).any()) for l in L]), "worst 72 h onshore CF": pc.roll_min(on, 72).mean(),
            "SD hourly onshore CF": on.std(axis=1).mean(), "offshore mean CF": s["off"].mean(), "PV mean CF": s["pv"].mean(),
            "h/yr wind + PV < 10%": (re / (p1.CAP["on"] + p1.CAP["off"] + p1.CAP["pv"]) < 0.10).sum(axis=1).mean(),
            "peak RL (GW)": rl.max(axis=1).mean(), "worst 72 h RL (GW)": pc.roll_max(rl, 72).mean(),
            "h/yr RL > 57 GW": (rl > 57).sum(axis=1).mean(), "years": on.shape[0]}


T = pd.DataFrame({k: metrics(s) for k, s in SETS.items()})
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 20)
print("\nNATIONAL METRICS (observed sets: 2000-2024; model sets: simulated years)")
print(T.round(3).to_string())
ratio_same = pd.DataFrame({f"model / observed, {r}": T[f"model {r}"] / T[f"observed {r}"] for r in RESS})
ratio_same["model / observed, unified model"] = T["model, unified model (0.5 deg units)"] / T["observed, unified model (0.5 deg units)"]
print("\nMODEL / OBSERVED AT THE SAME RESOLUTION")
print(ratio_same.round(3).to_string())
ratio_truth = pd.DataFrame({k: T[k] / T["truth 0.25 deg (observed)"] for k in T.columns if k != "truth 0.25 deg (observed)"})
print("\nRELATIVE TO THE 0.25 DEG TRUTH")
print(ratio_truth.round(3).to_string())
# error of each model against the truth: mean absolute log ratio over eight system statistics (column "Error" of the table)
ERR_STATS = ["h/yr onshore CF < 5%", "h/yr onshore CF < 10%", "lulls >= 24 h /yr", "lulls >= 72 h /yr", "worst 72 h onshore CF",
             "h/yr RL > 57 GW", "peak RL (GW)", "worst 72 h RL (GW)"]
ERR = pd.Series({k: float(np.abs(np.log(ratio_truth.loc[ERR_STATS, k].astype(float))).mean())
                 for k in ratio_truth.columns if k.startswith("model")})
print("\nMODEL ERROR AGAINST THE TRUTH: mean |log(model / truth)| over " + ", ".join(ERR_STATS))
print(ERR.round(3).to_string())
COH =pd.DataFrame({**{f"model {r}": B[r]["coherence model"] for r in RESS}, **{f"observed {r}": B[r]["coherence observed"] for r in RESS}})
print("\nCOHERENCE OF UNIT ONSHORE CF BY DISTANCE (mean hourly correlation)")
print(COH.round(3).to_string())
INFO = pd.DataFrame({r: {"land units": len(B[r]["units land"]), "offshore units": len(B[r]["units offshore"]),
                         **{f"EOF {v}": B[r]["eof explained"][v] for v in ("wind", "temp", "solar")},
                         "VAR p (winter)": B[r]["var p"]["winter"], "AR order median (winter)": B[r]["ar orders median"]["winter"],
                         "minutes": B[r]["seconds"] / 60} for r in RESS})
print("\nRUNS")
print(INFO.round(3).to_string())
with open(f"{D}/blocks_compare{SM}.pkl", "wb") as f:
    pickle.dump({"table": T, "model/observed": ratio_same, "relative to truth": ratio_truth, "error": ERR, "coherence": COH,
                 "runs": INFO}, f)
log(f"saved {D}/blocks_compare{SM}.pkl")
