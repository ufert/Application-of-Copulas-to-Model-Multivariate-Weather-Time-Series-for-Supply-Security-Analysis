# Thesis: Results, Number of Simulated Years (rules for N, convergence, paired scenarios)
# Run: cd analysis && python point5_mc.py   (~8 min)
# Writes: data/point5_mc.pkl
"""Number of simulated years (Monte Carlo repetitions) needed for statements at a given precision.

simulation  3,000 years of the unified model on 0.5 deg units (point5_model05.py: 10 batches of 300), converted per
            unit and put through the 2030 system of point1_night_ev.py (demand model with residual days, 115 / 30 /
            215 GW; EV charging staggered and smart for the scenario comparison). The simulated years are independent
            (own innovation days, own offset year), so per-year statistics are iid.
statements  means          annual peak residual load (RL), worst 72-h and 7-day mean RL, hours with RL > 57 GW, onshore
                           mean CF, lulls >= 72 h; target +-1 % / +-5 % of the estimate (95 %)
            probabilities  P(peak RL > 72 GW), P(worst 72 h > 52 GW), P(year with a 7-day lull); target +-10 % / +-20 %
                           relative (95 %)
            return levels  1-in-20 and 1-in-50 peak RL and worst 72 h; target +-0.5 / +-1 GW (95 %)
            differences    smart minus staggered charging (peak RL), with minus without offshore (worst 72 h): N to
                           detect the mean difference (5 % two-sided, 80 % power), paired against independent runs
criteria    analytic N (central limit theorem, binomial, density at the quantile) and empirical N: the 95 % half-width
            over 400 random subsets of size N of the 3,000 years (finite-population corrected), on a grid of N
model vs observed  N* = 9 n_obs (sd_model / sd_obs)^2: beyond N*, the sampling error of the model is below a third of
            that of the observed record (n_obs = 75)
Writes data/point5_mc.pkl.
"""
import pickle
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
import point5_common as pc  # noqa: E402

CACHE = pickle.load(open(f"{pc.D}/point5_model05.pkl", "rb"))
SIM = CACHE["mc"]
THETA = CACHE["theta"]
N_TOTAL, BATCH = CACHE["years"], CACHE["batch"]
rng_sub = np.random.default_rng(53)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
log(f"{len(SIM)} simulated years of the 0.5 deg model from point5_model05.pkl; critical threshold {THETA:.2f} GW")

# real reference (75 years): residual load from point 1, lulls from the 100 m conversion
R = pc.system_series()["real"]
Fr = pc.feedin(pc.load_real())
Lr = [pc.runs(u) for u in Fr["on"]]
REAL = pd.DataFrame({"peak RL (GW)": R["rl"].max(axis=1), "worst 72 h RL (GW)": pc.roll_max(R["rl"], 72),
                     "worst 7 d RL (GW)": pc.roll_max(R["rl"], 168), "hours RL > 57 GW": (R["rl"] > THETA).sum(axis=1),
                     "onshore mean CF": Fr["on"].mean(axis=1), "lulls >= 72 h": [(l >= 72).sum() for l in Lr],
                     "year with 7-day lull": [float((l >= 168).any()) for l in Lr], "PV mean CF": R["pv"].mean(axis=1),
                     "worst 72 h PV CF": pc.roll_min(R["pv"], 72), "worst 72 h RE CF": pc.roll_min(R["re"] / 360.0, 72)})

NGRID = np.array([10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000])


def empirical_halfwidth(x, est, n, reps=400):
    """95% half-width of estimator est at sample size n: SD over random subsets without replacement, FPC-corrected"""
    vals = np.array([est(x[rng_sub.choice(len(x), n, replace=False)]) for _ in range(reps)])
    return 1.96 * vals.std(ddof=1) / np.sqrt(max(1 - n / len(x), 1e-9))


# smallest number of years on the grid whose empirical 95 % half-width is below the target
def n_empirical(x, est, target):
    for n in NGRID:
        if empirical_halfwidth(x, est, n) <= target:
            return int(n)
    return f"> {NGRID[-1]}"


ROWS = []
# means
for s in ("peak RL (GW)", "worst 72 h RL (GW)", "worst 7 d RL (GW)", "hours RL > 57 GW", "onshore mean CF", "lulls >= 72 h"):
    x = SIM[s].to_numpy(float)
    m, sd = x.mean(), x.std(ddof=1)
    for e in (0.01, 0.05):
        ROWS.append({"statement": f"mean {s}", "estimate (N=3000)": m, "target (95%)": f"+-{e:.0%}",
                     "N analytic": int(np.ceil((1.96 * sd / (e * abs(m))) ** 2)), "N empirical": n_empirical(x, np.mean, e * abs(m)),
                     "N* model vs real": int(np.ceil(9 * len(REAL) * (sd / REAL[s].std(ddof=1)) ** 2)) if s in REAL else np.nan,
                     "real (75 yr)": REAL[s].mean() if s in REAL else np.nan})
# probabilities
for s, thr in (("peak RL (GW)", 72.0), ("worst 72 h RL (GW)", 52.0), ("year with 7-day lull", 0.5)):
    x = (SIM[s].to_numpy(float) > thr).astype(float)
    p = x.mean()
    lab = f"P({s} > {thr:g})" if s != "year with 7-day lull" else "P(year with a 7-day lull)"
    for e in (0.10, 0.20):
        ROWS.append({"statement": lab, "estimate (N=3000)": p, "target (95%)": f"+-{e:.0%} rel.",
                     "N analytic": int(np.ceil(1.96 ** 2 * (1 - p) / (p * e ** 2))), "N empirical": n_empirical(x, np.mean, e * p),
                     "real (75 yr)": (REAL[s].to_numpy(float) > thr).mean()})
# return levels
for s in ("peak RL (GW)", "worst 72 h RL (GW)"):
    x = SIM[s].to_numpy(float)
    for T_ in (20, 50):
        q = 1 - 1 / T_
        xq = np.quantile(x, q)
        f = gaussian_kde(x)(xq)[0]
        for d in (0.5, 1.0):
            ROWS.append({"statement": f"1-in-{T_}-yr {s}", "estimate (N=3000)": xq, "target (95%)": f"+-{d} GW",
                         "N analytic": int(np.ceil((1.96 * np.sqrt(q * (1 - q)) / (d * f)) ** 2)),
                         "N empirical": n_empirical(x, lambda v: np.quantile(v, q), d), "real (75 yr)": np.quantile(REAL[s], q)})
# solar and combined statistics
for s in ("PV mean CF", "worst 72 h PV CF", "worst 72 h RE CF"):
    x = SIM[s].to_numpy(float)
    m, sd = x.mean(), x.std(ddof=1)
    for e in (0.01, 0.05):
        ROWS.append({"statement": f"mean {s}", "estimate (N=3000)": m, "target (95%)": f"+-{e:.0%}",
                     "N analytic": int(np.ceil((1.96 * sd / (e * abs(m))) ** 2)), "N empirical": n_empirical(x, np.mean, e * abs(m)),
                     "N* model vs real": int(np.ceil(9 * len(REAL) * (sd / REAL[s].std(ddof=1)) ** 2)), "real (75 yr)": REAL[s].mean()})
x = SIM["worst 72 h RE CF"].to_numpy(float)
for T_ in (20, 50):
    q = 1 / T_
    xq = np.quantile(x, q)
    f = gaussian_kde(x)(xq)[0]
    for d in (0.0025, 0.005):
        ROWS.append({"statement": f"1-in-{T_}-yr worst 72 h RE CF (low)", "estimate (N=3000)": xq, "target (95%)": f"+-{d * 100:g} pp",
                     "N analytic": int(np.ceil((1.96 * np.sqrt(q * (1 - q)) / (d * f)) ** 2)),
                     "N empirical": n_empirical(x, lambda v: np.quantile(v, q), d), "real (75 yr)": np.quantile(REAL["worst 72 h RE CF"], q)})
TAB = pd.DataFrame(ROWS)
print("\nREQUIRED NUMBER OF SIMULATED YEARS (95% confidence; empirical = 400 random subsets of the 3,000 years)")
print(TAB.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# scenario differences: paired (same weather years) vs independent
k = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) ** 2
DIFF = []
for lab, a, b in (("smart minus staggered charging: peak RL", "peak RL, smart EV (GW)", "peak RL, staggered EV (GW)"),
                  ("without minus with offshore: worst 72 h", "worst 72 h RL without offshore (GW)", "worst 72 h RL (GW)"),
                  ("staggered charging minus no EVs: peak RL", "peak RL, staggered EV (GW)", "peak RL (GW)")):
    xa, xb = SIM[a].to_numpy(float), SIM[b].to_numpy(float)
    d = xa - xb
    DIFF.append({"comparison": lab, "mean difference (GW)": d.mean(), "pair correlation": np.corrcoef(xa, xb)[0, 1],
                 "N independent": int(np.ceil(k * (xa.var(ddof=1) + xb.var(ddof=1)) / d.mean() ** 2)),
                 "N paired": int(np.ceil(k * d.var(ddof=1) / d.mean() ** 2)),
                 "reduction": (xa.var(ddof=1) + xb.var(ddof=1)) / d.var(ddof=1)})
DIFF = pd.DataFrame(DIFF)
print("\nSCENARIO COMPARISONS: years needed per scenario to detect the mean difference (5% two-sided, 80% power)")
print(DIFF.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# running estimates (for the figure)
RUN = {}
for s in ("peak RL (GW)", "worst 72 h RL (GW)", "year with 7-day lull", "onshore mean CF", "PV mean CF", "worst 72 h RE CF"):
    x = SIM[s].to_numpy(float)
    ns = np.unique(np.round(np.logspace(1, np.log10(len(x)), 40)).astype(int))
    RUN[s] = pd.DataFrame({"N": ns, "mean": [x[:n].mean() for n in ns], "half-width": [1.96 * x[:n].std(ddof=1) / np.sqrt(n) for n in ns],
                           "1-in-20": [np.quantile(x[:n], 0.95) for n in ns], "1-in-20 low": [np.quantile(x[:n], 0.05) for n in ns]})
with open(f"{pc.D}/point5_mc.pkl", "wb") as f:
    pickle.dump({"sim": SIM, "real": REAL, "table": TAB, "differences": DIFF, "running": RUN, "grid": NGRID}, f)
log(f"saved {pc.D}/point5_mc.pkl")
