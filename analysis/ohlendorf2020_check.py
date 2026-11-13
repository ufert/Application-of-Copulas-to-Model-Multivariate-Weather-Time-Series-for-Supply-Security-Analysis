# Thesis: Discussion, the low-wind-power definitions of Ohlendorf and Schill (2020) on the series of this thesis
# Run: cd analysis && python ohlendorf2020_check.py   (~30 s, prints the results)
"""Low-wind-power events with the definitions of Ohlendorf & Schill (2020, Environ. Res. Lett. 15, 084045) applied to the
onshore series of this thesis, thresholds 2, 5 and 10 % of capacity:
  CBT  constantly below threshold: runs of hours with CF < threshold; frequency of a duration d = runs >= d per year
  MBT  mean below threshold: non-overlapping windows of d hours whose mean CF < threshold (maximum number of disjoint
       windows, greedy from the start); frequency = windows per year
Return-period duration T = the longest d whose frequency is at least 1/T per year; events are assigned to the year (or
the winter, DJF) that holds the middle of the event.
Data: national onshore CF of the 0.5 deg units (series of point1_night_ev.py), observed 1980-2019 as one continuous
series, and the 300 simulated years as independent years. Prints the return-period durations for T = 1, 10 and 20
years (any season and winter), the longest events and the season of the CBT events >= 24 h at 10 %.
"""
import pickle

import numpy as np
import pandas as pd
import paths  # repository folders: data, figures, external raw data (see paths.py)

D = paths.DATA
P1 = pickle.load(open(f"{D}/point1_night_ev.pkl", "rb"))["series"]
MON1 = pd.date_range("2025-01-01", periods=8760, freq="h").month.to_numpy()
T_RET = (1, 10, 20)


def cbt_freq(cf, mon, years, thr, winter, dmax=600):
    """events >= d per year (CBT) for d = 1..dmax"""
    x = np.diff(np.concatenate([[0], (cf < thr).astype(np.int8), [0]]))
    s, e = np.where(x == 1)[0], np.where(x == -1)[0]
    L = e - s
    if winter:
        L = L[np.isin(mon[(s + e) // 2], [12, 1, 2])]
    return np.array([(L >= d).sum() for d in range(1, dmax + 1)]) / years, L


def mbt_freq(cf, mon, years, thr, winter, dmax=600, dmin=5):
    """maximum number of disjoint d-hour windows with mean CF < thr, per year, for d = dmin..dmax (shorter: inf)"""
    c = np.concatenate([[0.0], np.cumsum(cf)])
    out = np.full(dmax, np.inf)
    win = np.isin(mon, [12, 1, 2])
    for d in range(dmin, dmax + 1):
        ok = (c[d:] - c[:-d]) / d < thr                                  # window starting at t
        if winter:
            ok &= win[np.arange(len(ok)) + d // 2]
        starts = np.flatnonzero(ok)
        n, pos = 0, 0
        while True:
            k = np.searchsorted(starts, pos)
            if k == len(starts):
                break
            n += 1
            pos = starts[k] + d
        out[d - 1] = n / years
    return out


def ret_durations(freq):
    """duration (h) with frequency >= 1/T, for T in T_RET, and the longest event"""
    d = np.arange(1, len(freq) + 1)
    return [int(d[freq >= 1 / T].max()) if (freq >= 1 / T).any() else 0 for T in T_RET] + [int(d[freq > 0].max())]


# low-wind events below 2, 5 and 10 % (constantly below and mean below the threshold), any season and winter:
# return-period durations and the longest event (definitions of Ohlendorf and Schill 2020)
def table(cf, mon, years):
    rows = {}
    for thr in (0.02, 0.05, 0.10):
        for winter in (False, True):
            lab = "winter" if winter else "any season"
            rows[("CBT", f"{thr:.0%}", lab)] = ret_durations(cbt_freq(cf, mon, years, thr, winter)[0])
            rows[("MBT", f"{thr:.0%}", lab)] = ret_durations(mbt_freq(cf, mon, years, thr, winter))
    return pd.DataFrame(rows, index=[f"T = {T} yr (h)" for T in T_RET] + ["longest (h)"]).T


real = np.asarray(P1["real 1950-2024"]["onshore 16 cells"][30:70], dtype=np.float64)          # 1980-2019
print(f"observed 1980-2019: mean onshore CF {real.mean():.3f}, hours CF < 10 % {100 * (real < 0.10).mean():.1f} %")
R = table(real.ravel(), np.tile(MON1, real.shape[0]), real.shape[0])
# model years are independent: one hour of CF 1e3 between years stops runs and windows from crossing them
model = np.asarray(P1["model"]["onshore 16 cells"], dtype=np.float64)
sep = np.full((model.shape[0], 1), 1e3)
M = table(np.hstack([model, sep]).ravel(), np.tile(np.append(MON1, 6), model.shape[0]), model.shape[0])
pd.set_option("display.width", 200)
print("\nOBSERVED 1980-2019 (40 years, one continuous series)\n" + R.to_string())
print("\nMODEL (300 independent years)\n" + M.to_string())

# season of CBT events >= 24 h below 10 %, observed
_, L = cbt_freq(real.ravel(), np.tile(MON1, real.shape[0]), 40, 0.10, False)
x = np.diff(np.concatenate([[0], (real.ravel() < 0.10).astype(np.int8), [0]]))
s, e = np.where(x == 1)[0], np.where(x == -1)[0]
m = np.tile(MON1, real.shape[0])[(s + e) // 2]
long_ = (e - s) >= 24
seas = {k: int((long_ & np.isin(m, v)).sum()) for k, v in (("DJF", [12, 1, 2]), ("MAM", [3, 4, 5]), ("JJA", [6, 7, 8]), ("SON", [9, 10, 11]))}
print(f"\nobserved CBT events >= 24 h below 10 % by season (1980-2019): {seas}, per year {long_.sum() / 40:.1f}")
