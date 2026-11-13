# Thesis: Discussion, the drought definition of Kittel and Schill (2026) on the series of this thesis,
# and the rank of the winter 1996/97 among the 74 observed winters
# Run: cd analysis && python kittel2026_check.py   (~1 min, prints the results)
"""Variable-renewable-energy droughts with the definition of Kittel & Schill (2026, Commun. Earth Environ. 7, 242; method:
Kittel & Schill 2024, Environ. Res.: Energy 1, 035007) applied to the German series of this thesis.
Definition (VMBT): a drought is a period whose mean capacity factor is below a relative threshold tau x the long-run mean
of the series; the search starts with the longest durations and moves to shorter ones, and every identified event is
excluded from later (shorter) searches.
Data: observed 1982-2019 on the 0.5 deg units (series of point1_night_ev.py; portfolio of the 2030 mix: 115 GW onshore,
30 GW offshore, 215 GW PV) as one continuous series; durations searched on a grid (1 h steps below 10 days, 6 h up to
100 days, 24 h above), the earliest start among windows of equal length. Also the worst 3-, 7-, 30- and 60-day
residual load of the 75 observed years and the rank of the winters 1995/96 to 1997/98.
"""
import pickle

import numpy as np
import pandas as pd
import paths  # repository folders: data, figures, external raw data (see paths.py)

D = paths.DATA
S = pickle.load(open(f"{D}/point1_night_ev.pkl", "rb"))["series"]["real 1950-2024"]
Y0, Y1 = 1982, 2019
sl = slice(Y0 - 1950, Y1 - 1950 + 1)
on, off, pv = (np.asarray(S[k][sl], dtype=np.float64).ravel() for k in ("onshore 16 cells", "offshore", "pv"))
SERIES = {"portfolio": (115 * on + 30 * off + 215 * pv) / 360, "onshore wind": on, "offshore wind": off, "solar PV": pv}
NY = Y1 - Y0 + 1
DATES = pd.DatetimeIndex(np.concatenate([pd.date_range(f"{y}-01-01", periods=8760, freq="h").to_numpy() for y in range(Y0, Y1 + 1)]))
GRID = sorted(set(list(range(17520, 2400, -24)) + list(range(2400, 240, -6)) + list(range(240, 0, -1))), reverse=True)


def vmbt(cf, thr):
    """events (start, length) of the variable-duration mean-below-threshold search, longest durations first"""
    c = np.concatenate([[0.0], np.cumsum(cf)])
    used = np.zeros(len(cf), dtype=np.int64)
    events = []
    for d in GRID:
        if d > len(cf):
            continue
        cu = np.concatenate([[0], np.cumsum(used)])
        ok = ((c[d:] - c[:-d]) / d < thr) & (cu[d:] - cu[:-d] == 0)
        starts = np.flatnonzero(ok)
        pos = 0
        while True:
            k = np.searchsorted(starts, pos)
            if k == len(starts):
                break
            s = starts[k]
            events.append((s, d))
            used[s:s + d] = 1
            pos = s + d
    return events


rows = {}
for name, cf in SERIES.items():
    mean = cf.mean()
    for tau in (0.4, 0.5, 0.6, 0.75, 0.8):
        ev = vmbt(cf, tau * mean)
        L = np.array([d for _, d in ev])
        s, d = max(ev, key=lambda e: e[1])
        mid = DATES[s + d // 2]
        rows[(name, tau)] = {"mean CF": round(mean, 3), "longest (d)": round(d / 24, 1), "start": DATES[s].strftime("%Y-%m-%d"),
                             "median hour": mid.strftime("%Y-%m"), "CF in longest / mean": round(cf[s:s + d].mean() / mean, 2),
                             ">= 48 h per yr": round((L >= 48).sum() / NY, 1), ">= 1 wk per yr": round((L >= 168).sum() / NY, 1),
                             ">= 2 wk per yr": round((L >= 336).sum() / NY, 1)}
T = pd.DataFrame(rows).T
pd.set_option("display.width", 220)
print(f"observed {Y0}-{Y1}, 2030 mix (portfolio = 115 on + 30 off + 215 PV)\n" + T.to_string())

# year of the longest portfolio drought per threshold
cf = SERIES["portfolio"]
for tau in (0.4, 0.6, 0.8):
    s, d = max(vmbt(cf, tau * cf.mean()), key=lambda e: e[1])
    print(f"portfolio tau {tau}: longest {d / 24:.1f} d, {DATES[s].date()} to {DATES[min(s + d, len(cf) - 1)].date()}")

# the most severe non-overlapping windows of 3, 7, 30 and 60 days of the 2030 residual load in the 75 observed years
X = pickle.load(open(f"{D}/point1_night_ev.pkl", "rb"))["series"]["real 1950-2024"]
rl = (np.asarray(X["demand"], dtype=np.float64) - 115 * np.asarray(X["onshore 16 cells"], dtype=np.float64)
      - 30 * np.asarray(X["offshore"], dtype=np.float64) - 215 * np.asarray(X["pv"], dtype=np.float64)).ravel()
dates75 = pd.DatetimeIndex(np.concatenate([pd.date_range(f"{y}-01-01", periods=8760, freq="h").to_numpy() for y in range(1950, 2025)]))
c = np.concatenate([[0.0], np.cumsum(rl)])
for days in (3, 7, 30, 60):
    w = 24 * days
    m = (c[w:] - c[:-w]) / w
    order, taken, top = np.argsort(m)[::-1], np.zeros(len(m), bool), []
    for i in order:
        if not taken[max(0, i - w):i + w].any():
            top.append(i)
            taken[i] = True
        if len(top) == 5:
            break
    print(f"worst {days:>2} d mean RL (GW): " + "; ".join(f"{dates75[i].date()} {m[i]:.1f}" for i in top))

# rank of each winter (July-June) by its worst 7-, 30- and 60-day mean residual load
win = np.where(dates75.month >= 7, dates75.year, dates75.year - 1)                  # winter starting in this year
for days in (7, 30, 60):
    w = 24 * days
    m = pd.Series((c[w:] - c[:-w]) / w, index=win[:len(c) - w])
    best = m.groupby(level=0).max().drop([1949, 2024], errors="ignore").sort_values(ascending=False)
    rank = {f"{y}/{(y + 1) % 100:02d}": int(np.where(best.index == y)[0][0]) + 1 for y in (1995, 1996, 1997)}
    print(f"worst {days:>2} d RL: rank among {len(best)} winters {rank}; top 3 {[f'{y}/{(y + 1) % 100:02d}' for y in best.index[:3]]}")
