# Thesis: Discussion, the Dunkelflaute definition of Li et al. (2021) applied to the series of this thesis
# Run: cd analysis && python li2021_check.py   (seconds, prints the results)
"""Dunkelflaute events with the definition of Li, Basu, Watson & Russchenberg (2021, Energies 14, 6508) applied to the
series of this thesis: an hour is a Dunkelflaute hour if the national wind CF and the national PV CF are both below a
threshold (20 %; 10 % as a sensitivity), night hours included; events are runs of such hours; statistics for events
longer than one day (number, longest, share of time, season, distribution of durations).
Data: the 2030 system series of point1_night_ev.py (0.5 deg units; wind = 115 GW onshore + 30 GW offshore, or onshore
alone; PV 215 GW): observed 1950-2024, its 1985-2016 subset and the 300 simulated years; 2025 calendar, UTC hours.
"""
import pickle

import numpy as np
import pandas as pd
import paths  # repository folders: data, figures, external raw data (see paths.py)

D = paths.DATA
P1 = pickle.load(open(f"{D}/point1_night_ev.pkl", "rb"))["series"]
MON = pd.date_range("2025-01-01", periods=8760, freq="h").month.to_numpy()


def runs(mask):
    """(start, length) of runs of True in a 1-d boolean array"""
    x = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    s, e = np.where(x == 1)[0], np.where(x == -1)[0]
    return s, e - s


# events with wind and PV capacity factors both below thr for more than one day (definition of Li et al. 2021)
def stats(wind, pv, thr, years):
    n_long, longest, hours_long, h_nj, h_of, lengths, all_lengths = 0, 0, 0, 0, 0, [], []
    for y in range(wind.shape[0]):
        s, L = runs((wind[y] < thr) & (pv[y] < thr))
        all_lengths += list(L)
        for a, l in zip(s, L):
            if l > 24:
                n_long += 1
                hours_long += l
                lengths.append(l)
                m = MON[a:a + l]
                h_nj += np.isin(m, [11, 12, 1]).sum()
                h_of += np.isin(m, [10, 11, 12, 1, 2]).sum()
        longest = max(longest, L.max() if len(L) else 0)
    al = np.array(all_lengths)
    hist = np.histogram(al, bins=[1, 12, 24, 36, 48, 72, 10 ** 6])[0]
    return {"events > 1 d per year": n_long / years, "events > 2 d per year": sum(l > 48 for l in lengths) / years,
            "events > 3 d per year": sum(l > 72 for l in lengths) / years, "longest (d)": longest / 24,
            "share of hours in events > 1 d (%)": 100 * hours_long / (years * 8760),
            "share Nov-Jan (%)": 100 * h_nj / max(hours_long, 1), "share Oct-Feb (%)": 100 * h_of / max(hours_long, 1),
            "events (all) share 12-24 h (%)": 100 * hist[1] / hist.sum(),
            "events >= 12 h: share 12-24 h (%)": 100 * hist[1] / hist[1:].sum()}


rows = {}
for name, X, sl in (("observed 1950-2024", P1["real 1950-2024"], slice(None)),
                    ("observed 1985-2016", P1["real 1950-2024"], slice(35, 67)),
                    ("model (300 yr)", P1["model"], slice(None))):
    on, off, pv = (np.asarray(X[k][sl], dtype=np.float64) for k in ("onshore 16 cells", "offshore", "pv"))
    wind = (115 * on + 30 * off) / 145
    for thr in (0.20, 0.10):
        rows[(name, "wind on+off", f"{thr:.0%}")] = stats(wind, pv, thr, on.shape[0])
    rows[(name, "onshore only", "20%")] = stats(on, pv, 0.20, on.shape[0])
T = pd.DataFrame(rows).T
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(T.round(2).to_string())

# duration histogram of all events, observed 1985-2016, wind on + off, 20 %
X = P1["real 1950-2024"]
on, off, pv = (np.asarray(X[k][35:67], dtype=np.float64) for k in ("onshore 16 cells", "offshore", "pv"))
wind = (115 * on + 30 * off) / 145
L = np.concatenate([runs((wind[y] < 0.2) & (pv[y] < 0.2))[1] for y in range(32)])
edges = [1, 3, 6, 9, 12, 15, 18, 21, 24, 36, 48, 72, 10 ** 6]
h = np.histogram(L, bins=edges)[0]
print(f"\nobserved 1985-2016, all events ({len(L) / 32:.0f} per year), share by duration:")
print("  " + "  ".join(f"{a}-{min(b, 1000) - 1} h {100 * c / len(L):.1f}%" for a, b, c in zip(edges[:-1], edges[1:], h)))
