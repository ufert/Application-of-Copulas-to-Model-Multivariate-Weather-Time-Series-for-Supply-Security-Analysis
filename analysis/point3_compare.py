# Thesis: Results, Potential and Risk Areas (check against the seven German onshore zones of the PECD)
# Run: cd analysis && python point3_compare.py   (after point3_value_map.py)
# Writes: data/point3_compare.pkl
"""Compares the potential and risk areas of point3_value_map.py with the per-point wind map of risk_map_025.py.
risk_map_025.py: 10 m wind, fixed shear 0.25, one power curve, 2000-2024; stress = the 5 % of hours with the lowest
centred 72-h mean CF of a uniform wind fleet; per point the CF in those hours ("lull CF") and lull CF / mean CF
("retention"); potential / risk = top / bottom 20 % of lull CF.
point3_value_map.py: 100 m wind, turbine classes, 1950-2024; stress from the 2030 residual load; firmness relative to an
average MW with a bootstrap test.
Matched on the 734 German land points; the stress CF of point3_value_map.py is also designated with the quintile rule
of risk_map_025.py, to separate the rule from the data. Also ranks the seven German onshore zones of the PECD by their
capacity factors in the same stress windows. Writes data/point3_compare.pkl.
"""
import pickle

import numpy as np
import pandas as pd
import paths  # repository folders: data, figures, external raw data (see paths.py)

D = paths.DATA
OLD = pickle.load(open(f"{D}/risk_map_025.pkl", "rb"))["points"]
NEW = pickle.load(open(f"{D}/point3_value_map.pkl", "rb"))["sites"]
NEW = NEW[NEW["type"] == "onshore"].copy()
OLD = OLD[OLD["germany"]].copy()
J = NEW.merge(OLD, on=["lat", "lon"], suffixes=("", " old"))
assert len(J) == len(NEW) == 734, len(J)
pd.set_option("display.width", 220)


# label the highest fifth 'potential', the lowest fifth 'risk' (definition of risk_map_025.py)
def quint(x):
    hi, lo = np.quantile(x, 0.8), np.quantile(x, 0.2)
    return np.where(x >= hi, "potential", np.where(x <= lo, "risk", "neither"))


J["class 72 h, quintile rule"] = quint(J["stress CF 72 h"])
J["class old"] = J["class old"] if "class old" in J else J["class"]
sp = lambda a, b: J[a].rank().corr(J[b].rank())
rows = {"resource: mean CF old vs new": sp("mean CF old", "mean CF"),
        "CF in stress: old lull CF vs new stress CF (72 h)": sp("lull CF", "stress CF 72 h"),
        "old lull CF vs new firmness r (72 h)": sp("lull CF", "r 72 h"),
        "old lull CF vs new firmness r (peak)": sp("lull CF", "r peak"),
        "retention: old vs new kept share (72 h)": sp("retention", "kept 72 h"),
        "old mean CF vs new firmness r (72 h)": sp("mean CF old", "r 72 h"),
        "data / conversion alone: old lull CF vs fresh data with the earlier definition": sp("lull CF", "wind-lull CF (earlier definition)"),
        "definition alone: fresh data, earlier definition vs fresh firmness (72 h)": sp("wind-lull CF (earlier definition)", "r 72 h")}
print("RANK CORRELATIONS on the 734 German land points (Spearman)")
print(pd.Series(rows).round(3).to_string())


# cross table of two designations
def xtab(a, b):
    return pd.crosstab(J[a], J[b], margins=True)


print("\nDESIGNATIONS: risk_map_025.py (rows) vs point3_value_map.py (columns)")
print(xtab("class old", "class 72 h").to_string())
print("\nDESIGNATIONS: risk_map_025.py vs the data of point3_value_map.py with the quintile rule")
print(xtab("class old", "class 72 h, quintile rule").to_string())
agree_rule = (J["class old"] == J["class 72 h, quintile rule"]).mean()
opp_rule = (((J["class old"] == "potential") & (J["class 72 h, quintile rule"] == "risk")) | ((J["class old"] == "risk") & (J["class 72 h, quintile rule"] == "potential"))).mean()
print(f"same class (quintile rule): {agree_rule:.2f}; opposite: {opp_rule:.2f}")

REG = {}
for lab, m in (("north coast > 53.5 N", J.lat > 53.5), ("central 50.5-52.5 N", (J.lat >= 50.5) & (J.lat <= 52.5)),
               ("Alpine foreland < 48.5 N", J.lat < 48.5), ("west < 9 E", J.lon < 9), ("east > 12 E", J.lon > 12)):
    REG[lab] = {"n": int(m.sum()), "old potential": (J.loc[m, "class old"] == "potential").mean(), "old risk": (J.loc[m, "class old"] == "risk").mean(),
                "new potential": (J.loc[m, "class 72 h"] == "potential").mean(), "new risk": (J.loc[m, "class 72 h"] == "risk").mean(),
                "old retention": J.loc[m, "retention"].median(), "new kept (72 h)": J.loc[m, "kept 72 h"].median(),
                "new r (72 h)": J.loc[m, "r 72 h"].median()}
REG = pd.DataFrame(REG).T
print("\nBY REGION")
print(REG.round(3).to_string())
O = pickle.load(open(f"{D}/risk_map_025.pkl", "rb"))
print("\nearlier robustness to the fleet geography:", O.get("robustness") if not isinstance(O.get("robustness"), pd.DataFrame) else "")
if isinstance(O.get("robustness"), pd.DataFrame):
    print(O["robustness"].round(3).to_string())
# independent check: PECD 4.2 zone capacity factors in the same stress windows
# PECD's own conversion (existing fleet, resource grade B) for the onshore zones DE01-DE07, 1950-2024; stress = the worst
# 72-h residual-load window of each year (as point3_value_map.py); site firmness of point3_value_map.py aggregated to the zones by the mask
import pecd  # noqa: E402

S1 = pickle.load(open(f"{D}/point1_night_ev.pkl", "rb"))["series"]["real 1950-2024"]
YRS = np.arange(1950, 2025)
order = [(list(range(2000, 2025)) + list(range(1950, 2000))).index(y) for y in YRS]
rl = (S1["demand"] - 215 * S1["pv"] - 30 * S1["offshore"] - 115 * S1["onshore 16 cells"])[order].astype(np.float64)
mask = np.zeros(rl.shape, dtype=bool)
for i, r in enumerate(rl):
    c = np.concatenate([[0.0], np.cumsum(r)])
    j = int(np.argmax(c[72:] - c[:-72]))
    mask[i, j:j + 72] = True
ZW = pecd.zone_weights("onshore")
P30 = pecd.cf("onshore", "30")
P30 = P30[(P30.index.year <= 2024) & ~((P30.index.month == 2) & (P30.index.day == 29))][ZW["zones"]].to_numpy(dtype=np.float64).reshape(75, 8760, -1)
area = ZW["W"].sum(axis=1)
s_z = np.stack([P30[i][mask[i]].mean(axis=0) for i in range(75)])        # (years, zones)
f_z = s_z @ (area / area.sum())
r_pecd = s_z.sum(axis=0) / f_z.sum()
NEWS = pickle.load(open(f"{D}/point3_value_map.pkl", "rb"))["sites"]
NEWS = NEWS[NEWS["type"] == "onshore"]
Wz = ZW["W"] / ZW["W"].sum(axis=1, keepdims=True)                           # zone x point weights (same point order)
r_ours = Wz @ NEWS["stress CF 72 h"].to_numpy() / (NEWS["stress CF 72 h"].to_numpy() @ (ZW["W"].sum(axis=0) / ZW["W"].sum()))
ZT = pd.DataFrame({"PECD firmness": r_pecd, "fresh firmness (sites -> zone)": r_ours,
                   "PECD mean CF": P30.mean(axis=(0, 1)), "region": ["NW (SH, Lower Saxony N)", "NE (MV, Brandenburg)", "W (NRW)",
                                                                       "E (Saxony-Anh., Saxony, Thur.)", "SW (Hesse, RP)",
                                                                       "Baden-Wuerttemberg", "Bavaria"]}, index=ZW["zones"])
print("\nINDEPENDENT CHECK: firmness of the PECD 4.2 onshore zones in the worst 72-h residual-load window of each year (1950-2024)")
print(ZT.round(3).to_string())
print(f"rank correlation PECD vs fresh (7 zones): {ZT['PECD firmness'].rank().corr(ZT['fresh firmness (sites -> zone)'].rank()):.3f}; "
      f"Pearson {np.corrcoef(ZT['PECD firmness'], ZT['fresh firmness (sites -> zone)'])[0, 1]:.3f}")
with open(f"{D}/point3_compare.pkl", "wb") as f:
    pickle.dump({"rank correlations": pd.Series(rows), "regions": REG, "crosstab": xtab("class old", "class 72 h"),
                 "crosstab quintile rule": xtab("class old", "class 72 h, quintile rule"), "joined": J, "pecd zones": ZT}, f)
