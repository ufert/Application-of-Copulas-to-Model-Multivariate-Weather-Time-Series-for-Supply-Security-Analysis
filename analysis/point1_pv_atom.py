# Thesis: Results, Critical Hours (the point mass of irradiance at zero)
# Run: cd analysis && python point1_pv_atom.py   (seconds)
"""The point mass of irradiance at zero.
1. Is the zero random? ERA5 night values are floating-point noise (about 1e-13 W/m2), so zero := GHI < 1e-6 W/m2. For
   each (cell, day of year, hour) slot, the share of the 25 years with zero shows whether the zero is a random event or
   fixed by the solar geometry.
2. Kendall's tau between national wind and irradiance pooled over all hours, pooled over the daylight hours and within
   (month, hour) slots.
3. The unified model: zero hours where the observed record is never zero, non-zero values where it is always zero, and
   the share of zero hours by hour of the day.
Writes data/point1_pv_atom.pkl.
"""
import pickle

import numpy as np
import pandas as pd
from scipy import stats

import units05 as u05
import paths  # repository folders: data, figures, external raw data (see paths.py)

D = paths.DATA
EPS = 1e-6
W = u05.W_LAND                                                       # points per 0.5 deg unit / 734
CELLS = [f"u{x}" for x in u05.LAND]                                  # 0.5 deg units (name kept for the output format)
_R = u05.real_units(adjust=False)
G = _R["solar"][50:].astype(np.float64).reshape(25, 365, 24, u05.NL)  # observed 2000-2024
V = _R["wind"][50:, :, :u05.NL].astype(np.float64).reshape(25, 365, 24, u05.NL)
del _R
_MS = u05.model_units()["solar"]                                     # unified model on 0.5 deg units, 300 years (float16 memmap)
ZM = np.concatenate([np.asarray(_MS[a:a + 50]) < EPS for a in range(0, _MS.shape[0], 50)]).reshape(-1, 365, 24, u05.NL)
CENTRAL = int(np.argmin((u05.LAT_U - 51.0) ** 2 + (u05.LON_U - 10.5) ** 2))
out = {}

# 1. random or fixed by the solar geometry?
Z = G < EPS
fz = Z.mean(axis=0)                                                   # (365, 24, 16)
always, never = fz == 1, fz == 0
mixed = ~always & ~never
out["atom"] = {"zero share": Z.mean(), "slots always zero": always.mean(), "slots never zero": never.mean(),
               "slots mixed": mixed.mean(), "zero hours in mixed slots": (Z & mixed[None]).sum() / Z.sum(),
               "non-zero value in mixed slots, median W/m2": float(np.median(G[:, mixed][~Z[:, mixed]]))}
print("1. THE ATOM AT ZERO (25 years, 203 units of 0.5 deg; zero = GHI < 1e-6 W/m2)")
for k, v in out["atom"].items():
    print(f"   {k:45} {v:.4f}")

# 2. dependence: pooled vs conditional on the solar geometry
wn, gn = (V * W).sum(axis=3), (G * W).sum(axis=3)                     # national (25, 365, 24)
day = (G.mean(axis=(0, 3)) > 1)                                       # (365, 24) daylight in the climatology
mon = pd.date_range("2025-01-01", periods=365, freq="D").month.to_numpy()
tau_all = stats.kendalltau(wn.ravel(), gn.ravel())[0]
tau_day = stats.kendalltau(wn[:, day].ravel(), gn[:, day].ravel())[0]
taus, ns = [], []
for m in range(1, 13):
    for h in range(24):
        sl = (mon == m)[:, None] & (np.arange(24) == h)[None, :] & day
        if sl.sum() < 5:
            continue
        x, y = wn[:, sl].ravel(), gn[:, sl].ravel()
        taus.append(stats.kendalltau(x, y)[0])
        ns.append(x.size)
tau_slot = np.average(taus, weights=ns)
ties = (gn < EPS).mean()
out["tau"] = {"pooled, all hours": tau_all, "pooled, daylight hours": tau_day, "within (month, hour) slots, daylight": tau_slot,
              "ties at zero (national)": ties}
print("\n2. KENDALL'S TAU, national wind vs national irradiance (2000-2024)")
for k, v in out["tau"].items():
    print(f"   {k:45} {v:+.3f}")

# 3. the unified model
night_light = (~ZM & always[None]).sum() / (always.sum() * ZM.shape[0])
day_dark = (ZM & never[None]).sum() / (never.sum() * ZM.shape[0])
light_vals = np.concatenate([np.asarray(_MS[a:a + 50], dtype=np.float64).reshape(-1, 365, 24, u05.NL)[:, always][~ZM[a:a + 50][:, always]]
                             for a in range(0, ZM.shape[0], 50)])
local_hour = (np.arange(24) + 1) % 24                                 # CET; zero share by hour (winter time)
share_real = Z.mean(axis=(0, 1, 3))
share_model = ZM.mean(axis=(0, 1, 3))
out["model"] = {"light in always-dark slots": night_light, "max value there W/m2": float(light_vals.max()) if light_vals.size else 0.0,
                "zero in never-dark slots": day_dark, "zero share real": Z.mean(), "zero share model": ZM.mean(),
                "max abs gap in zero share by hour (pp)": 100 * np.abs(share_real - share_model).max()}
print("\n3. UNIFIED MODEL (300 years) against the real zero pattern")
for k, v in out["model"].items():
    print(f"   {k:45} {v:.5f}")
mixed_m = mixed[None] & True
print("   zero share in the real mixed (dawn/dusk) slots: real", round(Z[:, mixed].mean(), 3), "model", round(ZM[:, mixed].mean(), 3))
with open(f"{D}/point1_pv_atom.pkl", "wb") as f:
    pickle.dump({**out, "zero share by UTC hour": pd.DataFrame({"real": share_real, "model": share_model}),
                 "share of years zero (day, UTC hour, cell)": fz.astype(np.float32), "cells": CELLS, "central unit": CENTRAL,
                 "resolution": "0.5 deg units"}, f)
print(f"\nsaved {D}/point1_pv_atom.pkl")
