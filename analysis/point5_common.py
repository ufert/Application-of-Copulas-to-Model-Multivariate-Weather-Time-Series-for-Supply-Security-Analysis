# Thesis: Results, shared data of the validation, the extremes and the number of years
# Module: observed and simulated series on the 16 cells and the 2030 system series
"""Shared data of the validation, extreme-value and convergence analyses (point5_*.py). Loads on import, no other effects.

observed  75 years 1950-2024 on the 0.5 deg units (units05.py), 29 February dropped, calendar order: 100 m wind,
          temperature, irradiance (1950-1999 not level-adjusted; dependence statistics use ranks within each period)
model     the unified model on 0.5 deg units (unified_model_05.py), 300 years (2025 calendar), memory-mapped
16 cells  temperature and irradiance aggregated from the units to the 16 cells of the 4 x 4 partition
          (point-weighted), so that statistics defined on the 16 cells keep their meaning
feed-in   converted per unit (units05.feedin), then aggregated: national onshore / offshore / PV CF, national
          temperature, onshore CF of the 16 cells; local time of the 2025 calendar for every year
system    the 2030 series of point1_night_ev.py (with offshore): demand without EVs, PV, offshore, onshore; residual
          load RL = demand - 115 onshore - 30 offshore - 215 PV
"""
import pickle

import numpy as np
import pandas as pd
from scipy import stats

import conversion as cv
import units05 as u05

D = cv.D
CELLS = cv.CELLS
YEARS = np.arange(1950, 2025)
T25 = pd.date_range("2025-01-01", periods=8760, freq="h")
MON, HR = T25.month.to_numpy(), T25.hour.to_numpy()
LMON, LHR = cv.local_time(T25)
SEAS = np.select([np.isin(MON, [12, 1, 2]), np.isin(MON, [3, 4, 5]), np.isin(MON, [6, 7, 8])], ["DJF", "MAM", "JJA"], "SON")
CAP = {"on": 115.0, "off": 30.0, "pv": 215.0}
LAT_C, LON_C = u05.LAT_C, u05.LON_C


def cell_distance_km():
    """distance between the German-land centroids of the 16 cells"""
    return u05.cell_distance_km()


def _cells(arr, add=0.0, chunk=50):
    """(n, 8760, NL) unit values (array or memmap) -> (n, 8760, 16) float32 cell means"""
    return np.concatenate([u05.to_cells(np.asarray(arr[a:a + chunk], dtype=np.float32) + add) for a in range(0, arr.shape[0], chunk)])


# hourly index of the years y0-y1 without 29 February
def _idx(y0, y1):
    i = pd.date_range(f"{y0}-01-01", f"{y1}-12-31 23:00", freq="h")
    return i[~((i.month == 2) & (i.day == 29))]


def load_real():
    """observed 1950-2024 on the 0.5 deg units (raw); 'temp' / 'solar' = 16-cell aggregates (K, W/m2)"""
    R = u05.real_units(adjust=False)
    return {"wind_u": R["wind"], "temp_u": R["temp"], "solar_u": R["solar"], "temp_in_c": False,
            "temp": _cells(R["temp"]), "solar": _cells(R["solar"])}


def load_model():
    """the 0.5 deg simulation of unified_model_05.py (memory-mapped units); 'temp' / 'solar' = 16-cell aggregates (K, W/m2)"""
    M = u05.model_units()
    return {"wind_u": M["wind"], "temp_u": M["temp"], "solar_u": M["solar"], "temp_in_c": True,
            "temp": _cells(M["temp"], 273.15), "solar": _cells(M["solar"])}


def feedin(X):
    """onshore CF of the 16 cells (units, 8760, 16), national onshore, offshore, PV CF and temperature (C) (units, 8760)"""
    F = u05.feedin({"wind": X["wind_u"], "temp": X["temp_u"], "solar": X["solar_u"]}, temp_in_c=X.get("temp_in_c", False))
    return {"cells": F["cells"], "on": F["on"], "off": F["off"], "pv": F["pv"], "tnat": F["tnat"]}


def system_series():
    """2030 system series of point 1 (with offshore, no EVs) for 'real' (75 years, calendar order) and 'model' (300)"""
    S = pickle.load(open(f"{D}/point1_night_ev.pkl", "rb"))["series"]
    order = [(list(range(2000, 2025)) + list(range(1950, 2000))).index(y) for y in YEARS]
    out = {}
    for name, key, o in (("real", "real 1950-2024", order), ("model", "model", None)):
        s = {k: (v[o] if o is not None else v).astype(np.float64) for k, v in S[key].items()}
        s["re"] = CAP["on"] * s["onshore 16 cells"] + CAP["off"] * s["offshore"] + CAP["pv"] * s["pv"]
        s["rl"] = s["demand"] - s["re"]
        out[name] = s
    return out


def slot_uniform(x, groups=None):
    """(units, 8760) -> uniforms by rank within each (month, hour) slot; groups (units,) ranks within each group of years
    separately (e.g. the two real periods, so that level trends do not enter)"""
    groups = np.zeros(x.shape[0], int) if groups is None else np.asarray(groups)
    u = np.empty(x.shape, dtype=np.float64)
    for g in np.unique(groups):
        rows = groups == g
        xs = x[rows]
        ug = np.empty(xs.shape)
        for m in range(1, 13):
            for h in range(24):
                sl = (MON == m) & (HR == h)
                block = xs[:, sl]
                r = stats.rankdata(block.ravel(), method="average")
                ug[:, sl] = (r / (len(r) + 1)).reshape(block.shape)
        u[rows] = ug
    return u


REAL_PERIOD = np.where(YEARS < 2000, 0, 1)                    # ranks within 1950-1999 and 2000-2024 separately


# lengths of the runs below thr (lulls)
def runs(b, thr=0.10):
    x = np.diff(np.concatenate([[0], (b < thr).astype(np.int8), [0]]))
    return np.where(x == -1)[0] - np.where(x == 1)[0]


# highest w-hour moving mean of each row (each year)
def roll_max(x, w):
    c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
    return ((c[:, w:] - c[:, :-w]) / w).max(axis=1)


# lowest w-hour moving mean of each row (each year)
def roll_min(x, w):
    return -roll_max(-x, w)
