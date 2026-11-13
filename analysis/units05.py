# Thesis: Case Study, the 0.5 deg units of the unified model and the conversion of their weather into feed-in
# Module: imported by the model and the analyses (observed and simulated unit series, feedin())
# Needs: data/grid025_*, data/grid025de_* (extract scripts) and the conversion of conversion.py
"""The 0.5 deg units of the thesis model and the conversion of their weather into feed-in. No heavy work on import.

units       the German 0.25 deg land points (734; conversion.point_classes) grouped into 0.5 deg blocks of the grid
            (203 land units) and the 87 German offshore points of point3_value_map.py grouped the same way (31 offshore
            units, wind only). A unit value is the mean of its points; national values weight the units by their
            number of points (equal capacity per 0.25 deg point)
cells16     the 16 cells of the 4 x 4 partition as point-weighted means of the units (CELL_W, U x 16)
observed    real_units(): ERA5 / PECD 4.2 100 m wind (scaled to the Global Wind Atlas), 2 m temperature (K) and GHI,
            1950-2024, 29 February dropped, (75, 8760, U) float32, cached in data/real05_{wind,temp,solar}.npy;
            adjust=True shifts the 1950-1999 temperature per (unit, month) and scales the irradiance per (unit, month,
            hour) to the 2000-2024 climate
model       model_units(): the 300-year simulation of unified_model_05.py, data/sim05_{wind,temp,solar}.npy (float16;
            temperature stored in C), memory-mapped
conversion  feedin(): onshore three turbine classes by the shares of the unit's points, spread sigma, wind scale
            kappa(month, local hour) (calibration of point2_aggregation.py); offshore 350 W/m2 at 120 m with the
            calibration of point3_value_map.py; PV formula of conversion.py (module temperature, performance ratio,
            -1 h, orientation). Returns national onshore / offshore / PV CF, national temperature (C), 16-cell onshore
            CF, optionally the onshore CF per unit
"""
import json
import os
import pickle

import numpy as np
import pandas as pd

import conversion as cv

D = cv.D
RES = "0.5"
_meta = pickle.load(open(f"{D}/grid025_meta.pkl", "rb"))
LATG, LONG = np.asarray(_meta["lat"]), np.asarray(_meta["lon"])
PC = cv.point_classes()
RI, CI = np.asarray(PC["ri"]), np.asarray(PC["ci"])


# calibrated conversion of points 2 and 3: onshore spread and wind scale, offshore points and conversion. On a fresh
# start, point2_aggregation.py and point3_value_map.py have not run yet (they need the 2030 system of point 1, which needs
# this module), so the same values are then read from conversion05_bootstrap.json, an extract of their outputs
def _calibration():
    try:
        p2 = pickle.load(open(f"{D}/point2_aggregation.pkl", "rb"))
        p3 = pickle.load(open(f"{D}/point3_value_map.pkl", "rb"))
        off = p3["sites"][p3["sites"].type == "offshore"]
        return {"sigma_pt": p2["sigma_pt"], "kappa": np.asarray(p2["kappa"]), "off_lat": off.lat.to_numpy(),
                "off_lon": off.lon.to_numpy(), "off_sigma": p3["offshore conversion"]["sigma"],
                "off_kappa": p3["offshore conversion"]["kappa"]}
    except FileNotFoundError:
        b = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversion05_bootstrap.json")))
        return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in b.items() if not k.startswith("_")}


_CAL = _calibration()
ORI = np.array([int(np.argmin(np.abs(LATG - la))) for la in _CAL["off_lat"]])
OCI = np.array([int(np.argmin(np.abs(LONG - lo))) for lo in _CAL["off_lon"]])
K = 2                                                                   # 0.5 deg = 2 grid steps of 0.25 deg
LAND_ID, OFF_ID = (RI // K) * 100 + CI // K, (ORI // K) * 100 + OCI // K
LAND, OFF = np.unique(LAND_ID), np.unique(OFF_ID)
NL, NO = len(LAND), len(OFF)
_ML = (LAND_ID[:, None] == LAND[None]).astype(np.float64)               # (734, NL)
_MO = (OFF_ID[:, None] == OFF[None]).astype(np.float64)                 # (87, NO)
N_PTS = _ML.sum(axis=0)
W_LAND = N_PTS / len(RI)
W_OFF = _MO.sum(axis=0) / len(ORI)
SHARES = np.stack([(_ML * (PC["class"] == k)[:, None]).sum(axis=0) for k in cv.CLASSES], axis=1) / N_PTS[:, None]
LAT_U = (_ML * LATG[RI][:, None]).sum(axis=0) / N_PTS
LON_U = (_ML * LONG[CI][:, None]).sum(axis=0) / N_PTS
CELLS = cv.CELLS
_cell_pt = np.array([CELLS.index(c) for c in PC["points"]["cell"]])     # cell of each land point (4 x 4 partition)
_MC = np.stack([(_cell_pt == k).astype(np.float64) for k in range(16)], axis=1)   # (734, 16)
CELL_W = (_ML.T @ _MC) / _MC.sum(axis=0)[None]                          # (NL, 16): share of cell k's points in unit u
LAT_C = (_MC * LATG[RI][:, None]).sum(axis=0) / _MC.sum(axis=0)         # point centroids of the cells' German land
LON_C = (_MC * LONG[CI][:, None]).sum(axis=0) / _MC.sum(axis=0)
COLS = {"wind": NL + NO, "temp": NL, "solar": NL}


# great-circle distance in km (haversine)
def _hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    d = np.sin((la1 - la2) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo1 - lo2) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(d))


# distances between all pairs of 0.5 deg units
def unit_distance_km():
    return _hav(LAT_U[:, None], LON_U[:, None], LAT_U[None], LON_U[None])


def cell_distance_km():
    """distance between the German-land centroids of the 16 cells"""
    return _hav(LAT_C[:, None], LON_C[:, None], LAT_C[None], LON_C[None])


def group_mean(X, ids):
    """(T, P) point values -> (T, U) unit means in np.unique(ids) order (float64 sums, float32 out)"""
    order = np.argsort(ids, kind="stable")
    sid = ids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sid)) + 1]
    cnt = np.diff(np.r_[starts, len(sid)])
    return (np.add.reduceat(X[:, order], starts, axis=1, dtype=np.float64) / cnt).astype(np.float32)


def to_cells(x):
    """(..., T, NL) unit values -> (..., T, 16) point-weighted cell means"""
    return (x.astype(np.float32) @ CELL_W.astype(np.float32)).astype(np.float32)


# hourly index of the years y0-y1 without 29 February
def _year_index(y0, y1):
    i = pd.date_range(f"{y0}-01-01", f"{y1}-12-31 23:00", freq="h")
    return i[~((i.month == 2) & (i.day == 29))]


def _build_real():
    """observed unit series 1950-2024 (Feb 29 dropped) -> data/real05_{v}.npy (75, 8760, U) float32"""
    out = {}
    parts = []
    for per, (y0, y1) in (("1950-1999", (1950, 1999)), ("2000-2024", (2000, 2024))):
        m = _meta if per == "2000-2024" else pickle.load(open(f"{D}/grid025_meta_1950-1999.pkl", "rb"))
        t = pd.DatetimeIndex(m["time"])
        keep = np.where(~((t.month == 2) & (t.day == 29)))[0]
        W = np.load(f"{D}/grid025_ws100_{per}.npy", mmap_mode="r")
        w = np.empty((len(keep), NL + NO), np.float32)
        for a in range(0, len(keep), 20000):
            blk = np.asarray(W[keep[a:a + 20000]])
            w[a:a + 20000] = np.hstack([group_mean(blk[:, RI, CI], LAND_ID), group_mean(blk[:, ORI, OCI], OFF_ID)])
        parts.append(w)
    out["wind"] = np.concatenate(parts).reshape(75, 8760, NL + NO)
    tm = pickle.load(open(f"{D}/grid025de_meta_1950-2024.pkl", "rb"))["time"]
    rows = np.where(~((tm.month == 2) & (tm.day == 29)))[0]
    for v, f in (("temp", "t2m"), ("solar", "ghi")):
        X = np.load(f"{D}/grid025de_{f}_1950-2024.npy", mmap_mode="r")
        x = np.concatenate([group_mean(np.asarray(X[rows[a:a + 50000]]), LAND_ID) for a in range(0, len(rows), 50000)])
        out[v] = (np.clip(x, 0, None) if v == "solar" else x).reshape(75, 8760, NL)
    for v in out:
        np.save(f"{D}/real05_{v}.npy", out[v])
    return out


def real_units(adjust=True):
    """observed 1950-2024 (years in calendar order), dict wind / temp (K) / solar, (75, 8760, U) float32"""
    if not all(os.path.exists(f"{D}/real05_{v}.npy") for v in ("wind", "temp", "solar")):
        _build_real()
    X = {v: np.load(f"{D}/real05_{v}.npy") for v in ("wind", "temp", "solar")}
    if adjust:                                                          # 1950-1999 to the 2000-2024 climate
        tg = pd.date_range("2025-01-01", periods=8760, freq="h")
        mon, hr = tg.month.to_numpy(), tg.hour.to_numpy()
        for m in range(1, 13):
            sl = np.where(mon == m)[0]
            X["temp"][:50, sl] += (X["temp"][50:, sl].mean(axis=(0, 1)) - X["temp"][:50, sl].mean(axis=(0, 1)))[None, None]
            for h in range(24):
                sh = np.where((mon == m) & (hr == h))[0]
                mi, mo = X["solar"][50:, sh].mean(axis=(0, 1)), X["solar"][:50, sh].mean(axis=(0, 1))
                X["solar"][:50, sh] *= np.where(mo > 1, mi / np.maximum(mo, 1e-6), 1.0)[None, None].astype(np.float32)
    return X


def model_units():
    """the 300-year simulation of unified_model_05.py, memory-mapped float16 (temperature in C): dict wind / temp / solar"""
    return {v: np.load(f"{D}/sim05_{v}.npy", mmap_mode="r") for v in ("wind", "temp", "solar")}


# conversion
CONV_ON = cv.Wind100Conversion(cv.CLASSES, SHARES, W_LAND, _CAL["sigma_pt"], _CAL["kappa"])
CONV_OFF = cv.Wind100Conversion(cv.OFFSHORE, np.ones((NO, 1)), W_OFF, _CAL["off_sigma"], np.full((12, 24), _CAL["off_kappa"]))
PVC = cv.calibrate_16cells()["pv"]
LMON, LHR = cv.local_time(pd.date_range("2025-01-01", periods=8760, freq="h"))


def feedin(X, temp_in_c=False, units_cf=False, chunk=25):
    """X: wind (n, 8760, NL+NO), temp (K, or C with temp_in_c), solar (n, 8760, NL) - arrays or memmaps.
    -> on, off, pv, tnat (C) (n, 8760); cells (n, 8760, 16) onshore CF; temp16 (K), solar16 (n, 8760, 16); units (optional)"""
    n = X["wind"].shape[0]
    out = {k: np.empty((n, 8760), np.float32) for k in ("on", "off", "pv", "tnat")}
    out["cells"] = np.empty((n, 8760, 16), np.float32)
    out["temp16"] = np.empty((n, 8760, 16), np.float32)
    out["solar16"] = np.empty((n, 8760, 16), np.float32)
    if units_cf:
        out["units"] = np.empty((n, 8760, NL), np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        w = np.asarray(X["wind"][a:b], dtype=np.float32)
        tk = np.asarray(X["temp"][a:b], dtype=np.float32) + (273.15 if temp_in_c else 0.0)
        g = np.asarray(X["solar"][a:b], dtype=np.float32)
        cells = CONV_ON.regional(w[..., :NL], LMON, LHR).astype(np.float32)
        out["on"][a:b] = (cells * W_LAND).sum(axis=-1)
        out["off"][a:b] = CONV_OFF.cf(w[..., NL:], LMON, LHR)
        base = (cv.pv_cells(g, tk) * W_LAND).sum(axis=-1)
        out["pv"][a:b] = np.roll(PVC.pr * base, PVC.lag, axis=-1) * PVC.ratio[LMON - 1, LHR]
        out["tnat"][a:b] = (tk * W_LAND).sum(axis=-1) - 273.15
        out["cells"][a:b] = to_cells(cells)
        out["temp16"][a:b] = to_cells(tk)
        out["solar16"][a:b] = to_cells(g)
        if units_cf:
            out["units"][a:b] = cells
    return out
