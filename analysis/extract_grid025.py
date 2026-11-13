# Thesis: Data chapter; 0.25 deg study box 2000-2024 used by the conversion and the aggregation study
# Run: cd analysis && python extract_grid025.py   (~7 min)
# Needs: raw PECD 4.2 ERA5 files in data/ERA-5-2000-2013 and data/ERA-5-2014-2025 (get_era5_data.ipynb)
# Writes: data/grid025_wind_2000-2024.npy, data/grid025_meta.pkl
"""Extracts the 0.25 deg ERA5 (PECD 4.2) field of the study box (47-55 N, 5-16 E, inclusive edges: 33 x 45 points),
2000-2024.

  wind  hourly 10 m wind speed (float32, ~1.3 GB)
  PV    the mean PV capacity factor per point and year (GHI with temperature derating, NOCT model)

Check at the end: the 16 cell means of the extracted wind reproduce grid4x4_wind_2000-2024.pkl.
"""
import glob
import pickle
import time

import netCDF4 as nc
import numpy as np
import pandas as pd
import paths  # repository folders: data, figures, external raw data (see paths.py)

D = paths.DATA
YEARS = range(2000, 2025)


# path of the raw PECD 4.2 file of one variable and year (2000-2013 or 2014-2025 folder)
def path(code, year):
    hits = sorted(glob.glob(f"{D}/ERA-5-2000-2013/*{code}*_S{year}01010000_*.nc")
                  + glob.glob(f"{D}/ERA-5-2014-2025/*{code}*_S{year}01010000_*.nc"))
    assert len(hits) == 1, (code, year, hits)
    return hits[0]


# read the study box 47-55 N, 5-16 E from one file
def read_box(f):
    d = nc.Dataset(f)
    lat, lon = d["latitude"][:].data, d["longitude"][:].data
    ii = np.where((lat >= 47 - 1e-6) & (lat <= 55 + 1e-6))[0]
    jj = np.where((lon >= 5 - 1e-6) & (lon <= 16 + 1e-6))[0]
    v = [k for k in d.variables if k not in ("time", "latitude", "longitude")][0]
    x = np.asarray(d[v][:, ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1], dtype=np.float32)
    t = pd.to_datetime(nc.num2date(d["time"][:], d["time"].units,
                                   only_use_cftime_datetimes=False, only_use_python_datetimes=True))
    out = (x, t, lat[ii[0]:ii[-1] + 1], lon[jj[0]:jj[-1] + 1])
    d.close()
    return out


# PV capacity factor from GHI and temperature (K) with the NOCT cell-temperature model
def pv_cf(ghi, tk):
    ghi = np.clip(ghi, 0, None)
    tc = tk - 273.15 + 0.03125 * ghi
    return np.clip(ghi / 1000 * (1 - 0.004 * (tc - 25)), 0, 1)


t0 = time.perf_counter()
winds, times, pv_year = [], [], []
for y in YEARS:
    w, t, lat, lon = read_box(path("WS-_0010m", y))
    g, tg, _, _ = read_box(path("GHI_0000m", y))
    a, ta, _, _ = read_box(path("TA-_0002m", y))
    assert t.equals(tg) and t.equals(ta) and w.shape == (len(t), 33, 45), y
    winds.append(w)
    times.append(t)
    pv_year.append(pv_cf(g, a).mean(axis=0))
    print(f"{y}: {len(t)} h  ({time.perf_counter() - t0:.0f} s)", flush=True)

W = np.concatenate(winds)
idx = times[0].append(times[1:])
np.save(f"{D}/grid025_wind_2000-2024.npy", W)
with open(f"{D}/grid025_meta.pkl", "wb") as f:
    pickle.dump({"time": idx, "lat": lat, "lon": lon, "pv_cf_by_year": np.stack(pv_year),
                 "years": list(YEARS)}, f)

# integrity: rebuild the 16 cell means and compare with the stored 4x4 series
g4 = pd.read_pickle(f"{D}/grid4x4_wind_2000-2024.pkl")
assert g4.index.equals(idx)
LAT_EDGES, LON_EDGES = [55, 53, 51, 49, 47], [5, 7.75, 10.5, 13.25, 16]
worst = 0.0
for r in range(4):
    for c in range(4):
        mi = (lat >= LAT_EDGES[r + 1]) & (lat <= LAT_EDGES[r])
        mj = (lon >= LON_EDGES[c]) & (lon <= LON_EDGES[c + 1])
        m = W[:, mi][:, :, mj].mean(axis=(1, 2))
        worst = max(worst, float(np.abs(m - g4[f"r{r}c{c}_wind"].to_numpy()).max()))
print(f"16-cell rebuild vs grid4x4 pickle: max |diff| = {worst:.5f} m/s")
print(f"saved {W.shape} float32 in {time.perf_counter() - t0:.0f} s")
