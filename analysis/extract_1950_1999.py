# Thesis: the out-of-sample record 1950-1999 (75 observed years together with 2000-2024)
# Run: cd analysis && python extract_1950_1999.py [--check]   (~7 min)
# Needs: raw PECD 4.2 ERA5 files 1950-1999 in data_external/
# Writes: data/grid4x4_*_1950-1999.pkl, data/grid025_wind_1950-1999.npy
"""Extracts the means over the 16 cells (wind, irradiance, temperature) and the 0.25 deg wind field of the study box for
1950-1999 from the PECD 4.2 files in data_external/. Cells: equal partition of 47-55 N / 5-16 E with inclusive edges.
Option --check compares the year 2000 with the 2000-2024 series.
"""
import glob
import os
import pickle
import sys
import time

import netCDF4 as nc
import numpy as np
import pandas as pd
import paths  # repository folders: data, figures, external raw data (see paths.py)

SRC = paths.EXTERNAL
D = paths.DATA
LAT_EDGES, LON_EDGES = [55, 53, 51, 49, 47], [5, 7.75, 10.5, 13.25, 16]
CODES = {"wind": "WS-_0010m", "solar": "GHI_0000m", "temp": "TA-_0002m"}
CELLS = [f"r{r}c{c}" for r in range(4) for c in range(4)]


# path of the raw PECD 4.2 file of one variable and year on the external drive
def path(code, y):
    hits = [h for h in glob.glob(f"{SRC}/*{code}*_S{y}01010000_*.nc") if not os.path.basename(h).startswith("._")]
    assert len(hits) == 1, (code, y, hits)
    return hits[0]


# read the study box 47-55 N, 5-16 E from one file
def read_box(f):
    d = nc.Dataset(f)
    lat, lon = d["latitude"][:].data, d["longitude"][:].data
    ii = np.where((lat >= 47 - 1e-6) & (lat <= 55 + 1e-6))[0]
    jj = np.where((lon >= 5 - 1e-6) & (lon <= 16 + 1e-6))[0]
    v = [k for k in d.variables if k not in ("time", "latitude", "longitude")][0]
    x = np.asarray(d[v][:, ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1], dtype=np.float32)
    t = pd.to_datetime(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False,
                                   only_use_python_datetimes=True))
    la, lo = lat[ii[0]:ii[-1] + 1], lon[jj[0]:jj[-1] + 1]
    d.close()
    return x, t, la, lo


# means over the 16 cells of the 4 x 4 partition (equal partition with inclusive edges)
def cell_means(x, la, lo, var):
    out = {}
    for r in range(4):
        for c in range(4):
            mi = (la >= LAT_EDGES[r + 1]) & (la <= LAT_EDGES[r])
            mj = (lo >= LON_EDGES[c]) & (lo <= LON_EDGES[c + 1])
            out[f"r{r}c{c}_{var}"] = x[:, mi][:, :, mj].mean(axis=(1, 2))
    return out


if "--check" in sys.argv:
    for var, code in CODES.items():
        x, t, la, lo = read_box(path(code, 2000))
        new = pd.DataFrame(cell_means(x, la, lo, var), index=t)
        old = pd.read_pickle(f"{D}/grid4x4_{var}_2000-2024.pkl").loc[str(2000)]
        assert old.index.equals(new.index)
        print(f"{var}: year 2000 from the volume vs local series, max |diff| = {float((new - old[new.columns]).abs().max().max()):.2e}")
    sys.exit(0)

T0 = time.time()
frames = {v: [] for v in CODES}
wind_box, times = [], []
for y in range(1950, 2000):
    for var, code in CODES.items():
        x, t, la, lo = read_box(path(code, y))
        frames[var].append(pd.DataFrame(cell_means(x, la, lo, var), index=t))
        if var == "wind":
            wind_box.append(x)
            times.append(t)
    print(f"{y}: {len(t)} h ({time.time() - T0:.0f} s)", flush=True)
for var in CODES:
    pd.concat(frames[var]).to_pickle(f"{D}/grid4x4_{var}_1950-1999.pkl")
np.save(f"{D}/grid025_wind_1950-1999.npy", np.concatenate(wind_box))
with open(f"{D}/grid025_meta_1950-1999.pkl", "wb") as f:
    pickle.dump({"time": times[0].append(times[1:]), "lat": la, "lon": lo}, f)
print(f"saved grid4x4_*_1950-1999.pkl and grid025_wind_1950-1999.npy in {time.time() - T0:.0f} s")
