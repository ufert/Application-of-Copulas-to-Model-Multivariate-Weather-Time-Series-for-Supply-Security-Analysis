# Thesis: irradiance and temperature at the 734 German land points, 1950-2024 (0.5 deg units, PV analyses)
# Run: cd analysis && python extract_solar025.py   (~8 min)
# Needs: raw PECD 4.2 ERA5 files 1950-2024 in data_external/, data/risk_map_025.pkl (risk_map_025.py)
# Writes: data/grid025de_{ghi,t2m}_1950-2024.npy and the meta file
"""Extracts irradiance (GHI) and 2 m temperature at the 734 German land points of the 0.25 deg grid (the points of
conversion.point_classes), 1950-2024, from the PECD 4.2 / ERA5 files in data_external/.
Output: data/grid025de_ghi_1950-2024.npy and data/grid025de_t2m_1950-2024.npy (time x 734, float32, UTC, all hours
including 29 February), data/grid025de_meta_1950-2024.pkl (time index, latitude, longitude).
"""
import glob
import os
import pickle
import time

import netCDF4 as nc
import numpy as np
import pandas as pd

import conversion as cv
import paths  # repository folders: data, figures, external raw data (see paths.py)

SRC = paths.EXTERNAL
D = cv.D
CODES = {"ghi": "GHI_0000m", "t2m": "TA-_0002m"}
PC = cv.point_classes()
meta = pickle.load(open(f"{D}/grid025_meta.pkl", "rb"))
LAT, LON = meta["lat"], meta["lon"]
RI, CI = PC["ri"], PC["ci"]


# path of the raw PECD 4.2 file of one variable and year on the external drive
def path(code, y):
    hits = [h for h in glob.glob(f"{SRC}/*{code}*_S{y}01010000_*.nc") if not os.path.basename(h).startswith("._")]
    assert len(hits) == 1, (code, y, hits)
    return hits[0]


# read the German land points of the study box from one file
def read_points(f):
    d = nc.Dataset(f)
    lat, lon = d["latitude"][:].data, d["longitude"][:].data
    ii = np.where((lat >= 47 - 1e-6) & (lat <= 55 + 1e-6))[0]
    jj = np.where((lon >= 5 - 1e-6) & (lon <= 16 + 1e-6))[0]
    assert np.allclose(lat[ii], LAT) and np.allclose(lon[jj], LON)
    v = [k for k in d.variables if k not in ("time", "latitude", "longitude")][0]
    x = np.asarray(d[v][:, ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1], dtype=np.float32)[:, RI, CI]
    t = pd.to_datetime(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
    d.close()
    return x, t


if __name__ == "__main__":
    T0 = time.time()
    idx = pd.date_range("1950-01-01", "2024-12-31 23:00", freq="h")
    out = {k: np.lib.format.open_memmap(f"{D}/grid025de_{k}_1950-2024.npy", mode="w+", dtype=np.float32, shape=(len(idx), len(RI)))
           for k in CODES}
    pos = 0
    for y in range(1950, 2025):
        n = None
        for k, code in CODES.items():
            x, t = read_points(path(code, y))
            assert t.equals(idx[pos:pos + len(t)]), (k, y)
            out[k][pos:pos + len(t)] = np.clip(x, 0, None) if k == "ghi" else x
            n = len(t)
        pos += n
        if y % 10 == 0:
            print(f"{y}: done ({time.time() - T0:.0f} s)", flush=True)
    assert pos == len(idx)
    for v in out.values():
        v.flush()
    pickle.dump({"time": idx, "lat": PC["points"]["lat"].to_numpy(), "lon": PC["points"]["lon"].to_numpy(), "cell": PC["points"]["cell"].to_numpy()},
                open(f"{D}/grid025de_meta_1950-2024.pkl", "wb"))
    g = np.load(f"{D}/grid025de_ghi_1950-2024.npy", mmap_mode="r")
    print(f"saved: GHI mean {float(g[-8760 * 5:].mean()):.1f} W/m2 (2020-24), {time.time() - T0:.0f} s")
