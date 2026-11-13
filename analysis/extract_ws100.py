# Thesis: 100 m wind of the PECD 4.2, input of the conversion and of the unified model
# Run: cd analysis && python extract_ws100.py --box=2000-2024 --box=1950-1999   (seconds)
# Needs: data_external/ws100 (download_pecd.py ws100)
# Writes: data/grid4x4_ws100_1950-2025.pkl, data/grid025_ws100_{2000-2024,1950-1999}.npy
"""Means over the 16 cells of the PECD 4.2 100 m wind (ERA5, bias-adjusted to the Global Wind Atlas) for 1950-2025 from
data_external/ws100, with the cell definition of extract_1950_1999.py; checks that the time stamps match those of the
10 m cell series.
Output: data/grid4x4_ws100_1950-2025.pkl (columns r{r}c{c}_ws100, m/s, UTC).
--box[=2000-2024|1950-1999]: also the 0.25 deg field of the study box (time x 33 x 45, float32, layout and time index of
grid025_wind_{period}.npy) as data/grid025_ws100_{period}.npy.
"""
import glob
import os
import time

import netCDF4 as nc
import numpy as np
import pandas as pd
import paths  # repository folders: data, figures, external raw data (see paths.py)

SRC = os.path.join(paths.EXTERNAL, "ws100")
D = paths.DATA
LAT_EDGES, LON_EDGES = [55, 53, 51, 49, 47], [5, 7.75, 10.5, 13.25, 16]


# read the study box 47-55 N, 5-16 E from one 100 m wind file
def read_box(f):
    d = nc.Dataset(f)
    lat, lon = d["latitude"][:].data, d["longitude"][:].data
    ii = np.where((lat >= 47 - 1e-6) & (lat <= 55 + 1e-6))[0]
    jj = np.where((lon >= 5 - 1e-6) & (lon <= 16 + 1e-6))[0]
    x = np.asarray(d["ws100"][:, ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1], dtype=np.float32)
    t = pd.to_datetime(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False,
                                   only_use_python_datetimes=True))
    la, lo = lat[ii[0]:ii[-1] + 1], lon[jj[0]:jj[-1] + 1]
    d.close()
    return x, t, la, lo


# means over the 16 cells of the 4 x 4 partition
def cell_means(x, la, lo):
    out = {}
    for r in range(4):
        for c in range(4):
            mi = (la >= LAT_EDGES[r + 1]) & (la <= LAT_EDGES[r])
            mj = (lo >= LON_EDGES[c]) & (lo <= LON_EDGES[c + 1])
            out[f"r{r}c{c}_ws100"] = x[:, mi][:, :, mj].mean(axis=(1, 2))
    return out


if __name__ == "__main__":
    T0 = time.time()
    frames = []
    for y in range(1950, 2026):
        hits = [h for h in glob.glob(f"{SRC}/*_S{y}01010000_*.nc") if not os.path.basename(h).startswith("._")]
        assert len(hits) == 1, (y, hits)
        x, t, la, lo = read_box(hits[0])
        assert x.shape[1:] == (33, 45) and not np.isnan(x).any(), (y, x.shape)
        frames.append(pd.DataFrame(cell_means(x, la, lo), index=t))
        if y % 10 == 0:
            print(f"{y}: {len(t)} h ({time.time() - T0:.0f} s)", flush=True)
    g = pd.concat(frames)
    for period in [a.split("=")[1] if "=" in a else "2000-2024" for a in __import__("sys").argv if a.startswith("--box")]:
        import pickle
        y0, y1 = (int(v) for v in period.split("-"))
        meta = pickle.load(open(f"{D}/grid025_meta{'' if period == '2000-2024' else '_' + period}.pkl", "rb"))
        box = np.empty((len(meta["time"]), 33, 45), dtype=np.float32)
        pos = 0
        for y in range(y0, y1 + 1):
            hits = [h for h in glob.glob(f"{SRC}/*_S{y}01010000_*.nc") if not os.path.basename(h).startswith("._")]
            x, t, la, lo = read_box(hits[0])
            assert np.allclose(la, meta["lat"]) and np.allclose(lo, meta["lon"]) and t.equals(meta["time"][pos:pos + len(t)])
            box[pos:pos + len(t)] = x
            pos += len(t)
        assert pos == len(meta["time"])
        np.save(f"{D}/grid025_ws100_{period}.npy", box)
        print(f"saved grid025_ws100_{period}.npy: {box.shape}, {time.time() - T0:.0f} s")
    for p in ("1950-1999", "2000-2024"):
        w10 = pd.read_pickle(f"{D}/grid4x4_wind_{p}.pkl")
        assert w10.index.isin(g.index).all(), p
    g.to_pickle(f"{D}/grid4x4_ws100_1950-2025.pkl")
    print(f"saved grid4x4_ws100_1950-2025.pkl: {g.shape}, mean {g.to_numpy().mean():.2f} m/s, {time.time() - T0:.0f} s")
