# Thesis: means over the 16 cells of the 4 x 4 partition, 2000-2024 (regional models, spatial tests, notebook)
# Run: cd analysis && python load_4x4_grid.py   (~10 min)
# Needs: raw PECD 4.2 ERA5 files in data/ERA-5-2000-2013 and data/ERA-5-2014-2025
# Writes: data/grid4x4_{wind,temp,solar}_2000-2024.pkl
"""Means of wind, temperature and irradiance over the 16 cells of the 4 x 4 partition, 2000-2024.
Each (year, variable) file is opened once and all 16 cell means are extracted from it.
Grid: the study box (latitude 47-55, longitude 5-16) split into 4 equal latitude bands (2 deg each) x 4 equal longitude
bands (2.75 deg each) = 16 cells r{row}c{col}, row 0 = north, column 0 = west.
"""
import os
import time

import numpy as np
import pandas as pd
import xarray as xr
import paths  # repository folders: data, figures, external raw data (see paths.py)

T0 = time.time()
# time-stamped log line
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

DATA_DIR = paths.DATA
base_dir = f"{DATA_DIR}/ERA-5-2014-2025"
pre2014_dir = f"{DATA_DIR}/ERA-5-2000-2013"
pre2014_suffix = ".area-subset.75.45.18.-31"
PRE2014_CUTOFF = 2014
train_years = range(2000, 2025)

LAT_EDGES = np.linspace(47, 55, 5)   # 4 bands: [47,49],[49,51],[51,53],[53,55]
LON_EDGES = np.linspace(5, 16, 5)    # 4 bands: [5,7.75],[7.75,10.5],[10.5,13.25],[13.25,16]

CELLS = {}
for row in range(4):
    for col in range(4):
        CELLS[f"r{row}c{col}"] = {
            "lat_min": LAT_EDGES[3 - row],      # row 0 = northmost = highest lat
            "lat_max": LAT_EDGES[4 - row],
            "lon_min": LON_EDGES[col],
            "lon_max": LON_EDGES[col + 1],
        }

VAR_SPECS = {
    "wind": ("WS-_0010m", "ws10", "mbc"),
    "temp": ("TA-_0002m", "t2m", "noc"),
    "solar": ("GHI_0000m", "ssrd", "noc"),
}


# read one variable and year from the raw PECD 4.2 file and average it over the 16 cells
def load_year_all_cells(var_code, var_name, year, mbc_or_noc, source_dir, filename_suffix=""):
    filename = (
        f"H_ERA5_ECMW_T639_{var_code}_Pecd_025d_"
        f"S{year}01010000_E{year}12312300_INS_MAP_01h_COM_"
        f"{mbc_or_noc}_org_NA_NA---_NA---_NA---_PECD4.2_fv1{filename_suffix}.nc"
    )
    path = os.path.join(source_dir, filename)
    out = {}
    with xr.open_dataset(path) as ds:
        # Force the whole year's grid into memory ONCE - each of the 16
        # .sel().mean() calls below then runs on an in-memory array
        # instead of triggering its own lazy disk read.
        data = ds[var_name].load()
        latitude = data["latitude"].values
        times = pd.to_datetime(data["time"].values)
        ascending = latitude[0] < latitude[-1]
        for cell_name, region in CELLS.items():
            lat_slice = (slice(region["lat_min"], region["lat_max"]) if ascending
                         else slice(region["lat_max"], region["lat_min"]))
            ts = data.sel(latitude=lat_slice, longitude=slice(region["lon_min"], region["lon_max"])).mean(dim=["latitude", "longitude"])
            out[cell_name] = pd.Series(ts.values, index=times)
    return out


# extract all three variables for 2000-2024 and cache them as pickles
def main():
    for var_key, (var_code, var_name, mbc_or_noc) in VAR_SPECS.items():
        cache_path = f"{DATA_DIR}/grid4x4_{var_key}_2000-2024.pkl"
        if os.path.exists(cache_path):
            log(f"{var_key}: already cached at {cache_path}, skipping")
            continue
        log(f"Loading {var_key} at 4x4 resolution (16 cells x 25 years)...")
        records = {f"{cell}_{var_key}": [] for cell in CELLS}
        for year in train_years:
            src_dir = base_dir if year >= PRE2014_CUTOFF else pre2014_dir
            suffix = "" if year >= PRE2014_CUTOFF else pre2014_suffix
            year_cells = load_year_all_cells(var_code, var_name, year, mbc_or_noc, src_dir, filename_suffix=suffix)
            for cell in CELLS:
                records[f"{cell}_{var_key}"].append(year_cells[cell])
            log(f"  {var_key} {year} done")
        df_var = pd.DataFrame({k: pd.concat(v).sort_index() for k, v in records.items()})
        df_var = df_var[~df_var.index.duplicated(keep="first")].sort_index()
        df_var.to_pickle(cache_path)
        log(f"Cached {var_key} to {cache_path}, shape {df_var.shape}")
    log("All variables loaded.")


if __name__ == "__main__":
    main()
