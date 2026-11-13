# Thesis: loader of the PECD 4.2 capacity factors and zone weights (conversion check, zone ranking)
# Module: cached in data/pecd_de_*.pkl
"""Loader for the PECD 4.2 German capacity factors and region masks downloaded by download_pecd.py.
  cf(kind, tech)        hourly CF, DataFrame (UTC, 1950-2025, one column per zone), cached in
                        data/pecd_de_{kind}_{tech}.pkl so that later runs do not need the raw files
  zone_weights(kind)    (zones, W): W (zones, 734 German points) = share of each 0.25 deg German land point that lies in
                        each zone (onshore PEON zones DE01-DE07; offshore PEOF zones over all grid points)
PECD time stamps: PV is hour-ending like ERA5 radiation (shift by -1 h before combining it with hour-beginning load or
feed-in data); wind needs no shift.
"""
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd
import paths  # repository folders: data, figures, external raw data (see paths.py)

D = paths.DATA
ROOT = paths.EXTERNAL
KINDS = {"onshore": "PEON", "offshore": "PEOF", "pv": "NUT0"}


# German PECD 4.2 capacity factors of one region type and technology (cached as a pickle)
def cf(kind, tech):
    path = f"{D}/pecd_de_{kind}_{tech}.pkl"
    if os.path.exists(path):
        return pd.read_pickle(path)
    files = sorted(f for f in glob.glob(f"{ROOT}/capacity_factors/{kind}/DE_*_org_{tech}_*.csv")
                   if not os.path.basename(f).startswith("._"))
    assert files, f"no PECD files for {kind} {tech}: is {ROOT} mounted?"
    parts = []
    for f in files:
        d = pd.read_csv(f, comment="#", index_col=0, parse_dates=True)
        parts.append(d.astype(np.float32))
    out = pd.concat(parts).sort_index()
    out.index.name = "utc"
    years = sorted({int(re.search(r"_S(\d{4})0101", f).group(1)) for f in files})
    assert len(out) == len(pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 23:00", freq="h")), "gaps in the PECD series"
    out.to_pickle(path)
    return out


# weights of the PECD zones (onshore or offshore) from the region masks (cached)
def zone_weights(kind):
    import xarray as xr
    path = f"{D}/pecd_zone_weights_{kind}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    m = xr.open_dataset(f"{ROOT}/masks/DE_ANCI_{KINDS[kind]}-mask_PECD4.2_fv1.nc").mask
    zones = [str(z) for z in m.region.values if str(z).startswith("DE")]
    m = m.sel(region=zones).values                                           # (zones, lat, lon), grid of grid025_meta
    P = pickle.load(open(f"{D}/risk_map_025.pkl", "rb"))["points"]
    G = P[P.germany].reset_index(drop=True)
    meta = pickle.load(open(f"{D}/grid025_meta.pkl", "rb"))
    ri = np.searchsorted(-meta["lat"], -G["lat"].to_numpy())
    ci = np.searchsorted(meta["lon"], G["lon"].to_numpy())
    out = {"zones": zones, "W": m[:, ri, ci].astype(float), "points": G[["lat", "lon", "cell"]], "full": m}
    pickle.dump(out, open(path, "wb"))
    return out
