# Thesis: Data chapter, Tables 4.2-4.4 (comparison statistics ERA5 against MERRA-2, Europe, 2025)
# Run: cd analysis && python data_comparison_relative.py [wind temp solar_swgdn solar]   (~25 min, prints the tables)
# Needs: the same ERA5 and MERRA-2 files as data_chapter_maps_pdf.py
"""Relative comparison statistics ERA5 vs MERRA-2 (Data chapter, section 4.5, Tables 4.2-4.4).

Europe, hourly 2025: MERRA-2 interpolated bilinearly onto the ERA5 grid, wind and temperature linearly in time,
irradiance resampled to 1 h, inner join of the time stamps, statistics per grid point, unweighted mean over the points.
Also the relative versions of the statistics of Definition 11 (mathematics chapter):
    relative = statistic / mean over the same points of the corresponding ERA5 statistic.
Wind and irradiance: relative to the ERA5 mean (bias), ERA5 std (std difference), ERA5 q95 (percentile difference).
Temperature (no natural zero in deg C): all three relative to the mean ERA5 std.
Irradiance: MERRA-2 SWGDN (rad collection, the values of Table 4.4). The option "solar" compares, in addition,
NIRDF + NIRDR of the flx collection; it needs these two variables in the MERRA-2 download, which the thesis does not use.
Without arguments, the three tables of the thesis are computed (wind, temp, solar_swgdn).

Processed in latitude bands (linear interpolation is separable, so banding with a margin gives the same values).
Output: printed (python data_comparison_relative.py > data_comparison_relative.log).
"""
import glob
import os
import sys

import numpy as np
import xarray as xr

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ERA = os.path.join(ROOT, "data", "PECD")
ERA_FILES = {
    "wind": ("WS-_0010m", "ws10"),
    "temp": ("TA-_0002m", "t2m"),
    "solar": ("GHI_0000m", "ssrd"),
}
BAND = 16  # ERA5 latitudes per band


# one ERA5 / PECD 4.2 field of 2025 on its native grid, latitudes ascending
def era(var):
    tag, name = ERA_FILES[var]
    f = glob.glob(os.path.join(ERA, f"*{tag}*S2025*.nc"))[0]
    da = xr.open_dataset(f)[name].rename({"latitude": "lat", "longitude": "lon"}).sortby("lat")
    return da - 273.15 if var == "temp" else da


# one MERRA-2 variable of 2025 from the daily files (expr selects or combines variables)
def merra(pattern, expr):
    files = sorted(glob.glob(os.path.join(ROOT, "data", pattern)))
    ds = xr.open_mfdataset(files, combine="by_coords")
    da = expr(ds).astype("float32").sortby("lat").load()
    return da


def stats(e, m):
    """per-point comparison statistics (Definition 11) and the ERA5 statistics the relative versions refer to"""
    return {
        "bias": e.mean("time") - m.mean("time"),
        "dstd": e.std("time") - m.std("time"),
        "corr": xr.corr(e, m, dim="time"),
        "dq95": e.quantile(0.95, dim="time") - m.quantile(0.95, dim="time"),
        "mean_A": e.mean("time"),
        "std_A": e.std("time"),
        "q95_A": e.quantile(0.95, dim="time"),
    }


# statistics per grid point, in latitude bands to limit memory: MERRA-2 interpolated onto the ERA5 grid and
# times, irradiance as hourly means
def run(var, m_full, solar):
    e_full = era(var)
    lats = e_full.lat.values
    dlat = float(abs(m_full.lat[1] - m_full.lat[0]))
    parts = []
    for k in range(0, len(lats), BAND):
        la = lats[k:k + BAND]
        e = e_full.sel(lat=la).astype("float32").load()
        m = m_full.sel(lat=slice(la.min() - 2 * dlat, la.max() + 2 * dlat))
        m = m.interp(lat=e.lat, lon=e.lon, method="linear")
        if solar:
            e = e.resample(time="1h").mean()
            m = m.resample(time="1h").mean()
        else:
            m = m.interp(time=e.time)
        e, m = xr.align(e, m, join="inner")
        parts.append(xr.Dataset(stats(e, m)).compute())
        print(f"  {var}: band {k // BAND + 1}/{-(-len(lats) // BAND)}", flush=True)
    return xr.concat(parts, dim="lat")


# print the domain means of the statistics and their values relative to the ERA5 reference (Tables 4.2-4.4)
def report(var, s, label):
    out = {}
    for key, ref in [("bias", "mean_A"), ("dstd", "std_A"), ("dq95", "q95_A")]:
        ok = np.isfinite(s[key])
        num = float(s[key].where(ok).mean())
        den = float(s["std_A" if var == "temp" else ref].where(ok).mean())
        out[key] = (num, den, num / den)
    r = float(s["corr"].mean())
    print(f"\n=== {label} ===")
    print(f"  points: {int(np.isfinite(s['bias']).sum())} of {s['bias'].size}")
    names = {"bias": "mean bias", "dstd": "std difference", "dq95": "95th percentile difference"}
    for key, (num, den, rel) in out.items():
        refname = "ERA5 std" if var == "temp" else {"bias": "ERA5 mean", "dstd": "ERA5 std", "dq95": "ERA5 q95"}[key]
        print(f"  {names[key]:28s} {num:+9.3f}   {refname} {den:8.3f}   relative {100 * rel:+6.1f} %")
    print(f"  {'mean temporal correlation':28s} {r:9.3f}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["wind", "temp", "solar_swgdn"]
    for v in which:
        if v == "wind":
            report("wind", run("wind", merra("MERRA-2-edit/*flx*2025*.nc", lambda d: d["SPEED"]), False),
                   "Wind speed (ERA5 10 m vs MERRA-2 lowest layer)")
        elif v == "temp":
            report("temp", run("temp", merra("MERRA-2-edit/*flx*2025*.nc", lambda d: d["TLML"] - 273.15), False),
                   "Air temperature (ERA5 2 m vs MERRA-2 lowest layer)")
        elif v == "solar":
            report("solar", run("solar", merra("MERRA-2-edit/*flx*2025*.nc", lambda d: d["NIRDF"] + d["NIRDR"]),
                                True), "Irradiance, MERRA-2 NIRDF + NIRDR (flx collection)")
        elif v == "solar_swgdn":
            report("solar", run("solar", merra("MERRA-2-rad/*rad*2025*.nc", lambda d: d["SWGDN"]), True),
                   "Irradiance with MERRA-2 SWGDN")
