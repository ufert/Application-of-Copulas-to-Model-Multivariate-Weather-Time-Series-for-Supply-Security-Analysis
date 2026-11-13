# Thesis: Data chapter, PECD 4.2 (capacity factors for the conversion check, 100 m wind, zone masks)
# Run: cd analysis && python download_pecd.py [masks|cf|ws100|all]   (~2.5 h, needs a CDS account in ~/.cdsapirc)
# Writes: data_external/masks, data_external/capacity_factors, data_external/ws100
"""Downloads PECD 4.2 (CDS dataset sis-energy-pecd, historical stream, ERA5, 1950-2025) into data_external/, Germany only:
gridded data for the study box 47-55 N, 5-16 E; regional time series arrive for all of Europe and are reduced to the
German regions on arrival.

  masks   static region masks on the 0.25 deg grid: nuts_0, nuts_2, peon, peof (zone membership of the grid points),
          cropped to the study box
  cf      hourly capacity factors as regional time series (CSV, one file per year):
            onshore  wind_power_onshore_capacity_factor, PEON zones (Germany DE01-DE07): technology 30 (existing
                     fleet), 35 (SP277 HH150) and 32 (SP199 HH150), resource grade B
            offshore wind_power_offshore_capacity_factor, PEOF zones: 20 (existing), 21 (SP316 HH155),
                     22 (SP370 HH155), resource grade B
            pv       solar_photovoltaic_generation_capacity_factor, NUTS 0: 60 industrial rooftop, 61 residential
                     rooftop, 62 utility fixed, 63 utility 1-axis tracking
          25-year requests (a request that fails twice is split into 5-year requests); each CSV is reduced to the
          German columns (prefix DE) with its metadata header and saved as DE_<file>.csv
  ws100   100 m wind speed, 0.25 deg, study box [55, 5, 47, 16], one request per year (~20 MB per year)
Four requests run in parallel. Existing targets are skipped, so the script can be restarted. Needs ~/.cdsapirc and the
accepted dataset licence.
Usage: python download_pecd.py [masks|cf|ws100|all]  (default all)
Output: data_external/{masks,capacity_factors/{onshore,offshore,pv},ws100}.
"""
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import cdsapi
import paths  # repository folders: data, figures, external raw data (see paths.py)

ROOT = paths.EXTERNAL
DATASET = "sis-energy-pecd"
BASE = {"pecd_version": "pecd4_2", "temporal_period": ["historical"], "origin": ["era5_reanalysis"], "file_version": ["fv1"]}
YEARS = list(range(1950, 2026))
AREA_DE = [55, 5, 47, 16]                     # N, W, S, E: the study box (16 cells, 0.25 deg points), covers Germany
WORKERS = 4
WHAT = next((a for a in sys.argv[1:] if not a.startswith("--")), "all")
CF = [("onshore", "wind_power_onshore_capacity_factor", "peon", ["30", "35", "32"], "resource_grade_b"),
      ("offshore", "wind_power_offshore_capacity_factor", "peof", ["20", "21", "22"], "resource_grade_b"),
      ("pv", "solar_photovoltaic_generation_capacity_factor", "nuts_0", ["60", "61", "62", "63"], None)]
LOCK = threading.Lock()


# time-stamped log line, safe across the download threads
def log(m):
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# one CDS API request with a retry
def retrieve(request, target, tries=2):
    client = cdsapi.Client(quiet=True, progress=False)
    for k in range(tries):
        try:
            client.retrieve(DATASET, request).download(target + ".part")
            os.replace(target + ".part", target)
            return True
        except Exception as e:                                       # noqa: BLE001 - CDS errors vary
            log(f"  {os.path.basename(target)}: attempt {k + 1} failed: {str(e)[:300]}")
            time.sleep(60 * (k + 1))
    return False


# unzip a downloaded archive next to it, optionally deleting the zip
def extract(target, delete=False):
    with zipfile.ZipFile(target) as z:
        z.extractall(os.path.dirname(target))
        names = z.namelist()
    if delete:
        os.remove(target)
    return names


def germany_csv(path):
    """keep the metadata header and the German columns of a PECD regional CSV; True if it worked"""
    import pandas as pd
    with open(path) as f:
        header = [ln for ln in f if ln.startswith("#")]
    d = pd.read_csv(path, comment="#")
    de = [c for c in d.columns if c.startswith("DE")]
    if not de:
        log(f"  {os.path.basename(path)}: no German columns (kept as is): {list(d.columns)[:8]}")
        return False
    out = os.path.join(os.path.dirname(path), "DE_" + os.path.basename(path))
    with open(out, "w") as f:
        f.writelines(header)
        d[[d.columns[0]] + de].to_csv(f, index=False)
    os.remove(path)
    return True


def crop_nc(path):
    """crop a gridded file to the study box; True if it worked"""
    import xarray as xr
    try:
        ds = xr.open_dataset(path)
        lat = next(c for c in ds.coords if c.lower().startswith("lat"))
        lon = next(c for c in ds.coords if c.lower().startswith("lon"))
        lat_sl = slice(AREA_DE[0], AREA_DE[2]) if ds[lat][0] > ds[lat][-1] else slice(AREA_DE[2], AREA_DE[0])
        sub = ds.sel({lat: lat_sl, lon: slice(AREA_DE[1], AREA_DE[3])}).load()
        ds.close()
        sub.to_netcdf(os.path.join(os.path.dirname(path), "DE_" + os.path.basename(path)))
        os.remove(path)
        return True
    except Exception as e:                                           # noqa: BLE001
        log(f"  {os.path.basename(path)}: not cropped ({str(e)[:200]})")
        return False


# region masks of the PECD zones (onshore, offshore, countries)
def job_masks():
    out = f"{ROOT}/masks"
    os.makedirs(out, exist_ok=True)
    target = f"{out}/region_masks.zip"
    if os.path.exists(target):
        return log("masks: already there")
    req = {"pecd_version": "pecd4_2", "file_version": ["fv1"],
           "variable": ["nuts_0_region_mask", "nuts_2_region_mask", "peon_region_mask", "peof_region_mask"]}
    log("masks: requesting")
    if retrieve(req, target):
        names = [n for n in extract(target) if n.endswith(".nc") and not os.path.basename(n).startswith("._")]
        ok = [crop_nc(os.path.join(out, n)) for n in names]
        log(f"masks: done, {sum(ok)} of {len(names)} cropped to the study box")
    else:
        log("masks: FAILED")


# hourly capacity factors of one technology and region type, 1950-2025, split into decades
def job_cf(name, var, region, tech, grade, years, split=True):
    out = f"{ROOT}/capacity_factors/{name}"
    os.makedirs(out, exist_ok=True)
    target = f"{out}/{name}_tech{tech}_{region}_{years[0]}-{years[-1]}.zip"
    if os.path.exists(target) or os.path.exists(target + ".done"):
        return log(f"{os.path.basename(target)}: already there")
    req = BASE | {"variable": [var], "spatial_resolution": [region], "technology": [tech], "year": [str(y) for y in years]}
    if grade:
        req["energy_scenario"] = [grade]
    log(f"{os.path.basename(target)}: requesting")
    if retrieve(req, target):
        names = [n for n in extract(target) if n.endswith(".csv") and not os.path.basename(n).startswith("._")]
        ok = [germany_csv(os.path.join(out, n)) for n in names]
        if all(ok):
            os.remove(target)
            open(target + ".done", "w").write("\n".join(names))          # marker: request completed
        log(f"{os.path.basename(target)}: done, {sum(ok)} of {len(names)} files reduced to Germany")
    elif split and len(years) > 5:
        log(f"{os.path.basename(target)}: splitting into 5-year requests")
        for i in range(0, len(years), 5):
            job_cf(name, var, region, tech, grade, years[i:i + 5], split=False)
    else:
        log(f"{os.path.basename(target)}: FAILED")


# 100 m wind of one year for the study box (skipped if already present)
def job_ws100(year):
    out = f"{ROOT}/ws100"
    os.makedirs(out, exist_ok=True)
    if any(f"_S{year}0101" in f and f.endswith(".nc") and not f.startswith("._") for f in os.listdir(out)):
        return log(f"ws100 {year}: already there")
    target = f"{out}/{year}-{year}.zip"
    req = BASE | {"variable": ["100m_wind_speed"], "spatial_resolution": ["0_25_degree"], "year": [str(year)],
                  "area": AREA_DE}
    log(f"ws100 {year}: requesting")
    if retrieve(req, target):
        log(f"ws100 {year}: done ({', '.join(extract(target, delete=True))})")
    else:
        log(f"ws100 {year}: FAILED")


if __name__ == "__main__":
    assert os.path.isdir(ROOT), f"{ROOT} not mounted"
    jobs = []
    if WHAT in ("masks", "all"):
        jobs.append((job_masks, ()))
    if WHAT in ("cf", "all"):
        for name, var, region, techs, grade in CF:
            for tech in techs:
                for i in range(0, len(YEARS), 25):
                    jobs.append((job_cf, (name, var, region, tech, grade, YEARS[i:i + 25])))
    if WHAT in ("ws100", "all"):
        jobs += [(job_ws100, (y,)) for y in YEARS]
    log(f"{len(jobs)} jobs, {WORKERS} in parallel")
    with ThreadPoolExecutor(WORKERS) as ex:
        for f in [ex.submit(fn, *args) for fn, args in jobs]:
            f.result()
    log("finished")
