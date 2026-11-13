# Thesis: all chapters; sets up the folders of raw data that the scripts and notebooks read
# Run: cd analysis && python organize_raw_data.py   (seconds; after get_era5_data.ipynb and nasa_data_pre.ipynb)
# Needs: data_external/ with the unzipped PECD 4.2 ERA5 files 1950-2025 (get_era5_data.ipynb);
#        data/MERRA-2-2015-2025 and data/MERRA-2-rad-2015-2025 (nasa_data_pre.ipynb)
# Writes: links in data/ERA-5-2000-2013, data/ERA-5-2014-2025, data/ERA-5-extreme-years, data/PECD, data/MERRA-2-edit,
#         data/MERRA-2-rad
"""Links the downloaded raw files into the folders that the scripts and notebooks read, under the file names they expect.
No data is copied: every entry is a relative symbolic link (a hard link where symbolic links are not available).

ERA5 / PECD 4.2, from data_external/ (10 m wind, 2 m temperature, irradiance; three files per year):
  data/ERA-5-2000-2013      2000-2013, file names as downloaded (with the area suffix ".area-subset.75.45.18.-31")
  data/ERA-5-2014-2025      2014-2025, file names without the area suffix
  data/ERA-5-extreme-years  1963 and 2010, file names as downloaded
  data/PECD                 2025, file names without the area suffix
MERRA-2, the files of 2025 from the downloads of 2015-2025:
  data/MERRA-2-edit         flux collection (M2T1NXFLX)
  data/MERRA-2-rad          radiation collection (M2T1NXRAD)

Entries that already exist (links or files of the same name) are left as they are, so the script can be run again at
any time. It stops with exit code 1 and lists the missing downloads if a source file is not there.
"""
import glob
import os
import sys

import paths  # repository folders: data, figures, external raw data (see paths.py)

SUFFIX = ".area-subset.75.45.18.-31"
# PECD 4.2 variable codes and the bias-correction tag in their file names
CODES = {"WS-_0010m": "mbc", "TA-_0002m": "noc", "GHI_0000m": "noc"}
ERA5 = [("ERA-5-2000-2013", range(2000, 2014), SUFFIX),
        ("ERA-5-2014-2025", range(2014, 2026), ""),
        ("ERA-5-extreme-years", (1963, 2010), SUFFIX),
        ("PECD", (2025,), "")]
MERRA = [("MERRA-2-edit", "MERRA-2-2015-2025", "MERRA2_4*.tavg1_2d_flx_Nx.2025*.nc"),
         ("MERRA-2-rad", "MERRA-2-rad-2015-2025", "MERRA2_4*.tavg1_2d_rad_Nx.2025*.nc")]


# PECD 4.2 file name of one variable and year
def era5_name(code, year, suffix):
    return (f"H_ERA5_ECMW_T639_{code}_Pecd_025d_S{year}01010000_E{year}12312300_INS_MAP_01h_COM_{CODES[code]}_org_"
            f"NA_NA---_NA---_NA---_PECD4.2_fv1{suffix}.nc")


# the downloaded file of one variable and year in data_external/, with or without the area suffix
def era5_source(code, year):
    for suffix in (SUFFIX, ""):
        p = os.path.join(paths.EXTERNAL, era5_name(code, year, suffix))
        if os.path.exists(p):
            return p
    return None


# link src to dst (relative, so the repository can be moved); returns "linked" or "exists"
def link(src, dst):
    if os.path.lexists(dst):
        return "exists"
    try:
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
    except OSError:
        os.link(src, dst)
    return "linked"


if __name__ == "__main__":
    missing = []
    for folder, years, suffix in ERA5:
        out = os.path.join(paths.DATA, folder)
        os.makedirs(out, exist_ok=True)
        count = {"linked": 0, "exists": 0}
        for year in years:
            for code in CODES:
                src = era5_source(code, year)
                if src is None:
                    missing.append(f"data_external/{era5_name(code, year, SUFFIX)}")
                    continue
                count[link(src, os.path.join(out, era5_name(code, year, suffix)))] += 1
        print(f"data/{folder:22} {count['linked']:4} linked, {count['exists']:4} already there")

    for folder, source, pattern in MERRA:
        out = os.path.join(paths.DATA, folder)
        os.makedirs(out, exist_ok=True)
        files = sorted(glob.glob(os.path.join(paths.DATA, source, pattern)))
        if len(files) < 365:
            missing.append(f"data/{source}: {len(files)} files of 2025 instead of 365")
        count = {"linked": 0, "exists": 0}
        for src in files:
            count[link(src, os.path.join(out, os.path.basename(src)))] += 1
        print(f"data/{folder:22} {count['linked']:4} linked, {count['exists']:4} already there")

    if missing:
        print(f"\n{len(missing)} source(s) missing; download them first (get_era5_data.ipynb, nasa_data_pre.ipynb):")
        for m in missing[:20]:
            print(f"  {m}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        sys.exit(1)
    print("\nall raw-data folders complete")
