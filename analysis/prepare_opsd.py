# Thesis: Data, observed load and feed-in in Germany from Open Power System Data (OPSD), time series version 2020-10-06
# Used by: the demand model (ENTSO-E load 2015-2019) and the calibration of the wind and PV conversion (observed feed-in)
# Run: cd analysis && python prepare_opsd.py [path/to/time_series_60min_singleindex.csv]   (downloads ~125 MB if no path)
# Writes: data/opsd_germany_60min.pkl
"""Extract the German columns of the OPSD hourly time series into data/opsd_germany_60min.pkl."""
import os
import sys
import urllib.request

import pandas as pd

import paths  # repository folders: data, figures, external raw data (see paths.py)

# OPSD data package "time series", version 2020-10-06 (doi 10.25832/time_series/2020-10-06)
URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"

# the German columns used in the thesis: load, and capacity, generation and profile of wind (total, offshore, onshore) and PV
COLS = ["DE_load_actual_entsoe_transparency", "DE_load_forecast_entsoe_transparency", "DE_solar_capacity",
        "DE_solar_generation_actual", "DE_solar_profile", "DE_wind_capacity", "DE_wind_generation_actual",
        "DE_wind_profile", "DE_wind_offshore_capacity", "DE_wind_offshore_generation_actual", "DE_wind_offshore_profile",
        "DE_wind_onshore_capacity", "DE_wind_onshore_generation_actual", "DE_wind_onshore_profile"]

if __name__ == "__main__":
    # use a local copy of the CSV if given, otherwise download it once into the data folder
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(paths.DATA, "time_series_60min_singleindex.csv")
    if not os.path.exists(src):
        # download under a temporary name, so an interrupted download is never mistaken for the complete file
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, src + ".part")
        os.replace(src + ".part", src)

    # hourly UTC time stamps as a naive index, as in the rest of the code
    df = pd.read_csv(src, usecols=["utc_timestamp"] + COLS, index_col="utc_timestamp", parse_dates=True)
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df = df[COLS]
    out = os.path.join(paths.DATA, "opsd_germany_60min.pkl")
    df.to_pickle(out)
    print(f"saved {out}: {df.shape[0]} hours, {df.index[0]} to {df.index[-1]}")
