# Thesis: Data chapter, Figures 4.1-4.6 (mean maps of ERA5 and MERRA-2 2025 and their differences)
# Run: cd analysis && python data_chapter_maps_pdf.py
# Needs: ERA5 / PECD 4.2 2025 in data/PECD, MERRA-2 2025 flx and rad in data/MERRA-2-edit and data/MERRA-2-rad
#        (get_era5_data.ipynb, nasa_data_pre.ipynb, then organize_raw_data.py)
# Writes: thesis/pics/Era-5-mean-*.pdf, Merra-2-mean-*.pdf, Residual_*.pdf
"""
PDF versions of the nine maps of the data chapter (Figures 4.1-4.6 of the thesis): the 2025 means of ERA5 and MERRA-2
and their difference, for irradiance, air temperature and wind speed.
  ERA5 means       ERA5 / PECD 4.2 2025 (data/PECD/): GHI, 2 m temperature, 10 m wind
  MERRA-2 means    MERRA-2 2025: SWGDN (rad collection, data/MERRA-2-rad/), SPEED and TLML (flx collection,
                   data/MERRA-2-edit/)
  residual maps    mean MERRA-2 minus mean ERA5, MERRA-2 at the nearest grid point of each ERA5 point (plotting code of
                   residuen.ipynb)
The residual is computed as mean(MERRA-2 on the ERA5 grid) - mean(ERA5). This equals the mean of the hourly residuals
(nearest-neighbour interpolation and the mean are both linear) without holding 8,760 hourly fields twice.
Output: thesis/pics/<map name>.pdf; with --check also PNGs (dpi 100) in tmp/ch4check for a visual check;
--era-solar-only redraws only the ERA5 irradiance map.
"""
import os
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import paths  # repository folders: data, figures, external raw data (see paths.py)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

CODE = paths.CODE
OUT = f"{CODE}/thesis/pics"
CHECK = "--check" in sys.argv
CHECK_DIR = os.path.join(paths.TMP, "ch4check")
PECD = f"{CODE}/data/PECD/"
ERA = {"wind": "H_ERA5_ECMW_T639_WS-_0010m_Pecd_025d_S202501010000_E202512312300_INS_MAP_01h_COM_mbc_org_NA_NA---_NA---_NA---_PECD4.2_fv1.nc",
       "temp": "H_ERA5_ECMW_T639_TA-_0002m_Pecd_025d_S202501010000_E202512312300_INS_MAP_01h_COM_noc_org_NA_NA---_NA---_NA---_PECD4.2_fv1.nc",
       "solar": "H_ERA5_ECMW_T639_GHI_0000m_Pecd_025d_S202501010000_E202512312300_INS_MAP_01h_COM_noc_org_NA_NA---_NA---_NA---_PECD4.2_fv1.nc"}
ERA_SOLAR_MAP = PECD + ERA["solar"]    # ERA5 irradiance of 2025
FLX = f"{CODE}/data/MERRA-2-edit/MERRA2_400.tavg1_2d_flx_Nx.2025*.nc"
RAD = f"{CODE}/data/MERRA-2-rad/MERRA2_400.tavg1_2d_rad_Nx.2025*.nc"


# save the current map as PDF (and a PNG copy for the visual check, if requested)
def save(name):
    plt.savefig(os.path.join(OUT, f"{name}.pdf"), dpi=200, bbox_inches="tight")
    if CHECK:
        os.makedirs(CHECK_DIR, exist_ok=True)
        plt.savefig(os.path.join(CHECK_DIR, f"{name}.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print("saved", name)


def era_mean_map(data_plt, vmin, vmax, cbar_label, title, name, num_levels=11):
    """map of an ERA5 annual mean"""
    projection = ccrs.PlateCarree()
    transform = ccrs.PlateCarree()
    levels = np.linspace(vmin, vmax, num_levels)
    rdbu_cmap = plt.get_cmap("RdBu_r")
    selected_colors = rdbu_cmap(np.linspace(0, 1, num_levels + 1))
    custom_cmap = LinearSegmentedColormap.from_list("RdBu_r", selected_colors)
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=projection)
    imax = data_plt.plot.imshow(ax=ax, levels=levels, transform=transform, cmap=custom_cmap, vmin=vmin, vmax=vmax,
                                add_colorbar=False, zorder=0)
    ticks = np.linspace(vmin, vmax, num_levels)
    last = [vmax]
    ticks = np.concatenate((ticks, last), axis=0)
    rc2 = {"axes.edgecolor": "black", "xtick.color": "white", "xtick.labelcolor": "black"}
    with plt.rc_context(rc2):
        cbar = plt.colorbar(imax, ax=ax, orientation="horizontal", aspect=30, ticks=ticks, pad=0.08, shrink=1,
                            spacing="uniform", drawedges=True)
        cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(cbar_label, fontsize=14)
        cbar.dividers.set_linewidth(4)
    ax.coastlines(zorder=20)
    ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor="black", zorder=21)
    gr = ax.gridlines(draw_labels=True, zorder=10, alpha=0.8, color="gray", linestyle="--")
    gr.xlabel_style = {"size": 14}
    gr.ylabel_style = {"size": 14}
    gr.top_labels = False
    gr.right_labels = False
    ax.set_title(title, fontsize=18, pad=10)
    save(name)


def merra_mean_map(data_plt, vmin, vmax, cbar_label, title, name, num_levels=11):
    """map of a MERRA-2 annual mean, in the same layout as the ERA5 maps"""
    projection = ccrs.PlateCarree()
    transform = ccrs.PlateCarree()
    levels = np.linspace(vmin, vmax, num_levels)
    rdbu_cmap = plt.get_cmap("RdBu_r")
    selected_colors = rdbu_cmap(np.linspace(0, 1, num_levels + 1))
    custom_cmap = LinearSegmentedColormap.from_list("RdBu_r", selected_colors)
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=projection)
    imax = data_plt.plot.imshow(ax=ax, levels=levels, transform=transform, cmap=custom_cmap, vmin=vmin, vmax=vmax,
                                add_colorbar=False, zorder=0)
    ticks = np.linspace(vmin, vmax, num_levels)
    cbar = plt.colorbar(imax, ax=ax, orientation="horizontal", aspect=30, ticks=ticks, pad=0.08, shrink=1,
                        spacing="uniform", drawedges=True)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_label, fontsize=14)
    cbar.dividers.set_linewidth(4)
    ax.coastlines(zorder=20)
    ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor="black", zorder=21)
    gr = ax.gridlines(draw_labels=True, zorder=10, alpha=0.8, color="gray", linestyle="--")
    gr.xlabel_style = {"size": 14}
    gr.ylabel_style = {"size": 14}
    gr.top_labels = False
    gr.right_labels = False
    ax.set_title(title, fontsize=18, pad=10)
    save(name)


def residual_map(residual_mean, label, title, name):
    """map of the mean residual MERRA-2 minus ERA5 (plotting code of residuen.ipynb, cell 12)"""
    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    residual_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="RdBu_r", robust=True, cbar_kwargs={"label": label},
                       rasterized=True)                # PDF only: no seams between the mesh cells
    ax.coastlines()
    # no title (the caption names the figure)
    save(name)


# annual mean of one ERA5 field, read in monthly chunks
def era_annual_mean(path, var):
    ds = xr.open_dataset(path, chunks={"time": 744})
    return ds[var].mean(dim="time").compute()


def residual_mean(era_path, era_var, merra_pattern, merra_var):
    """MERRA-2 shifted by -30 min, nearest neighbour onto the ERA5 grid, inner join in
    time, mean over time of MERRA-2 - ERA5 (computed as the difference of the time means)"""
    ds1 = xr.open_dataset(era_path, chunks={"time": 744}).rename({"latitude": "lat", "longitude": "lon"})
    ds2 = xr.open_mfdataset(merra_pattern, combine="by_coords", chunks={"time": 24})
    ds2 = ds2.assign_coords(time=ds2.time - np.timedelta64(30, "m"))
    common = np.intersect1d(ds1.time.values, ds2.time.values)              # inner join in time
    ds1, ds2 = ds1.sel(time=common), ds2.sel(time=common)                  # (the grids are matched by the interpolation below)
    assert len(common) == 8760, len(common)
    m2 = ds2[merra_var].mean(dim="time").compute()
    m1 = ds1[era_var].mean(dim="time").compute()
    return m2.interp(lat=m1.lat, lon=m1.lon, method="nearest") - m1


# the six maps of the means: ERA5 and MERRA-2 for irradiance, temperature and wind (Figures 4.1, 4.3, 4.5)
def means():
    # ERA5 means
    era_mean_map(era_annual_mean(ERA_SOLAR_MAP, "ssrd"), 0, 400, "W m$^{-2}$", "ERA5 Mean Solar Radiation 2025",
                 "Era5-mean-solar-radiation")
    era_mean_map(era_annual_mean(PECD + ERA["temp"], "t2m") - 273.15, -20, 45, "°C", "ERA5 Mean Temperature 2025",
                 "Era-5-mean-temperature")
    era_mean_map(era_annual_mean(PECD + ERA["wind"], "ws10"), 0, 20, "m s$^{-1}$", "ERA5  Mean Wind Speed 2025",
                 "Era-5-mean-wind-speed")
    # MERRA-2 means
    rad = xr.open_mfdataset(RAD, combine="by_coords", chunks={"time": 24})
    merra_mean_map(rad["SWGDN"].mean(dim="time").compute(), 0, 400, "W m$^{-2}$", "MERRA-2 Mean Solar Radiation 2025",
                   "Merra-2-mean-solar-radiation")
    flx = xr.open_mfdataset(FLX, combine="by_coords", chunks={"time": 24})
    merra_mean_map(flx["SPEED"].mean(dim="time").compute(), 0, 20, "m s$^{-1}$", "MERRA-2 Mean Wind Speed 2025",
                   "Merra-2-mean-wind-speed")
    merra_mean_map((flx["TLML"] - 273.15).mean(dim="time").compute(), -20, 45, "°C", "MERRA-2 Mean Temperature 2025",
                   "Merra-2-mean-temperature")

# the three residual maps MERRA-2 minus ERA5 (Figures 4.2, 4.4, 4.6)
def residuals():
    # residual maps (residuen.ipynb)
    residual_map(residual_mean(PECD + ERA["solar"], "ssrd", RAD, "SWGDN"), "Residual (W m$^{-2}$)",
                 "Mean Residual (MERRA-2 - ERA5) Solar radiation in 2025", "Residual_solarradiation")
    residual_map(residual_mean(PECD + ERA["wind"], "ws10", FLX, "SPEED"), "Residual (m/s)",
                 "Mean Residual (MERRA-2 - ERA5) Windspeed in 2025", "Residual_windspeed")
    residual_map(residual_mean(PECD + ERA["temp"], "t2m", FLX, "TLML"), "Residual (°C)",
                 "Mean Residual (MERRA-2 - ERA5) Temperatur in 2025", "Residual_temperatur")


# redraw only the ERA5 irradiance map (option --era-solar-only)
def era_solar():
    era_mean_map(era_annual_mean(ERA_SOLAR_MAP, "ssrd"), 0, 400, "W m$^{-2}$", "ERA5 Mean Solar Radiation 2025",
                 "Era5-mean-solar-radiation")


if __name__ == "__main__":
    if "--era-solar-only" in sys.argv:
        era_solar()
        sys.exit()
    if "--residuals-only" not in sys.argv:
        means()
    residuals()
