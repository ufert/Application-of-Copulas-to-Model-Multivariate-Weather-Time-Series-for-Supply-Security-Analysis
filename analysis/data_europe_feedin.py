# Thesis: Data chapter, Table 4.5 (mean capacity factors of wind and PV 2025 from ERA5 and MERRA-2)
# Run: cd analysis && python data_europe_feedin.py   (prints the table, draws a check map)
# Needs: the same ERA5 and MERRA-2 files as data_chapter_maps_pdf.py
"""
Renewable feed-in from ERA5 and from MERRA-2 over the European domain, 2025 hourly, without any height scaling of the
wind (data chapter, Table 4.5).

  wind  the thesis power curve (one class: 300 W/m2, cut-in 3, rated 10.3, cut-out 25 m/s, lognormal multi-turbine
        spread sigma 0.36) applied directly to the delivered wind speeds: ERA5 / PECD 4.2 10 m wind (ws10, scaled to the
        Global Wind Atlas) and MERRA-2 SPEED (lowest model layer, ~60 m). The difference therefore mixes the dataset
        difference with the height difference.
  PV    CF = GHI/1000 * (1 - 0.004 (T_air + 0.03125 GHI - 25)), clipped at 0, no performance ratio: ERA5 GHI and 2 m
        temperature, MERRA-2 SWGDN and TLML (~60 m).
Per grid point: mean capacity factor of each dataset; MERRA-2 at the nearest grid point of each ERA5 point (as in
residuen.ipynb). Area means with cos(latitude) weights over all points, land and sea (Natural Earth 50 m land).
Outputs: the table printed, data/data_europe_feedin.npz and two check maps thesis/pics/europe_feedin_{means,difference}
(not in the thesis).
"""
import glob
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from shapely import contains_xy
from shapely.ops import unary_union
import paths  # repository folders: data, figures, external raw data (see paths.py)

D = paths.DATA
OUT = paths.PICS
CACHE = f"{D}/data_europe_feedin.npz"
INK, MUTED, GRID = "#1F2328", "#5F6368", "#E3E5E8"
BLUE, BOR = "#00549F", "#A11035"
SEQ = LinearSegmentedColormap.from_list("seq", ["#F4F7FB", "#BDD2EA", "#5B8FC7", BLUE, "#002B55"])
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#BDD2EA", "#EEEFF1", "#E7C3CB", BOR])

# conversion
SP, VIN, VOUT, SIG = 300.0, 3.0, 25.0, 0.36
VR = (2 * SP / (1.225 * 0.45)) ** (1 / 3)
_q = np.linspace(-3.5, 3.5, 41)
_wq = np.exp(-_q ** 2 / 2); _wq /= _wq.sum()
_mult = np.exp(-SIG ** 2 / 2 + SIG * _q)
_vg = np.linspace(0, 40, 4001)


# power curve of one turbine (cut-in VIN, rated VR, cut-out)
def _curve(v):
    return np.where((v >= VIN) & (v < VR), (v ** 3 - VIN ** 3) / (VR ** 3 - VIN ** 3), np.where((v >= VR) & (v < VOUT), 1.0, 0.0))


_cf_grid = sum(w * _curve(_vg * m) for w, m in zip(_wq, _mult))       # multi-turbine curve tabulated, 0.01 m/s steps


# wind capacity factor from the fleet curve, interpolated on the wind-speed grid
def wind_cf(v):
    return np.interp(v, _vg, _cf_grid)


# PV capacity factor from GHI and air temperature with the NOCT cell-temperature model
def pv_cf(ghi, t_c):
    return np.clip(ghi / 1000.0 * (1 - 0.004 * (t_c + 0.03125 * ghi - 25)), 0, None)


# data, month by month
def era_file(code, tag):
    return glob.glob(f"{D}/PECD/*{code}*S20250101*{tag}*.nc")[0]


if os.path.exists(CACHE):
    z = np.load(CACHE)
    ela, elo, mla, mlo = z["ela"], z["elo"], z["mla"], z["mlo"]
    S = {k: z[k] for k in ("E_wind", "M_wind", "E_pv", "M_pv")}
else:
    ncs = {k: netCDF4.Dataset(era_file(*a)) for k, a in
           {"w": ("WS-_0010m", "mbc"), "t": ("TA-_0002m", "noc"), "g": ("GHI_0000m", "noc")}.items()}
    var = {k: [v for v in nc.variables if nc.variables[v].ndim == 3][0] for k, nc in ncs.items()}
    ela, elo = np.asarray(ncs["w"]["latitude"][:]), np.asarray(ncs["w"]["longitude"][:])
    ff = sorted(glob.glob(f"{D}/MERRA-2-2015-2025/MERRA2_40[01].tavg1_2d_flx_Nx.2025*.SUB.nc"))
    fr = sorted(glob.glob(f"{D}/MERRA-2-rad-2015-2025/MERRA2_40[01].tavg1_2d_rad_Nx.2025*.SUB.nc"))
    assert len(ff) == len(fr) == 365
    with netCDF4.Dataset(ff[0]) as nc:
        mla, mlo = np.asarray(nc["lat"][:]), np.asarray(nc["lon"][:])
    S = {"E_wind": 0.0, "E_pv": 0.0, "M_wind": 0.0, "M_pv": 0.0}
    n_e, n_m = 0, 0
    days = pd.date_range("2025-01-01", periods=365, freq="D")
    for mo in range(1, 13):
        idx = np.flatnonzero(days.month == mo)
        h0, h1 = 24 * idx[0], 24 * (idx[-1] + 1)
        w = np.asarray(ncs["w"][var["w"]][h0:h1], dtype=np.float64)
        t = np.asarray(ncs["t"][var["t"]][h0:h1], dtype=np.float64) - 273.15
        g = np.clip(np.asarray(ncs["g"][var["g"]][h0:h1], dtype=np.float64), 0, None)
        S["E_wind"] = S["E_wind"] + wind_cf(w).sum(0)
        S["E_pv"] = S["E_pv"] + pv_cf(g, t).sum(0)
        n_e += h1 - h0
        for i in idx:
            with netCDF4.Dataset(ff[i]) as a, netCDF4.Dataset(fr[i]) as b:
                sp = np.asarray(a["SPEED"][:], dtype=np.float64)
                tl = np.asarray(a["TLML"][:], dtype=np.float64) - 273.15
                sw = np.asarray(b["SWGDN"][:], dtype=np.float64)
            S["M_wind"] = S["M_wind"] + wind_cf(sp).sum(0)
            S["M_pv"] = S["M_pv"] + pv_cf(sw, tl).sum(0)
            n_m += sp.shape[0]
        print("month", mo, flush=True)
    for nc in ncs.values():
        nc.close()
    S = {k: v / (n_e if k.startswith("E") else n_m) for k, v in S.items()}
    np.savez(CACHE, ela=ela, elo=elo, mla=mla, mlo=mlo, **S)

ri = np.array([int(np.argmin(np.abs(mla - x))) for x in ela])
ci = np.array([int(np.argmin(np.abs(mlo - x))) for x in elo])
M_on_E = {k: S["M_" + k][np.ix_(ri, ci)] for k in ("wind", "pv")}
DIFF = {k: M_on_E[k] - S["E_" + k] for k in ("wind", "pv")}

# summary
land = unary_union(list(shpreader.Reader(shpreader.natural_earth("50m", "physical", "land")).geometries()))
LO, LA = np.meshgrid(elo, ela)
is_land = contains_xy(land, LO, LA)
W = np.cos(np.deg2rad(LA))
rows = []
for k, name in (("wind", "Wind"), ("pv", "Photovoltaics")):
    for region, mask in (("all", np.ones_like(is_land)), ("land", is_land), ("sea", ~is_land)):
        w = W * mask
        e = (S["E_" + k] * w).sum() / w.sum()
        m = (M_on_E[k] * w).sum() / w.sum()
        rows.append({"technology": name, "region": region, "CF ERA5": e, "CF MERRA-2": m, "difference": m - e,
                     "MERRA-2 / ERA5 - 1 (%)": 100 * (m / e - 1),
                     "full-load hours ERA5": e * 8760, "full-load hours MERRA-2": m * 8760})
T = pd.DataFrame(rows).set_index(["technology", "region"])
pd.set_option("display.width", 200)
print("Mean capacity factor 2025, cos(latitude)-weighted; wind without height scaling (ERA5 10 m, MERRA-2 ~60 m)")
print(T.round(3).to_string())
T.to_csv(f"{D}/data_europe_feedin_summary.csv")

# maps
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": INK, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})
proj = ccrs.PlateCarree()
borders = cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "50m")
coast = cfeature.NaturalEarthFeature("physical", "coastline", "50m")


# cell edges from cell centres, for pcolormesh
def edges(c):
    d = np.diff(c)
    return np.r_[c[0] - d[0] / 2, c[:-1] + d / 2, c[-1] + d[-1] / 2]


# map background: coastlines, borders and the extent of the domain
def base(ax, left=True):
    ax.add_feature(coast, lw=0.35, edgecolor=INK, facecolor="none")
    ax.add_feature(borders, lw=0.3, edgecolor=INK, facecolor="none")
    ax.set_extent([elo.min(), elo.max(), ela.min(), ela.max()], crs=proj)
    gl = ax.gridlines(draw_labels=True, lw=0.3, color=GRID, x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    gl.left_labels = left
    gl.xlabel_style = gl.ylabel_style = {"size": 6, "color": MUTED}


# small text box in the lower left corner of a map
def stamp(ax, text):
    ax.text(0.02, 0.03, text, transform=ax.transAxes, fontsize=6.3, color=INK, zorder=30,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.4, alpha=0.92))


# area mean with cos(latitude) weights
def wmean(f, lat):
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(f)
    return float((f * w).sum() / w.sum())


LAB = {"wind": ("ERA5, wind at 10 m", "MERRA-2, wind at about 60 m"), "pv": ("ERA5", "MERRA-2")}
TITLE = {"wind": "Wind capacity factor", "pv": "Photovoltaic capacity factor"}
fig, axs = plt.subplots(2, 2, figsize=(6.3, 5.4), subplot_kw={"projection": proj},
                        gridspec_kw={"wspace": 0.08, "hspace": 0.32})
for r, k in enumerate(("wind", "pv")):
    hi = np.percentile(np.r_[S["E_" + k].ravel(), S["M_" + k].ravel()], 99.5)
    for c, (f, la, lo, lab) in enumerate([(S["E_" + k], ela, elo, LAB[k][0]), (S["M_" + k], mla, mlo, LAB[k][1])]):
        ax = axs[r, c]
        pm = ax.pcolormesh(edges(lo), edges(la), f, cmap=SEQ, vmin=0, vmax=hi, transform=proj, rasterized=True)
        base(ax, left=(c == 0))
        ax.text(0.01, 1.02, lab, transform=ax.transAxes, fontsize=7, color=INK, va="bottom")
        stamp(ax, f"Mean: {wmean(f, la):.3f}")
    cb = fig.colorbar(pm, ax=axs[r], orientation="vertical", fraction=0.025, pad=0.02, extend="max")
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label(f"{TITLE[k]}, mean 2025", fontsize=6.5, color=MUTED)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/europe_feedin_means.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.9), subplot_kw={"projection": proj}, gridspec_kw={"wspace": 0.12})
for c, k in enumerate(("wind", "pv")):
    ax, d = axs[c], DIFF[k]
    lim = np.percentile(np.abs(d), 99)
    pm = ax.pcolormesh(edges(elo), edges(ela), d, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), transform=proj,
                       rasterized=True)
    base(ax, left=(c == 0))
    ax.text(0.01, 1.02, TITLE[k] + (" (no height scaling)" if k == "wind" else ""), transform=ax.transAxes,
            fontsize=7, color=INK, va="bottom")
    dm, em = wmean(d, ela), wmean(S["E_" + k], ela)
    stamp(ax, f"Mean difference: {dm:+.3f}\n= {100 * dm / em:+.0f} % of the ERA5 mean ({em:.3f})")
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08, aspect=30, extend="both")
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label("MERRA-2 minus ERA5 (capacity factor)", fontsize=6.5, color=MUTED)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/europe_feedin_difference.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
