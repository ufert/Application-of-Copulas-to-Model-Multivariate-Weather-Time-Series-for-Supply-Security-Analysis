# Thesis: Case Study, Conversion to Feed-in: power curves, turbine classes, fleet spread, wind scale and PV
# Module: calibrated against OPSD feed-in 2015-2019 (cached in data/conversion_16cells.pkl)
# Needs: OPSD time series in data/, data/grid4x4_ws100_1950-2025.pkl
"""Conversion of weather into wind and PV feed-in, calibrated to observed German feed-in (OPSD 2015-2019).
calibrate_16cells() returns the calibrated conversions and caches them in data/conversion_16cells.pkl.

wind   turbine classes by site wind (terciles of the mean 10 m wind of the 734 German land points at 0.25 deg):
       strong-wind (specific power 400 W/m2, hub 100 m), medium (300 W/m2, 120 m), low-wind (220 W/m2, 140 m);
       offshore 350 W/m2, 120 m. Power curve: cut-in 3 m/s, rated speed from the specific power (Cp 0.45,
       1.225 kg/m3), cut-out 25 m/s, cubic ramp. Fleet curve: the turbine curve averaged over a lognormal spread sigma
       of the wind around the input value (sites, turbines and turbulence within a region).
       100 m wind: hub wind = kappa(month, local hour) * v100 * (hub / 100) ** (1/7), with kappa per slot such that the
       mean CF of the fleet per slot follows the observed daily cycle (mean CF 0.207); offshore one kappa for all hours.
       10 m wind (variants *_10m): hub wind from 10 m with a shear exponent per (month, local hour).
       sigma: the value whose per-year normalised quantiles of the CF are closest to the observed ones.
       map: empirical quantile map from 10 m to 100 m wind per (cell, local month, local hour), fitted on 2000-2024.
pv     CF = GHI/1000 (1 - 0.004 (T_cell - 25)), T_cell = T_air + 0.03125 GHI; shifted by the lag of best correlation
       with the observed feed-in (ERA5 radiation is hour-ending), scaled to the observed mean (performance ratio) and
       corrected per (month, local hour) for the orientation of the modules.
Usage: C = calibrate_16cells(); C["onshore"].cf(v100, lmon, lhr); from 10 m wind C["onshore"].cf(C["map"](v10, lmon,
       lhr), lmon, lhr); C["pv"].cf(ghi, tk, lmon, lhr). Wind arrays are (..., time, regions); lmon and lhr are the
       local month and hour (local_time()).
"""
import pickle

import numpy as np
import pandas as pd
import paths  # repository folders: data, figures, external raw data (see paths.py)

D = paths.DATA
CELLS = [f"r{r}c{c}" for r in range(4) for c in range(4)]
RHO, CP = 1.225, 0.45
CLASSES = {"strong-wind": (400, 100), "medium": (300, 120), "low-wind": (220, 140)}             # W/m2, hub m
OFFSHORE = {"offshore": (350, 120)}
XG = np.arange(0, 90, 0.01)
GH_X, GH_W = np.polynomial.hermite.hermgauss(21)
QS = np.linspace(0.01, 0.99, 99)


# rated wind speed of a turbine class from its specific power (W/m2): P/A = 0.5 rho Cp v_r^3
def rated_speed(sp):
    return (2 * sp / (RHO * CP)) ** (1 / 3)


# power curve of one turbine: zero below 3 m/s, cubic up to the rated speed vr, full output up to cut-out at 25 m/s
def curve(v, vr):
    out = np.zeros(np.shape(v), dtype=np.float32)
    v = np.asarray(v)
    r = (v >= 3) & (v < vr)
    out[r] = (v[r] ** 3 - 27) / (vr ** 3 - 27)
    out[(v >= vr) & (v < 25)] = 1.0
    return out


# fleet power curve on a wind-speed grid: the turbine curve averaged over a lognormal spread sigma of the wind
# speeds seen by the turbines of a region (Gauss-Hermite quadrature)
def fleet_table(vr, sigma):
    if sigma == 0:
        return curve(XG, vr)
    m = np.exp(np.sqrt(2) * sigma * GH_X - sigma ** 2 / 2)
    return (GH_W[:, None] * curve(XG[None] * m[:, None], vr)).sum(axis=0) / np.sqrt(np.pi)


# local month and hour (Europe/Berlin) of UTC time stamps; slots for the wind scale and the PV timing
def local_time(utc_index):
    loc = pd.DatetimeIndex(utc_index).tz_localize("UTC").tz_convert("Europe/Berlin")
    return loc.month.to_numpy(), loc.hour.to_numpy()


# turbine classes
def point_classes():
    """class per German land point at 0.25 deg (terciles of the mean 10 m wind), cached"""
    path = f"{D}/turbine_classes_025.pkl"
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        pass
    meta = pickle.load(open(f"{D}/grid025_meta.pkl", "rb"))
    P = pickle.load(open(f"{D}/risk_map_025.pkl", "rb"))["points"]
    G = P[P.germany].reset_index(drop=True)
    ri = np.searchsorted(-meta["lat"], -G["lat"].to_numpy())
    ci = np.searchsorted(meta["lon"], G["lon"].to_numpy())
    vbar = np.asarray(np.load(f"{D}/grid025_wind_2000-2024.npy", mmap_mode="r")[:, ri, ci], dtype=np.float32).mean(axis=0)
    cls = np.where(vbar >= np.quantile(vbar, 2 / 3), "strong-wind", np.where(vbar >= np.quantile(vbar, 1 / 3), "medium", "low-wind"))
    out = {"points": G[["lat", "lon", "cell"]], "ri": ri, "ci": ci, "mean wind": vbar, "class": cls}
    pickle.dump(out, open(path, "wb"))
    return out


def cell_shares():
    """(16, 3) share of each class among the German land points of each cell; German land weight per cell"""
    pc = point_classes()
    cells = pc["points"]["cell"].to_numpy()
    shares = np.array([[(pc["class"][cells == c] == k).mean() if (cells == c).any() else 0 for k in CLASSES] for c in CELLS])
    w = np.array([(cells == c).sum() for c in CELLS], dtype=float)
    return shares, w / w.sum()


# wind
class WindConversion:
    # turbine classes with their shares, region weights, fleet spread sigma and shear exponent per month and hour
    def __init__(self, classes, shares, weights, sigma, alpha_mh):
        self.classes, self.shares, self.weights = classes, np.asarray(shares, float), np.asarray(weights, float)
        self.sigma, self.alpha_mh = sigma, alpha_mh
        self.tables = {k: fleet_table(rated_speed(sp), sigma) for k, (sp, h) in classes.items()}

    def regional(self, v10, lmon, lhr):
        """v10 (..., time, regions) 10 m wind; lmon, lhr (time,) local month and hour -> CF (..., time, regions)"""
        a = self.alpha_mh[lmon - 1, lhr][:, None]
        out = np.zeros(v10.shape, dtype=np.float32)
        for j, (k, (sp, h)) in enumerate(self.classes.items()):
            s = self.shares[:, j]
            if not s.any():
                continue
            vh = v10 * ((h / 10) ** a)
            out += s * np.interp(vh, XG, self.tables[k]).astype(np.float32)
        return out

    # national capacity factor: regional capacity factors weighted by the installed capacity of each region
    def cf(self, v10, lmon, lhr):
        return (self.regional(v10, lmon, lhr) * self.weights).sum(axis=-1)


# divide a series by its mean in each year (removes the growth of the observed fleet from OPSD feed-in)
def per_year_norm(x, years):
    out = np.empty(len(x))
    for y in np.unique(years):
        m = years == y
        out[m] = x[m] / np.nanmean(x[m])
    return out


def calibrate_wind(v10, lmon, lhr, years, obs, classes, shares, weights, target, sigmas=np.arange(0, 0.61, 0.04),
                   diurnal=True):
    """v10 (time, regions) over the calibration period; obs observed CF (time,) with NaN gaps.
    diurnal: shear per (month, local hour), slot targets from the per-year normalised observation (the OPSD
    capacity series lags new plants, so years differ in level); otherwise one shear for all hours"""
    ok = ~np.isnan(obs)
    on = per_year_norm(obs, years)
    oq = np.quantile(on[ok], QS)
    slots = [(m, h, np.where((lmon == m) & (lhr == h))[0]) for m in range(1, 13) for h in range(24)]
    best = None
    for sig in sigmas:
        wc = WindConversion(classes, shares, weights, sig, np.zeros((12, 24)))

        # bisection for the shear exponent of the slots in rows, so that the mean CF matches the goal (10 m conversion)
        def fit(rows, goal):
            lo, hi = 0.0, 0.6
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                wc.alpha_mh[lmon[rows] - 1, lhr[rows]] = mid
                lo, hi = (mid, hi) if wc.cf(v10[rows], lmon[rows], lhr[rows]).mean() < goal else (lo, mid)
            wc.alpha_mh[lmon[rows] - 1, lhr[rows]] = 0.5 * (lo + hi)

        if diurnal:
            for m, h, rows in slots:
                fit(rows, np.nanmean(on[rows]) * target)
        else:
            fit(np.arange(len(obs)), target)
        mod = wc.cf(v10, lmon, lhr)
        err = np.mean((np.quantile(per_year_norm(mod, years)[ok], QS) - oq) ** 2)
        if best is None or err < best[0]:
            best = (err, wc, mod)
    err, wc, mod = best
    return wc, diagnostics(mod, obs, years, lhr) | {"sigma": wc.sigma, "quantile error": err}


# modelled against observed national capacity factors: correlation, low-output hours and daily cycle
def diagnostics(mod, obs, years, lhr):
    ok = ~np.isnan(obs)
    o = obs.copy()
    for y in np.unique(years):
        m = years == y
        o[m] *= np.nanmean(mod[m]) / np.nanmean(o[m])
    return {"corr": np.corrcoef(mod[ok], obs[ok])[0, 1], "mean": mod.mean(),
            "h/yr <5% model": (mod[ok] < 0.05).sum() / len(np.unique(years)), "h/yr <5% observed": (o[ok] < 0.05).sum() / len(np.unique(years)),
            "q99 model": np.quantile(mod[ok], 0.99), "q99 observed": np.nanquantile(o[ok], 0.99),
            "low share by hour": pd.DataFrame({"model": [(mod[ok & (lhr == h)] < 0.05).mean() for h in range(24)],
                                               "observed": [(o[ok & (lhr == h)] < 0.05).mean() for h in range(24)]}),
            "mean by hour": pd.DataFrame({"model": [mod[lhr == h].mean() for h in range(24)],
                                          "observed": [np.nanmean(o[lhr == h]) for h in range(24)]})}


# wind from 100 m
class Wind100Conversion(WindConversion):
    """hub wind = kappa(month, local hour) * v100 * (hub / 100) ** (1/7); input 100 m wind (..., time, regions)"""

    # 100 m conversion: the same fleet curves, with the 100 m wind scaled by kappa per local month and hour
    def __init__(self, classes, shares, weights, sigma, kappa_mh):
        # WindConversion scales by (hub / 10) ** alpha: hubs / 10 and alpha = 1/7 give (hub / 100) ** (1/7)
        super().__init__({k: (sp, h / 10) for k, (sp, h) in classes.items()}, shares, weights, sigma, np.full((12, 24), 1 / 7))
        self.hub_classes, self.kappa_mh = classes, kappa_mh

    # regional capacity factors from 100 m wind, after scaling it by kappa of the local month and hour
    def regional(self, v100, lmon, lhr):
        return super().regional(v100 * self.kappa_mh[lmon - 1, lhr][:, None], lmon, lhr)


def calibrate_wind100(v100, lmon, lhr, years, obs, classes, shares, weights, target, sigmas=np.arange(0, 0.61, 0.04),
                      diurnal=True):
    """as calibrate_wind, with the wind scale kappa (per slot, or one for all hours) in place of the shear"""
    ok = ~np.isnan(obs)
    on = per_year_norm(obs, years)
    oq = np.quantile(on[ok], QS)
    slots = [(m, h, np.where((lmon == m) & (lhr == h))[0]) for m in range(1, 13) for h in range(24)]
    best = None
    for sig in sigmas:
        wc = Wind100Conversion(classes, shares, weights, sig, np.ones((12, 24)))

        # bisection for the wind scale kappa of the slots in rows, so that the mean CF matches the observed one
        def fit(rows, goal):
            lo, hi = 0.3, 2.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                wc.kappa_mh[lmon[rows] - 1, lhr[rows]] = mid
                lo, hi = (mid, hi) if wc.cf(v100[rows], lmon[rows], lhr[rows]).mean() < goal else (lo, mid)
            wc.kappa_mh[lmon[rows] - 1, lhr[rows]] = 0.5 * (lo + hi)

        if diurnal:
            for m, h, rows in slots:
                fit(rows, np.nanmean(on[rows]) * target)
        else:
            fit(np.arange(len(obs)), target)
        mod = wc.cf(v100, lmon, lhr)
        err = np.mean((np.quantile(per_year_norm(mod, years)[ok], QS) - oq) ** 2)
        if best is None or err < best[0]:
            best = (err, wc, mod)
    err, wc, mod = best
    return wc, diagnostics(mod, obs, years, lhr) | {"sigma": wc.sigma, "quantile error": err}


class WindMap:
    """10 m -> 100 m wind per (region, local month, local hour): empirical quantile map; values outside the fitted
    range map to the end points"""

    # quantiles of 10 m and 100 m wind per region, local month and hour
    def __init__(self, q10, q100):
        self.q10, self.q100 = q10, q100                                             # (regions, 12, 24, quantiles)

    # map 10 m wind onto 100 m wind by matching quantiles within each local month and hour
    def __call__(self, v10, lmon, lhr):
        v10 = np.asarray(v10, dtype=np.float32)
        out = np.empty(v10.shape, dtype=np.float32)
        for m in range(1, 13):
            for h in range(24):
                t = np.where((lmon == m) & (lhr == h))[0]
                if not len(t):
                    continue
                for c in range(v10.shape[-1]):
                    out[..., t, c] = np.interp(v10[..., t, c], self.q10[c, m - 1, h], self.q100[c, m - 1, h])
        return out


# fit the quantile map from 10 m to 100 m wind per region, local month and hour (n quantiles)
def fit_wind_map(v10, v100, lmon, lhr, n=101):
    P = np.linspace(0, 1, n)
    q10 = np.empty((v10.shape[1], 12, 24, n), dtype=np.float32)
    q100 = np.empty_like(q10)
    for m in range(1, 13):
        for h in range(24):
            r = (lmon == m) & (lhr == h)
            q10[:, m - 1, h] = np.quantile(v10[r], P, axis=0).T
            q100[:, m - 1, h] = np.quantile(v100[r], P, axis=0).T
    return WindMap(q10, q100)


# PV
def pv_cells(ghi, tk):
    return np.clip(ghi / 1000 * (1 - 0.004 * (tk - 273.15 + 0.03125 * ghi - 25)), 0, None)


# photovoltaic conversion: GHI and air temperature to capacity factor, with time lag, performance ratio and a
# ratio per local month and hour for the orientation of the modules
class PVConversion:
    # region weights, time lag in hours, performance ratio and the ratio per local month and hour
    def __init__(self, weights, lag, pr, ratio):
        self.weights, self.lag, self.pr, self.ratio = np.asarray(weights, float), lag, pr, ratio

    def cf(self, ghi, tk, lmon, lhr):
        """ghi, tk (..., time, cells) on an hourly grid -> national CF (..., time); the lag shifts in time
        (a roll: the first / last hour wraps around)"""
        base = (pv_cells(ghi, tk) * self.weights).sum(axis=-1)
        return np.roll(self.pr * base, self.lag, axis=-1) * self.ratio[lmon - 1, lhr]


# fit the PV conversion to observed feed-in: time lag by correlation, then performance ratio and slot ratios
def calibrate_pv(ghi, tk, lmon, lhr, obs, weights, lag=None):
    base = pd.Series((pv_cells(ghi, tk) * weights).sum(axis=1))
    o = pd.Series(obs)
    lags = {k: base.shift(k).corr(o) for k in (-2, -1, 0, 1, 2)}
    lag = max(lags, key=lags.get) if lag is None else lag
    pr = np.nanmean(obs) / base.shift(lag).mean()
    sh = (base.shift(lag) * pr).to_numpy()
    ratio = np.ones((12, 24))
    for m in range(1, 13):
        for h in range(24):
            rows = (lmon == m) & (lhr == h)
            mm = np.nanmean(sh[rows])
            if mm > 0.002:
                ratio[m - 1, h] = np.clip(np.nanmean(obs[rows]) / mm, 0.2, 1.5)
    pvc = PVConversion(weights, lag, pr, ratio)
    mod = np.roll(pr * base.to_numpy(), lag) * ratio[lmon - 1, lhr]
    ok = ~np.isnan(obs)
    summer = np.isin(lmon, [6, 7])
    ev = lambda x: np.nansum(x[summer & (lhr >= 17)]) / np.nansum(x[summer])
    return pvc, {"lag": lag, "lag correlations": lags, "PR": pr, "corr": np.corrcoef(mod[ok], obs[ok])[0, 1],
                 "summer share after 17 h model": ev(mod), "summer share after 17 h observed": ev(obs)}


# calibration data
def opsd(index, key):
    op = pd.read_pickle(f"{D}/opsd_germany_60min.pkl")
    return (op[f"DE_{key}_generation_actual"] / op[f"DE_{key}_capacity"]).reindex(index).to_numpy()


VERSION = "2026-10-13 100m v1"                                    # bump when the calibration changes (invalidates the cache)


def wind100_cells(index):
    """16 cell means of the PECD 4.2 100 m wind (extract_ws100.py) on an hourly UTC index"""
    return pd.read_pickle(f"{D}/grid4x4_ws100_1950-2025.pkl")[[f"{c}_ws100" for c in CELLS]].loc[index].to_numpy(dtype=np.float32)


def calibrate_16cells(cal_index=pd.date_range("2015-01-01", "2019-12-31 23:00", freq="h"), pv_lag=None, refresh=False):
    """onshore and offshore from the 100 m wind and from the 10 m wind (variants *_10m), PV, and the quantile
    map 10 m -> 100 m (fitted 2000-2024), calibrated on the 16-cell ERA5 data 2015-2019 against OPSD; cached"""
    default = cal_index.equals(pd.date_range("2015-01-01", "2019-12-31 23:00", freq="h"))
    path = f"{D}/conversion_16cells{'' if pv_lag is None else f'_lag{pv_lag}'}.pkl"
    if default and not refresh:
        try:
            out = pickle.load(open(path, "rb"))
            if out.get("version") == VERSION:
                return out
        except FileNotFoundError:
            pass
    g = {v: pd.read_pickle(f"{D}/grid4x4_{v}_2000-2024.pkl")[[f"{c}_{v}" for c in CELLS]].loc[cal_index].to_numpy()
         for v in ("wind", "temp", "solar")}
    g["solar"] = np.clip(g["solar"], 0, None)
    g["ws100"] = wind100_cells(cal_index)
    lmon, lhr = local_time(cal_index)
    yrs = cal_index.year.to_numpy()
    shares, w = cell_shares()
    off_cell = [CELLS.index("r0c0")]
    obs_on, obs_off = opsd(cal_index, "wind_onshore"), opsd(cal_index, "wind_offshore")
    on, d_on = calibrate_wind100(g["ws100"], lmon, lhr, yrs, obs_on, CLASSES, shares, w, 0.207)
    off, d_off = calibrate_wind100(g["ws100"][:, off_cell], lmon, lhr, yrs, obs_off, OFFSHORE, np.ones((1, 1)), np.ones(1), 0.40,
                                   diurnal=False)                                       # one wind scale for all hours
    on10, d_on10 = calibrate_wind(g["wind"], lmon, lhr, yrs, obs_on, CLASSES, shares, w, 0.207)
    off10, d_off10 = calibrate_wind(g["wind"][:, off_cell], lmon, lhr, yrs, obs_off, OFFSHORE, np.ones((1, 1)), np.ones(1), 0.40,
                                    diurnal=False)
    pv, d_pv = calibrate_pv(g["solar"], g["temp"], lmon, lhr, opsd(cal_index, "solar"), w, lag=pv_lag)
    fit_index = pd.date_range("2000-01-01", "2024-12-31 23:00", freq="h")
    fm, fh = local_time(fit_index)
    v10 = pd.read_pickle(f"{D}/grid4x4_wind_2000-2024.pkl")[[f"{c}_wind" for c in CELLS]].loc[fit_index].to_numpy(dtype=np.float32)
    wmap = fit_wind_map(v10, wind100_cells(fit_index), fm, fh)
    out = {"version": VERSION, "onshore": on, "offshore": off, "pv": pv, "map": wmap, "onshore_10m": on10, "offshore_10m": off10,
           "diag": {"onshore": d_on, "offshore": d_off, "pv": d_pv, "onshore_10m": d_on10, "offshore_10m": d_off10},
           "weights": w, "shares": shares}
    if default:
        pickle.dump(out, open(path, "wb"))
    return out
