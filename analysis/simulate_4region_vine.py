# Thesis: Case Study, Model Development; the 4-region simulation with an R-vine copula on the residuals of
# separate autoregressions, one of the variants compared in spatial_var_4q.py
# Run: cd analysis && python simulate_4region_vine.py   (needs data/grid4x4_* and pyvinecopulib)
# Writes: data/sim_4region_vine_{wind,temp,solar}.pkl, data/sim_4region_vine_diagnostics.pkl
"""4-region simulation with an R-vine copula across the regions, per season and variable: families and structure by
Dissmann's algorithm with BIC (pyvinecopulib), on the residuals of the seasonal ARs per region. Everything else (AR
fit, climatology, empirical pools, reconstruction) as in simulate_4_16_regions.py.
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import paths  # repository folders: data, figures, external raw data (see paths.py)

T0 = time.time()
# time-stamped log line
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

DATA_DIR = paths.DATA
SEASONS = ["winter", "spring", "summer", "autumn"]
AR_MAX_LAG = 168
N_TRAJECTORIES = 300
BASE_SEED = 42
CLIMATOLOGY_SMOOTHING_WINDOW_DAYS = 15
SOLAR_LOG_EPS = 1.0
EPS = 1e-6

# copula families of the two-region spatial copulas of ar_var_copula_extremes.ipynb (Sections 18, 26, 28), with BB1
SPATIAL_CANDIDATE_FAMILIES = [pv.BicopFamily.gaussian, pv.BicopFamily.student, pv.BicopFamily.clayton,
                               pv.BicopFamily.gumbel, pv.BicopFamily.frank, pv.BicopFamily.bb1]


# meteorological season of a month
def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


# id of the contiguous block of a season (one per season and year), -1 outside the season
def season_block_ids(index, season):
    season_labels = pd.Series(index.month.map(get_season), index=index).to_numpy()
    is_season = season_labels == season
    block_id = np.full(len(index), -1, dtype=int)
    current_block, prev_in_season = -1, False
    for i in range(len(index)):
        if is_season[i]:
            if not prev_in_season:
                current_block += 1
            block_id[i] = current_block
        prev_in_season = is_season[i]
    return block_id


# seasonal AR fit of one series by least squares, using only lags within the same season block; order by BIC
def fit_ar_model_seasonal_one(y, block_id, col, maxlag):
    n = len(y)
    y_arr = y.to_numpy(dtype=float)
    lag_cols = []
    for lag in range(1, maxlag + 1):
        shifted_val = np.roll(y_arr, lag)
        shifted_block = np.roll(block_id, lag)
        valid = (block_id == shifted_block) & (block_id != -1) & (np.arange(n) >= lag)
        lag_cols.append(np.where(valid, shifted_val, np.nan))
    X_full = np.column_stack(lag_cols)
    target_valid = (block_id != -1) & ~np.isnan(X_full).any(axis=1)
    y_fit = y_arr[target_valid]
    X_fit = X_full[target_valid]
    design = np.column_stack([np.ones(len(y_fit)), X_fit])
    Q, R = np.linalg.qr(design)
    Qty = Q.T @ y_fit
    rss = np.maximum(np.sum(y_fit**2) - np.cumsum(Qty**2), 1e-12)
    n_obs = len(y_fit)
    bic = n_obs * np.log(rss / n_obs) + np.arange(1, maxlag + 2) * np.log(n_obs)
    best_k = int(np.argmin(bic)) + 1
    best_p = best_k - 1
    beta = np.linalg.solve(R[:best_k, :best_k], Qty[:best_k])
    params = pd.Series(beta, index=["const"] + [f"{col}.L{lag}" for lag in range(1, best_p + 1)])
    fitted = design[:, :best_k] @ beta
    resid = pd.Series(y_fit - fitted, index=y.index[target_valid])
    return {"params": params, "lags": list(range(1, best_p + 1)), "resid": resid}


# climatology per (month, day, hour), smoothed with a +-15-day window
def build_smoothed_climatology_ns(df, cols, window_days=15):
    raw = df.groupby([df.index.month, df.index.day, df.index.hour])[cols].mean()
    raw.index.names = ["month", "day", "hour"]
    days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    calendar_order = [(m, d) for m in range(1, 13) for d in range(1, days_in_month[m - 1] + 1)]
    smoothed_blocks = []
    for hour in range(24):
        vals = raw.xs(hour, level="hour").reindex(calendar_order)[cols].to_numpy(dtype=float)
        padded = np.concatenate([vals[-window_days:], vals, vals[:window_days]], axis=0)
        smoothed_vals = (pd.DataFrame(padded).rolling(2 * window_days + 1, center=True, min_periods=1).mean()
                          .to_numpy()[window_days:-window_days])
        block = pd.DataFrame(smoothed_vals, columns=cols)
        block["month"] = [m for m, d in calendar_order]
        block["day"] = [d for m, d in calendar_order]
        block["hour"] = hour
        smoothed_blocks.append(block)
    return pd.concat(smoothed_blocks, ignore_index=True).set_index(["month", "day", "hour"])[cols]


# climatology values at the time stamps of index
def map_climatology_ns(index, climatology, cols):
    cal = pd.DataFrame({"month": index.month, "day": index.day, "hour": index.hour}, index=index)
    mapped = cal.merge(climatology, left_on=["month", "day", "hour"], right_index=True, how="left")[cols]
    mapped.index = index
    return mapped


# linear trend of the anomalies from a rough climatology
def fit_linear_trend_ns(df, cols, window_days=15):
    rough_climatology = build_smoothed_climatology_ns(df, cols, window_days=window_days)
    rough_mean = map_climatology_ns(df.index, rough_climatology, cols)
    rough_anom = df[cols] - rough_mean
    t_years = (df.index - df.index[0]).total_seconds() / (365.25 * 24 * 3600)
    trend_coefs = {}
    for col in cols:
        slope, intercept = np.polyfit(t_years, rough_anom[col].to_numpy(), 1)
        trend_coefs[col] = {"slope": slope, "intercept": intercept, "t0": df.index[0]}
    return trend_coefs


# trend values at the time stamps of index
def trend_value_ns(index, trend_coefs, cols):
    out = pd.DataFrame(index=index)
    for col in cols:
        t0 = trend_coefs[col]["t0"]
        t_years = (index - t0).total_seconds() / (365.25 * 24 * 3600)
        out[col] = trend_coefs[col]["intercept"] + trend_coefs[col]["slope"] * t_years
    return out


def simulate_region_group_vine(var_key, cols, df_train_raw, df_test_raw, is_solar):
    """As simulate_region_group() of simulate_4_16_regions.py, with an R-vine copula across the regions (fitted per
    season on the residuals of the regions, families by BIC) instead of the Student-t copula."""
    df_train = df_train_raw.copy()
    df_test = df_test_raw.copy()
    if is_solar:
        for c in cols:
            df_train[c] = np.log(df_train[c] + SOLAR_LOG_EPS)
            df_test[c] = np.log(df_test[c] + SOLAR_LOG_EPS)

    trend_coefs = fit_linear_trend_ns(df_train, cols, window_days=CLIMATOLOGY_SMOOTHING_WINDOW_DAYS)
    df_train_detrended = df_train[cols] - trend_value_ns(df_train.index, trend_coefs, cols)
    climatology = build_smoothed_climatology_ns(df_train_detrended, cols, window_days=CLIMATOLOGY_SMOOTHING_WINDOW_DAYS)
    train_mean = map_climatology_ns(df_train.index, climatology, cols) + trend_value_ns(df_train.index, trend_coefs, cols)
    df_anom = df_train[cols] - train_mean
    year_baseline_offset = df_anom.groupby(df_anom.index.year).mean()
    sim_calendar_mean = map_climatology_ns(df_test.index, climatology, cols)

    models_seasonal = {season: {} for season in SEASONS}
    residuals_by_season = {}
    for season in SEASONS:
        block_id = season_block_ids(df_anom.index, season)
        season_resid = pd.DataFrame(index=df_anom.index)
        for col in cols:
            fit = fit_ar_model_seasonal_one(y=df_anom[col], block_id=block_id, col=col, maxlag=AR_MAX_LAG)
            models_seasonal[season][col] = fit
            season_resid[col] = fit["resid"]
        residuals_by_season[season] = season_resid.dropna()
    log(f"    {var_key}: AR fit done for {len(cols)} regions")

    # fit one R-vine copula per season across the region residuals
    vines_by_season = {}
    for season in SEASONS:
        resid = residuals_by_season[season][cols]
        n = len(resid)
        U = pd.DataFrame({col: np.clip(pd.Series(resid[col]).rank(method="average") / (n + 1), EPS, 1 - EPS)
                           for col in cols})
        controls = pv.FitControlsVinecop(family_set=SPATIAL_CANDIDATE_FAMILIES, selection_criterion="bic",
                                          parametric_method="mle", allow_rotations=True, num_threads=1)
        vine = pv.Vinecop.from_data(data=np.asfortranarray(U[cols].to_numpy(dtype=float)), controls=controls)
        vines_by_season[season] = vine
        log(f"    {var_key} {season}: vine fit, {len(cols)} dims")
    log(f"    {var_key}: all seasonal vines fit")

    pool_by_season = {season: residuals_by_season[season][cols] for season in SEASONS}

    # simulate
    seeds = [BASE_SEED + r for r in range(N_TRAJECTORIES)]
    target_index = sim_calendar_mean.index
    T = len(target_index)
    sim_calendar_arr = {col: sim_calendar_mean[col].to_numpy() for col in cols}
    season_labels_target = np.array([get_season(m) for m in target_index.month])

    target_mean_batch = {col: np.empty((N_TRAJECTORIES, T)) for col in cols}
    for j, seed in enumerate(seeds):
        rng = np.random.default_rng(seed + 1_000_000)
        year = int(rng.choice(year_baseline_offset.index.to_numpy()))
        offset = year_baseline_offset.loc[year]
        for col in cols:
            target_mean_batch[col][j] = sim_calendar_arr[col] + offset[col]

    var_salt = {"wind": 0, "temp": 50_000, "solar": 90_000}[var_key]
    season_seed_offsets = {"winter": 0, "spring": 1000, "summer": 2000, "autumn": 3000}
    resid_sim_batch = {col: np.empty((N_TRAJECTORIES, T)) for col in cols}
    for season in SEASONS:
        mask = season_labels_target == season
        n_season_hours = int(mask.sum())
        vine = vines_by_season[season]
        pool = pool_by_season[season]
        n_pool = len(pool)
        sorted_pool = {col: np.sort(pool[col].to_numpy()) for col in cols}
        for j, seed in enumerate(seeds):
            sim_seed = int(seed) + var_salt + season_seed_offsets[season] + 777
            u = vine.simulate(n=n_season_hours, seeds=[sim_seed])
            for k, col in enumerate(cols):
                idx = np.clip((u[:, k] * n_pool).astype(int), 0, n_pool - 1)
                resid_sim_batch[col][j, mask] = sorted_pool[col][idx]
    log(f"    {var_key}: simulation draws done")

    season_labels_hist = target_index.month.map(get_season).to_numpy()
    out = {}
    for col in cols:
        train_history = df_anom[col].dropna().to_numpy(dtype=float)
        max_lag = max((max(models_seasonal[s][col]["lags"], default=0) for s in models_seasonal), default=0)
        step_const = np.empty(T, dtype=float)
        step_coefs = []
        for t_i in range(T):
            model = models_seasonal[season_labels_hist[t_i]][col]
            params = model["params"]
            lags = model["lags"]
            step_const[t_i] = float(params["const"])
            step_coefs.append({lag: float(params[f"{col}.L{lag}"]) for lag in lags})
        sim = np.empty((N_TRAJECTORIES, max_lag + T), dtype=float)
        if max_lag > 0:
            sim[:, :max_lag] = train_history[-max_lag:][np.newaxis, :]
        for t_i in range(T):
            i = t_i + max_lag
            value = np.full(N_TRAJECTORIES, step_const[t_i], dtype=float)
            for lag, coef in step_coefs[t_i].items():
                value += coef * sim[:, i - lag]
            value += resid_sim_batch[col][:, t_i]
            sim[:, i] = value
        phys_pre = sim[:, max_lag:] + target_mean_batch[col]
        out[col] = np.clip(np.exp(phys_pre) - SOLAR_LOG_EPS, 0, None) if is_solar else phys_pre
    log(f"    {var_key}: AR reconstruction done")
    return out, target_index, vines_by_season


# the 4-region simulation with an R-vine copula across the regions, per season and variable
def main():
    log("Loading 4x4 grid real data + test-year 2025 data...")
    df_wind_train = pd.read_pickle(f"{DATA_DIR}/grid4x4_wind_2000-2024.pkl")
    df_temp_train = pd.read_pickle(f"{DATA_DIR}/grid4x4_temp_2000-2024.pkl")
    df_solar_train = pd.read_pickle(f"{DATA_DIR}/grid4x4_solar_2000-2024.pkl")
    df_test_all = pd.read_pickle(f"{DATA_DIR}/grid4x4_test2025.pkl")

    quadrant_groups = {
        "Q_NW": [f"r{r}c{c}" for r in [0, 1] for c in [0, 1]],
        "Q_NE": [f"r{r}c{c}" for r in [0, 1] for c in [2, 3]],
        "Q_SW": [f"r{r}c{c}" for r in [2, 3] for c in [0, 1]],
        "Q_SE": [f"r{r}c{c}" for r in [2, 3] for c in [2, 3]],
    }
    region_names = list(quadrant_groups.keys())

    # mean of the cells of one region
    def region_mean(df, var_key, region_cells):
        return df[[f"{c}_{var_key}" for c in region_cells]].mean(axis=1)

    vine_diagnostics = {}
    for var_key, df_train_grid, is_solar in [
        ("wind", df_wind_train, False),
        ("temp", df_temp_train, False),
        ("solar", df_solar_train, True),
    ]:
        out_path = f"{DATA_DIR}/sim_4region_vine_{var_key}.pkl"
        if os.path.exists(out_path):
            log(f"{var_key}: already done, skipping")
            continue
        df_train_r = pd.DataFrame({rn: region_mean(df_train_grid, var_key, rc) for rn, rc in quadrant_groups.items()})
        df_test_r = pd.DataFrame({rn: region_mean(df_test_all, var_key, rc) for rn, rc in quadrant_groups.items()})

        phys, target_index, vines_by_season = simulate_region_group_vine(var_key, region_names, df_train_r, df_test_r, is_solar)
        with open(out_path, "wb") as f:
            pickle.dump({"phys": phys, "target_index": target_index, "region_names": region_names}, f)
        log(f"{var_key}: saved to {out_path}")

        vine_diagnostics[var_key] = {season: str(vines_by_season[season]) for season in SEASONS}

    with open(f"{DATA_DIR}/sim_4region_vine_diagnostics.pkl", "wb") as f:
        pickle.dump(vine_diagnostics, f)
    log("All variables done (4-region vine).")


if __name__ == "__main__":
    main()
