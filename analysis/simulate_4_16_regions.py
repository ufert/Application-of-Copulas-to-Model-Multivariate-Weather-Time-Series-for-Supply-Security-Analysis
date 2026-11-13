# Thesis: shared helpers (seasons, seasonal AR, climatology, trend); run directly, it builds the 4- and 16-region
# simulations (AR + Student-t copula per variable), used in the fit assessment, the spatial tests and the notebook
# Run: cd analysis && python simulate_4_16_regions.py
# Writes: data/sim_{4,16}region_{wind,temp,solar}.pkl
"""Regional simulations on 4 and 16 regions of the study box: per region and variable a seasonal AR on the anomalies
(trend and smoothed climatology removed), the residuals of the regions coupled by a Student-t copula per season. The
correlation matrix comes from Kendall's tau between the residuals of every pair of regions (rho = sin(pi tau / 2)),
projected to the nearest valid correlation matrix; the degrees of freedom per variable and season are NU_TWO_REGION.
The three variables are simulated independently.
Also provides the shared helpers of the regional models: seasons, season blocks, seasonal AR fit with BIC, smoothed
climatology, linear trend.
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import paths  # repository folders: data, figures, external raw data (see paths.py)

T0 = time.time()
# time-stamped log line
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

DATA_DIR = paths.DATA


# degrees of freedom of the seasonal Student-t copulas between the north and south series of each variable (two-region
# fits on the residuals of the seasonal ARs); every copula of this script uses the value of its variable and season
NU_TWO_REGION = {
    "wind": {"winter": 3.634198540156927, "spring": 3.756890791102919, "summer": 4.630343266396363,
             "autumn": 3.8934849037476456},
    "temp": {"winter": 3.9130609842421897, "spring": 3.400234683485781, "summer": 3.893468104236237,
             "autumn": 3.448218696472178},
    "solar": {"winter": 2.0, "spring": 2.0, "summer": 2.0306321085437053, "autumn": 2.0},
}
SEASONS = ["winter", "spring", "summer", "autumn"]
AR_MAX_LAG = 168
N_TRAJECTORIES = 300
BASE_SEED = 42
CLIMATOLOGY_SMOOTHING_WINDOW_DAYS = 15
SOLAR_LOG_EPS = 1.0


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


def nearest_psd_correlation(R):
    """Eigenvalue-clip R to the nearest valid correlation matrix (finite-sample
    tau matrices aren't guaranteed PSD, especially with many regions)."""
    eigvals, eigvecs = np.linalg.eigh((R + R.T) / 2)
    eigvals_clipped = np.clip(eigvals, 1e-6, None)
    R_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    d = np.sqrt(np.diag(R_psd))
    return R_psd / np.outer(d, d)


def simulate_region_group(var_key, cols, df_train_raw, df_test_raw, nu_by_season, is_solar):
    """Fit + simulate a multivariate Student-t spatial copula across len(cols)
    regions for one variable. Returns physical hourly reconstructions,
    shape (N_TRAJECTORIES, T) per column, plus the target calendar index."""
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

    # correlation matrix per season (Kendall tau -> Pearson, nearest-PSD)
    corr_by_season = {}
    for season in SEASONS:
        resid = residuals_by_season[season][cols]
        tau_mat = resid.corr(method="kendall").to_numpy()
        rho_mat = np.sin(tau_mat * np.pi / 2)
        np.fill_diagonal(rho_mat, 1.0)
        corr_by_season[season] = nearest_psd_correlation(rho_mat)
    log(f"    {var_key}: correlation matrices built")

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
    n_regions = len(cols)
    for season in SEASONS:
        mask = season_labels_target == season
        n_season_hours = int(mask.sum())
        R = corr_by_season[season]
        L = np.linalg.cholesky(R)
        nu = nu_by_season[season]
        pool = pool_by_season[season]
        n_pool = len(pool)
        sorted_pool = {col: np.sort(pool[col].to_numpy()) for col in cols}
        for j, seed in enumerate(seeds):
            rng = np.random.default_rng(int(seed) + var_salt + season_seed_offsets[season] + 777)
            z = rng.standard_normal((n_season_hours, n_regions)) @ L.T
            w = rng.chisquare(nu, size=n_season_hours)
            tvals = z * np.sqrt(nu / w)[:, None]
            u = student_t.cdf(tvals, df=nu)
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
    return out, target_index


# the 4- and 16-region simulations (AR + Student-t copula, variables independent)
def main():
    log("Loading 4x4 grid real data + test-year 2025 data...")
    df_wind_train = pd.read_pickle(f"{DATA_DIR}/grid4x4_wind_2000-2024.pkl")
    df_temp_train = pd.read_pickle(f"{DATA_DIR}/grid4x4_temp_2000-2024.pkl")
    df_solar_train = pd.read_pickle(f"{DATA_DIR}/grid4x4_solar_2000-2024.pkl")

    cells = [f"r{r}c{c}" for r in range(4) for c in range(4)]
    quadrant_groups = {
        "Q_NW": [f"r{r}c{c}" for r in [0, 1] for c in [0, 1]],
        "Q_NE": [f"r{r}c{c}" for r in [0, 1] for c in [2, 3]],
        "Q_SW": [f"r{r}c{c}" for r in [2, 3] for c in [0, 1]],
        "Q_SE": [f"r{r}c{c}" for r in [2, 3] for c in [2, 3]],
    }

    # test year 2025 on the 16 cells (cached in data/grid4x4_test2025.pkl)
    QUAD_TEST_CACHE = f"{DATA_DIR}/grid4x4_test2025.pkl"
    if os.path.exists(QUAD_TEST_CACHE):
        log("Loading cached 4x4 test-year (2025) data...")
        df_test_all = pd.read_pickle(QUAD_TEST_CACHE)
    else:
        log("Loading 4x4 test-year (2025) data fresh (16 cells x 3 variables, 1 year)...")
        import sys
        sys.path.insert(0, paths.ANALYSIS)
        from load_4x4_grid import load_year_all_cells, base_dir, VAR_SPECS
        frames = {}
        for var_key, (var_code, var_name, mbc_or_noc) in VAR_SPECS.items():
            year_cells = load_year_all_cells(var_code, var_name, 2025, mbc_or_noc, base_dir, filename_suffix="")
            for cell in cells:
                frames[f"{cell}_{var_key}"] = year_cells[cell]
            log(f"  test-year {var_key} loaded")
        df_test_all = pd.DataFrame(frames)
        df_test_all.to_pickle(QUAD_TEST_CACHE)
    log(f"Test-year data: {df_test_all.shape}")

    results_store = {}
    for granularity_name, groups in [("4region", quadrant_groups), ("16region", {c: [c] for c in cells})]:
        log(f"\n{'='*70}\nGranularity: {granularity_name}\n{'='*70}")
        region_names = list(groups.keys())

        # mean of the cells of one region
        def region_mean(df, var_key, region_cells):
            return df[[f"{c}_{var_key}" for c in region_cells]].mean(axis=1)

        for var_key, df_train_grid, df_test_grid, is_solar in [
            ("wind", df_wind_train, df_test_all, False),
            ("temp", df_temp_train, df_test_all, False),
            ("solar", df_solar_train, df_test_all, True),
        ]:
            out_path = f"{DATA_DIR}/sim_{granularity_name}_{var_key}.pkl"
            if os.path.exists(out_path):
                log(f"  {var_key} @ {granularity_name}: already done, skipping")
                continue
            nu_by_season = NU_TWO_REGION[var_key]
            log(f"  {var_key} @ {granularity_name}: reused seasonal nu = {nu_by_season}")

            df_train_r = pd.DataFrame({rn: region_mean(df_train_grid, var_key, rc) for rn, rc in groups.items()})
            df_test_r = pd.DataFrame({rn: region_mean(df_test_grid, var_key, rc) for rn, rc in groups.items()})

            phys, target_index = simulate_region_group(var_key, region_names, df_train_r, df_test_r, nu_by_season, is_solar)
            with open(out_path, "wb") as f:
                pickle.dump({"phys": phys, "target_index": target_index, "region_names": region_names}, f)
            log(f"  {var_key} @ {granularity_name}: saved to {out_path}")

    log("\nAll granularities x variables done.")


if __name__ == "__main__":
    main()
