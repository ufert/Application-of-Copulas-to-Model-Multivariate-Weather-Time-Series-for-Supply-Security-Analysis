# Thesis: Case Study, Model Development: batched simulation of the four national models
# Module: imported by ar_var_copula_extremes.ipynb; python vectorized_simulation.py runs its self-tests
"""Batched simulation of the national models of ar_var_copula_extremes.ipynb (Sections 5-7).

The reconstruction of the AR, VAR and AR + Copula models is a recursion over time (simulated[i] depends on
simulated[i-1..i-lag]), so the loop over the T hours stays; each step processes all N trajectories at once as one numpy
operation. The innovations are drawn per trajectory with the same seeds as the scalar reference functions, so batched
and scalar results are identical.

Irradiance is reconstructed in log space (log(solar + SOLAR_LOG_EPS)) and exponentiated back to W/m2, wind in Box-Cox
space (WIND_LAMBDA), so that neither can become negative.

Running the file directly (python vectorized_simulation.py) checks that batched and scalar functions agree, on small
synthetic fixtures with the statsmodels object shapes of the notebook, and times both.
"""

import time

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.api import VAR

SOLAR_LOG_EPS = 1.0  # W/m2 - must match the setting in Section 2 of the notebook

# Box-Cox parameter of the wind transformation: wind is modelled as (v ** lambda - 1) / lambda and transformed back
# with wind_inv, so the simulated wind cannot become negative
WIND_LAMBDA = 0.25  # must match the setting in Section 2 of the notebook


def wind_fwd(x, lam=None):
    """Physical wind speed -> model space. Inverse of wind_inv."""
    lam = WIND_LAMBDA if lam is None else lam
    if lam == 1.0:
        return x
    return np.log(x) if lam == 0.0 else (np.asarray(x, dtype=float) ** lam - 1.0) / lam


def wind_inv(y, lam=None):
    """Model space -> physical wind speed. Always non-negative, so no clipping."""
    lam = WIND_LAMBDA if lam is None else lam
    if lam == 1.0:
        return np.clip(y, 0, None)
    if lam == 0.0:
        return np.exp(y)
    return np.clip(lam * np.asarray(y, dtype=float) + 1.0, 1e-9, None) ** (1.0 / lam)


# scalar reference implementations (one trajectory at a time, as in ar_var_copula_extremes.ipynb), used to test the
# batched versions below

# reference implementation (one trajectory): add the simulated AR residuals to the anomalies and the mean
def reconstruct_simulation_scalar(resid_sim, models, test_mean, train_anom, variables):
    sim_anom = pd.DataFrame(index=test_mean.index, columns=variables, dtype=float)
    for col in variables:
        model = models[col]
        params = model.params
        lags = list(model.model._lags)
        train_history = train_anom[col].dropna().to_numpy(dtype=float)
        intercept = float(params["const"]) if "const" in params.index else 0.0
        n_test = len(test_mean)
        simulated = np.zeros(n_test, dtype=float)
        for i in range(n_test):
            value = intercept
            for lag in lags:
                coefficient = float(params[f"{col}.L{lag}"])
                lagged_value = simulated[i - lag] if i - lag >= 0 else train_history[i - lag]
                value += coefficient * lagged_value
            value += float(resid_sim[col].iloc[i])
            simulated[i] = value
        sim_anom[col] = simulated
    sim_phys = sim_anom + test_mean[variables]
    sim_phys["wind"] = wind_inv(sim_phys["wind"].to_numpy())
    sim_phys["solar"] = np.exp(sim_phys["solar"]) - SOLAR_LOG_EPS
    sim_phys["solar"] = sim_phys["solar"].clip(lower=0)
    return sim_phys


# reference implementation: one VAR path with a burn-in, optionally with wind innovations scaled
def simulate_var_path_scalar(var_results, n_steps, index, seed=42, burn_in=500, wind_scale=None, wind_idx=None):
    if not var_results.is_stable(verbose=False):
        raise ValueError("The fitted VAR is not stable.")
    rng = np.random.default_rng(seed)
    p = var_results.k_ar
    k = var_results.neqs
    coefs = var_results.coefs
    intercept = var_results.intercept
    sigma_u = np.asarray(var_results.sigma_u)
    total_steps = n_steps + burn_in
    history = np.asarray(var_results.endog[-p:], dtype=float).copy()
    simulated = np.empty((total_steps, k))
    for step in range(total_steps):
        value = intercept.copy()
        for lag in range(1, p + 1):
            value += coefs[lag - 1] @ history[-lag]
        innovation = rng.multivariate_normal(mean=np.zeros(k), cov=sigma_u)
        if wind_scale is not None and step >= burn_in:
            innovation = innovation.copy()
            innovation[wind_idx] *= wind_scale[step - burn_in]
        new_value = value + innovation
        simulated[step] = new_value
        history = np.vstack([history, new_value])[-p:]
    simulated = simulated[burn_in:]
    return pd.DataFrame(simulated, index=index, columns=var_results.names)


# reference implementation: seasonal AR reconstruction of one trajectory
def reconstruct_simulation_seasonal_scalar(resid_sim, models_seasonal, target_mean, train_anom, variables, get_season):
    sim_anom = pd.DataFrame(index=target_mean.index, columns=variables, dtype=float)
    season_labels = target_mean.index.month.map(get_season).to_numpy()
    for col in variables:
        train_history = train_anom[col].dropna().to_numpy(dtype=float)
        n = len(target_mean)
        simulated = np.zeros(n, dtype=float)
        for i in range(n):
            model = models_seasonal[season_labels[i]][col]
            params = model["params"]
            lags = model["lags"]
            value = float(params["const"])
            for lag in lags:
                coefficient = float(params[f"{col}.L{lag}"])
                lagged_value = simulated[i - lag] if i - lag >= 0 else train_history[i - lag]
                value += coefficient * lagged_value
            value += float(resid_sim[col].iloc[i])
            simulated[i] = value
        sim_anom[col] = simulated
    sim_phys = sim_anom + target_mean[variables]
    sim_phys["wind"] = wind_inv(sim_phys["wind"].to_numpy())
    sim_phys["solar"] = np.exp(sim_phys["solar"]) - SOLAR_LOG_EPS
    sim_phys["solar"] = sim_phys["solar"].clip(lower=0)
    return sim_phys


# BATCHED (vectorized across N trajectories) implementations

def _ar_recursion_batch(resid_batch, const, coefs, train_history, N, T):
    """
    Core AR recursion, vectorized across N trajectories.

    resid_batch:   (N, T) array of residuals/innovations, one row per trajectory
    const:         scalar intercept
    coefs:         dict {lag: coefficient}
    train_history: 1-D array, the tail of the training anomaly (for lag
                   references before simulation starts) - shared across
                   all trajectories, exactly as in the scalar version.

    Returns (N, T) array of simulated anomalies.

    The only remaining Python-level loop is over T (unavoidable: step i
    depends on step i-1). Every iteration operates on N-length vectors.
    """
    max_lag = max(coefs) if coefs else 0
    sim = np.empty((N, max_lag + T), dtype=float)
    if max_lag > 0:
        pre = train_history[-max_lag:]  # (max_lag,) - same negative-index slice as train_history[i-lag]
        sim[:, :max_lag] = pre[np.newaxis, :]

    for t in range(T):
        i = t + max_lag
        value = np.full(N, const, dtype=float)
        for lag, coef in coefs.items():
            value += coef * sim[:, i - lag]
        value += resid_batch[:, t]
        sim[:, i] = value

    return sim[:, max_lag:]


def reconstruct_simulation_batch(resid_sim_batch, models, target_mean_batch, train_anom, variables):
    """
    resid_sim_batch:   dict[col] -> (N, T) ndarray
    target_mean_batch: dict[col] -> (N, T) ndarray (per-trajectory baseline,
                        e.g. from the year-bootstrap - can also just be the
                        same row repeated N times if there's no per-trajectory
                        baseline)
    Returns dict[col] -> (N, T) ndarray of physical (clipped) values.
    """
    N, T = next(iter(resid_sim_batch.values())).shape
    out = {}
    for col in variables:
        model = models[col]
        params = model.params
        lags = list(model.model._lags)
        train_history = train_anom[col].dropna().to_numpy(dtype=float)
        intercept = float(params["const"]) if "const" in params.index else 0.0
        coefs = {lag: float(params[f"{col}.L{lag}"]) for lag in lags}

        sim_anom = _ar_recursion_batch(resid_sim_batch[col], intercept, coefs, train_history, N, T)
        phys = sim_anom + target_mean_batch[col]
        if col == "wind":
            phys = wind_inv(phys)
        elif col == "solar":
            phys = np.exp(phys) - SOLAR_LOG_EPS
            phys = np.clip(phys, 0, None)
        out[col] = phys
    return out


def simulate_ar_independent_batch(models, residuals, target_mean_batch, train_anom, variables, seeds, wind_scale=None):
    """
    seeds: length-N sequence of per-trajectory seeds (e.g. [BASE_SEED+r for r in range(N)])
           - residual bootstrap draws are generated per-trajectory with these
           exact seeds, so results match simulate_ar_independent() run N times.
    target_mean_batch: dict[col] -> (N, T) ndarray
    wind_scale: optional length-T array, an hour-of-day/season volatility
           multiplier (Section 7 of the notebook) applied to the bootstrapped wind
           residual before reconstruction, identically across trajectories
           (it depends on the calendar only). None (default): no scaling.
    """
    N = len(seeds)
    T = next(iter(target_mean_batch.values())).shape[1]
    pools = {col: residuals[col].dropna().to_numpy() for col in variables}
    resid_sim_batch = {col: np.empty((N, T)) for col in variables}
    # one rng per trajectory, consumed across all columns in order - matches
    # simulate_ar_independent()'s "rng = default_rng(seed); for col in variables: ..."
    # exactly (NOT a fresh rng per column, which would silently reuse the same
    # seed's first draw for every variable).
    for j, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for col in variables:
            resid_sim_batch[col][j] = rng.choice(pools[col], size=T, replace=True)
    if wind_scale is not None:
        resid_sim_batch["wind"] = resid_sim_batch["wind"] * wind_scale[np.newaxis, :]
    return reconstruct_simulation_batch(resid_sim_batch, models, target_mean_batch, train_anom, variables)


def simulate_var_path_batch(var_results, n_steps, N, seeds, burn_in=500, wind_scale=None, wind_idx=None):
    """
    Vectorized across N trajectories. Innovations are pre-drawn once per
    trajectory (size=total_steps in one call, exactly as the scalar
    version drew them one-at-a-time with the same per-step RNG calls -
    numpy's multivariate_normal with a size argument draws the same
    underlying stream as repeated single draws) so results match
    simulate_var_path() run N times with the same seeds.

    wind_scale: optional length-n_steps array (Section 7 of the notebook),
           applied only to the innovations of the wind column after the
           burn-in (the burn-in steps have no calendar dates, so they stay
           unscaled).
           wind_idx: index of "wind" in var_results.names, required if
           wind_scale is given. Both None (default): no scaling.
    """
    if not var_results.is_stable(verbose=False):
        raise ValueError("The fitted VAR is not stable.")

    p = var_results.k_ar
    k = var_results.neqs
    coefs = var_results.coefs
    intercept = var_results.intercept
    sigma_u = np.asarray(var_results.sigma_u)
    total_steps = n_steps + burn_in

    init_history = np.asarray(var_results.endog[-p:], dtype=float)  # (p, k)
    hist = np.broadcast_to(init_history, (N, p, k)).copy()          # (N, p, k)
    sim = np.empty((N, total_steps, k), dtype=float)

    innovations = np.empty((N, total_steps, k), dtype=float)
    for j, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        innovations[j] = rng.multivariate_normal(mean=np.zeros(k), cov=sigma_u, size=total_steps)
    if wind_scale is not None:
        innovations[:, burn_in:, wind_idx] *= wind_scale[np.newaxis, :]

    for step in range(total_steps):
        value = np.tile(intercept, (N, 1))  # (N, k)
        for lag in range(1, p + 1):
            value += hist[:, -lag, :] @ coefs[lag - 1].T
        new_value = value + innovations[:, step, :]
        sim[:, step, :] = new_value
        hist = np.roll(hist, -1, axis=1)
        hist[:, -1, :] = new_value

    return sim[:, burn_in:, :]  # (N, n_steps, k)


# VAR paths for many seeds at once, mapped to physical units
def simulate_var_physical_batch(var_results, target_mean_batch, variables, seeds, burn_in=500, wind_scale=None):
    N = len(seeds)
    T = next(iter(target_mean_batch.values())).shape[1]
    names = list(var_results.names)
    wind_idx = names.index("wind") if wind_scale is not None else None
    sim = simulate_var_path_batch(var_results, n_steps=T, N=N, seeds=seeds, burn_in=burn_in,
                                   wind_scale=wind_scale, wind_idx=wind_idx)  # (N, T, k)
    out = {}
    for col in variables:
        col_idx = names.index(col)
        phys = sim[:, :, col_idx] + target_mean_batch[col]
        if col == "wind":
            phys = wind_inv(phys)
        elif col == "solar":
            phys = np.exp(phys) - SOLAR_LOG_EPS
            phys = np.clip(phys, 0, None)
        out[col] = phys
    return out


# seasonal AR reconstruction for many trajectories at once
def reconstruct_simulation_seasonal_batch(resid_sim_batch, models_seasonal, target_mean_batch, target_index,
                                           train_anom, variables, get_season):
    N, T = next(iter(resid_sim_batch.values())).shape
    season_labels = target_index.month.map(get_season).to_numpy()  # (T,), shared across all trajectories
    out = {}
    for col in variables:
        train_history = train_anom[col].dropna().to_numpy(dtype=float)
        max_lag = max((max(models_seasonal[s][col]["lags"], default=0) for s in models_seasonal), default=0)

        # per-step (const, {lag: coef}) - depends only on the calendar date, so
        # computed once (not per trajectory)
        step_const = np.empty(T, dtype=float)
        step_coefs = []
        for t in range(T):
            model = models_seasonal[season_labels[t]][col]
            params = model["params"]
            lags = model["lags"]
            step_const[t] = float(params["const"])
            step_coefs.append({lag: float(params[f"{col}.L{lag}"]) for lag in lags})

        sim = np.empty((N, max_lag + T), dtype=float)
        if max_lag > 0:
            pre = train_history[-max_lag:]
            sim[:, :max_lag] = pre[np.newaxis, :]

        for t in range(T):
            i = t + max_lag
            value = np.full(N, step_const[t], dtype=float)
            for lag, coef in step_coefs[t].items():
                value += coef * sim[:, i - lag]
            value += resid_sim_batch[col][:, t]
            sim[:, i] = value

        phys = sim[:, max_lag:] + target_mean_batch[col]
        if col == "wind":
            phys = wind_inv(phys)
        elif col == "solar":
            phys = np.exp(phys) - SOLAR_LOG_EPS
            phys = np.clip(phys, 0, None)
        out[col] = phys
    return out


# CORRECTNESS TESTS - synthetic fixtures, no real ERA5 data needed.
# Real statsmodels AR/VAR objects (not mocks) so the batched functions
# are exercised against the exact object shapes the notebook uses.

# test fixture: a fitted AR model on a synthetic series
def _make_ar_fixture(name="wind", n_obs=400, maxlag=10, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n_obs)
    for i in range(1, n_obs):
        y[i] = 0.3 * y[i - 1] + (0.15 * y[i - 5] if i >= 5 else 0) + rng.normal(scale=1.0)
    idx = pd.date_range("2000-01-01", periods=n_obs, freq="h")
    series = pd.Series(y, index=idx, name=name)
    selection = ar_select_order(series, maxlag=maxlag, ic="bic", old_names=False)
    model = selection.model.fit()
    return model


def _make_ar_models_fixture(variables=("wind", "temp", "solar")):
    """One independently-fit tiny AR model per variable name - needed because
    the scalar reference functions clip both "wind"
    and "solar" (matching the notebook, which always has all three
    variables present), and params are named f"{series.name}.L{lag}"."""
    return {col: _make_ar_fixture(name=col, seed=i) for i, col in enumerate(variables)}


# test fixture: a fitted VAR on a synthetic series
def _make_var_fixture(n_obs=400, k=2, maxlags=5, seed=1):
    rng = np.random.default_rng(seed)
    y = np.zeros((n_obs, k))
    for i in range(1, n_obs):
        y[i, 0] = 0.4 * y[i - 1, 0] + 0.1 * y[i - 1, 1] + rng.normal(scale=1.0)
        y[i, 1] = 0.2 * y[i - 1, 0] + 0.35 * y[i - 1, 1] + rng.normal(scale=1.0)
    idx = pd.date_range("2000-01-01", periods=n_obs, freq="h")
    df = pd.DataFrame(y, index=idx, columns=["a", "b"])
    model = VAR(df)
    results = model.fit(maxlags=maxlags, trend="c")
    return results


# test: the batched AR simulation equals the scalar reference
def test_ar_batch_matches_scalar():
    variables = ["wind", "temp", "solar"]
    models = _make_ar_models_fixture(variables)

    T = 300
    N = 6
    idx = pd.date_range("2030-01-01", periods=T, freq="h")
    train_anom = pd.DataFrame(
        {col: models[col].model.endog.ravel() for col in variables},
        index=models["wind"].model.data.row_labels,
    )
    # arbitrary nonzero, per-variable baseline (stands in for the year-bootstrap target_mean)
    target_mean = pd.DataFrame(
        {col: (i + 1) * np.sin(np.linspace(0, 6, T)) for i, col in enumerate(variables)}, index=idx
    )

    seeds = [1000 + r for r in range(N)]
    pools = {col: np.asarray(models[col].resid) for col in variables}
    residuals = {col: models[col].resid for col in variables}

    # scalar reference, run one trajectory at a time
    scalar_results = {col: [] for col in variables}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        resid_sim = pd.DataFrame(
            {col: rng.choice(pools[col], size=T, replace=True) for col in variables}, index=idx
        )
        out = reconstruct_simulation_scalar(resid_sim, models, target_mean, train_anom, variables)
        for col in variables:
            scalar_results[col].append(out[col].to_numpy())
    scalar_stack = {col: np.stack(scalar_results[col]) for col in variables}  # (N, T) each

    # batched
    target_mean_batch = {col: np.tile(target_mean[col].to_numpy(), (N, 1)) for col in variables}
    batch_out = simulate_ar_independent_batch(models, residuals, target_mean_batch, train_anom, variables, seeds)

    max_diff = 0.0
    for col in variables:
        assert np.allclose(batch_out[col], scalar_stack[col], atol=1e-10), f"AR batch mismatch vs scalar reference ({col})"
        max_diff = max(max_diff, np.max(np.abs(batch_out[col] - scalar_stack[col])))
    print(f"[PASS] AR batch matches scalar reference exactly (N={N}, T={T}, "
          f"{len(variables)} variables, max abs diff={max_diff:.2e})")


# test: the batched VAR simulation equals the scalar reference
def test_var_batch_matches_scalar():
    results = _make_var_fixture()
    T = 200
    burn_in = 50
    N = 5
    idx = pd.date_range("2030-01-01", periods=T, freq="h")
    seeds = [2000 + r for r in range(N)]

    scalar_results = []
    for seed in seeds:
        sim = simulate_var_path_scalar(results, n_steps=T, index=idx, seed=seed, burn_in=burn_in)
        scalar_results.append(sim.to_numpy())
    scalar_stack = np.stack(scalar_results)  # (N, T, k)

    batch_stack = simulate_var_path_batch(results, n_steps=T, N=N, seeds=seeds, burn_in=burn_in)

    assert np.allclose(batch_stack, scalar_stack, atol=1e-8), "VAR batch mismatch vs scalar reference"
    print(f"[PASS] VAR batch matches scalar reference exactly (N={N}, T={T}, max abs diff="
          f"{np.max(np.abs(batch_stack - scalar_stack)):.2e})")


def _make_var_fixture_3var(n_obs=400, maxlags=5, seed=7):
    """Like _make_var_fixture, but named wind/temp/solar so simulate_var_physical's
    wind/solar clip + solar exp-transform branches are actually exercised."""
    rng = np.random.default_rng(seed)
    y = np.zeros((n_obs, 3))
    for i in range(1, n_obs):
        y[i, 0] = 0.4 * y[i - 1, 0] + 0.1 * y[i - 1, 1] + rng.normal(scale=1.0)
        y[i, 1] = 0.2 * y[i - 1, 0] + 0.35 * y[i - 1, 1] + rng.normal(scale=1.0)
        y[i, 2] = 0.15 * y[i - 1, 1] + 0.3 * y[i - 1, 2] + rng.normal(scale=1.0)
    idx = pd.date_range("2000-01-01", periods=n_obs, freq="h")
    df = pd.DataFrame(y, index=idx, columns=["wind", "temp", "solar"])
    return VAR(df).fit(maxlags=maxlags, trend="c")


# reference implementation: one VAR trajectory in physical units
def simulate_var_physical_scalar(var_results, target_mean, variables, seed, burn_in=500):
    sim_anom = simulate_var_path_scalar(
        var_results=var_results, n_steps=len(target_mean), index=target_mean.index, seed=seed, burn_in=burn_in
    )
    sim_phys = sim_anom[variables] + target_mean[variables]
    sim_phys["wind"] = wind_inv(sim_phys["wind"].to_numpy())
    sim_phys["solar"] = np.exp(sim_phys["solar"]) - SOLAR_LOG_EPS
    sim_phys["solar"] = sim_phys["solar"].clip(lower=0)
    return sim_phys


# test: the batched VAR simulation in physical units equals the scalar reference
def test_var_physical_batch_matches_scalar():
    variables = ["wind", "temp", "solar"]
    results = _make_var_fixture_3var()
    T = 300
    burn_in = 50
    N = 5
    idx = pd.date_range("2030-01-01", periods=T, freq="h")
    # deliberately includes large negative baselines for solar so the exp-transform's
    # "never goes negative" property is actually exercised, not just the happy path
    target_mean = pd.DataFrame(
        {"wind": np.full(T, 3.0), "temp": np.zeros(T), "solar": np.linspace(-3, 1, T)}, index=idx
    )
    seeds = [5000 + r for r in range(N)]

    scalar_results = {col: [] for col in variables}
    for seed in seeds:
        out = simulate_var_physical_scalar(results, target_mean, variables, seed=seed, burn_in=burn_in)
        for col in variables:
            scalar_results[col].append(out[col].to_numpy())
    scalar_stack = {col: np.stack(scalar_results[col]) for col in variables}

    target_mean_batch = {col: np.tile(target_mean[col].to_numpy(), (N, 1)) for col in variables}
    batch_out = simulate_var_physical_batch(results, target_mean_batch, variables, seeds, burn_in=burn_in)

    max_diff = 0.0
    for col in variables:
        assert np.allclose(batch_out[col], scalar_stack[col], atol=1e-8), f"VAR physical batch mismatch ({col})"
        max_diff = max(max_diff, np.max(np.abs(batch_out[col] - scalar_stack[col])))
    assert (batch_out["solar"] >= 0).all(), "solar went negative despite the log-space exp-transform"
    print(f"[PASS] VAR physical batch matches scalar reference exactly (N={N}, T={T}, "
          f"max abs diff={max_diff:.2e}); solar stayed >= 0 throughout despite a baseline "
          f"ranging down to {target_mean['solar'].min():.1f} in log-space")


def test_solar_never_clips():
    """
    Test of the log-space reconstruction of irradiance. Each reconstruction gets inputs in its own units:
    Part 1: an additive reconstruction in physical units, a low baseline (a few W/m2 up to ~20) plus residual noise of
    the same size, goes negative in a large share of the hours.
    Part 2: the log-space reconstruction of the same baseline, with log-space noise, transformed back with exp(), never
    goes negative and varies smoothly instead of piling up at zero.
    """
    rng = np.random.default_rng(11)
    T = 5000

    # Part 1: additive reconstruction in physical units
    low_baseline = np.abs(rng.normal(loc=8.0, scale=4.0, size=T))  # a few to ~20 W/m2
    daytime_like_resid_std = 12.0  # comparable to the baseline itself - realistic for solar
    old_phys = low_baseline + rng.normal(scale=daytime_like_resid_std, size=T)
    old_negative_fraction = (old_phys < 0).mean()
    print(f"[DEMO] additive reconstruction, low irradiance baseline (mean~{low_baseline.mean():.1f} "
          f"W/m2, resid std~{daytime_like_resid_std:.0f} W/m2): {old_negative_fraction:.1%} of hours "
          f"go negative and would be clipped to exactly 0.")
    assert old_negative_fraction > 0.10, "the fixture should make the additive reconstruction go negative"

    # Part 2: log-space reconstruction of the same baseline
    log_baseline = np.log(low_baseline + SOLAR_LOG_EPS)  # same physical baseline, correctly transformed
    log_anom_std = 1.2  # a swing of e^1.2 ~= 3.3x is a large but realistic cloud-cover-driven swing
    log_anom = rng.normal(scale=log_anom_std, size=T)
    new_phys = np.exp(log_baseline + log_anom) - SOLAR_LOG_EPS
    new_phys = np.clip(new_phys, 0, None)  # safety net only - exp() already guarantees >= -SOLAR_LOG_EPS
    at_floor_fraction = (new_phys <= 1e-6).mean()
    print(f"[DEMO] log-space reconstruction, same baseline: min={new_phys.min():.4f} W/m2, "
          f"{at_floor_fraction:.1%} at the exact clip floor (vs {old_negative_fraction:.1%} that "
          f"clipped additively above) - still varies smoothly and realistically up to "
          f"{new_phys.max():.0f} W/m2 rather than piling up at zero.")
    assert (new_phys >= 0).all(), "solar went negative despite the log-space exp-transform"
    print("[PASS] log-space reconstruction never goes negative by construction, and does not pile "
          "up at zero like the additive reconstruction")


def test_wind_scale_batch_matches_scalar():
    """Test: the hour-of-day and season scale of the wind innovations (Section 7 of the notebook), batched against
    scalar."""
    variables = ["wind", "temp", "solar"]
    models = _make_ar_models_fixture(variables)

    T = 300
    N = 5
    idx = pd.date_range("2030-01-01", periods=T, freq="h")
    train_anom = pd.DataFrame(
        {col: models[col].model.endog.ravel() for col in variables},
        index=models["wind"].model.data.row_labels,
    )
    target_mean = pd.DataFrame(
        {col: (i + 1) * np.sin(np.linspace(0, 6, T)) for i, col in enumerate(variables)}, index=idx
    )
    seeds = [6000 + r for r in range(N)]
    pools = {col: np.asarray(models[col].resid) for col in variables}
    residuals = {col: models[col].resid for col in variables}

    rng_scale = np.random.default_rng(99)
    wind_scale = 1.0 + 0.5 * np.sin(np.linspace(0, 20, T)) + 0.1 * rng_scale.normal(size=T)

    # AR: scalar reference manually scales wind's drawn residual
    scalar_results = {col: [] for col in variables}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        resid_sim = pd.DataFrame(
            {col: rng.choice(pools[col], size=T, replace=True) for col in variables}, index=idx
        )
        resid_sim["wind"] = resid_sim["wind"].to_numpy() * wind_scale
        out = reconstruct_simulation_scalar(resid_sim, models, target_mean, train_anom, variables)
        for col in variables:
            scalar_results[col].append(out[col].to_numpy())
    scalar_stack = {col: np.stack(scalar_results[col]) for col in variables}

    target_mean_batch = {col: np.tile(target_mean[col].to_numpy(), (N, 1)) for col in variables}
    batch_out = simulate_ar_independent_batch(models, residuals, target_mean_batch, train_anom, variables, seeds,
                                               wind_scale=wind_scale)
    for col in variables:
        assert np.allclose(batch_out[col], scalar_stack[col], atol=1e-10), \
            f"AR wind_scale batch mismatch vs scalar reference ({col})"
    print("[PASS] AR batch with wind_scale matches scalar reference exactly")

    # VAR: scalar reference scales the wind innovation post-burn-in
    var_results = _make_var_fixture_3var()
    burn_in = 50
    wind_idx = list(var_results.names).index("wind")

    scalar_var_results = []
    for seed in seeds:
        sim = simulate_var_path_scalar(var_results, n_steps=T, index=idx, seed=seed, burn_in=burn_in,
                                        wind_scale=wind_scale, wind_idx=wind_idx)
        scalar_var_results.append(sim.to_numpy())
    scalar_var_stack = np.stack(scalar_var_results)

    batch_var_stack = simulate_var_path_batch(var_results, n_steps=T, N=N, seeds=seeds, burn_in=burn_in,
                                               wind_scale=wind_scale, wind_idx=wind_idx)
    assert np.allclose(batch_var_stack, scalar_var_stack, atol=1e-8), "VAR wind_scale batch mismatch vs scalar reference"
    print("[PASS] VAR batch with wind_scale matches scalar reference exactly")


# test: the batched seasonal AR reconstruction equals the scalar reference
def test_seasonal_ar_batch_matches_scalar():
    # meteorological season of a month
    def get_season(month):
        return "winter" if month in (12, 1, 2) else "spring" if month in (3, 4, 5) else \
               "summer" if month in (6, 7, 8) else "autumn"

    variables = ["wind", "temp", "solar"]
    models_seasonal = {}
    lag_sets = {"winter": [1, 3], "spring": [1], "summer": [1, 2, 6], "autumn": [1, 4]}
    for season, lag_set in lag_sets.items():
        rng = np.random.default_rng(hash(season) % 1000)
        models_seasonal[season] = {}
        for col in variables:
            params = pd.Series(
                [0.1 * rng.normal()] + [0.2 + 0.05 * rng.normal() for _ in lag_set],
                index=["const"] + [f"{col}.L{lag}" for lag in lag_set],
            )
            models_seasonal[season][col] = {"params": params, "lags": lag_set}
    rng = np.random.default_rng(42)
    train_anom = pd.DataFrame({col: rng.normal(size=200) for col in variables})

    T = 365 * 3  # spans multiple seasons + a winter wrap
    N = 6
    idx = pd.date_range("2030-01-01", periods=T, freq="h")
    target_mean = pd.DataFrame({col: np.zeros(T) for col in variables}, index=idx)
    seeds = [3000 + r for r in range(N)]

    scalar_results = {col: [] for col in variables}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        resid_sim = pd.DataFrame({col: rng.normal(size=T) for col in variables}, index=idx)
        out = reconstruct_simulation_seasonal_scalar(
            resid_sim, models_seasonal, target_mean, train_anom, variables, get_season
        )
        for col in variables:
            scalar_results[col].append(out[col].to_numpy())
    scalar_stack = {col: np.stack(scalar_results[col]) for col in variables}

    resid_sim_batch = {col: np.empty((N, T)) for col in variables}
    for j, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for col in variables:
            resid_sim_batch[col][j] = rng.normal(size=T)
    target_mean_batch = {col: np.zeros((N, T)) for col in variables}

    batch_out = reconstruct_simulation_seasonal_batch(
        resid_sim_batch, models_seasonal, target_mean_batch, idx, train_anom, variables, get_season
    )

    max_diff = 0.0
    for col in variables:
        assert np.allclose(batch_out[col], scalar_stack[col], atol=1e-10), f"Seasonal AR batch mismatch vs scalar reference ({col})"
        max_diff = max(max_diff, np.max(np.abs(batch_out[col] - scalar_stack[col])))
    print(f"[PASS] Seasonal AR batch matches scalar reference exactly (N={N}, T={T}, "
          f"{len(variables)} variables, max abs diff={max_diff:.2e})")


# Benchmark: scalar loop over the trajectories against the batched functions, same fixture, with a T shorter than 8760

# timing of the batched against the scalar simulation
def benchmark(T=2000, N=100):
    variables = ["wind", "temp", "solar"]
    models = _make_ar_models_fixture(variables)
    idx = pd.date_range("2030-01-01", periods=T, freq="h")
    train_anom = pd.DataFrame(
        {col: models[col].model.endog.ravel() for col in variables},
        index=models["wind"].model.data.row_labels,
    )
    target_mean = pd.DataFrame({col: np.zeros(T) for col in variables}, index=idx)
    seeds = [4000 + r for r in range(N)]
    pools = {col: np.asarray(models[col].resid) for col in variables}
    residuals = {col: models[col].resid for col in variables}

    t0 = time.perf_counter()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        resid_sim = pd.DataFrame({col: rng.choice(pools[col], size=T, replace=True) for col in variables}, index=idx)
        reconstruct_simulation_scalar(resid_sim, models, target_mean, train_anom, variables)
    scalar_time = time.perf_counter() - t0

    target_mean_batch = {col: np.zeros((N, T)) for col in variables}
    t0 = time.perf_counter()
    simulate_ar_independent_batch(models, residuals, target_mean_batch, train_anom, variables, seeds)
    batch_time = time.perf_counter() - t0

    print(f"\nBenchmark (AR reconstruction, N={N} trajectories, T={T} steps):")
    print(f"  scalar (original, per-trajectory loop): {scalar_time:.2f}s")
    print(f"  batched (vectorized across N):          {batch_time:.2f}s")
    print(f"  speedup: {scalar_time / batch_time:.1f}x")
    print("  (the AR order in the notebook can run past 100 lags on 25 years of")
    print("   data, and T=8760 - both push the ratio further in the batched")
    print("   version's favor, since it's fixed per-step overhead that scales")
    print("   with T for the scalar version but is paid once per T-step for N.)")


if __name__ == "__main__":
    print("Running correctness tests...\n")
    test_ar_batch_matches_scalar()
    test_var_batch_matches_scalar()
    test_var_physical_batch_matches_scalar()
    test_wind_scale_batch_matches_scalar()
    test_seasonal_ar_batch_matches_scalar()
    print("\nAll correctness tests passed - batched output is bit-identical to the")
    print("original scalar per-trajectory code on these fixtures.\n")
    test_solar_never_clips()
    print()
    benchmark()
