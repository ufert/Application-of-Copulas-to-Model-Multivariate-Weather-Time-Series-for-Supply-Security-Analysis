# Thesis: Case Study, Model Development: the lag-1 copula of irradiance in the model AR + Copula
# Module: imported by ar_var_copula_extremes.ipynb
# Simulates the irradiance of many trajectories along time with the seasonal lag-1 copula of the model AR + Copula
# (Section 7 of ar_var_copula_extremes.ipynb). The loop runs over the hours, because each hour is drawn from the
# conditional distribution (h-function) given the previous hour; at each hour all N trajectories are processed at once
# (pyvinecopulib's Bicop.hinv2 and np.quantile take arrays).

import numpy as np


# irradiance ranks along time for many trajectories from the seasonal lag-1 copula (AR + Copula model)
def simulate_solar_temporal_sequence_batch(solar_temporal_copulas, solar_pool_by_season, season_labels, n_traj, seed, eps=1e-6):
    rng = np.random.default_rng(seed)
    T = len(season_labels)
    u = np.empty((n_traj, T))
    out = np.empty((n_traj, T))
    prev_season = None

    for i in range(T):
        season = season_labels[i]
        if season != prev_season:
            u[:, i] = rng.uniform(size=n_traj)
        else:
            w = rng.uniform(size=n_traj)
            data = np.column_stack([w, u[:, i - 1]])
            u[:, i] = np.asarray(solar_temporal_copulas[season].hinv2(data))
        out[:, i] = np.quantile(solar_pool_by_season[season], np.clip(u[:, i], eps, 1 - eps))
        prev_season = season

    return out
