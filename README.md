# Copulas for multivariate weather time series in supply-security analysis

Code of the master's thesis *Application of Copulas to Model Multivariate Weather Time Series for Supply Security
Analysis* (Tim Ufer, RWTH Aachen University, 2026).

The thesis develops a stochastic generator for hourly 100 m wind speed, 2 m temperature and global horizontal
irradiance in Germany (box 47–55° N, 5–16° E). It converts the weather into the residual load of the German power system
of 2030 (115 GW onshore wind, 30 GW offshore wind, 215 GW photovoltaics) and evaluates it against 75 years of ERA5
reanalysis (1950–2024).

## Repository layout

| Path | Content |
|---|---|
| `analysis/` | All analysis scripts. Run them from this folder (`cd analysis`), because they import each other by name. |
| `analysis/paths.py` | The folders used by all scripts, relative to the repository (see *Data*). |
| `analysis/innovation_copula_16cell/` | Comparison of the empirical, Gaussian and Student-t innovation copulas, done on the 16-cell version of the model. |
| `analysis/conversion05_bootstrap.json` | Calibrated conversion parameters needed on a fresh start (see *Run order*). |
| `ar_var_copula_extremes.ipynb`, `vectorized_simulation.py`, `vectorized_solar_temporal.py` | The four national models (AR, VAR, AR + Copula, VAR + Copula) of the model development. |
| `get_era5_data.ipynb`, `nasa_data_pre.ipynb` | Downloads of ERA5 (PECD 4.2) and MERRA-2 (see *Data*). |
| `analysis/organize_raw_data.py` | Links the downloaded raw files into the folders the scripts read (see *Data*). |
| `residuen.ipynb` | Comparison of ERA5 and MERRA-2; its stored outputs are read by `analysis/data_yearly_scatter.py`. |
| `run_all.sh` | Runs every step in order after the data download. |
| `data/`, `data_external/`, `thesis/pics/` | Created by the scripts: processed data, raw data and figures (not in the repository). |

Each script starts with comment lines naming the part of the thesis it reproduces, how to run it, what it needs and
what it writes.

The names **"point 1" to "point 5"** in the code are the work packages of the thesis:
- 1: critical hours and the point mass of irradiance at night;
- 2: conversion and aggregation;
- 3: potential and risk areas;
- 4: fit assessment;
- 5: dependence in the extremes, comparison with the record and the number of simulated years.

## Installation

Python 3.14 was used. Install the pinned packages with

```bash
pip install -r requirements.txt
```

The downloads need a free account at the Copernicus Climate Data Store (`~/.cdsapirc`) and at NASA Earthdata
(`~/.netrc`).

## Data

All data are public. Download them in this order; every step skips what is already there, so it can be resumed after an
interruption.

| Step | Data set | How to get it | Where it goes |
|---|---|---|---|
| 1 | ERA5 via the Pan-European Climate Database (PECD) 4.2: 10 m wind, 2 m temperature, irradiance; hourly, 0.25°, 1950–2025, Europe (CDS dataset `sis-energy-pecd`) | `get_era5_data.ipynb` | `data_external/` |
| 2 | PECD 4.2 capacity factors, 100 m wind and zone masks, Germany, 1950–2025 | `cd analysis && python download_pecd.py all` | `data_external/{capacity_factors,ws100,masks}` |
| 3 | MERRA-2 M2T1NXFLX (SPEED, TLML) and M2T1NXRAD (SWGDN); hourly, 2015–2025, Europe | `nasa_data_pre.ipynb` with two link lists from the GES DISC subsetter (settings in the notebook) | `data/MERRA-2-2015-2025`, `data/MERRA-2-rad-2015-2025` |
| 4 | Open Power System Data, time series 2020-10-06 (German load and feed-in) | `cd analysis && python prepare_opsd.py` (also run by `run_all.sh`) | `data/opsd_germany_60min.pkl` |
| 5 | Folders the scripts read | `cd analysis && python organize_raw_data.py` (also run by `run_all.sh`) | links in `data/ERA-5-2000-2013`, `data/ERA-5-2014-2025`, `data/ERA-5-extreme-years`, `data/PECD`, `data/MERRA-2-edit`, `data/MERRA-2-rad` |

Step 5 links the downloaded files into the folders and under the file names that the scripts and notebooks read (for
example the ERA5 files of 2025 in `data/PECD/`); it copies nothing and stops with a list if a download is missing.
`risk_map_025.py` downloads the Natural Earth land and country shapes once through cartopy.

Disk space: about 250 GB in total (ERA5 176 GB, MERRA-2 33 GB, PECD 2.5 GB, processed data about 25 GB, temporary files
12 GB). The ERA5 zip files take another 176 GB until they are deleted (`KEEP_ZIPS` in `get_era5_data.ipynb`).
`data_external/` can be a link to an external drive.

## Run order

After the download, `./run_all.sh` runs every step (about 6–8 hours on a laptop). On macOS, use
`caffeinate -ims ./run_all.sh`. The scripts were run on a laptop with 16 GB of memory; most steps need 1–8 GB. The 0.25°
run of `unified_model_blocks.py` (about 11 GB, plus 12 GB of temporary files in `tmp/`) and the national-model notebook
need the most, so close other programs and keep enough free space on the system disk for swap. The order matters:

1. **Raw-data folders and observed data:** `organize_raw_data.py`, `prepare_opsd.py`, `load_4x4_grid.py`,
   `extract_grid025.py`, `extract_1950_1999.py`, `extract_ws100.py`, `risk_map_025.py` (the 734 German land points),
   `extract_solar025.py` (irradiance and temperature at these points).
2. **Unified model and power system:**
   - `unified_model_05.py`, `point1_night_ev.py`, `point2_aggregation.py`, `point3_value_map.py`.
   - The point-1 script needs the converted weather of the units (`units05.py`). That conversion uses parameters
     calibrated by the point-2 and point-3 scripts, which in turn need the 2030 system of point 1.
   - On a fresh start, `units05.py` therefore reads these parameters from `analysis/conversion05_bootstrap.json`, an
     exact extract of the outputs of the two scripts. Once they have run, it reads the same values from their outputs.
3. **Simulations and analyses:** `point5_model05.py` (3,000 simulated years, about 1 hour), then the validation,
   extremes, critical-hour, potential-and-risk and convergence scripts, and `results_figures.py`.

Random seeds are fixed, so the results are reproduced exactly.

## Where each result comes from

| Thesis | Script(s) | Output |
|---|---|---|
| Mathematical Foundations: Figure 3.1 (copula families) and the examples | `math_chapter_figures.py`, `math_copula_tail_table.py` | `thesis/pics/copula_families.pdf` |
| Data: Figures 4.1–4.6 (ERA5 and MERRA-2 maps) | `data_chapter_maps_pdf.py` | `thesis/pics/Era-5-*`, `Merra-2-*`, `Residual_*` |
| Data: Tables 4.2–4.4 (comparison statistics) | `data_comparison_relative.py` | printed |
| Data: Table 4.5 (capacity factors ERA5 vs MERRA-2) | `data_europe_feedin.py` | printed |
| Data: Figure 4.7 (yearly means) | `data_yearly_scatter.py` | `thesis/pics/Merra2vsEra5.pdf` |
| Case Study, Model Development: national models | `ar_var_copula_extremes.ipynb`, `vectorized_simulation.py` | notebook outputs |
| Case Study, Model Development: copula on the shocks vs dependence in the dynamics | `spatial_var_4q.py`, `spatial_var_16.py` (with `simulate_4_16_regions.py`, `simulate_4region_vine.py`) | printed |
| Case Study: flowcharts of the models | `model_architecture_figures.py --thesis` | `thesis/pics/model_arch_*.pdf` |
| Case Study: Choice of the Spatial Resolution | `unified_model_blocks.py`, `blocks_compare.py` | printed |
| Case Study: the unified model and its parameters | `unified_model_05.py`, `units05.py`, `spatial_common.py` | `data/sim05_*.npy` |
| Case Study: Innovation Copula (empirical vs Gaussian and Student-t) | `innovation_copula_16cell/*` | printed |
| Case Study: Conversion to Feed-in and its parameters | `conversion.py`, `point2_aggregation.py`, `point3_value_map.py` | printed |
| Case Study: Power System 2030, demand model, critical threshold | `point1_night_ev.py` | printed |
| Results, Model Validation: model selection and parametrization (BIC) | `model_selection_bic.py`, `model_selection_figure.py` | `thesis/pics/model_selection.pdf` |
| Results, Model Validation: fit assessment | `fit_assessment.py`, `fit_assessment_sig.py`, `fit_assessment_table.py` | `thesis/fit_assessment_table.tex` |
| Results, Model Validation: statistics against the observed record, envelope test | `point5_model05.py`, `point5_properties.py`, `point5_record_table.py` | `thesis/record_table.tex` |
| Results: From Weather to Feed-in | `point2_aggregation.py`, `point2_solar.py`, `results_figures.py` | `thesis/pics/pv_wind_aggregation_comparison.pdf` |
| Results: Dependence in the Extremes | `point5_tails.py`, `point5_tails_solar.py`, `results_figures.py` | `thesis/pics/tails_comparison.pdf` |
| Results: Critical Hours and the point mass of irradiance | `point1_night_ev.py`, `point1_pv_atom.py`, `results_figures.py` | `thesis/pics/critical_demand_dunkelflauten.pdf` |
| Results: Potential and Risk Areas | `point3_value_map.py`, `point3_solar.py`, `point3_compare.py`, `results_figures.py` | `thesis/pics/risk_potential_areas_wind_pv.pdf` |
| Results: Number of Simulated Years | `point5_mc.py`, `results_figures.py` | `thesis/pics/monte_carlo_convergence_plot.pdf` |
| Discussion: definitions of Li et al. (2021), Ohlendorf and Schill (2020), Kittel and Schill (2026) | `li2021_check.py`, `ohlendorf2020_check.py`, `kittel2026_check.py` | printed |
