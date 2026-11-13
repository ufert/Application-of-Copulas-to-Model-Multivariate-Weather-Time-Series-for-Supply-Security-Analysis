#!/usr/bin/env bash
# Reproduce all results, numbers and figures of the thesis after the raw data has been downloaded (README.md, Data).
# Total runtime about 6-8 hours on a laptop; on macOS, run it as "caffeinate -ims ./run_all.sh" so the machine does not sleep.
# Every step can also be run on its own; the order below respects the inputs each script reads.
set -euo pipefail
cd "$(dirname "$0")/analysis"

# raw-data folders (links to the downloaded ERA5 and MERRA-2 files; stops here if a download is missing)
python organize_raw_data.py

# observed data: OPSD load and feed-in, the 16 cells, the 0.25 deg box, 1950-1999, 100 m wind, German land points, irradiance
python prepare_opsd.py
python load_4x4_grid.py
python extract_grid025.py
python extract_1950_1999.py
python extract_ws100.py --box=2000-2024 --box=1950-1999
python risk_map_025.py
python extract_solar025.py

# data chapter and mathematics chapter
python data_chapter_maps_pdf.py
python data_comparison_relative.py
python data_europe_feedin.py
python data_yearly_scatter.py
python math_chapter_figures.py
python math_copula_tail_table.py

# case study: the unified model, the 2030 power system and the calibrated conversion
python unified_model_05.py
python point1_night_ev.py
python point2_aggregation.py
python point3_value_map.py
for r in 16 1 0.5 0.25; do python unified_model_blocks.py --res=$r; done
python blocks_compare.py
python model_architecture_figures.py --thesis

# model development: earlier regional models and the spatial dependence tests
python simulate_4_16_regions.py
python simulate_4region_vine.py
python spatial_var_4q.py
python spatial_var_16.py

# results: 3,000 simulated years, validation, extremes, critical hours, potential and risk areas, number of years
python point5_model05.py
python model_selection_bic.py
python model_selection_figure.py
python fit_assessment.py
python fit_assessment_sig.py
python fit_assessment_table.py
python point5_properties.py
python point5_tails.py
python point5_tails_solar.py
python point5_mc.py
python point5_record_table.py
python point1_pv_atom.py
python point2_solar.py
python point3_solar.py
python point3_compare.py
python results_figures.py

# discussion: definitions of the literature applied to the series of this thesis
python li2021_check.py
python ohlendorf2020_check.py
python kittel2026_check.py

# case study: comparison of the innovation copulas (16-cell basis)
cd innovation_copula_16cell
python unified_model.py
python fit_assessment.py
python fit_assessment_sig.py
python parametric_copula.py
python parametric_copula_diag.py
cd ../..

# case study: the four national models (notebook, ~1 hour)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 ar_var_copula_extremes.ipynb
