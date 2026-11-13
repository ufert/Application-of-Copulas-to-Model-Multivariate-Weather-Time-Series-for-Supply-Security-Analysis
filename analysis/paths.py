# Folders of the repository, shared by all analysis scripts.
# Every path is derived from the location of this file, so the scripts run on any machine without editing:
#   data/           processed data, caches and simulation output
#   data_external/  raw ERA5 / PECD 4.2 files for 1950-2025, PECD capacity factors, 100 m wind and masks
#   tmp/            temporary files of the resolution study and of chapter checks
#   thesis/         LaTeX tables and, in thesis/pics/, the figures
import os

# folder of the analysis scripts and the repository root
ANALYSIS = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(ANALYSIS)

# processed data, raw external data and temporary files
DATA = os.path.join(CODE, "data")
EXTERNAL = os.path.join(CODE, "data_external")
TMP = os.path.join(CODE, "tmp")

# LaTeX snippets (tables) and figures of the thesis
THESIS = os.path.join(CODE, "thesis")
PICS = os.path.join(THESIS, "pics")

# create the output folders on first use
for _d in (DATA, TMP, PICS):
    os.makedirs(_d, exist_ok=True)
