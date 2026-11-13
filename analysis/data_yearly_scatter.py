# Thesis: Data chapter, Figure 4.7 (yearly mean wind speed against irradiance, ERA5 and MERRA-2)
# Run: python analysis/data_yearly_scatter.py   (seconds, reads the stored outputs of residuen.ipynb)
# Writes: thesis/pics/Merra2vsEra5.{png,pdf}
"""Figure 4.7 of the data chapter: yearly mean wind speed against yearly mean GHI of ERA5 and MERRA-2, 2015-2025, with the
plotting code of residuen.ipynb (cell 31).
Data: the yearly domain means (cos-latitude weights) stored in the outputs of residuen.ipynb (cell 28: ERA5, cell 29:
MERRA-2); the ERA5 value of 2014 is left out, so that both data sets cover 2015-2025.
Output: thesis/pics/Merra2vsEra5.{png,pdf}.
"""
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "thesis", "pics", "Merra2vsEra5")


def table(cells, i):
    """the DataFrame printed by cell i of residuen.ipynb"""
    txt = "".join(cells[i]["outputs"][0]["data"]["text/plain"])
    lines = [l.replace("variable", "", 1) if l.lstrip().startswith("variable") else l
             for l in txt.splitlines() if l.strip()]        # cell 28 prints the pivot's column name first
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=r"\s+")
    return df[["year", "SPEED", "TLML", "SWGDN"]]


cells = json.load(open(os.path.join(ROOT, "residuen.ipynb")))["cells"]
df_er_wide = table(cells, 28)
df = table(cells, 29)
assert list(df_er_wide.year) == list(range(2014, 2026)) and list(df.year) == list(range(2015, 2026))
df_er_wide = df_er_wide[df_er_wide.year >= 2015].reset_index(drop=True)    # ERA5 2014 left out, so that both data sets cover 2015-2025

# plotting code of residuen.ipynb, cell 31
df_plot_era = df_er_wide
df_plot_merra = df

plt.figure(figsize=(10,6))

# Scatter for ERA5
plt.scatter(
    df_plot_era["SWGDN"],
    df_plot_era["SPEED"],
    color="dodgerblue",
    s=100,
    label="ERA-5"
)

# Scatter for MERRA-2
plt.scatter(
    df_plot_merra["SWGDN"],
    df_plot_merra["SPEED"],
    color="red",
    s=100,
    label="MERRA-2"
)

# Annotate each point with year (ERA5)
for i, row in df_plot_era.iterrows():
    plt.text(row["SWGDN"] - 0.04, row["SPEED"] - 0.007, str(int(row["year"])), fontsize=9)

# Annotate each point with year (MERRA-2)
for i, row in df_plot_merra.iterrows():
    plt.text(row["SWGDN"] - 0.04, row["SPEED"] - 0.007, str(int(row["year"])), fontsize=9)

plt.xlabel("Solar radiation (W/m²)")
plt.ylabel("Wind Speed (m/s)")
plt.grid(alpha=0.3)
plt.legend()  # <-- adds the legend
plt.tight_layout()

plt.savefig(OUT + ".png", dpi=100, bbox_inches="tight")
plt.savefig(OUT + ".pdf", bbox_inches="tight")
print("wrote", OUT + ".{png,pdf}")
