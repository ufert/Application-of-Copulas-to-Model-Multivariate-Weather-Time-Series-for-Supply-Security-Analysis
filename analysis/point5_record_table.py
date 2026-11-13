# Thesis: Results, table of the statistics of the weather model against the observed record
# Run: cd analysis && python point5_record_table.py   (after point5_model05.py)
# Writes: thesis/record_table.tex
"""Table of the weather model against the observed record, thesis/record_table.tex: one statistic per variable and
property. Model: the 3,000 simulated years (data/point5_model05.pkl, point5_model05.py); observed 2000-2024 from
data/point5_properties.pkl, the worst 72-h PV CF (data/point5_mc.pkl) and dark-and-calm (data/point5_tails.pkl)
1950-2024. Prints the standard error of each model value from the spread of the 10 batch values (SD / sqrt(10)).
"""
import pickle

import numpy as np
import pandas as pd
import os
import paths  # repository folders: data, figures, external raw data (see paths.py)

D = paths.DATA
OUT = os.path.join(paths.THESIS, "record_table.tex")
M3 = pickle.load(open(f"{D}/point5_model05.pkl", "rb"))
BATCH, NB = M3["batch"], M3["years"] // M3["batch"]
THETA = M3["theta"]

# (row, column, label, statistic, format, unit)
CELLS_T = [
    ("Level", "Wind", "mean CF", "L mean onshore CF", "{:.3f}", ""),
    ("Level", "Solar", "mean CF", "L mean PV CF", "{:.3f}", ""),
    ("Level", "Temperature", "winter mean", "L winter mean temperature (C)", "{:.2f}", r"\,$^\circ$C"),
    ("Level", "System 2030", "annual peak RL", "R annual peak RL (GW)", "{:.1f}", r"\,GW"),
    ("Persistence", "Wind", r"autocorr.\ 24\,h", "T onshore CF autocorrelation 24 h", "{:.2f}", ""),
    ("Persistence", "Solar", r"day-to-day autocorr.", "T PV day-to-day autocorrelation", "{:.2f}", ""),
    ("Persistence", "Temperature", r"autocorr.\ 72\,h", "T temperature autocorrelation 72 h", "{:.2f}", ""),
    ("Persistence", "System 2030", r"worst 72\,h RL", "R worst 72-h mean RL (GW)", "{:.1f}", r"\,GW"),
    ("Spatial", "Wind", r"cells $<$ 250\,km", "S onshore CF correlation between cells < 250 km", "{:.2f}", ""),
    ("Spatial", "Solar", "between cells", "S irradiance correlation between cells (daylight)", "{:.2f}", ""),
    ("Spatial", "Temperature", "between cells", "S temperature correlation between cells", "{:.2f}", ""),
    ("Spatial", "System 2030", r"h/yr wind + PV CF $<$ 10\,\%", "E hours RE CF < 10% (wind + PV)", "{:,.0f}", ""),
    ("Coupling", "Wind", r"wind--temp., winter", "X onshore CF-temperature correlation, winter", "{:.2f}", ""),
    ("Coupling", "Solar", r"wind--PV, winter", "X onshore CF-PV correlation, winter daylight", "{:.2f}", ""),
    ("Coupling", "Temperature", r"wind--temp., summer", "X onshore CF-temperature correlation, summer", "{:.2f}", ""),
    ("Coupling", "System 2030", r"demand--RE, winter", "X demand-RE feed-in correlation, winter", "{:.2f}", ""),
    ("Extremes", "Wind", r"lulls $\geq$ 72\,h per year", "E lulls >= 72 h", "{:.1f}", ""),
    ("Extremes", "Solar", r"worst 72\,h PV CF", "worst 72 h PV CF", "{:.3f}", ""),
    ("Extremes", "Temperature", r"coldest 72\,h", "E coldest 72-h mean temperature (C)", "{:.1f}", r"\,$^\circ$C"),
    ("Extremes", "System 2030", r"critical hours per year", "R hours RL > 57 GW", "{:.0f}", ""),
    ("Joint extremes", "Wind", "calm next door", "P calm-calm chi(5%) between cells < 250 km", "{:.2f}", ""),
    ("Joint extremes", "Solar", "dark and calm", "dark-calm chi(5%)", "{:.3f}", ""),
    ("Joint extremes", "Temperature", "cold and calm", "P cold-calm chi(5%), winter", "{:.2f}", ""),
    ("Joint extremes", "System 2030", "high demand, low RE", "P high demand-low RE chi(5%), winter", "{:.2f}", ""),
]


# value of a statistic over the years rows: mean, standard deviation or pooled ratio
def value(kind, vals, rows):
    if kind == "mean":
        return vals[rows].mean()
    if kind == "sd":
        return vals[rows].std(ddof=1)
    num, den = vals
    return num[rows].sum() / den[rows].sum()


def model_stat(st):
    """pooled value over all years and the 10 batch values"""
    if st == "worst 72 h PV CF":
        kind, vals = "mean", M3["mc"]["worst 72 h PV CF"].to_numpy()
    elif st == "dark-calm chi(5%)":
        c, i05 = M3["tails counts"]["dark-calm (DJF daylight)"], int(np.argmin(np.abs(M3["u grid"] - 0.05)))
        kind, vals = "ratio", (c["nj"][:, i05], c["na"][:, i05])
    else:
        kind, vals, _ = M3["per year"][st]
    n = len(vals) if kind != "ratio" else len(vals[0])
    allr = np.arange(n)
    return value(kind, vals, allr), np.array([value(kind, vals, allr[b * BATCH:(b + 1) * BATCH]) for b in range(NB)])


PR = pickle.load(open(f"{D}/point5_properties.pkl", "rb"))["table"]
OBS = PR[PR["reference"] == "2000-2024"].set_index("statistic")["real"].to_dict()
OBS["worst 72 h PV CF"] = pickle.load(open(f"{D}/point5_mc.pkl", "rb"))["real"]["worst 72 h PV CF"].mean()
TT = pickle.load(open(f"{D}/point5_tails.pkl", "rb"))["table"]
OBS["dark-calm chi(5%)"] = float(TT.loc[("dark-calm (DJF daylight)", "real"), "chi(0.05)"])

ROWS = []
for row, col, label, st, fmt, unit in CELLS_T:
    v, per_batch = model_stat(st)
    ROWS.append({"row": row, "column": col, "label": label, "statistic": st, "model": v,
                 "SE (batches)": per_batch.std(ddof=1) / np.sqrt(NB), "batch 0 (300 years)": per_batch[0],
                 "observed": OBS[st], "fmt": fmt, "unit": unit})
R = pd.DataFrame(ROWS)


# number formatted for LaTeX (thousands separator, minus sign)
def tex(x, fmt):
    s = fmt.format(x).replace(",", "{,}")
    return s.replace("-", "$-$") if s.startswith("-") else s


lines = [
    "% Model against the observed record, one statistic per variable and property. Model: 3,000",
    "% simulated years of the unified model on 0.5 deg units (analysis/point5_record_table.py from point5_model05.pkl);",
    "% observed 2000-2024 from point5_properties.py, worst 72 h PV CF (point5_mc.py) and dark-calm",
    "% (point5_tails.py) 1950-2024. Needs booktabs.",
    r"\begin{table}[htbp]",
    r"  \centering",
    r"  \caption{Statistics of the weather model (3,000 simulated years) against the observed record (2000--2024), each given",
    r"    as model vs observed. CF: capacity factor; RL: residual load of the 2030 system; RE: wind and photovoltaic feed-in;",
    f"    critical hours: residual load above {THETA:.1f}\\,GW. Persistence, spatial and coupling values are correlations of",
    r"    anomalies, with the daily and seasonal cycle removed; spatial values on the 16 cells of the $4 \times 4$ partition.",
    r"    Joint extremes: probability that the second variable is in its lowest 5\,\% (for demand: highest) given that the",
    r"    first one is. Worst 72\,h PV CF and dark and calm: observed 1950--2024. The weather levels (mean CF, winter mean) agree by",
    r"    construction, because the model's margins are mapped onto the observed distributions.}",
    r"  \label{tab:record}",
    r"  \medskip",
    r"  \footnotesize",
    r"  \setlength{\tabcolsep}{3pt}",
    r"  \newcommand{\stat}[2]{\begin{tabular}[t]{@{}l@{}}#1\\ #2\end{tabular}}",
    r"  \begin{tabular}{lllll}",
    r"    \toprule",
    r"     & Wind & Solar & Temperature & System 2030 \\",
    r"    \midrule",
]
for i, row in enumerate(dict.fromkeys(R["row"])):
    if i:
        lines.append(r"    \addlinespace")
    lines.append(f"    {row}")
    sub = R[R["row"] == row]
    for j, (_, r) in enumerate(sub.iterrows()):
        end = r" \\" if j == len(sub) - 1 else ""
        lines.append(f"      & \\stat{{{r['label']}}}{{{tex(r['model'], r['fmt'])} vs {tex(r['observed'], r['fmt'])}{r['unit']}}}{end}")
lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
open(OUT, "w").write("\n".join(lines) + "\n")
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
print(f"RECORD TABLE, model {M3['years']} years of the 0.5 deg model (critical threshold {THETA:.2f} GW)")
print(R.drop(columns=["statistic", "fmt", "unit"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
with open(f"{D}/point5_record_table.pkl", "wb") as f:
    pickle.dump(R, f)
print(f"saved {OUT} and {D}/point5_record_table.pkl")
