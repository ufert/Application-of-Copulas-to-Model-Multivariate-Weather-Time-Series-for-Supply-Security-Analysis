# Thesis: Results, Fit Assessment, table of coverage and scores
# Run: cd analysis && python fit_assessment_table.py   (after fit_assessment.py)
# Writes: thesis/fit_assessment_table.tex
"""LaTeX table of the fit assessment, thesis/fit_assessment_table.tex, from data/fit_assessment.pkl: coverage per variable;
CRPS and spatial variogram score as the mean over the three variables of the ratio to the unified model; coupling
variogram score as a ratio. Options --src=<folder> and --out=<file> set the input folder and the output file.
"""
import pickle
import sys

import numpy as np
import os
import paths  # repository folders: data, figures, external raw data (see paths.py)

SRC = next((a.split("=")[1] for a in sys.argv if a.startswith("--src=")), paths.DATA)
OUT = next((a.split("=")[1] for a in sys.argv if a.startswith("--out=")),
           os.path.join(paths.THESIS, "fit_assessment_table.tex"))
R = pickle.load(open(f"{SRC}/fit_assessment.pkl", "rb"))
U = R["ladder UNIFIED"]
V = ("wind", "temp", "solar")
ROWS = [("Variance inflation, fitted model (raw margins)", [(f"$k = {k}$", f"inflation k={k:.1f}" if k != 1.25 else "inflation k=1.25")
                                                            for k in (0.8, 1, 1.25, 1.5, 2)]),
        ("Model structure (real margins, $k = 1$)", [("Old sims (variables independent)", "ladder OLD"),
                                                     ("Per-unit AR + copula", "ladder AR-ONLY"),
                                                     ("Separate VAR per variable", "ladder SEPARATE"),
                                                     ("Unified model", "ladder UNIFIED")]),
        ("Benchmark", [("Real climate (other 74 years)", "real climate (leave one year out)")])]


# one table row: coverage per variable and the scores relative to the unified model
def line(label, r):
    cov = " & ".join(f"{r[f'coverage {v}']:.2f}" for v in V)
    crps = np.mean([r[f"CRPS {v}"] / U[f"CRPS {v}"] for v in V])
    spat = np.mean([r[f"variogram {v}"] / U[f"variogram {v}"] for v in V])
    coup = r["variogram coupling"] / U["variogram coupling"]
    return f"\\quad {label} & {cov} & {crps:.3f} & {spat:.3f} & {coup:.3f} \\\\"


out = [r"\begin{tabular}{lcccccc}", r"\toprule",
       r" & \multicolumn{3}{c}{Coverage of 90\% band} & CRPS & \multicolumn{2}{c}{Variogram score} \\",
       r"\cmidrule(lr){2-4}\cmidrule(lr){6-7}",
       r" & wind & temp. & solar & (rel.) & spatial (rel.) & coupling (rel.) \\", r"\midrule"]
for head, rows in ROWS:
    out.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{head}}}}} \\\\")
    out += [line(lab, R[key]) for lab, key in rows]
out += [r"\bottomrule", r"\end{tabular}"]
open(OUT, "w").write("\n".join(out) + "\n")
print(f"saved {OUT}")
