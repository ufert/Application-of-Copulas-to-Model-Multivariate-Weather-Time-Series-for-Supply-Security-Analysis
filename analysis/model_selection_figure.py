# Thesis: Results, Model Validation, figure of the BIC of the VAR order and of the spatial modes
# Run: cd analysis && python model_selection_figure.py   (after model_selection_bic.py)
# Writes: thesis/pics/model_selection.{pdf,png}
"""Figure of the model selection of the unified model:
(a) BIC of the seasonal VAR of the modes over the grid of orders, per season, relative to its minimum;
(b) cumulative variance of the anomalies explained by the leading EOF modes, per variable, with the chosen numbers.
Input: data/model_selection_bic.pkl (model_selection_bic.py). Output: thesis/pics/model_selection.{pdf,png}.
"""
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import paths  # repository folders: data, figures, external raw data (see paths.py)

D = paths.DATA
OUT = paths.PICS
INK, MUTED, GRID = "#1F2328", "#5F6368", "#E3E5E8"
BLUE, BOR, PVC_, PET = "#00549F", "#A11035", "#B07A00", "#0098A1"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "text.color": INK, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})

R = pickle.load(open(f"{D}/model_selection_bic.pkl", "rb"))
U = R["unified"]


# axis labels, ticks, grid and frame in the thesis style
def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(length=0, labelsize=14)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.32})

# (a) BIC of the mode VAR; the order of 1 h lies far off the scale and is not drawn
seasons = (("winter", "winter (DJF)", BLUE, "o"), ("spring", "spring (MAM)", PET, "s"),
           ("summer", "summer (JJA)", PVC_, "^"), ("autumn", "autumn (SON)", BOR, "D"))
for s, lab, col, mk in seasons:
    b = U["var_bics"][s]
    ps = [p for p in sorted(b) if p > 1]
    m = min(b.values())
    y = [(b[p] - m) / 1e3 for p in ps]
    a1.plot(ps, y, color=col, lw=2, marker=mk, ms=8, label=lab)
a1.axvline(36, color=MUTED, lw=1, ls=(0, (4, 3)))
a1.text(37.5, 74, "selected:\n$p$ = 36 h", fontsize=13, color=MUTED, va="top")
a1.set_xscale("log")
a1.set_xticks([2, 3, 6, 12, 24, 36, 48, 72])
a1.set_xticklabels(["2", "3", "6", "12", "24", "36", "48", "72"])
a1.minorticks_off()
a1.set_ylim(-2, 80)
style(a1, "order $p$ of the VAR of the modes (hours)", "BIC minus its minimum (thousands)")
a1.legend(frameon=False, fontsize=13, loc="lower left")
a1.set_title("(a) Order of the VAR of the modes", loc="left", fontsize=17, fontweight="bold", pad=10)

# (b) explained variance of the EOF modes
for v, lab, col, mk, k in (("wind", "wind", BLUE, "o", 2), ("temp", "temperature", BOR, "s", 2),
                           ("solar", "irradiance", PVC_, "^", 3)):
    e = [100 * x for x in U["eof_expl"][v]]
    n = list(range(1, len(e) + 1))
    a2.plot(n, e, color=col, lw=2, marker=mk, ms=8, mfc="white", mew=2,
            label=f"{lab}: {k} modes, {e[k - 1]:.0f} %")
    a2.plot([k], [e[k - 1]], marker=mk, ms=11, color=col, ls="none")
style(a2, "number of modes per variable", "explained variance of the anomalies (%)")
a2.set_xticks(range(1, 9))
a2.set_ylim(40, 100)
a2.set_title("(b) Spatial modes", loc="left", fontsize=17, fontweight="bold", pad=10)
a2.plot([], [], color=MUTED, marker="o", ms=11, ls="none", label="filled: number of modes used")
a2.legend(frameon=False, fontsize=13, loc="lower right")

for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/model_selection.{ext}", dpi=150, bbox_inches="tight")
print("saved", f"{OUT}/model_selection.pdf")
