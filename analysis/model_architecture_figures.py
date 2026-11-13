# Thesis: Case Study, flowcharts of the four national models and of the unified model (Figures 5.1-5.5)
# Run: cd analysis && python model_architecture_figures.py --thesis   (seconds)
# Writes: thesis/pics/model_arch_{ar,var,ar_copula,var_copula,unified}.pdf
"""
Architecture flowcharts of the models: one figure per model, same stages top to bottom
(data, preprocessing, dynamics, innovations, simulation, output), so that the models can be compared side by side.
Where a copula is used, the box is bordeaux with a COPULA tag; a copula that was fitted but not used for simulation
(AR + Copula) is dashed grey.

  model_arch_ar             model AR (ar_var_copula_extremes.ipynb, section 5): independent AR per variable
  model_arch_var            model VAR (section 6 of the notebook): pooled VAR, Gaussian innovations
  model_arch_ar_copula      model AR + Copula (sections 7-8): seasonal AR, solar lag-1 temporal copula
  model_arch_var_copula     model VAR + Copula (section 44): seasonal VAR, solar temporal copula + R-vine innovations
  model_arch_unified        thesis model (unified_model_05.py): EOF modes + VAR + per-unit AR, whole-day
                            empirical copula, margin map
Palette validated (light): #00549F dynamics, #A11035 copula (CVD dE 18); neutral grey for shared steps.
Run: python model_architecture_figures.py            PDF and PNG with titles in the repository root
     python model_architecture_figures.py --thesis   thesis versions in thesis/pics (no title inside the figure,
                                                     since the caption names it; theory tag under each stage label)
"""
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import paths  # repository folders: data, figures, external raw data (see paths.py)

THESIS = "--thesis" in sys.argv
OUT = paths.PICS if THESIS else paths.CODE
THEORY = {"Preprocessing": "stationarity"}                               # theory tags of the shared stages (--thesis)
INK, MUTED, GRID = "#1F2328", "#5F6368", "#C9CDD3"
BLUE, BOR = "#00549F", "#A11035"
STYLE = {   # kind: fill, edge, linewidth, linestyle, text colour
    "data": ("#F2F4F7", "#8A9099", 0.8, "-", INK),
    "prep": ("#F2F4F7", "#8A9099", 0.8, "-", INK),
    "dyn": ("#E6EEF7", BLUE, 1.0, "-", INK),
    "innov": ("#FFFFFF", BLUE, 0.9, "-", INK),
    "cop": ("#F7E8EB", BOR, 1.7, "-", INK),
    "unused": ("#FFFFFF", "#8A9099", 0.9, (0, (4, 2.5)), MUTED),
    "out": ("#F2F4F7", "#8A9099", 0.8, "-", INK),
}
mpl.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "text.color": INK})
W = 6.3                                                                   # figure width (inches), thesis text width
X0, X1 = 15.5, 99.5                                                       # main area (x units, 0-100 across the width)


# one flowchart: a matplotlib canvas in millimetres with boxes, arrows, legend and notes
class Chart:
    # empty canvas of the given height; theory names the mathematical basis of each stage
    def __init__(self, height, title, theory=None):
        self.theory = {**THEORY, **(theory or {})}
        self.fig = plt.figure(figsize=(W, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.H = height * 100 / W                                          # equal units in x and y
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, self.H)
        self.ax.axis("off")
        if not THESIS:
            self.ax.text(1, self.H - 2.2, title, fontsize=9, fontweight="bold", va="top", ha="left")

    def stage(self, y_top, y_bot, label):
        """stage label in the left margin and a faint separator above the stage"""
        yc = (y_top + y_bot) / 2
        tag = self.theory.get(label) if THESIS else None
        self.ax.text(1, yc + (0.9 if tag else 0), label, fontsize=6.6, color=MUTED, va="center", ha="left",
                     fontweight="bold")
        if tag:                                                           # theory behind the stage (mathematics chapter)
            self.ax.text(1, yc - 1.3, tag, fontsize=5.4, color=MUTED, va="center", ha="left", style="italic",
                         linespacing=1.2)
        self.ax.plot([1, 99.5], [y_top + 1.0] * 2, color=GRID, lw=0.5, ls=(0, (1, 2)), zorder=0)

    # rounded box of the given kind (data, model, copula, ...) with centred text and an optional tag
    def box(self, xc, yc, w, h, text, kind, fs=6.5, tag=None, align="center"):
        fill, edge, lw, ls, tc = STYLE[kind]
        self.ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.1",
                                         fc=fill, ec=edge, lw=lw, ls=ls, zorder=2))
        if align == "left":
            self.ax.text(xc - w / 2 + 1.6, yc, text, fontsize=fs, color=tc, ha="left", va="center", zorder=3, linespacing=1.35)
        else:
            self.ax.text(xc, yc, text, fontsize=fs, color=tc, ha="center", va="center", zorder=3, linespacing=1.35)
        if tag:                                                           # label on the top border, like a fieldset
            col = BOR if kind == "cop" else MUTED
            self.ax.text(xc - w / 2 + 2.2, yc + h / 2, tag, fontsize=5.5, color=col, ha="left", va="center",
                         fontweight="bold", zorder=4, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
        return (xc, yc, w, h)

    # arrow between two points, optionally labelled
    def arrow(self, x0, y0, x1, y1, color=INK, ls="-", lw=0.8, text=None, tx=None, ty=None, ha="left"):
        self.ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=color, lw=lw,
                                         ls=ls, shrinkA=0, shrinkB=0), zorder=1)
        if text:
            self.ax.text(tx if tx is not None else x0 + 0.8, ty if ty is not None else (y0 + y1) / 2, text,
                         fontsize=5.6, color=MUTED, ha=ha, va="center", zorder=3)

    def down(self, b0, b1, dx=0.0, **kw):
        """vertical arrow from the bottom of box b0 to the top of box b1 (at x of b1 + dx)"""
        x = b1[0] + dx
        self.arrow(x, b0[1] - b0[3] / 2, x, b1[1] + b1[3] / 2, **kw)

    # legend of the box kinds
    def legend(self, y, items):
        x = X0
        for kind, lab in items:
            fill, edge, lw, ls, tc = STYLE[kind]
            self.ax.add_patch(FancyBboxPatch((x, y - 1.3), 4.2, 2.6, boxstyle="round,pad=0,rounding_size=0.6",
                                             fc=fill, ec=edge, lw=lw, ls=ls, zorder=2))
            self.ax.text(x + 5.4, y, lab, fontsize=5.9, color=INK, va="center", ha="left")
            x += 7.4 + 0.8 * len(lab)

    # explanatory note below the chart
    def note(self, y, text):
        self.ax.text(X0, y, text, fontsize=6.0, color=MUTED, va="top", ha="left", linespacing=1.35)

    def finish(self, y_bottom):
        """crop the figure to [y_bottom, top], keeping equal units in x and y"""
        top = self.H - (3.8 if THESIS else 0)                             # no title row in the thesis version
        self.ax.set_ylim(y_bottom, top)
        self.fig.set_size_inches(W, (top - y_bottom) * W / 100)

    # save the chart (PDF for the thesis)
    def save(self, name):
        for ext in ("pdf",) if THESIS else ("pdf", "png"):
            self.fig.savefig(f"{OUT}/{name}.{ext}", dpi=220, facecolor="white")
        plt.close(self.fig)
        print(f"saved {OUT}/{name}.pdf" + ("" if THESIS else "/png"))


# x positions and width of n parallel lanes (one per variable)
def lanes(n=3, x0=X0, x1=X1, gap=2.2):
    w = (x1 - x0 - gap * (n - 1)) / n
    return [x0 + w / 2 + i * (w + gap) for i in range(n)], w


FULL = ((X0 + X1) / 2, X1 - X0)
VARS = ["wind", "temperature", "solar"]
DATA_NAT = ("ERA5 hourly 2000\u20132024, means over the study box 47\u201355\u00b0N, 5\u201316\u00b0E:\n"
            "10 m wind speed, 2 m temperature, global horizontal irradiance (GHI)")
PREP = ("transform (wind Box\u2013Cox $\\lambda = 0.25$, solar $\\ln(1 + \\mathrm{GHI})$), remove linear trend and\n"
        "smoothed (month, day, hour) climatology \u2192 anomalies $x_{v,t}$")
RECON = ("+ climatology + year offset (one training year per trajectory, all variables jointly)\n"
         "\u2192 inverse transforms \u2192 300 trajectories of hourly wind, temperature, solar (2025 calendar)")
LEG_NAT = [("prep", "data, shared steps"), ("dyn", "dynamics"), ("innov", "innovations"), ("cop", "copula")]


def national_top(c, y):
    """data and preprocessing rows of the national arms; returns the preprocessing box and the next free y"""
    c.stage(y, y - 6, "Data")
    b_data = c.box(*FULL[:1], y - 3.0, FULL[1], 6.0, DATA_NAT, "data")
    y2 = y - 9.5
    c.stage(y2, y2 - 6, "Preprocessing")
    b_prep = c.box(FULL[0], y2 - 3.0, FULL[1], 6.0, PREP, "prep")
    c.down(b_data, b_prep)
    return b_prep, y2 - 9.5


# variable names above the lanes
def lane_headers(c, xs, y):
    for x, v in zip(xs, VARS):
        c.ax.text(x, y, v, fontsize=6.4, color=BLUE, ha="center", va="center", fontweight="bold", zorder=4,
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})


# model AR
def fig_ar():
    c = Chart(7.0, "AR: one autoregression per variable",
              {"Dynamics": "autoregression", "Innovations": "bootstrap"})
    b_prep, y = national_top(c, c.H - 6.0)
    xs, lw_ = lanes()
    lane_headers(c, xs, y + 0.2)
    y -= 1.5
    c.stage(y, y - 9, "Dynamics")
    dyn = [c.box(x, y - 4.5, lw_, 9.0, "AR($p_v$), whole year\nBIC over $p \\leq 168$ h\n"
                 "$x_{v,t} = c_v + \\sum_\\ell \\phi_{v,\\ell}\\, x_{v,t-\\ell} + e_{v,t}$", "dyn", fs=6.3) for x in xs]
    for d in dyn:
        c.down(b_prep, d)
    y -= 12.5
    c.stage(y, y - 6, "Innovations")
    inn = [c.box(x, y - 3.0, lw_, 6.0, t, "innov") for x, t in zip(xs, [
        "i.i.d. bootstrap of the own\nresiduals \u00d7 hour-of-day scale", "i.i.d. bootstrap of the\nown residuals",
        "i.i.d. bootstrap of the\nown residuals"])]
    for d, i in zip(dyn, inn):
        c.down(d, i, text="residuals", tx=i[0] + 0.8)
    y -= 9.5
    c.stage(y, y - 6, "Simulation")
    out = c.box(FULL[0], y - 3.0, FULL[1], 6.0, RECON, "out")
    for i in inn:
        c.down(i, out, dx=i[0] - out[0])
    c.note(y - 7.8, "No dependence between the variables except the shared year offset: each variable is simulated on "
                    "its own.")
    c.legend(y - 12.5, LEG_NAT[:3])
    c.finish(y - 15.5)
    c.save("model_arch_ar")


# model VAR
def fig_var():
    c = Chart(7.0, "VAR: one vector autoregression for all variables",
              {"Dynamics": "VAR, BIC", "Innovations": "Gaussian"})
    b_prep, y = national_top(c, c.H - 6.0)
    c.stage(y, y - 7, "Dynamics")
    dyn = c.box(FULL[0], y - 3.5, FULL[1], 7.0,
                "VAR($p$) on $\\mathbf{x}_t = (x_{\\mathrm{wind},t},\\, x_{\\mathrm{temp},t},\\, x_{\\mathrm{solar},t})$, "
                "one coefficient set for the whole year, BIC over $p \\leq 168$ h\n"
                "$\\mathbf{x}_t = \\mathbf{c} + \\sum_\\ell \\boldsymbol{\\Phi}_\\ell\\, \\mathbf{x}_{t-\\ell} + \\mathbf{u}_t$"
                "     cross-lags in $\\boldsymbol{\\Phi}_\\ell$ couple the variables", "dyn")
    c.down(b_prep, dyn)
    y -= 10.5
    c.stage(y, y - 6, "Innovations")
    inn = c.box(FULL[0], y - 3.0, FULL[1], 6.0,
                "Gaussian $\\mathbf{u}_t \\sim \\mathcal{N}(\\mathbf{0}, \\boldsymbol{\\Sigma}_u)$ with the residual "
                "covariance; wind \u00d7 hour-of-day scale", "innov")
    c.down(dyn, inn, text="residual covariance $\\boldsymbol{\\Sigma}_u$", tx=FULL[0] + 0.8)
    y -= 9.5
    c.stage(y, y - 6, "Simulation")
    out = c.box(FULL[0], y - 3.0, FULL[1], 6.0, "VAR recursion with a 500 h warm-up; + climatology + year offset (one "
                "training year per trajectory)\n\u2192 inverse transforms \u2192 300 trajectories of hourly wind, "
                "temperature, solar (2025 calendar)", "out")
    c.down(inn, out)
    c.note(y - 7.8, "Dependence between the variables through the cross-lags and $\\boldsymbol{\\Sigma}_u$; "
                    "with one coefficient set the seasonal sign change of the\nwind\u2013temperature coupling averages out "
                    "(+0.07 instead of +0.50 in winter and \u22120.43 in summer).")
    c.legend(y - 15.0, LEG_NAT[:3])
    c.finish(y - 18.0)
    c.save("model_arch_var")


# model AR + Copula
def fig_ar_copula():
    c = Chart(8.0, "AR + Copula: seasonal ARs, temporal copula for solar",
              {"Dynamics": "periodic AR", "Innovations": "copulas,\nvine copulas"})
    b_prep, y = national_top(c, c.H - 6.0)
    xs, lw_ = lanes(x1=78.0)
    side_x, side_w = (80.4 + X1) / 2, X1 - 80.4
    lane_headers(c, xs, y + 0.2)
    y -= 1.5
    c.stage(y, y - 7, "Dynamics")
    dyn = [c.box(x, y - 3.5, lw_, 7.0, "seasonal AR($p_{v,s}$)\none per season $s$, BIC,\n$p \\leq 168$ h", "dyn")
           for x in xs]
    for d in dyn:
        c.down(b_prep, d)
    y -= 10.5
    c.stage(y, y - 18, "Innovations")
    inn_w = c.box(xs[0], y - 4.5, lw_, 9.0, "seasonal i.i.d.\nbootstrap of the own\nresiduals\n\u00d7 hour-of-day scale", "innov")
    inn_t = c.box(xs[1], y - 4.5, lw_, 9.0, "seasonal i.i.d.\nbootstrap of the own\nresiduals", "innov")
    inn_s = c.box(xs[2], y - 9.0, lw_, 18.0,
                  "lag-1 temporal copula\nper season (family by\nBIC, parameter \u00d7 shrink):\n"
                  "$u_t = h^{-1}(w_t \\mid u_{t-1})$,\n$w_t \\sim U(0, 1)$\n\u2193\nempirical margin:\n"
                  "$e_t = \\hat F_s^{-1}(u_t)$", "cop", fs=6.2, tag="COPULA")
    vine = c.box(side_x, y - 9.0, side_w, 18.0,
                 "3-dim R-vine\non the residual\nranks (wind,\ntemp, solar)\n\nfitted, not used\nfor simulation:\n"
                 "$|\\tau| < 0.08$\nin every season", "unused", fs=5.9, tag="COPULA")
    for d, i in zip(dyn, (inn_w, inn_t, inn_s)):
        c.down(d, i, text="residuals" if i is not inn_s else "residual ranks", tx=i[0] + 0.8)
    c.arrow(dyn[2][0] + lw_ / 2, dyn[2][1], side_x + side_w / 4, y, color="#8A9099", ls=(0, (3, 2)), lw=0.8)
    c.ax.text(side_x + side_w / 2 - 0.5, dyn[2][1] + 1.6, "ranks of all\nthree residuals", fontsize=5.6, color=MUTED,
              ha="right", va="bottom", linespacing=1.2)
    y -= 21.5
    c.stage(y, y - 6, "Simulation")
    out = c.box(FULL[0], y - 3.0, FULL[1], 6.0, "seasonal AR recursion per variable; + climatology + year offset (one "
                "training year per trajectory)\n\u2192 inverse transforms \u2192 300 trajectories of hourly wind, "
                "temperature, solar (2025 calendar)", "out")
    for i in (inn_w, inn_t, inn_s):
        c.down(i, out, dx=i[0] - out[0])
    c.note(y - 7.8, "The only copula in the simulation is the temporal one for solar (persistence of dull days); "
                    "the variables are not coupled\nexcept through the shared year offset.")
    c.legend(y - 15.0, LEG_NAT + [("unused", "fitted, not used")])
    c.finish(y - 18.0)
    c.save("model_arch_ar_copula")


# VAR + Copula
def fig_var_copula():
    c = Chart(8.5, "VAR + Copula: seasonal VAR with copula innovations",
              {"Dynamics": "periodic VAR", "Innovations": "R-vine,\nlag-1 copula"})
    b_prep, y = national_top(c, c.H - 6.0)
    c.stage(y, y - 9, "Dynamics")
    dyn = c.box(FULL[0], y - 4.5, FULL[1], 9.0,
                "seasonal VAR on $\\mathbf{x}_t = (x_{\\mathrm{wind}}, x_{\\mathrm{temp}}, x_{\\mathrm{solar}})_t$, one per "
                "season $s$, block-masked least squares, BIC → $p = 36$\n"
                "$\\mathbf{x}_t = \\mathbf{c}_s + \\sum_{\\ell=1}^{36} \\boldsymbol{\\Phi}_{s,\\ell}\\, \\mathbf{x}_{t-\\ell}"
                " + \\mathbf{e}_t$        the seasonal cross-lags carry the wind–temperature coupling", "dyn")
    c.down(b_prep, dyn)
    y -= 12.5
    xs, lw_ = lanes()
    c.stage(y, y - 27, "Innovations")
    pr = xs[1] + lw_ / 2 - 7.0
    pobs = c.box((X0 + pr) / 2, y - 4.0, pr - X0, 8.0,
                 "residual ranks per season (pseudo-observations)\n$U_{v,t} = \\mathrm{rank}(e_{v,t}) / (n + 1)$", "innov")
    sol = c.box(xs[2], y - 4.0, lw_, 8.0, "solar lag-1 temporal copula\n"
                "$u^{\\mathrm{sol}}_t = h^{-1}(w_t \\mid u^{\\mathrm{sol}}_{t-1})$\n(fitted on lag-1 pairs, shrink 2.5)",
                "cop", fs=6.1, tag="COPULA")
    c.down(dyn, pobs, text="residuals $\\mathbf{e}_t$", tx=pobs[0] + 0.8)
    c.arrow(pobs[0] + pobs[2] / 2, pobs[1], sol[0] - sol[2] / 2, sol[1], text="solar\nranks",
            tx=(pobs[0] + pobs[2] / 2 + sol[0] - sol[2] / 2) / 2, ty=pobs[1] + 2.4, ha="center")
    vine = c.box(FULL[0], y - 14.5, FULL[1], 9.0,
                 "3-dim R-vine per season on the residual ranks (families by BIC: Gaussian, t, Clayton, Gumbel, Frank, "
                 "BB1, Joe)\nsolar at the root: $(u^{\\mathrm{wind}}_t, u^{\\mathrm{temp}}_t)$ drawn conditionally on "
                 "$u^{\\mathrm{sol}}_t$ by the inverse Rosenblatt\ntransform; $u^{\\mathrm{sol}}_t$ passes unchanged",
                 "cop", fs=6.2, tag="COPULA")
    c.down(pobs, vine, dx=pobs[0] - vine[0], text="fit", tx=pobs[0] + 0.8)
    c.down(sol, vine, dx=sol[0] - vine[0], text="$u^{\\mathrm{sol}}_t$", tx=sol[0] + 0.8)
    marg = c.box(FULL[0], y - 24.0, FULL[1], 5.0,
                 "empirical margins: $e^*_{v,t} = \\hat F^{-1}_{v,s}(u^v_t)$, the quantiles of the season's VAR residuals",
                 "innov")
    c.down(vine, marg, text="$(u^{\\mathrm{wind}}_t,\\, u^{\\mathrm{temp}}_t,\\, u^{\\mathrm{sol}}_t)$",
           tx=FULL[0] + 0.8)
    y -= 30.0
    c.stage(y, y - 6, "Simulation")
    out = c.box(FULL[0], y - 3.0, FULL[1], 6.0,
                "seasonal VAR recursion from the last observed 36 h, driven by $\\mathbf{e}^*_t$; + climatology + year "
                "offset\n→ inverse transforms → 300 trajectories of hourly wind, temperature, solar (2025 "
                "calendar)", "out")
    c.down(marg, out, text="innovations $\\mathbf{e}^*_t$", tx=FULL[0] + 0.8)
    c.note(y - 7.8, "Copulas at two places: in time for solar (persistence of dull days) and across the variables for "
                    "the joint innovations (shape of\nthe joint distribution). The coupling between the variables "
                    "itself comes from the cross-lags of the seasonal VAR.")
    c.legend(y - 15.0, LEG_NAT)
    c.finish(y - 18.0)
    c.save("model_arch_var_copula")


# unified model (thesis)
def fig_unified():
    c = Chart(9.0, "Unified model (thesis model, 0.5° units): EOF modes + VAR + AR, empirical day copula",
              {"Spatial modes": "EOF", "Dynamics": "periodic VAR, AR", "Innovations": "empirical\ncopula",
               "Margins": "Sklar, GPD"})
    y = c.H - 6.0
    c.stage(y, y - 6, "Data")
    b_data = c.box(FULL[0], y - 3.0, FULL[1], 6.0,
                   "ERA5 / PECD 4.2 hourly 2000–2024 on 0.5° units (2 × 2 grid points):\n100 m wind on "
                   "203 land + 31 offshore units, 2 m temperature and GHI on 203 land units", "data")
    y -= 9.5
    c.stage(y, y - 6, "Preprocessing")
    b_prep = c.box(FULL[0], y - 3.0, FULL[1], 6.0, PREP.replace("anomalies $x_{v,t}$", "anomalies per unit "
                                                              "$\\mathbf{A}_{v,t}$ (640 series)"), "prep")
    c.down(b_data, b_prep)
    y -= 9.5
    c.stage(y, y - 6, "Spatial modes")
    eof = c.box(FULL[0], y - 3.0, FULL[1], 6.0,
                "EOFs per variable, $\\mathbf{A}_{v,t} = \\bar{\\mathbf{A}}_v + \\mathbf{E}_v\\, \\mathbf{s}_{v,t} + "
                "\\mathbf{r}_{v,t}$: wind 2, temperature 2, solar 3 modes (66, 90, 70 % of the variance)\n"
                "→ 7 mode scores $\\mathbf{s}_t$ (large-scale, coherent)  +  640 remainders $\\mathbf{r}_t$ (local)",
                "prep", fs=6.3)
    c.down(b_prep, eof)
    y -= 9.5
    xs, lw_ = lanes(2)
    c.stage(y, y - 7, "Dynamics")
    var = c.box(xs[0], y - 3.5, lw_, 7.0, "ONE seasonal VAR(36) across the 7 scores\n"
                "of all three variables; the cross-lags\ncouple wind, temperature and solar", "dyn")
    ar = c.box(xs[1], y - 3.5, lw_, 7.0, "seasonal AR per remainder (unit × variable),\n"
               "BIC over $q \\leq 72$ h (median order 4–5)", "dyn")
    c.down(eof, var, text="$\\mathbf{s}_t$", tx=xs[0] + 0.8)
    c.down(eof, ar, text="$\\mathbf{r}_t$", tx=xs[1] + 0.8)
    y -= 10.5
    c.stage(y, y - 9, "Innovations")
    cop = c.box(FULL[0], y - 4.5, FULL[1], 9.0,
                "whole real days resampled: each simulated day takes the complete residual vector of one real day\n"
                "within ±10 days of its date, same season (24 h × 647 components = 15,528; pool of 8,822 days)\n"
                "= empirical copula across units, variables and hours; keeps night residuals at night (PV zero),\n"
                "the daily cycle of the variance and the persistence within the day", "cop", fs=6.2, tag="COPULA")
    c.down(var, cop, dx=var[0] - cop[0], text="residuals", tx=var[0] + 0.8)
    c.down(ar, cop, dx=ar[0] - cop[0], text="residuals", tx=ar[0] + 0.8)
    y -= 12.5
    c.stage(y, y - 6, "Simulation")
    sim = c.box(FULL[0], y - 3.0, FULL[1], 6.0,
                "VAR + AR recursion with a 61-day warm-up (discarded); anomalies = modes × EOFs + remainders;\n"
                "+ climatology + year offset (one training year per trajectory, all series) → inverse transforms",
                "out")
    c.down(cop, sim, text="innovations", tx=FULL[0] + 0.8)
    y -= 9.5
    c.stage(y, y - 6, "Margins")
    mm = c.box(FULL[0], y - 3.0, FULL[1], 6.0,
               "Sklar step: per (unit, month, hour) the simulated ranks are kept (the copula of the simulation)\n"
               "and mapped onto the observed 2000–2024 distribution; temperature with generalized Pareto tails",
               "cop", fs=6.2, tag="COPULA KEPT, MARGINS REPLACED")
    c.down(sim, mm)
    y -= 9.5
    c.stage(y, y - 6, "Output")
    out = c.box(FULL[0], y - 3.0, FULL[1], 6.0,
                "300 years × 8,760 h on every unit → feed-in per unit (turbine classes, PV model) →\n"
                "national wind and PV feed-in, residual load of the 2030 system; 3,000 years for rare events", "out")
    c.down(mm, out)
    c.note(y - 7.8, "Dependence between units and variables is in the dynamics (EOF modes and VAR cross-lags); the "
                    "copula of the innovations is\nempirical, and the final Sklar step replaces the margins without "
                    "touching the dependence.")
    c.legend(y - 15.0, LEG_NAT)
    c.finish(y - 18.0)
    c.save("model_arch_unified")


if __name__ == "__main__":
    fig_ar()
    fig_var()
    fig_ar_copula()
    fig_var_copula()
    fig_unified()
