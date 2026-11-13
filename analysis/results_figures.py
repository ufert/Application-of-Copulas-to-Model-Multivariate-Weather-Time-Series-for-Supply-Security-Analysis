# Thesis: Results, the five figures of the chapter (critical hours, aggregation, potential and risk areas,
# dependence in the extremes, convergence) and the numbers the chapter quotes
# Run: cd analysis && python results_figures.py   (seconds, after the point1/2/3/5 scripts)
# Writes: thesis/pics/{critical_demand_dunkelflauten,pv_wind_aggregation_comparison,risk_potential_areas_wind_pv,tails_comparison,monte_carlo_convergence_plot}.pdf
"""The five figures of the Results chapter, from the outputs of the analysis scripts:
  critical_demand_dunkelflauten   Dunkelflaute hours (wind + PV < 10 % of capacity) and critical hours (RL above the
                                  threshold THETA of point1_night_ev.py) by local hour, 75 observed years, 2030 system
  pv_wind_aggregation_comparison  error of averaging the weather before converting it, wind and PV
  risk_potential_areas_wind_pv    potential and risk areas for wind and PV
  tails_comparison                chi(u) for four winter pairs
  monte_carlo_convergence_plot    running means and running percentiles of the 3,000 simulated years
Each figure is saved as PNG and as vector PDF, without titles inside the figure (panel labels only).
Inputs: data/point1_night_ev.pkl, point2_aggregation.pkl, point2_solar.pkl, point3_value_map.pkl, point3_solar.pkl,
point5_tails.pkl, point5_tails_solar.pkl, point5_mc.pkl. Output: thesis/pics/. Prints the numbers shown in the figures.
"""
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from scipy import stats
import paths  # repository folders: data, figures, external raw data (see paths.py)

D = paths.DATA
OUT = paths.PICS
INK, MUTED, GRID, BAND = "#1F2328", "#5F6368", "#E3E5E8", "#EEF1F4"
BLUE, BOR, GREY, LIGHT, PVC_, PET = "#00549F", "#A11035", "#8A9099", "#D3D6DA", "#B07A00", "#0098A1"
BG = "#FFFFFF"
RC = {"font.family": "DejaVu Sans", "font.size": 15, "text.color": INK, "axes.edgecolor": GRID, "xtick.color": MUTED,
      "ytick.color": MUTED, "axes.labelcolor": INK, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG}
mpl.rcParams.update(RC)
pd.set_option("display.width", 220)


# axis labels, ticks, grid and frame in the thesis style
def style(ax, xlabel, ylabel, grid="y"):
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(length=0, labelsize=14)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# bold panel title on the left
def ttl(ax, t):
    ax.set_title(t, loc="left", fontsize=17, fontweight="bold", pad=10)


def save(fig, name, **kw):
    """PNG and vector PDF; the map image stays a raster inside the PDF"""
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=150, **kw)


# section heading in the printed log
def head(t):
    print(f"\n=== {t} ===")


# point 1: Dunkelflaute and critical hours by local hour
P1 = pickle.load(open(f"{D}/point1_night_ev.pkl", "rb"))
THETA = float(P1["theta"]["with offshore"])
LOC = pd.date_range("2025-01-01", periods=8760, freq="h").tz_localize("UTC").tz_convert("Europe/Berlin")
LH = LOC.hour.to_numpy()
LM = LOC.month.to_numpy()
NIGHT_H = [22, 23, 0, 1, 2, 3, 4, 5]


# Dunkelflaute hours (wind + PV below 10 % of 360 GW) and critical hours (residual load above theta)
def p1_sets(X):
    re = 115 * X["onshore 16 cells"] + 30 * X["offshore"] + 215 * X["pv"]          # key of the onshore series in point1_night_ev.pkl
    return {"dunkelflaute": re / 360 < 0.10, "critical": X["demand"] - re > THETA}


# number of flagged hours per local hour of the day
def profile(m):
    return np.array([m[:, LH == h].sum() for h in range(24)], float)


REAL1 = p1_sets(P1["series"]["real 1950-2024"])
MOD1 = p1_sets(P1["series"]["model"])
head(f"point 1 (75 observed years 1950-2024; model 300 years scaled to 75; threshold {THETA:.2f} GW)")
for k in ("dunkelflaute", "critical"):
    r, m = profile(REAL1[k]), profile(MOD1[k]) * 75 / MOD1[k].shape[0]
    nr = r[NIGHT_H].sum() / r.sum()
    ev = r[[17, 18, 19, 20, 21]].sum() / r.sum()
    mo = r[[6, 7, 8, 9]].sum() / r.sum()
    mon = np.array([REAL1[k][:, LM == mm].sum() for mm in range(1, 13)], float)
    print(f"{k}: real {r.sum() / 75:,.0f} h/yr, night {100 * nr:.1f}%, evening 17-22 h {100 * ev:.1f}%, morning 6-10 h "
          f"{100 * mo:.1f}%, peak hour {r.argmax()} ({r.max():,.0f}), min hour {r.argmin()} ({r.min():,.0f}); "
          f"Nov-Feb share {100 * mon[[10, 11, 0, 1]].sum() / mon.sum():.1f}%; model/real {m.sum() / r.sum():.3f}, "
          f"night share model {100 * m[NIGHT_H].sum() / m.sum():.1f}%, profile r {np.corrcoef(r, m)[0, 1]:.3f}")
    print("   by hour (real):", " ".join(f"{v:,.0f}" for v in r))
both = REAL1["dunkelflaute"] & REAL1["critical"]
print(f"critical hours that are also Dunkelflaute hours: {100 * both.sum() / REAL1['critical'].sum():.1f}%; "
      f"Dunkelflaute hours that are critical: {100 * both.sum() / REAL1['dunkelflaute'].sum():.2f}%")

fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.3})
h = np.arange(24)
for ax, key, col, title in ((axs[0], "dunkelflaute", BLUE, "Dunkelflaute hours (wind + PV < 10%)"),
                            (axs[1], "critical", BOR, f"Critical hours (residual load > {THETA:.1f} GW)")):
    r = profile(REAL1[key])
    ax.axvspan(-0.5, 5.5, color=BAND, lw=0, zorder=0)
    ax.axvspan(21.5, 23.5, color=BAND, lw=0, zorder=0)
    ax.bar(h, r, width=0.78, color=col, zorder=2)
    top = 19500 if key == "dunkelflaute" else r.max() * 1.22
    ax.set_ylim(0, top)
    ax.text(2.5, 0.93 * top, "night", ha="center", va="center", fontsize=14, color=MUTED)
    ax.text(0.98, 0.93, f"{r.sum() / 75:,.0f} h per year\n{100 * r[NIGHT_H].sum() / r.sum():.0f}% at night", transform=ax.transAxes,
            ha="right", va="top", fontsize=14, color=INK, linespacing=1.4)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=14)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=10)
    ax.set_ylabel("hours in 75 years", fontsize=15)
    ax.set_xlabel("local hour of the day", fontsize=15)
save(fig, "critical_demand_dunkelflauten", bbox_inches="tight", pad_inches=0.15)
plt.close(fig)

# point 2: aggregation before or after converting
W2 = pickle.load(open(f"{D}/point2_aggregation.pkl", "rb"))["relative"]
S2 = pickle.load(open(f"{D}/point2_solar.pkl", "rb"))["relative"]
levels = [("Germany", 1), ("4 quadrants", 4), ("16 cells", 16), ("1 deg blocks", 61), ("0.5 deg blocks", 203)]
nreg = [n for _, n in levels] + [734]
head("point 2 (2000-2024, relative to the conversion at the 734 points)")
print(W2.loc[[(l, "A") for l, _ in levels], ["energy vs truth %", "hours <5% vs truth", "lulls >=72 h vs truth", "critical h vs truth"]]
      .round(3).to_string())
print(S2.loc[[("no clipping", l) for l, _ in levels]].round(3).to_string())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.25}, sharey=True)
for lab, col, mk, ls, c in (("lulls ≥ 72 h", "lulls >=72 h vs truth", "^", ":", BOR), ("hours below 5%", "hours <5% vs truth", "s", "--", BLUE),
                            ("energy", None, "o", "-", INK)):
    y = [1 + W2.loc[(l, "A"), "energy vs truth %"] / 100 if col is None else W2.loc[(l, "A"), col] for l, _ in levels] + [1.0]
    a1.plot(nreg, y, color=c, lw=2.4, ls=ls, marker=mk, ms=8, label=lab)
for lab, col, mk, ls, c in (("dark days", "dark days x", "^", ":", BOR), ("daylight hours below 5%", "daylight low hours x", "s", "--", BLUE),
                            ("energy", "energy %", "o", "-", INK)):
    y = [1 + S2.loc[("no clipping", l), col] / 100 if col == "energy %" else S2.loc[("no clipping", l), col] for l, _ in levels] + [1.0]
    a2.plot(nreg, y, color=c, lw=2.4, ls=ls, marker=mk, ms=8, label=lab)
for ax, t in ((a1, "Wind"), (a2, "Photovoltaics")):
    ax.axhline(1, color=MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.7, 1100)
    ax.set_xticks([1, 4, 16, 61, 203, 734])
    ax.set_xticklabels(["Germany", "4", "16\ncells", "61\n(1°)", "203\n(0.5°)", "734\n(0.25°)"])
    ax.minorticks_off()
    ax.set_ylim(0.6, 2.0)
    ax.set_yticks(np.arange(0.8, 2.01, 0.2))                      # no tick at 0.6, which ran into the "Germany" label
    style(ax, "regions averaged before converting", "ratio to full resolution" if ax is a1 else "")
    ax.legend(loc="upper right", fontsize=13, frameon=False)
    ttl(ax, t)
fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.17)
save(fig, "pv_wind_aggregation_comparison")
plt.close(fig)

# point 3: potential and risk areas
R3 = pickle.load(open(f"{D}/point3_value_map.pkl", "rb"))
S3 = pickle.load(open(f"{D}/point3_solar.pkl", "rb"))["sites"]
sites = R3["sites"]
code = {"risk": 0, "average": 1, "potential": 2}
head("point 3 (stress = worst 72 h of the 2030 residual load per year, 1950-2024)")
off = sites[sites["type"] == "offshore"]
onl = sites[sites["type"] == "onshore"]
print("wind onshore classes:", onl["class 72 h"].value_counts().to_dict(), " offshore:", off["class 72 h"].value_counts().to_dict(),
      f" offshore r median {off['r 72 h'].median():.2f} ({off['r 72 h'].min():.2f}-{off['r 72 h'].max():.2f})")
print(f"wind onshore r p10/median/p90 {onl['r 72 h'].quantile(0.1):.2f} / {onl['r 72 h'].median():.2f} / {onl['r 72 h'].quantile(0.9):.2f}")
print("PV classes:", S3["class pv"].value_counts().to_dict(), " combined:", S3["class combined"].value_counts().to_dict())
for lab, la0, la1, lo0, lo1 in (("north >= 53.5 N", 53.5, 56, 0, 20), ("centre 50.5-52.5 N", 50.5, 52.5, 0, 20),
                                ("south <= 48.5 N", 0, 48.5, 0, 20), ("west <= 9 E", 0, 90, 0, 9), ("east >= 12 E", 0, 90, 12, 20)):
    mw = (onl["lat"] >= la0) & (onl["lat"] <= la1) & (onl["lon"] >= lo0) & (onl["lon"] <= lo1)
    mp = (S3["lat"] >= la0) & (S3["lat"] <= la1) & (S3["lon"] >= lo0) & (S3["lon"] <= lo1)
    sw, sp = onl.loc[mw, "class 72 h"], S3.loc[mp, "class pv"]
    print(f"  {lab:20s} n {mw.sum():3d}/{mp.sum():3d}  wind: r med {onl.loc[mw, 'r 72 h'].median():.2f}, pot {100 * (sw == 'potential').mean():.0f}% "
          f"risk {100 * (sw == 'risk').mean():.0f}%  |  PV: r med {S3.loc[mp, 'r pv'].median():.2f}, pot {100 * (sp == 'potential').mean():.0f}% "
          f"risk {100 * (sp == 'risk').mean():.0f}%  |  combined r med {S3.loc[mp, 'r combined'].median():.2f}")
print(f"PV r p10/median/p90 {S3['r pv'].quantile(0.1):.2f} / {S3['r pv'].median():.2f} / {S3['r pv'].quantile(0.9):.2f}; "
      f"rank corr wind vs PV firmness {stats.spearmanr(S3['r wind'], S3['r pv'])[0]:.2f}; PV periods "
      f"{stats.spearmanr(S3['r pv 1950-1999'], S3['r pv 2000-2024'])[0]:.2f}; wind periods "
      f"{stats.spearmanr(S3['r wind 1950-1999'], S3['r wind 2000-2024'])[0]:.2f}")
print(f"stress CF: PV {np.average(S3['stress CF pv']):.3f} (mean CF {np.average(S3['mean CF pv']):.3f}), "
      f"wind {np.average(S3['stress CF wind']):.3f} (mean CF {np.average(S3['mean CF wind']):.3f})")
print(f"wind r vs mean CF {np.corrcoef(onl['r 72 h'], onl['mean CF'])[0, 1]:.2f}, vs kept share {np.corrcoef(onl['r 72 h'], onl['kept 72 h'])[0, 1]:.2f}; "
      f"PV r vs mean CF {np.corrcoef(S3['r pv'], S3['mean CF pv'])[0, 1]:.2f}")
fig, axs = plt.subplots(1, 2, figsize=(10.5, 6.6), gridspec_kw={"wspace": 0.05})
for ax, (title, lat, lon, cls) in zip(axs, [("Wind (onshore + offshore)", sites["lat"], sites["lon"], sites["class 72 h"]),
                                           ("Photovoltaics", S3["lat"], S3["lon"], S3["class pv"])]):
    img = np.full((33, 45), np.nan)
    for la, lo, c in zip(lat, lon, cls):
        img[int(round((55 - la) / 0.25)), int(round((lo - 5) / 0.25))] = code[c]
    ax.imshow(img, extent=(5 - 0.125, 16 + 0.125, 47 - 0.125, 55 + 0.125), cmap=ListedColormap([BOR, LIGHT, BLUE]), vmin=-0.5, vmax=2.5,
              interpolation="nearest", aspect=1 / np.cos(np.deg2rad(51)))
    ax.set_xlim(5.4, 15.4)
    ax.set_ylim(47.1, 55.25)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    n = {c: int((np.asarray(cls) == c).sum()) for c in code}
    ax.set_title(title, fontsize=18, fontweight="bold", pad=8)
    noff = int((np.asarray(cls)[np.asarray(sites["class"]) == "offshore"] == "potential").sum()) if title.startswith("Wind (") else 0
    pot = f"{n['potential'] - noff} + {noff} offshore" if title.startswith("Wind (") else f"{n['potential']}"
    ax.text(10.4, 46.95, f"potential {pot} · average {n['average']} · risk {n['risk']}", ha="center", va="top", fontsize=13, color=MUTED)
handles = [mpl.patches.Patch(color=BLUE, label="potential area"), mpl.patches.Patch(color=LIGHT, label="average"),
           mpl.patches.Patch(color=BOR, label="risk area")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=15, frameon=False)
fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.13)
save(fig, "risk_potential_areas_wind_pv")
plt.close(fig)

# point 5.1: tail dependence
T5 = pickle.load(open(f"{D}/point5_tails.pkl", "rb"))
TS = pickle.load(open(f"{D}/point5_tails_solar.pkl", "rb"))
U = T5["u"]
panels = [(T5["results"], "cold-calm (DJF)", "Cold and calm"), (T5["results"], "dark-calm (DJF daylight)", "Dark and calm"),
          (TS["results"], "dark persists 1 day (DJF)", "Dark two days running"), (T5["results"], "high demand-low RE (DJF)", "High demand, low feed-in")]
head("point 5.1 (observed 1950-2024 [90%] / Gaussian / model; model 3,000 years, dark two days running 300 years)")
i5, i1 = list(U).index(0.05), list(U).index(0.01)
for R, key, t in panels:
    r, m = R[("real", key)], R[("model", key)]
    print(f"{t:22s} rho_N {r['rho']:+.3f} (model {m['rho']:+.3f}); chi(0.2) {r['chi'][2]:.3f} / {r['chi_G'][2]:.3f} / {m['chi'][2]:.3f}; "
          f"chi(0.05) {r['chi'][i5]:.3f} [{r['chi lo'][i5]:.3f}-{r['chi hi'][i5]:.3f}] / {r['chi_G'][i5]:.3f} / {m['chi'][i5]:.3f}; "
          f"chi(0.01) {r['chi'][i1]:.3f} / {r['chi_G'][i1]:.3f} / {m['chi'][i1]:.3f}")
fig, axs = plt.subplots(1, 4, figsize=(16, 5.0), gridspec_kw={"wspace": 0.3})
for ax, (R, key, t) in zip(axs, panels):
    r, m = R[("real", key)], R[("model", key)]
    ax.fill_between(U, r["chi lo"], r["chi hi"], color=GRID, lw=0)
    ax.plot(U, r["chi"], color=INK, lw=2.4, marker="o", ms=6, label="observed")
    ax.plot(U, r["chi_G"], color=BLUE, lw=2.2, ls=(0, (4, 2)), label="Gaussian, same correlation")
    ax.plot(U, m["chi"], color=BOR, lw=2.2, marker="s", ms=5, label="model")
    ax.set_xscale("log")
    ax.set_xlim(0.55, 0.009)
    ax.set_xticks([0.5, 0.1, 0.01])
    ax.set_xticklabels(["50%", "10%", "1%"])
    ax.minorticks_off()
    ax.set_ylim(0, 0.75)
    style(ax, "most extreme share", "P(both | one)" if ax is axs[0] else "")
    ax.set_title(t, loc="left", fontsize=15, fontweight="bold", pad=8)
    ax.text(0.97, 0.9, f"correlation {r['rho']:+.2f}", transform=ax.transAxes, ha="right", fontsize=13, color=MUTED)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=13, frameon=False)
fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.27)        # legend below the panels
save(fig, "tails_comparison")
plt.close(fig)

# point 5.3: Monte Carlo convergence
MC = pickle.load(open(f"{D}/point5_mc.pkl", "rb"))
SIM = MC["sim"]
ns = np.unique(np.round(np.logspace(1, np.log10(3000), 70)).astype(int))
head(f"point 5.3 ({len(SIM)} simulated years; deviation of the running value from the {len(SIM)}-year value)")


def settle(dev, tol):
    """first N of the grid after which the running deviation stays within +-tol %"""
    out = np.abs(dev) > tol
    return int(ns[np.where(out)[0].max() + 1]) if out.any() and np.where(out)[0].max() + 1 < len(ns) else (int(ns[0]) if not out.any() else None)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.25})
for s, col, ls, lab in (("onshore mean CF", BLUE, "-", "wind: mean CF"), ("PV mean CF", PVC_, "-", "PV: mean CF"),
                        ("peak RL (GW)", INK, "-", "residual load: annual peak"), ("worst 72 h RL (GW)", BOR, "-", "residual load: worst 72 h")):
    x = SIM[s].to_numpy(float)
    dev = 100 * (np.array([x[:k].mean() for k in ns]) / x.mean() - 1)
    a1.plot(ns, dev, color=col, ls=ls, lw=2.3, label=lab)
    at = {k: dev[np.argmin(np.abs(ns - k))] for k in (30, 100, 300)}
    print(f"mean   {s:22s} {x.mean():8.3f}  dev at 30/100/300: {at[30]:+.2f} / {at[100]:+.2f} / {at[300]:+.2f} %; max |dev| "
          f"beyond 100: {np.abs(dev[ns >= 100]).max():.2f}, beyond 300: {np.abs(dev[ns >= 300]).max():.2f}; within 1% from N = {settle(dev, 1)}")
for s, col, lab, q in (("worst 72 h RL (GW)", BOR, "residual load, worst 72 h: 95th percentile of years", 0.95), ("worst 72 h RE CF", PET, "wind + PV output, worst 72 h: 5th percentile of years", 0.05),
                       ("worst 72 h PV CF", PVC_, "PV output, worst 72 h: 5th percentile of years", 0.05)):
    x = SIM[s].to_numpy(float)
    fin = np.quantile(x, q)
    dev = 100 * (np.array([np.quantile(x[:k], q) for k in ns]) / fin - 1)
    a2.plot(ns, dev, color=col, lw=2.3, label=lab)
    at = {k: dev[np.argmin(np.abs(ns - k))] for k in (30, 100, 300)}
    print(f"q{q:.2f} {s:22s} {fin:8.3f}  dev at 30/100/300: {at[30]:+.2f} / {at[100]:+.2f} / {at[300]:+.2f} %; max |dev| "
          f"beyond 100: {np.abs(dev[ns >= 100]).max():.2f}, beyond 300: {np.abs(dev[ns >= 300]).max():.2f}; within 5% from N = "
          f"{settle(dev, 5)}, within 10% from N = {settle(dev, 10)}")
for ax, e in ((a1, (1, 5)), (a2, (5, 10))):
    for v in e:
        ax.axhline(v, color=MUTED, lw=1, ls=(0, (2, 2)))
        ax.axhline(-v, color=MUTED, lw=1, ls=(0, (2, 2)))
        ax.text(3150, v, f"±{v}%", fontsize=13, color=MUTED, va="center")
    ax.axvline(300, color=GREY, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlim(10, 3000)
    ax.set_xticks([10, 30, 100, 300, 1000, 3000])
    ax.set_xticklabels(["10", "30", "100", "300", "1,000", "3,000"])
    ax.minorticks_off()
    style(ax, "simulated years", f"deviation from {len(SIM):,}-year value (%)", grid="both")
a1.set_ylim(-8, 8)
a2.set_ylim(-25, 25)
a1.legend(loc="lower right", fontsize=12, frameon=False)
a2.legend(loc="lower right", fontsize=11, frameon=False, borderaxespad=0.3)
ttl(a1, "Means of annual values")
ttl(a2, "1-in-20-year values")
fig.subplots_adjust(left=0.07, right=0.93, top=0.9, bottom=0.15)
save(fig, "monte_carlo_convergence_plot")
plt.close(fig)
print(f"\nsaved 5 figures to {OUT}")
