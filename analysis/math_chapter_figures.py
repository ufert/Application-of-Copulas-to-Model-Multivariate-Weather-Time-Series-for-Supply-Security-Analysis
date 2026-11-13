# Thesis: Mathematical Foundations, Figure 3.1 (six copulas at Kendall's tau 0.5) and the numbers of the examples
# Run: cd analysis && python math_chapter_figures.py   (seconds)
# Writes: thesis/pics/copula_families.{pdf,png}; prints the example numbers
"""
Figures and numbers for the examples of the mathematics chapter (Mathematical Foundations).

1. copula_families.pdf: samples of six bivariate copulas, all but independence with Kendall's tau = 0.5 (Gaussian and
   Student-t rho = sin(pi tau / 2), t with nu = 4, Clayton theta = 2, Gumbel theta = 2, Frank theta = 5.74), shown with
   standard normal margins; joint 5 % extremes in bordeaux, with the exact chi_L(0.05) / chi_U(0.05) in the corners.
   Prints the table of the chapter (parameter, lambda_L, lambda_U, chi_L(0.05), chi_U(0.05)) and checks the samplers
   (sample tau, sample chi).
2. Prints the numbers of the other examples (correlation of X and exp(X), the VAR(1) cross-lag example, AR(1) half-life,
   two-site EOF, number of pair-copulas for the day vector).
Run: python math_chapter_figures.py > math_chapter_figures.log (~20 s).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize, stats
from scipy.signal import lfilter
import paths  # repository folders: data, figures, external raw data (see paths.py)

OUT = paths.PICS
INK, MUTED, GRID = "#1F2328", "#5F6368", "#E3E5E8"
BLUE, BOR = "#00549F", "#A11035"
mpl.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "text.color": INK,
                     "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})
W = 6.3                                                                      # thesis text width (inches)
TAU, NU, U = 0.5, 4, 0.05
RHO = np.sin(np.pi * TAU / 2)
TH_C, TH_G = 2 * TAU / (1 - TAU), 1 / (1 - TAU)
D1 = lambda th: integrate.quad(lambda t: t / np.expm1(t), 0, th)[0] / th     # Debye function
TH_F = optimize.brentq(lambda th: 1 - 4 / th * (1 - D1(th)) - TAU, 0.1, 50)

# copula cdfs on the diagonal (exact tail values)
MVN = stats.multivariate_normal(mean=[0, 0], cov=[[1, RHO], [RHO, 1]])


def c_t(u):
    """t copula C(u, u) by integrating the normal mixture over the chi-square scale"""
    q = stats.t.ppf(u, NU)
    f = lambda w: MVN.cdf([q * np.sqrt(w / NU)] * 2) * stats.chi2.pdf(w, NU)
    return integrate.quad(f, 0, 200, limit=200, epsabs=1e-12)[0]


DIAG = {
    "Independence": lambda u: u * u,
    "Gaussian": lambda u: MVN.cdf([stats.norm.ppf(u)] * 2),
    "Student-t": c_t,
    "Clayton": lambda u: (2 * u ** -TH_C - 1) ** (-1 / TH_C),
    "Gumbel": lambda u: u ** (2 ** (1 / TH_G)),
    "Frank": lambda u: -np.log(1 + np.expm1(-TH_F * u) ** 2 / np.expm1(-TH_F)) / TH_F,
}
LAMBDA = {"Independence": (0, 0), "Gaussian": (0, 0),
          "Student-t": (2 * stats.t.cdf(-np.sqrt(NU + 1) * np.sqrt((1 - RHO) / (1 + RHO)), NU + 1),) * 2,
          "Clayton": (2 ** (-1 / TH_C), 0), "Gumbel": (0, 2 - 2 ** (1 / TH_G)), "Frank": (0, 0)}
PARAM = {"Independence": r"$\tau = 0$", "Gaussian": rf"$\rho = {RHO:.2f}$", "Student-t": rf"$\rho = {RHO:.2f},\ \nu = {NU}$",
         "Clayton": rf"$\theta = {TH_C:.0f}$", "Gumbel": rf"$\theta = {TH_G:.0f}$", "Frank": rf"$\theta = {TH_F:.2f}$"}


def chi(name, u=U):
    """exact chi_L(u) = C(u,u)/u and chi_U(u) = P(U > 1-u, V > 1-u)/u"""
    c = DIAG[name]
    return c(u) / u, (1 - 2 * (1 - u) + c(1 - u)) / u


# samplers
def sample(name, n, rng):
    if name == "Independence":
        return rng.random(n), rng.random(n)
    if name in ("Gaussian", "Student-t"):
        z = rng.standard_normal((n, 2)) @ np.linalg.cholesky([[1, RHO], [RHO, 1]]).T
        if name == "Gaussian":
            return stats.norm.cdf(z[:, 0]), stats.norm.cdf(z[:, 1])
        t = z / np.sqrt(rng.chisquare(NU, n) / NU)[:, None]
        return stats.t.cdf(t[:, 0], NU), stats.t.cdf(t[:, 1], NU)
    e = rng.exponential(size=(n, 2))
    if name == "Clayton":                                                    # Marshall-Olkin, gamma frailty
        v = rng.gamma(1 / TH_C, size=n)[:, None]
        x = (1 + e / v) ** (-1 / TH_C)
        return x[:, 0], x[:, 1]
    if name == "Gumbel":                                                     # Marshall-Olkin, positive stable frailty
        a = 1 / TH_G
        th, w = rng.uniform(0, np.pi, n), rng.exponential(size=n)
        s = (np.sin(a * th) / np.sin(th) ** (1 / a)) * (np.sin((1 - a) * th) / w) ** ((1 - a) / a)   # Kanter
        x = np.exp(-(e / s[:, None]) ** a)
        return x[:, 0], x[:, 1]
    if name == "Frank":                                                      # conditional inversion
        u, w = rng.random(n), rng.random(n)
        a, c = np.exp(-TH_F * u), np.exp(-TH_F)
        return u, -np.log(1 + w * (c - 1) / (a + w * (1 - a))) / TH_F
    raise ValueError(name)


# Figure 3.1: samples of six copulas with Kendall's tau = 0.5 and their tail values
def copula_figure():
    rng = np.random.default_rng(20261022)
    names = list(DIAG)
    print(f"Kendall's tau {TAU}: rho = {RHO:.4f}, Clayton theta = {TH_C}, Gumbel theta = {TH_G}, Frank theta = {TH_F:.4f}")
    print(f"{'family':<13}{'parameter':<20}{'lambda_L':>9}{'lambda_U':>9}{'chi_L(5%)':>11}{'chi_U(5%)':>11}"
          f"   check: sample tau, sample chi_L, chi_U (n = 400,000)")
    for nm in names:
        cl, cu = chi(nm)
        u, v = sample(nm, 400_000, rng)
        tau_s = stats.kendalltau(u[:20000], v[:20000])[0]
        cls, cus = np.mean(v[u <= U] <= U), np.mean(v[u > 1 - U] > 1 - U)
        print(f"{nm:<13}{PARAM[nm]:<20}{LAMBDA[nm][0]:>9.3f}{LAMBDA[nm][1]:>9.3f}{cl:>11.3f}{cu:>11.3f}"
              f"   {tau_s:6.3f} {cls:6.3f} {cus:6.3f}")

    fig, axs = plt.subplots(2, 3, figsize=(W, 4.55), sharex=True, sharey=True)
    lim, q = 3.6, stats.norm.ppf(1 - U)
    for ax, nm in zip(axs.ravel(), names):
        u, v = sample(nm, 2000, rng)
        x, y = stats.norm.ppf(u), stats.norm.ppf(v)
        ext = ((u <= U) & (v <= U)) | ((u > 1 - U) & (v > 1 - U))
        for k in (-q, q):
            ax.axvline(k, color=GRID, lw=0.8, zorder=0)
            ax.axhline(k, color=GRID, lw=0.8, zorder=0)
        ax.scatter(x[~ext], y[~ext], s=2.2, color=BLUE, alpha=0.35, lw=0, rasterized=True)
        ax.scatter(x[ext], y[ext], s=5.5, color=BOR, alpha=0.9, lw=0, rasterized=True)
        cl, cu = chi(nm)
        box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.2}
        ax.text(-lim + 0.2, -lim + 0.2, rf"$\chi_L$ = {cl:.2f}", fontsize=7, color=INK, ha="left", va="bottom", bbox=box, zorder=5)
        ax.text(lim - 0.2, lim - 0.2, rf"$\chi_U$ = {cu:.2f}", fontsize=7, color=INK, ha="right", va="top", bbox=box, zorder=5)
        ax.set_title(nm + ", " + PARAM[nm], fontsize=8.2, loc="left", pad=4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-3, 0, 3])
        ax.tick_params(labelsize=7, length=2)
    for ax in axs[1]:
        ax.set_xlabel(r"$\Phi^{-1}(U)$", fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\Phi^{-1}(V)$", fontsize=8)
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    for ext_ in ("pdf", "png"):
        fig.savefig(f"{OUT}/copula_families.{ext_}", dpi=300 if ext_ == "png" else 400, bbox_inches="tight")
    plt.close(fig)


# numbers of the remaining examples of the mathematics chapter
def other_numbers():
    print(f"\nrho(X, exp X), X ~ N(0,1): 1/sqrt(e - 1) = {1 / np.sqrt(np.e - 1):.4f}")
    a, b = 0.99, 0.01
    g11 = 1 / (1 - a * a)
    g12 = a * b / (1 - a * a) ** 2
    g22 = (1 + b * b * g11 + 2 * a * b * g12) / (1 - a * a)
    rng = np.random.default_rng(0)
    e = rng.standard_normal((2, 4_000_000))
    y1 = lfilter([1], [1, -a], e[0])
    y2 = lfilter([1], [1, -a], np.r_[0, b * y1[:-1]] + e[1])
    print(f"VAR(1) a = {a}, b = {b}: var y1 {g11:.2f}, cov {g12:.2f}, var y2 {g22:.2f}, corr {g12 / np.sqrt(g11 * g22):.4f}"
          f" (simulated, 4 M steps: {np.corrcoef(y1[10_000:], y2[10_000:])[0, 1]:.4f})")
    print(f"AR(1) phi 0.99: half-life {np.log(0.5) / np.log(0.99):.1f} h, variance / shock variance {1 / (1 - 0.99 ** 2):.1f}")
    r = 0.9
    print(f"two-site EOF, correlation {r}: variances sigma^2 (1 +- r), share of the first mode {(1 + r) / 2:.2f}")
    print(f"pair-copulas of a vine: day vector 24 x 647 = {24 * 647}: {24 * 647 * (24 * 647 - 1) // 2:,}")


if __name__ == "__main__":
    copula_figure()
    other_numbers()
