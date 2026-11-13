# Thesis: Mathematical Foundations, tail values of the copula families at u = 0.05 (exact formulas)
# Run: cd analysis && python math_copula_tail_table.py   (seconds, prints the values)
"""Numbers of the copula-family example of the mathematics chapter (Mathematical Foundations).

Five bivariate copulas with Kendall's tau = 0.5: Gaussian and Student-t (nu = 4) with rho = sin(pi tau / 2), Clayton and
Gumbel with theta = 2, Frank with theta from the Debye-function relation. Prints the tail dependence coefficients and the
tail probabilities chi_L(u) = C(u, u) / u and chi_U(u) = (1 - 2(1 - u) + C(1 - u, 1 - u)) / u at u = 0.05, all from the
copula formulas (no simulation).
"""
import numpy as np
from scipy import integrate, optimize, stats

TAU, U = 0.5, 0.05


# lower and upper tail probabilities chi_L(u) and chi_U(u) of a copula C at level u
def chi(C, u=U):
    return C(u, u) / u, (1 - 2 * (1 - u) + C(1 - u, 1 - u)) / u


rho = np.sin(np.pi * TAU / 2)
R = [[1, rho], [rho, 1]]
gauss = lambda a, b: stats.multivariate_normal([0, 0], R).cdf([stats.norm.ppf(a), stats.norm.ppf(b)])
nu = 4
student = lambda a, b: stats.multivariate_t([0, 0], R, df=nu).cdf([stats.t.ppf(a, nu), stats.t.ppf(b, nu)])
lam_t = 2 * stats.t.cdf(-np.sqrt(nu + 1) * np.sqrt((1 - rho) / (1 + rho)), nu + 1)

th = 2 * TAU / (1 - TAU)                                    # Clayton: tau = theta / (theta + 2)
clayton = lambda a, b: (a ** -th + b ** -th - 1) ** (-1 / th)
tg = 1 / (1 - TAU)                                          # Gumbel: tau = 1 - 1 / theta
gumbel = lambda a, b: np.exp(-((-np.log(a)) ** tg + (-np.log(b)) ** tg) ** (1 / tg))
debye = lambda t: integrate.quad(lambda s: s / np.expm1(s), 0, t)[0] / t
tf = optimize.brentq(lambda t: 1 - 4 / t * (1 - debye(t)) - TAU, 0.1, 50)
frank = lambda a, b: -1 / tf * np.log(1 + np.expm1(-tf * a) * np.expm1(-tf * b) / np.expm1(-tf))

rows = [("Gaussian", f"rho = {rho:.2f}", 0, 0, gauss),
        ("Student-t", f"rho = {rho:.2f}, nu = {nu}", lam_t, lam_t, student),
        ("Clayton", f"theta = {th:.2f}", 2 ** (-1 / th), 0, clayton),
        ("Gumbel", f"theta = {tg:.2f}", 0, 2 - 2 ** (1 / tg), gumbel),
        ("Frank", f"theta = {tf:.2f}", 0, 0, frank)]
print(f"{'copula':10s} {'parameter':22s} lam_L  lam_U  chi_L({U})  chi_U({U})")
for name, par, ll, lu, C in rows:
    cl, cu = chi(C)
    print(f"{name:10s} {par:22s} {ll:5.2f}  {lu:5.2f}  {cl:10.2f}  {cu:10.2f}")
