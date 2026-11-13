# Thesis: Results, Statistical Properties against the Observed Record (49 statistics, envelope test with FDR)
# Run: cd analysis && python point5_properties.py   (after point5_model05.py)
# Writes: data/point5_properties.pkl
"""Statistical properties of the simulated weather against the observed record.

test        is the observed record a plausible draw of the model? For every statistic, the model is turned into
            pseudo-records of the length of the observed one (25 or 50 years, resampled from its independent years,
            2,000 times) and the observed value is placed in that distribution: percentile, two-sided p, z.
            Benjamini-Hochberg over all tests (false discovery rate 10 %) separates misses from multiple-testing noise.
references  2000-2024 (25 years, the fitting period) for every statistic; 1950-1999 (50 years, out of sample) for the
            statistics that do not depend on the level of temperature or irradiance. Also: do the two observed
            periods differ by more than the sampling noise of the model?
statistics  from feed-in and the 2030 system of point1_night_ev.py: levels; persistence and spectrum; spatial
            structure; coupling between variables; tail dependence (chi at 5 %); events (low output, lulls, cold);
            residual load; year-to-year variability
model       the per-year statistics of the 3,000 simulated years from data/point5_model05.pkl (point5_model05.py);
            --model=300 uses only the 300 years of unified_model_05.py
Writes data/point5_properties.pkl.
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

import point5_common as pc

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)
rng = np.random.default_rng(52)
NB = 2000
DIST = pc.cell_distance_km()
IU, JU = np.triu_indices(16, 1)
BANDS = {"< 250 km": DIST[IU, JU] < 250, "250-450 km": (DIST[IU, JU] >= 250) & (DIST[IU, JU] < 450),
         "450-700 km": (DIST[IU, JU] >= 450) & (DIST[IU, JU] < 700), "> 700 km": DIST[IU, JU] >= 700}
djf, jja = pc.SEAS == "DJF", pc.SEAS == "JJA"
THETA = pickle.load(open(f"{pc.D}/point1_night_ev.pkl", "rb"))["theta"]["with offshore"]   # critical threshold of point 1


def slot_anom(x, groups):
    """(units, 8760[, k]) -> standardised anomalies per (month, hour) slot, within each group of years"""
    groups = np.zeros(x.shape[0], int) if groups is None else np.asarray(groups)
    out = np.empty(x.shape, dtype=np.float64)
    for g in np.unique(groups):
        r = groups == g
        xs = x[r]
        o = np.empty(xs.shape)
        for m in range(1, 13):
            for h in range(24):
                sl = (pc.MON == m) & (pc.HR == h)
                blk = xs[:, sl]
                mu, sd = blk.mean(axis=(0, 1)), blk.std(axis=(0, 1))
                o[:, sl] = (blk - mu) / np.where(sd > 1e-9, sd, 1)
        out[r] = o
    return out


# row-wise correlation of two arrays (one value per year), optionally on a subset of hours
def corr_rows(a, b, mask=None):
    m = np.ones(a.shape[1], bool) if mask is None else mask
    a, b = a[:, m], b[:, m]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))


# row-wise autocorrelation at the given lag
def acf(a, lag):
    return corr_rows(a[:, :-lag], a[:, lag:])


# share of the variance in a band of periods between lo_days and hi_days (spectrum)
def band_share(a, lo_days, hi_days):
    f = np.fft.rfftfreq(a.shape[1], d=1.0)                          # cycles per hour
    p = np.abs(np.fft.rfft(a - a.mean(axis=1, keepdims=True), axis=1)) ** 2
    band = (f >= 1 / (24 * hi_days)) & (f < 1 / (24 * lo_days))
    return p[:, band].sum(axis=1) / p[:, 1:].sum(axis=1)


# counts for joint extremes: hours with u in its lowest q and, among them, v also in its lowest q
def tail_counts(u, v, mask, q=0.05):
    m = np.broadcast_to(mask, u.shape)
    a = (u <= q) & m
    return a.sum(axis=1), (a & (v <= q)).sum(axis=1)


def per_year(X, F, SYS, groups):
    """dict name -> ('mean', values) | ('ratio', (num, den)) | ('sd', values) per year; plus level flags"""
    out = {}
    on, tn, pv, cells = F["on"], F["tnat"], F["pv"], F["cells"]
    day = pv.mean(axis=0) > 0.05
    a_on, a_t, a_pv = slot_anom(on, groups), slot_anom(tn, groups), slot_anom(pv, groups)
    a_cells = slot_anom(cells, groups)
    a_tc = slot_anom(X["temp"], groups)
    a_sc = slot_anom(X["solar"], groups)
    # levels
    out["L mean onshore CF"] = ("mean", on.mean(axis=1), "wind")
    out["L hourly SD of onshore CF"] = ("mean", on.std(axis=1), "wind")
    out["L mean offshore CF"] = ("mean", F["off"].mean(axis=1), "wind")
    out["L mean PV CF"] = ("mean", pv.mean(axis=1), "level")
    out["L winter mean temperature (C)"] = ("mean", tn[:, djf].mean(axis=1), "level")
    out["L summer mean temperature (C)"] = ("mean", tn[:, jja].mean(axis=1), "level")
    # persistence and spectrum
    for lag in (6, 24, 72, 168):
        out[f"T onshore CF autocorrelation {lag} h"] = ("mean", acf(a_on, lag), "free")
    out["T temperature autocorrelation 72 h"] = ("mean", acf(a_t, 72), "free")
    dm = a_pv.reshape(len(pv), 365, 24).mean(axis=2)
    out["T PV day-to-day autocorrelation"] = ("mean", acf(dm, 1), "free")
    out["T onshore CF variance share 2-10 days"] = ("mean", band_share(a_on, 2, 10), "free")
    out["T onshore CF variance share 10-60 days"] = ("mean", band_share(a_on, 10, 60), "free")
    out["T temperature variance share 2-10 days"] = ("mean", band_share(a_t, 2, 10), "free")
    # spatial
    cc = np.stack([np.corrcoef(a_cells[i].T)[IU, JU] for i in range(len(on))])
    for b, sel in BANDS.items():
        out[f"S onshore CF correlation between cells {b}"] = ("mean", cc[:, sel].mean(axis=1), "free")
    ev = np.stack([np.linalg.eigvalsh(np.corrcoef(a_cells[i].T))[-1] / 16 for i in range(len(on))])
    out["S onshore CF first EOF share"] = ("mean", ev, "free")
    out["S temperature correlation between cells"] = ("mean", np.stack([np.corrcoef(a_tc[i].T)[IU, JU].mean() for i in range(len(on))]), "free")
    out["S irradiance correlation between cells (daylight)"] = ("mean", np.stack([np.corrcoef(a_sc[i][day].T)[IU, JU].mean()
                                                                                  for i in range(len(on))]), "free")
    # coupling
    out["X onshore CF-temperature correlation, winter"] = ("mean", corr_rows(a_on, a_t, djf), "free")
    out["X onshore CF-temperature correlation, summer"] = ("mean", corr_rows(a_on, a_t, jja), "free")
    out["X onshore CF-PV correlation, winter daylight"] = ("mean", corr_rows(a_on, a_pv, djf & day), "free")
    out["X onshore CF-PV correlation, summer daylight"] = ("mean", corr_rows(a_on, a_pv, jja & day), "free")
    a_dem, a_re = slot_anom(SYS["demand"], groups), slot_anom(SYS["re"], groups)
    out["X demand-RE feed-in correlation, winter"] = ("mean", corr_rows(a_dem, a_re, djf), "free")
    # tail dependence (chi at 5%, ratio of the pooled counts)
    u_on, u_t = pc.slot_uniform(on, groups), pc.slot_uniform(tn, groups)
    out["P cold-calm chi(5%), winter"] = ("ratio", tail_counts(u_t, u_on, djf)[::-1], "free")
    out["P calm persists 24 h chi(5%)"] = ("ratio", tail_counts(u_on[:, :-24], u_on[:, 24:], np.ones(8760 - 24, bool))[::-1], "free")
    u_dem, u_re = pc.slot_uniform(SYS["demand"], groups), pc.slot_uniform(SYS["re"], groups)
    out["P high demand-low RE chi(5%), winter"] = ("ratio", tail_counts(1 - u_dem, u_re, djf)[::-1], "free")
    uc = np.stack([pc.slot_uniform(cells[..., k], groups) for k in range(16)], axis=-1)
    for b, sel in (("< 250 km", BANDS["< 250 km"]), ("> 700 km", BANDS["> 700 km"])):
        nj, na = np.zeros(len(on)), np.zeros(len(on))
        for i, j in zip(IU[sel], JU[sel]):
            a_, j_ = tail_counts(uc[..., i], uc[..., j], np.ones(8760, bool))
            na += a_
            nj += j_
        out[f"P calm-calm chi(5%) between cells {b}"] = ("ratio", (nj, na), "free")
    # events
    L = [pc.runs(r) for r in on]
    out["E hours onshore CF < 5%"] = ("mean", (on < 0.05).sum(axis=1), "wind")
    out["E lulls >= 24 h (CF < 10%)"] = ("mean", np.array([(l >= 24).sum() for l in L]), "wind")
    out["E lulls >= 72 h"] = ("mean", np.array([(l >= 72).sum() for l in L]), "wind")
    out["E longest lull (h)"] = ("mean", np.array([l.max() if len(l) else 0 for l in L]), "wind")
    out["E year with a 7-day lull"] = ("mean", np.array([float((l >= 168).any()) for l in L]), "wind")
    out["E coldest 72-h mean temperature (C)"] = ("mean", pc.roll_min(tn, 72), "level")
    out["E hours below -10 C"] = ("mean", (tn < -10).sum(axis=1), "level")
    recf = SYS["re"] / (pc.CAP["on"] + pc.CAP["off"] + pc.CAP["pv"])
    out["E hours RE CF < 10% (wind + PV)"] = ("mean", (recf < 0.10).sum(axis=1), "free2")
    # residual load (2030 system; real 1950-1999 level-adjusted in point 1)
    rl = SYS["rl"]
    out["R annual peak RL (GW)"] = ("mean", rl.max(axis=1), "free2")
    out["R worst 72-h mean RL (GW)"] = ("mean", pc.roll_max(rl, 72), "free2")
    out["R worst 7-day mean RL (GW)"] = ("mean", pc.roll_max(rl, 168), "free2")
    out["R hours RL > 57 GW"] = ("mean", (rl > THETA).sum(axis=1), "free2")
    out["R RL energy above 50 GW (TWh)"] = ("mean", np.maximum(rl - 50, 0).sum(axis=1) / 1000, "free2")
    # year-to-year variability (SD across the years of a record)
    out["I SD of annual mean onshore CF"] = ("sd", on.mean(axis=1), "wind")
    out["I SD of winter mean temperature"] = ("sd", tn[:, djf].mean(axis=1), "free")
    out["I SD of annual peak RL"] = ("sd", rl.max(axis=1), "free2")
    out["I SD of worst 72-h mean RL"] = ("sd", pc.roll_max(rl, 72), "free2")
    return out


# value of a statistic over the record years rows: mean, standard deviation or pooled ratio
def record_value(kind, vals, rows):
    if kind == "mean":
        return vals[rows].mean()
    if kind == "sd":
        return vals[rows].std(ddof=1)
    num, den = vals
    return num[rows].sum() / den[rows].sum()


MODEL300 = "--model=300" in sys.argv
PY = {}
for name in ("real", "model"):
    if name == "model" and not MODEL300:
        M3 = pickle.load(open(f"{pc.D}/point5_model05.pkl", "rb"))
        PY["model"] = M3["per year"]
        log(f"model: per-year statistics of {M3['years']} simulated years from point5_model05.pkl ({len(PY['model'])} statistics)")
        continue
    X = pc.load_real() if name == "real" else pc.load_model()
    F = pc.feedin(X)
    SYS = pc.system_series()[name]
    PY[name] = per_year(X, F, SYS, pc.REAL_PERIOD if name == "real" else None)
    del X, F
    log(f"{name}: per-year statistics done ({len(PY[name])} statistics)")

# "wind" and "free": both periods; "free2": residual load / RE (point 1 adjusted 1950-1999 temperature and irradiance:
# comparable for both periods); "level": 2000-2024 only
REFS = {"2000-2024": pc.YEARS >= 2000, "1950-1999": pc.YEARS < 2000}
ROWS = []
for st, (kind, vals, flag) in PY["model"].items():
    rkind, rvals, _ = PY["real"][st]
    nmod = len(vals) if kind != "ratio" else len(vals[0])
    dist = {}
    for ref, sel in REFS.items():
        if flag == "level" and ref == "1950-1999":
            continue
        rows_real = np.where(sel)[0]
        v_real = record_value(rkind, rvals, rows_real)
        n = len(rows_real)
        idx = rng.integers(0, nmod, (NB, n))
        sims = np.array([record_value(kind, vals, r) for r in idx])
        dist[ref] = sims
        pct = (np.sum(sims < v_real) + 0.5 * np.sum(sims == v_real)) / NB
        ROWS.append({"statistic": st, "group": st.split()[0], "reference": ref, "real": v_real, "model mean": sims.mean(),
                     "model 5%": np.quantile(sims, 0.05), "model 95%": np.quantile(sims, 0.95), "percentile": pct,
                     "p two-sided": max(2 * min(pct, 1 - pct), 1 / NB), "z": (v_real - sims.mean()) / sims.std(ddof=1),
                     "model / real": sims.mean() / v_real if v_real != 0 else np.nan})
    if "1950-1999" in dist:                                                         # do the real periods differ?
        r_in = record_value(rkind, rvals, np.where(REFS["2000-2024"])[0])
        r_out = record_value(rkind, rvals, np.where(REFS["1950-1999"])[0])
        ROWS[-1]["periods differ (z)"] = (r_in - r_out) / np.sqrt(dist["2000-2024"].var() + dist["1950-1999"].var())
T = pd.DataFrame(ROWS)
# Benjamini-Hochberg, false discovery rate 10%, over all tests
p = T["p two-sided"].to_numpy()
order = np.argsort(p)
thr = 0.10 * np.arange(1, len(p) + 1) / len(p)
passed = p[order] <= thr
k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
T["significant (FDR 10%)"] = False
T.loc[T.index[order[:k]], "significant (FDR 10%)"] = True
T["status"] = np.where(T["significant (FDR 10%)"], "miss", np.where(T["p two-sided"] < 0.05, "nominal p<0.05 only", "consistent"))

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 52)
show = ["statistic", "reference", "real", "model mean", "model 5%", "model 95%", "percentile", "p two-sided", "status", "periods differ (z)"]
print(T[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\n{len(T)} tests ({T.statistic.nunique()} statistics): consistent {int((T.status == 'consistent').sum())}, nominal p<0.05 only "
      f"{int((T.status == 'nominal p<0.05 only').sum())}, misses (FDR 10%) {int((T.status == 'miss').sum())}")
for ref in REFS:
    pp = T.loc[T.reference == ref, "percentile"]
    ks = stats.kstest(pp, "uniform")
    print(f"{ref}: percentiles {len(pp)} tests; in the outer 5% on each side {((pp < 0.05) | (pp > 0.95)).mean():.0%} (10% expected); "
          f"KS vs uniform D {ks.statistic:.2f}")
print("by group (share consistent): " + ", ".join(f"{g}: {(d.status == 'consistent').mean():.0%} of {len(d)}" for g, d in T.groupby("group")))
pdiff = T.dropna(subset=["periods differ (z)"])
print(f"real periods differ by more than 2 model SD in {int((pdiff['periods differ (z)'].abs() > 2).sum())} of {len(pdiff)} statistics: "
      + ", ".join(f"{r.statistic} ({r['periods differ (z)']:+.1f})" for _, r in pdiff[pdiff['periods differ (z)'].abs() > 2].iterrows()))
with open(f"{pc.D}/point5_properties.pkl", "wb") as f:
    pickle.dump({"table": T, "per year": {k: {s: (v[0], v[2]) for s, v in PY[k].items()} for k in PY}}, f)
log(f"saved {pc.D}/point5_properties.pkl")
