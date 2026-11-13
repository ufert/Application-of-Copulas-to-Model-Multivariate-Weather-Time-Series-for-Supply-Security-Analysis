# Thesis: Case Study, Mathematical Model: the unified weather model on 0.5 deg units (the thesis model)
# Run: cd analysis && python unified_model_05.py   (fit to 2000-2024 and 300 simulated years, ~20 min)
# Module: build(variant), simulate(M, n, seed, scale, margins), margin_map()
# Writes: data/unified05.pkl, data/sim05_{wind,temp,solar}.npy
"""The unified weather model on 0.5 deg units (units05.py: 203 land units of the German 0.25 deg points and 31 offshore
units).

model      100 m wind (modelled directly), 2 m temperature and GHI per unit, 2000-2024 (all hours including
           29 February): transformation (wind Box-Cox 0.25, irradiance log(GHI + 1)), linear trend and smoothed
           (calendar day, hour) climatology removed; EOF per variable (wind 2, temperature 2, irradiance 3 modes) -> one
           seasonal VAR across the 7 scores (order by BIC on a lag grid); seasonal AR remainders per unit (BIC, up to
           72 h); innovations = whole observed days within +-10 days (empirical copula of the 7 + 640 components); one
           training-year offset per simulated year; 61-day warm-up; margins mapped per (unit, month, hour) onto
           2000-2024, temperature with generalized Pareto tails beyond the outer 2 %
variants   build(variant): "unified" (the thesis model), "separate" (one VAR per variable's modes), "ar-only" (AR per
           unit on all anomaly series, no modes); simulate(scale=k) scales the anomalies, margins=False gives the
           output before the margin map
API        M = build(); paths = simulate(M, n, seed, out_prefix) -> float16 .npy memmaps (n, 8760, U), temperature in C
Run:       python unified_model_05.py -> data/sim05_{wind,temp,solar}.npy (300 years) and data/unified05.pkl
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy import stats

import spatial_common as sc
import units05 as u05
from simulate_4_16_regions import SEASONS, get_season, season_block_ids
import paths  # repository folders: data, figures, external raw data (see paths.py)

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
D = u05.D
SCR = os.environ.get("BLOCKS_TMP", paths.TMP)
VARS = ["wind", "temp", "solar"]
KMODES = {"wind": 2, "temp": 2, "solar": 3}
MAXLAG, GRID, WINDOW, BURN = 72, [1, 2, 3, 6, 12, 24, 36, 48, 72], 10, sc.BURN_DAYS
FWD = {"wind": sc.wind_fwd, "temp": lambda x: x, "solar": lambda x: np.log1p(np.clip(x, 0, None))}
INV = {"wind": sc.wind_inv, "temp": lambda y: y, "solar": lambda y: np.clip(np.expm1(y), 0, None)}
SEED = 20261018
MSTART = np.cumsum([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30])


# data 2000-2024 (all hours, incl. Feb 29)
def fit_data():
    meta = pickle.load(open(f"{D}/grid025_meta.pkl", "rb"))
    idx = pd.DatetimeIndex(meta["time"])
    W100 = np.load(f"{D}/grid025_ws100_2000-2024.npy", mmap_mode="r")
    w = np.empty((len(idx), u05.NL + u05.NO), np.float32)
    for a in range(0, len(idx), 20000):
        blk = np.asarray(W100[a:a + 20000])
        w[a:a + 20000] = np.hstack([u05.group_mean(blk[:, u05.RI, u05.CI], u05.LAND_ID), u05.group_mean(blk[:, u05.ORI, u05.OCI], u05.OFF_ID)])
    tm = pickle.load(open(f"{D}/grid025de_meta_1950-2024.pkl", "rb"))["time"]
    rows = np.where(tm.year >= 2000)[0]
    assert len(rows) == len(idx) and tm[rows[0]] == idx[0]
    t = u05.group_mean(np.asarray(np.load(f"{D}/grid025de_t2m_1950-2024.npy", mmap_mode="r")[rows]), u05.LAND_ID) - 273.15
    g = np.clip(u05.group_mean(np.asarray(np.load(f"{D}/grid025de_ghi_1950-2024.npy", mmap_mode="r")[rows]), u05.LAND_ID), 0, None)
    return idx, {"wind": w, "temp": t.astype(np.float32), "solar": g}     # temperature in C


# preprocessing
def cal_key(index):
    return (MSTART[index.month.to_numpy() - 1] + index.day.to_numpy() - 1) * 24 + index.hour.to_numpy()


# preprocessing of the hourly unit series: transformation, trend and smoothed climatology
class Prep:
    # calendar keys of the fit period and of the simulated year, sorted for fast group means
    def __init__(self, idx):
        self.idx = idx
        self.key, self.key_t = cal_key(idx), cal_key(sc.TARGET)
        self.order = np.argsort(self.key, kind="stable")
        sk = self.key[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(sk)) + 1]
        self.cnt = np.diff(np.r_[self.starts, len(sk)])
        assert len(self.starts) == 366 * 24
        self.ty = ((idx - idx[0]).total_seconds() / (365.25 * 24 * 3600)).to_numpy()
        self.year = idx.year.to_numpy()
        self.years = np.unique(self.year)

    # climatology per day of year and hour, smoothed with a +-15-day moving window
    def clim(self, Y):
        M = (np.add.reduceat(Y[self.order], self.starts, axis=0, dtype=np.float64) / self.cnt[:, None]).reshape(366, 24, -1)
        pad = np.concatenate([M[-15:], M, M[:15]], axis=0)
        c = np.concatenate([np.zeros((1,) + pad.shape[1:]), np.cumsum(pad, axis=0)], axis=0)
        return ((c[31:] - c[:-31]) / 31).reshape(366 * 24, -1)

    # anomalies of one variable and the trend and climatology needed to undo them
    def run(self, Yraw, fwd):
        y = fwd(Yraw.astype(np.float64)).astype(np.float32)
        A = y - self.clim(y)[self.key].astype(np.float32)
        tm = self.ty.mean()
        tc = self.ty - tm
        slope = np.concatenate([tc @ A[:, a:a + 256].astype(np.float64) for a in range(0, A.shape[1], 256)]) / (tc @ tc)
        intercept = A.mean(axis=0, dtype=np.float64) - slope * tm
        del A
        tr = (intercept[None] + slope[None] * self.ty[:, None]).astype(np.float32)
        cl = self.clim(y - tr)
        anom = y - cl[self.key].astype(np.float32) - tr
        offsets = pd.DataFrame(np.stack([anom[self.year == yy].mean(axis=0, dtype=np.float64) for yy in self.years]), index=self.years)
        return anom, {"offsets": offsets, "sim_mean": cl[self.key_t]}


# leading k EOF modes of the anomalies: loadings, mean and explained variance, scores and remainder
def eof(anom, k):
    mu = anom.mean(axis=0, dtype=np.float64)
    Xc = anom - mu.astype(np.float32)
    if k == 0:
        return (np.zeros((anom.shape[1], 0)), mu, np.array([0.0])), np.zeros((len(anom), 0)), Xc
    C = np.zeros((Xc.shape[1], Xc.shape[1]))
    for a in range(0, len(Xc), 20000):
        x = Xc[a:a + 20000].astype(np.float64)
        C += x.T @ x
    ev, evec = np.linalg.eigh(C / (len(Xc) - 1))
    o = np.argsort(ev)[::-1]
    E = evec[:, o[:k]]
    S = np.concatenate([Xc[a:a + 20000].astype(np.float64) @ E for a in range(0, len(Xc), 20000)])
    return (E, mu, ev[o] / ev.sum()), S, Xc - (S @ E.T).astype(np.float32)


def blockdiag(blocks):
    """seasonal VAR coefficient blocks (1 + d_i p_i, d_i) -> one block-diagonal (1 + d p, d) matrix"""
    ds = [B.shape[1] for B in blocks]
    ps = [(B.shape[0] - 1) // d for B, d in zip(blocks, ds)]
    dt, p = sum(ds), max(ps)
    out = np.zeros((1 + dt * p, dt))
    off = 0
    for B, d, pi in zip(blocks, ds, ps):
        out[0, off:off + d] = B[0]
        for lag in range(1, pi + 1):
            out[1 + (lag - 1) * dt + off:1 + (lag - 1) * dt + off + d, off:off + d] = B[1 + (lag - 1) * d:1 + lag * d]
        off += d
    return out, p


# build: data, fit, innovation pool
def build(variant="unified", gpd=True):
    assert variant in ("unified", "separate", "ar-only")
    kmodes = {v: (0 if variant == "ar-only" else KMODES[v]) for v in VARS}
    idx, REAL = fit_data()
    log(f"[{variant}] data 2000-2024: {u05.NL} land + {u05.NO} offshore units")
    P = Prep(idx)
    PRE, EOFS, S_L, R_L = {}, {}, [], []
    for v in VARS:
        anom, PRE[v] = P.run(REAL[v], FWD[v])
        EOFS[v], s, r = eof(anom, kmodes[v])
        del anom
        S_L.append(s)
        R_L.append(r)
    expl = {v: float(EOFS[v][2][:kmodes[v]].sum()) for v in VARS}
    scores = pd.DataFrame(np.hstack(S_L), index=idx)
    remain = np.hstack(R_L)
    del S_L, R_L
    NM, NS = scores.shape[1], remain.shape[1]
    log(f"[{variant}] EOF variance " + ", ".join(f"{v} {expl[v]:.1%}" for v in VARS) + f"; {NM} scores, {NS} remainders")
    VAR = {}
    if NM:
        if variant == "unified":
            VAR = sc.fit_var(scores, GRID)
        else:                                                           # separate: one VAR per variable's modes
            parts, off = [], 0
            for v in VARS:
                parts.append(sc.fit_var(scores.iloc[:, off:off + kmodes[v]], GRID))
                off += kmodes[v]
            for s in SEASONS:
                B, p = blockdiag([f[s]["B"] for f in parts])
                VAR[s] = {"B": B, "p": p, "pool": np.hstack([f[s]["pool"] for f in parts]), "index": parts[0][s]["index"]}
        log(f"[{variant}] mode VAR orders " + str({s: VAR[s]["p"] for s in SEASONS}) + "; spectral radius "
            + str({s: round(sc.spectral_radius(VAR[s]["B"], NM), 4) for s in SEASONS}))
    AR, POOL_P, DOY_P, SEAS_P = {}, [], [], []
    T = len(idx)
    dayno = np.arange(T) // 24
    feb29 = np.asarray((idx.month == 2) & (idx.day == 29))
    for s in SEASONS:
        bid = season_block_ids(idx, s)
        ok = bid != -1
        for lag in range(1, MAXLAG + 1):
            ok &= (np.roll(bid, lag) == bid) & (np.arange(T) >= lag)
        tv = np.where(ok)[0]
        if NM:
            assert np.array_equal(idx.get_indexer(VAR[s]["index"]), tv), "VAR and AR residual rows differ"
        n = len(tv)
        lagidx = tv[:, None] - np.arange(1, MAXLAG + 1)[None]
        X = np.empty((n, MAXLAG + 1))
        X[:, 0] = 1.0
        c, phi, orders = np.zeros(NS), np.zeros((NS, MAXLAG)), []
        pen = np.arange(1, MAXLAG + 2) * np.log(n)
        for j in range(NS):
            col = remain[:, j]
            X[:, 1:] = col[lagidx]
            y = col[tv].astype(np.float64)
            Q, Rq = np.linalg.qr(X)
            Qty = Q.T @ y
            rss = np.maximum(y @ y - np.cumsum(Qty ** 2), 1e-12)
            k = int(np.argmin(n * np.log(rss / n) + pen)) + 1
            beta = np.linalg.solve(Rq[:k, :k], Qty[:k])
            c[j], phi[j, :k - 1] = beta[0], beta[1:]
            orders.append(k - 1)
        pred = np.tile(c.astype(np.float32), (n, 1))
        for lag in range(1, MAXLAG + 1):
            if np.any(phi[:, lag - 1]):
                pred += remain[tv - lag] * phi[:, lag - 1].astype(np.float32)[None]
        comb = np.hstack(([VAR[s]["pool"].astype(np.float32)] if NM else []) + [remain[tv] - pred])
        del pred
        dno = dayno[tv]
        _, first, cnt = np.unique(dno, return_index=True, return_counts=True)
        full = (cnt == 24) & ~feb29[tv[first]]
        sel = (first[full][:, None] + np.arange(24)[None]).ravel()
        POOL_P.append(comb[sel].reshape(full.sum(), 24, comb.shape[1]))
        DOY_P.append(np.array([pd.Timestamp(2025, d.month, d.day).dayofyear for d in idx[tv[first[full]]]]))
        SEAS_P.append(np.full(full.sum(), s))
        AR[s] = {"c": c.astype(np.float32), "phi": phi, "orders": orders}
        del comb
        log(f"[{variant}] remainder ARs {s}: {NS} series, median order {np.median(orders):.0f}")
    pool = np.concatenate(POOL_P)
    doy, seas = np.concatenate(DOY_P), np.concatenate(SEAS_P)
    del POOL_P
    cands = []
    for d in pd.date_range("2025-01-01", periods=365, freq="D"):
        dist = np.abs(doy - d.dayofyear)
        dist = np.minimum(dist, 365 - dist)
        cands.append(np.where((dist <= WINDOW) & (seas == get_season(d.month)))[0])
    tails = None
    if gpd:
        tails = sc.fit_gpd_tails(REAL["temp"].astype(np.float64), idx)
        log(f"[{variant}] GPD temperature tails, xi: " + ", ".join(f"{s} {t} {x:+.3f}" for (s, t), x in tails["xi"].items()))
    M = {"variant": variant, "idx": idx, "REAL": REAL, "PRE": PRE, "EOFS": EOFS, "kmodes": kmodes, "expl": expl, "VAR": VAR,
         "AR": AR, "NM": NM, "NS": NS, "pool": pool, "cands": cands, "hist_s": scores.to_numpy()[-MAXLAG:],
         "hist_r": remain[-MAXLAG:].copy(), "tails": tails, "pool days": len(pool)}
    del remain
    log(f"[{variant}] built: innovation pool {len(pool)} days x {pool.shape[2]} components")
    return M


# simulation
def draws(M, n, seed):
    """the same sequence as unified_model_blocks.py: target-year days, offset years, warm-up days"""
    rng = np.random.default_rng(seed)
    draw = np.stack([rng.choice(c, size=n) for c in M["cands"]], axis=1)
    years = rng.choice(M["PRE"]["wind"]["offsets"].index.to_numpy(), size=n)
    bdraw = np.stack([rng.choice(c, size=n) for c in M["cands"][-BURN:]], axis=1)
    return draw, years, bdraw


def margin_map(mm, real, ridx, tails=None, chunk_slots=None):
    """per (unit, month, hour): ranks within the ensemble onto the 2000-2024 distribution (linear interpolation between
    order statistics); with tails (fit_gpd_tails) the outer p follows the generalized Pareto tail. In place on mm.
    Reads and writes mm once per month (the months are contiguous in the 2025 calendar), so a memory-mapped mm is not
    re-read for every hour slot; the results are identical to mapping slot by slot."""
    rm, rh = ridx.month.to_numpy(), ridx.hour.to_numpy()
    sm, sh = sc.TARGET.month.to_numpy(), sc.TARGET.hour.to_numpy()
    U = real.shape[1]
    for mo in range(1, 13):
        s = get_season(mo)
        rows = np.where(sm == mo)[0]
        r0, r1 = rows[0], rows[-1] + 1
        assert r1 - r0 == len(rows)
        mblk = np.asarray(mm[:, r0:r1, :])
        out = np.empty(mblk.shape, dtype=np.float16)
        for h in range(24):
            ss = np.where(sh[r0:r1] == h)[0]
            ref = np.sort(real[(rm == mo) & (rh == h)].astype(np.float64), axis=0)
            nr = ref.shape[0]
            blk = mblk[:, ss, :].astype(np.float64)
            flat = blk.reshape(-1, U)
            u = (flat.argsort(axis=0).argsort(axis=0) + 0.5) / flat.shape[0]
            pos = u * (nr - 1)
            i0 = np.minimum(np.floor(pos).astype(int), nr - 2)
            w = pos - i0
            x = np.take_along_axis(ref, i0, axis=0) * (1 - w) + np.take_along_axis(ref, i0 + 1, axis=0) * w
            if tails is not None:
                p = tails["p"]
                q = lambda pp: np.array([np.interp(pp, np.linspace(0, 1, nr), ref[:, c]) for c in range(U)])
                iqr = np.percentile(ref, 75, axis=0) - np.percentile(ref, 25, axis=0)
                for tail, sel, base, frac, sign in (("upper", u > 1 - p, q(1 - p), lambda uu: (uu - (1 - p)) / p, 1.0),
                                                    ("lower", u < p, q(p), lambda uu: (p - uu) / p, -1.0)):
                    if not sel.any():
                        continue
                    sig = np.array([tails["sigma"][(c, s, tail)] for c in range(U)])
                    rr, cc = np.where(sel)
                    x[rr, cc] = base[cc] + sign * iqr[cc] * stats.genpareto.ppf(frac(u[rr, cc]), tails["xi"][(s, tail)], scale=sig[cc])
            out[:, ss, :] = x.reshape(blk.shape).astype(np.float16)
        mm[:, r0:r1, :] = out


def simulate(M, n, seed=SEED, out_prefix=None, scale=1.0, margins=True, chunk=50):
    """n simulated years -> dict of float16 memmaps (n, 8760, U) at {out_prefix}_{wind,temp,solar}.npy (temperature in C)"""
    out_prefix = out_prefix or f"{SCR}/sim05_tmp_{M['variant']}_{seed}"
    draw, years, bdraw = draws(M, n, seed)
    NM, NS, L = M["NM"], M["NS"], MAXLAG
    nu = {"wind": u05.NL + u05.NO, "temp": u05.NL, "solar": u05.NL}
    mm = {v: np.lib.format.open_memmap(f"{out_prefix}_{v}.npy", mode="w+", dtype=np.float16, shape=(n, 8760, nu[v])) for v in VARS}
    offm, offr, a0, b0 = {}, {}, 0, 0
    for v in VARS:
        offm[v] = (a0, a0 + M["kmodes"][v])
        offr[v] = (b0, b0 + nu[v])
        a0 += M["kmodes"][v]
        b0 += nu[v]
    phir = {s: M["AR"][s]["phi"][:, ::-1].T.astype(np.float32).copy() for s in SEASONS}
    sims = sc.SIM_SEAS
    TT = len(sims)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        k = b - a
        S = np.zeros((k, L + TT, NM))
        R = np.zeros((k, L + TT, NS), dtype=np.float32)
        S[:, :L], R[:, :L] = M["hist_s"][None], M["hist_r"][None]
        days = np.concatenate([bdraw[a:b], draw[a:b]], axis=1)
        for dd in range(days.shape[1]):
            E = M["pool"][days[:, dd]]
            for h in range(24):
                t = dd * 24 + h
                s = sims[t]
                i = L + t
                if NM:
                    B, p = M["VAR"][s]["B"], M["VAR"][s]["p"]
                    S[:, i] = B[0] + S[:, i - p:i][:, ::-1].reshape(k, p * NM) @ B[1:] + E[:, h, :NM]
                R[:, i] = M["AR"][s]["c"] + np.einsum("nls,ls->ns", R[:, i - L:i], phir[s]) + E[:, h, NM:]
        S, R = S[:, L + 24 * BURN:], R[:, L + 24 * BURN:]
        for v in VARS:
            Ev, mu = M["EOFS"][v][0], M["EOFS"][v][1]
            an = R[..., offr[v][0]:offr[v][1]].astype(np.float64) + mu
            if M["kmodes"][v]:
                an = an + S[..., offm[v][0]:offm[v][1]] @ Ev.T
            level = scale * an + M["PRE"][v]["sim_mean"][None] + M["PRE"][v]["offsets"].loc[years[a:b]].to_numpy()[:, None, :]
            mm[v][a:b] = INV[v](level).astype(np.float16)
        del S, R
        log(f"[{M['variant']}] simulated years {a + 1}-{b} of {n}")
    if margins:
        for v in VARS:
            margin_map(mm[v], M["REAL"][v], M["idx"], M["tails"] if v == "temp" else None)
            mm[v].flush()
        log(f"[{M['variant']}] margins mapped")
    return mm, years


# close and delete memory-mapped simulation files
def remove(mm):
    for v in list(mm):
        fn = mm[v].filename
        del mm[v]
        os.remove(fn)


if __name__ == "__main__":
    M = build("unified")
    mm, years = simulate(M, 300, SEED, out_prefix=f"{D}/sim05")
    F = u05.feedin(mm, temp_in_c=True)
    log("300 years: onshore CF %.3f, offshore %.3f, PV %.3f, national temperature %.2f C" % (
        F["on"].mean(), F["off"].mean(), F["pv"].mean(), F["tnat"].mean()))
    with open(f"{D}/unified05.pkl", "wb") as f:
        pickle.dump({"units land": u05.LAND, "units offshore": u05.OFF, "eof explained": M["expl"],
                     "var p": {s: M["VAR"][s]["p"] for s in SEASONS}, "ar orders": {s: M["AR"][s]["orders"] for s in SEASONS},
                     "pool days": M["pool days"], "gpd xi": M["tails"]["xi"], "years": years, "seed": SEED,
                     "seconds": time.time() - T0}, f)
    log(f"saved data/sim05_{{wind,temp,solar}}.npy and data/unified05.pkl")
