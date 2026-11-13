# Thesis: Case Study, Choice of the Spatial Resolution (the unified model at four resolutions)
# Run: cd analysis && python unified_model_blocks.py --res=16|1|0.5|0.25   (0.7 / 2.4 / 7.9 / 31 min)
# Writes: data/blocks_{res}.pkl; the 0.25 deg run needs ~12 GB of temporary disk in tmp/
"""The unified model on spatial units of different size, for the choice of the spatial resolution.

units      --res=16    the 16 cells of the 4 x 4 partition, means over their German land points (and offshore points)
           --res=1     1 deg blocks of the 0.25 deg grid (61 land blocks)
           --res=0.5   0.5 deg blocks (203 land blocks)
           --res=0.25  the single 0.25 deg points
           offshore: the 87 German offshore points of point3_value_map.py, grouped by the same rule (wind only)
data       2000-2024 hourly: 100 m wind (ERA5 / PECD 4.2, scaled to the Global Wind Atlas; grid025_ws100), modelled
           directly; 2 m temperature and GHI at the 734 land points (grid025de_*). Unit value = mean of its points
model      transformation (wind Box-Cox 0.25, irradiance log(GHI + 1)), linear trend and smoothed (month, day, hour)
           climatology removed, EOF per variable (wind 2, temperature 2, irradiance 3 modes) -> one seasonal VAR across
           the 7 scores (BIC grid), seasonal AR remainders per unit (BIC, up to 72 h), innovations = whole observed
           days within +-10 days (empirical copula), one training-year offset per trajectory, 61-day warm-up, margins
           mapped per (unit, month, hour) onto 2000-2024 (empirical). The remainders are simulated as univariate ARs,
           trajectories in chunks, physical values in float16 memmaps (deleted at the end)
conversion the point-level conversion of point2_aggregation.py and point3_value_map.py applied to the unit means:
           onshore three turbine classes (shares of the unit's points), spread 0.36, wind scale kappa(month, local
           hour); offshore class 350 W/m2 at 120 m, spread 0.28, kappa 0.808; PV formula of conversion.py; national =
           weighted by the number of points (equal capacity per 0.25 deg point)
outputs    national onshore / offshore / PV CF and temperature for the model (N years) and the observed 2000-2024 at
           the same resolution; coherence of the onshore CF of the units by distance (model and observed)
Writes data/blocks_{res}{_smoke}.pkl. Option --smoke: a short test run.
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

import conversion as cv
import spatial_common as sc
from simulate_4_16_regions import (SEASONS, build_smoothed_climatology_ns, fit_ar_model_seasonal_one, fit_linear_trend_ns,
                                   get_season, map_climatology_ns, season_block_ids, trend_value_ns)
import paths  # repository folders: data, figures, external raw data (see paths.py)

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
RES = next((a.split("=")[1] for a in sys.argv if a.startswith("--res=")), "1")
assert RES in ("16", "1", "0.5", "0.25"), "--res=16, 1, 0.5 or 0.25"
SMOKE = "--smoke" in sys.argv
N = 30 if SMOKE else 300
TAG = f"{RES}{'_smoke' if SMOKE else ''}"
D = cv.D
SCR = os.environ.get("BLOCKS_TMP", paths.TMP)
KMODES = {"wind": 2, "temp": 2, "solar": 3}
VARS = ["wind", "temp", "solar"]
MAXLAG, GRID, WINDOW, BURN = 72, [1, 2, 3, 6, 12, 24, 36, 48, 72], 10, sc.BURN_DAYS
FWD = {"wind": sc.wind_fwd, "temp": lambda x: x, "solar": lambda x: np.log1p(np.clip(x, 0, None))}
INV = {"wind": sc.wind_inv, "temp": lambda y: y, "solar": lambda y: np.clip(np.expm1(y), 0, None)}
CHUNK = {"16": 150, "1": 100, "0.5": 50, "0.25": 20}[RES]

# units
meta = pickle.load(open(f"{D}/grid025_meta.pkl", "rb"))
LATG, LONG = np.asarray(meta["lat"]), np.asarray(meta["lon"])
PC = cv.point_classes()
ri, ci = np.asarray(PC["ri"]), np.asarray(PC["ci"])
S3 = pickle.load(open(f"{D}/point3_value_map.pkl", "rb"))["sites"]
OFFP = S3[S3.type == "offshore"]
ori = np.array([int(np.argmin(np.abs(LATG - la))) for la in OFFP.lat])
oci = np.array([int(np.argmin(np.abs(LONG - lo))) for lo in OFFP.lon])


# id of the unit of each grid point at the chosen resolution (16 cells, 1 deg, 0.5 deg or single points)
def unit_id(r, c):
    if RES == "16":
        return np.clip(((55 - LATG[r]) // 2).astype(int), 0, 3) * 10 + np.clip(((LONG[c] - 5) // 2.75).astype(int), 0, 3)
    k = {"1": 4, "0.5": 2, "0.25": 1}[RES]
    return (r // k) * 100 + c // k


land_id, off_id = unit_id(ri, ci), unit_id(ori, oci)
if RES == "16":                                                     # land points keep their cell of the 4 x 4 partition
    land_id = np.array([int(c[1]) * 10 + int(c[3]) for c in PC["points"]["cell"]])
LAND, OFF = np.unique(land_id), np.unique(off_id)
M_land = (land_id[:, None] == LAND[None]).astype(np.float64)        # (734, nL) membership
M_off = (off_id[:, None] == OFF[None]).astype(np.float64)           # (87, nO)
W_land = M_land.sum(axis=0) / len(ri)                               # equal capacity per point
W_off = M_off.sum(axis=0) / len(ori)
SHARES = np.stack([(M_land * (PC["class"] == k)[:, None]).sum(axis=0) for k in cv.CLASSES], axis=1) / M_land.sum(axis=0)[:, None]
nL, nO = len(LAND), len(OFF)
LAT_U = (M_land * LATG[ri][:, None]).sum(axis=0) / M_land.sum(axis=0)
LON_U = (M_land * LONG[ci][:, None]).sum(axis=0) / M_land.sum(axis=0)
log(f"resolution {RES}: {nL} land units, {nO} offshore units; N = {N}")


def group_mean(X, ids):
    """(T, P) point values -> (T, U) unit means, units in np.unique(ids) order (float64 sums, float32 out)"""
    order = np.argsort(ids, kind="stable")
    sid = ids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sid)) + 1]
    cnt = np.diff(np.r_[starts, len(sid)])
    return (np.add.reduceat(X[:, order], starts, axis=1, dtype=np.float64) / cnt).astype(np.float32)


# data 2000-2024 (all hours, incl. Feb 29), float32
idx = pd.DatetimeIndex(meta["time"])
T = len(idx)
W100 = np.load(f"{D}/grid025_ws100_2000-2024.npy", mmap_mode="r")
wl, wo = np.empty((T, nL), np.float32), np.empty((T, nO), np.float32)
for a in range(0, T, 20000):
    blk = np.asarray(W100[a:a + 20000])
    wl[a:a + 20000] = group_mean(blk[:, ri, ci], land_id)
    wo[a:a + 20000] = group_mean(blk[:, ori, oci], off_id)
del blk
SM = pickle.load(open(f"{D}/grid025de_meta_1950-2024.pkl", "rb"))["time"]
rows = np.where(SM.year >= 2000)[0]
assert len(rows) == T and SM[rows[0]] == idx[0], "solar / temperature time axis does not match"
tl = group_mean(np.asarray(np.load(f"{D}/grid025de_t2m_1950-2024.npy", mmap_mode="r")[rows]), land_id)
gl = np.clip(group_mean(np.asarray(np.load(f"{D}/grid025de_ghi_1950-2024.npy", mmap_mode="r")[rows]), land_id), 0, None)
COLS = {"wind": [f"L{u}" for u in LAND] + [f"O{u}" for u in OFF], "temp": [f"L{u}" for u in LAND], "solar": [f"L{u}" for u in LAND]}
REAL = {"wind": pd.DataFrame(np.hstack([wl, wo]), index=idx, columns=COLS["wind"]),
        "temp": pd.DataFrame(tl, index=idx, columns=COLS["temp"]), "solar": pd.DataFrame(gl, index=idx, columns=COLS["solar"])}
del wl, wo, tl, gl
log("unit series 2000-2024 built")

# preprocessing (numpy; same definitions as simulate_4_16_regions' *_ns functions)
MSTART = np.cumsum([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30])       # calendar incl. Feb 29 (366 days)


# calendar key (day of year and hour) of each time stamp
def cal_key(index):
    return (MSTART[index.month.to_numpy() - 1] + index.day.to_numpy() - 1) * 24 + index.hour.to_numpy()


KEY_R, KEY_T = cal_key(idx), cal_key(sc.TARGET)
ORD_R = np.argsort(KEY_R, kind="stable")
_sk = KEY_R[ORD_R]
STARTS_R = np.r_[0, np.flatnonzero(np.diff(_sk)) + 1]
CNT_R = np.diff(np.r_[STARTS_R, len(_sk)])
assert len(STARTS_R) == 366 * 24
TY = ((idx - idx[0]).total_seconds() / (365.25 * 24 * 3600)).to_numpy()
YEAR = idx.year.to_numpy()
YEARS_ALL = np.unique(YEAR)


def smoothed_clim(Y):
    """(T, U) -> (8784, U): mean per (calendar day, hour), then a centred 31-day mean, circular over the year"""
    M = (np.add.reduceat(Y[ORD_R], STARTS_R, axis=0, dtype=np.float64) / CNT_R[:, None]).reshape(366, 24, -1)
    pad = np.concatenate([M[-15:], M, M[:15]], axis=0)
    c = np.concatenate([np.zeros((1,) + pad.shape[1:]), np.cumsum(pad, axis=0)], axis=0)
    return ((c[31:] - c[:-31]) / 31).reshape(366 * 24, -1)


def preprocess_var(Yraw, fwd):
    """linear trend (fitted to the deviations from a first climatology) + smoothed climatology removed (float32 out)"""
    y = fwd(Yraw.astype(np.float64)).astype(np.float32)
    A = y - smoothed_clim(y)[KEY_R].astype(np.float32)
    tm = TY.mean()
    tc = TY - tm
    slope = np.concatenate([tc @ A[:, a:a + 256].astype(np.float64) for a in range(0, A.shape[1], 256)]) / (tc @ tc)
    intercept = A.mean(axis=0, dtype=np.float64) - slope * tm
    del A
    tr = (intercept[None] + slope[None] * TY[:, None]).astype(np.float32)
    clim = smoothed_clim(y - tr)
    anom = y - clim[KEY_R].astype(np.float32) - tr
    del y, tr
    offsets = pd.DataFrame(np.stack([anom[YEAR == yy].mean(axis=0, dtype=np.float64) for yy in YEARS_ALL]), index=YEARS_ALL)
    return anom, {"offsets": offsets, "sim_mean": clim[KEY_T]}


# leading k EOF modes of the anomalies: loadings, scores and remainder
def eof(anom, k):
    mu = anom.mean(axis=0, dtype=np.float64)
    Xc = anom - mu.astype(np.float32)
    C = np.zeros((Xc.shape[1], Xc.shape[1]))
    for a in range(0, len(Xc), 20000):
        x = Xc[a:a + 20000].astype(np.float64)
        C += x.T @ x
    ev, evec = np.linalg.eigh(C / (len(Xc) - 1))
    o = np.argsort(ev)[::-1]
    E = evec[:, o[:k]]
    S = Xc.astype(np.float64) @ E if Xc.shape[1] < 64 else np.concatenate([Xc[a:a + 20000].astype(np.float64) @ E for a in range(0, len(Xc), 20000)])
    R = Xc - (S @ E.T).astype(np.float32)
    return (E, mu, ev[o] / ev.sum()), S, R


PRE, EOFS, S_LIST, R_LIST = {}, {}, [], []
for v in VARS:
    anom, PRE[v] = preprocess_var(REAL[v].to_numpy(), FWD[v])
    EOFS[v], S_v, R_v = eof(anom, KMODES[v])
    del anom
    S_LIST.append(S_v)
    R_LIST.append(R_v)
EXPL = {v: float(EOFS[v][2][:KMODES[v]].sum()) for v in VARS}
log("EOF explained variance: " + ", ".join(f"{v} {KMODES[v]} modes {EXPL[v]:.1%}" for v in VARS))
scores = pd.DataFrame(np.hstack(S_LIST), index=idx)
remain = np.hstack(R_LIST)                                           # (T, S) float32
del S_LIST, R_LIST
NS = remain.shape[1]
NM = scores.shape[1]

# fitting
VAR = sc.fit_var(scores, GRID)
log("mode VAR orders " + str({s: VAR[s]["p"] for s in SEASONS}))


def valid_rows(bid):
    """rows with MAXLAG lags of history in the same season block (the rows of fit_ar_model_seasonal_one and fit_var)"""
    ok = bid != -1
    for lag in range(1, MAXLAG + 1):
        ok &= (np.roll(bid, lag) == bid) & (np.arange(len(bid)) >= lag)
    return np.where(ok)[0]


AR, POOL_P, POOL_DOY_P, POOL_SEAS_P = {}, [], [], []
DAYNO = np.arange(T) // 24                                           # UTC day number (index starts at 00:00)
FEB29 = np.asarray((idx.month == 2) & (idx.day == 29))
for s in SEASONS:
    bid = season_block_ids(idx, s)
    tv = valid_rows(bid)
    assert np.array_equal(idx.get_indexer(VAR[s]["index"]), tv), "VAR and AR residual rows differ"
    n = len(tv)
    lagidx = tv[:, None] - np.arange(1, MAXLAG + 1)[None]
    X = np.empty((n, MAXLAG + 1))
    X[:, 0] = 1.0
    c, phi, orders = np.zeros(NS), np.zeros((NS, MAXLAG)), []
    pen = np.arange(1, MAXLAG + 2) * np.log(n)
    for j in range(NS):                                              # BIC over nested AR(0..MAXLAG), one QR per series
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
    comb = np.hstack([VAR[s]["pool"].astype(np.float32), remain[tv] - pred])  # (n, DIM)
    del pred
    # complete days (24 valid hours, not Feb 29) -> innovation pool
    dno = DAYNO[tv]
    days, first, cnt = np.unique(dno, return_index=True, return_counts=True)
    full = (cnt == 24) & ~FEB29[tv[first]]
    sel = (first[full][:, None] + np.arange(24)[None]).ravel()
    POOL_P.append(comb[sel].reshape(full.sum(), 24, comb.shape[1]))
    dts = idx[tv[first[full]]]
    POOL_DOY_P.append(np.array([pd.Timestamp(2025, d.month, d.day).dayofyear for d in dts]))
    POOL_SEAS_P.append(np.full(full.sum(), s))
    AR[s] = {"c": c, "phi": phi, "orders": orders}
    del comb
    log(f"remainder ARs {s}: {NS} series, median order {np.median(orders):.0f}")
POOL = np.concatenate(POOL_P)
POOL_DOY, POOL_SEAS = np.concatenate(POOL_DOY_P), np.concatenate(POOL_SEAS_P)
del POOL_P
DIM = POOL.shape[2]
assert DIM == NM + NS
hist_r_keep = remain[-MAXLAG:].copy()
del remain
tdays = pd.date_range("2025-01-01", periods=365, freq="D")
CANDS = []
for d in tdays:
    dist = np.abs(POOL_DOY - d.dayofyear)
    dist = np.minimum(dist, 365 - dist)
    CANDS.append(np.where((dist <= WINDOW) & (POOL_SEAS == get_season(d.month)))[0])
rng = np.random.default_rng(20261018)
DRAW = np.stack([rng.choice(c, size=N) for c in CANDS], axis=1)                 # (N, 365)
YEARS = rng.choice(PRE["wind"]["offsets"].index.to_numpy(), size=N)
BDRAW = np.stack([rng.choice(c, size=N) for c in CANDS[-BURN:]], axis=1)       # (N, BURN)
SIMS = sc.SIM_SEAS
TT = len(SIMS)
log(f"innovation pool: {len(POOL)} complete days, {DIM} components; candidates per day {min(map(len, CANDS))}-{max(map(len, CANDS))}")

# simulation (chunks of trajectories) -> physical values in float16 memmaps
L = MAXLAG
pmax = max(VAR[s]["p"] for s in SEASONS)
assert pmax <= L
PHIR = {s: AR[s]["phi"][:, ::-1].T.astype(np.float32).copy() for s in SEASONS}   # (L, S): row 0 = lag L ... row L-1 = lag 1
CR = {s: AR[s]["c"].astype(np.float32) for s in SEASONS}
NU = {v: len(COLS[v]) for v in VARS}
MM = {v: np.lib.format.open_memmap(f"{SCR}/blocks_{TAG}_{v}.npy", mode="w+", dtype=np.float16, shape=(N, 8760, NU[v])) for v in VARS}
OFF_M = {}
off = 0
for v in VARS:
    OFF_M[v] = (off, off + KMODES[v])
    off += KMODES[v]
OFF_R = {}
off = 0
for v in VARS:
    OFF_R[v] = (off, off + NU[v])
    off += NU[v]
hist_s, hist_r = scores.to_numpy()[-L:], hist_r_keep
for a in range(0, N, CHUNK):
    b = min(N, a + CHUNK)
    n = b - a
    S = np.zeros((n, L + TT, NM))
    R = np.zeros((n, L + TT, NS), dtype=np.float32)
    S[:, :L], R[:, :L] = hist_s[None], hist_r[None]
    days = np.concatenate([BDRAW[a:b], DRAW[a:b]], axis=1)                     # (n, BURN + 365)
    for dd in range(days.shape[1]):
        E = POOL[days[:, dd]]                                                  # (n, 24, DIM)
        for h in range(24):
            t = dd * 24 + h
            s = SIMS[t]
            i = L + t
            B, p = VAR[s]["B"], VAR[s]["p"]
            S[:, i] = B[0] + S[:, i - p:i][:, ::-1].reshape(n, p * NM) @ B[1:] + E[:, h, :NM]
            R[:, i] = CR[s] + np.einsum("nls,ls->ns", R[:, i - L:i], PHIR[s]) + E[:, h, NM:]
    S, R = S[:, L + 24 * BURN:], R[:, L + 24 * BURN:]
    for v in VARS:
        Ev, mu = EOFS[v][0], EOFS[v][1]
        an = S[..., OFF_M[v][0]:OFF_M[v][1]] @ Ev.T + R[..., OFF_R[v][0]:OFF_R[v][1]] + mu
        level = an + PRE[v]["sim_mean"][None] + PRE[v]["offsets"].loc[YEARS[a:b]].to_numpy()[:, None, :]
        x = INV[v](level)
        MM[v][a:b] = (x - 273.15 if v == "temp" else x).astype(np.float16)
    del S, R
    log(f"simulated trajectories {a + 1}-{b} of {N}")


# margins: per (unit, month, hour) onto 2000-2024 (vectorized over units)
def margin_map(mm, real_df, to_store):
    rm, rh = real_df.index.month.to_numpy(), real_df.index.hour.to_numpy()
    sm, sh = sc.TARGET.month.to_numpy(), sc.TARGET.hour.to_numpy()
    real = real_df.to_numpy(dtype=np.float64)
    for mo in range(1, 13):
        for h in range(24):
            ss = np.where((sm == mo) & (sh == h))[0]
            ref = np.sort(real[(rm == mo) & (rh == h)], axis=0)               # (nref, U)
            blk = np.asarray(mm[:, ss, :], dtype=np.float64)                   # (N, nd, U)
            flat = blk.reshape(-1, blk.shape[2])
            u = (flat.argsort(axis=0).argsort(axis=0) + 0.5) / flat.shape[0]
            pos = u * (ref.shape[0] - 1)
            i0 = np.minimum(np.floor(pos).astype(int), ref.shape[0] - 2)
            w = pos - i0
            x = np.take_along_axis(ref, i0, axis=0) * (1 - w) + np.take_along_axis(ref, i0 + 1, axis=0) * w
            mm[:, ss, :] = to_store(x).reshape(blk.shape).astype(np.float16)


for v in VARS:
    margin_map(MM[v], (REAL[v] - 273.15) if v == "temp" else REAL[v], lambda x: x)
    MM[v].flush()
log("margins mapped")

# conversion
P2 = pickle.load(open(f"{D}/point2_aggregation.pkl", "rb"))
OFFC = pickle.load(open(f"{D}/point3_value_map.pkl", "rb"))["offshore conversion"]
CONV_ON = cv.Wind100Conversion(cv.CLASSES, SHARES, W_land, P2["sigma_pt"], P2["kappa"])
CONV_OFF = cv.Wind100Conversion(cv.OFFSHORE, np.ones((nO, 1)), W_off, OFFC["sigma"], np.full((12, 24), OFFC["kappa"]))
PVC = cv.calibrate_16cells()["pv"]


def convert(wind, temp_c, ghi, lmon, lhr):
    """unit values (..., T, U) -> national onshore, offshore, PV CF and temperature (..., T); onshore unit CF"""
    cells = CONV_ON.regional(wind[..., :nL], lmon, lhr)
    on = (cells * W_land).sum(axis=-1)
    offc = CONV_OFF.cf(wind[..., nL:], lmon, lhr)
    base = (cv.pv_cells(ghi, temp_c + 273.15) * W_land).sum(axis=-1)
    pv = np.roll(PVC.pr * base, PVC.lag, axis=-1) * PVC.ratio[lmon - 1, lhr]
    return {"on": on.astype(np.float32), "off": offc.astype(np.float32), "pv": pv.astype(np.float32),
            "tnat": (temp_c * W_land).sum(axis=-1).astype(np.float32)}, cells


_la, _lo = np.radians(LAT_U), np.radians(LON_U)
DIST = 6371 * 2 * np.arcsin(np.sqrt(np.sin((_la[:, None] - _la[None]) / 2) ** 2
                                    + np.cos(_la)[:, None] * np.cos(_la)[None] * np.sin((_lo[:, None] - _lo[None]) / 2) ** 2))
IU, JU = np.triu_indices(nL, 1)
BANDS = {"< 150 km": DIST[IU, JU] < 150, "150-300 km": (DIST[IU, JU] >= 150) & (DIST[IU, JU] < 300),
         "300-500 km": (DIST[IU, JU] >= 300) & (DIST[IU, JU] < 500), "> 500 km": DIST[IU, JU] >= 500}


def coherence(cells):
    """mean hourly correlation of unit onshore CF by distance band, averaged over years; (Y, T, U)"""
    out = {b: [] for b in BANDS}
    for y in range(cells.shape[0]):
        cc = np.corrcoef(cells[y].T)[IU, JU]
        for b, sel in BANDS.items():
            out[b].append(np.nanmean(cc[sel]) if sel.any() else np.nan)
    return {b: float(np.nanmean(v)) for b, v in out.items()}


LM, LH = cv.local_time(sc.TARGET)
MODEL = {k: [] for k in ("on", "off", "pv", "tnat")}
coh_m = []
for a in range(0, N, 25):
    b = min(N, a + 25)
    nat, cells = convert(np.asarray(MM["wind"][a:b], dtype=np.float32), np.asarray(MM["temp"][a:b], dtype=np.float32),
                         np.asarray(MM["solar"][a:b], dtype=np.float32), LM, LH)
    for k in MODEL:
        MODEL[k].append(nat[k])
    if a < 100:
        coh_m.append(coherence(cells))
MODEL = {k: np.concatenate(v) for k, v in MODEL.items()}
COH_M = {bnd: float(np.mean([c[bnd] for c in coh_m])) for bnd in BANDS}
keep = ~((idx.month == 2) & (idx.day == 29))
ridx = idx[keep]
RLM, RLH = cv.local_time(ridx)
rw = REAL["wind"].to_numpy(dtype=np.float32)[keep]
rt = (REAL["temp"].to_numpy(dtype=np.float32)[keep] - 273.15)
rg = REAL["solar"].to_numpy(dtype=np.float32)[keep]
nat_r, cells_r = convert(rw, rt, rg, RLM, RLH)
OBS = {k: v.reshape(25, 8760) for k, v in nat_r.items()}
COH_R = coherence(cells_r.reshape(25, 8760, nL))
log("converted: model onshore CF %.3f (observed %.3f), offshore %.3f (%.3f), PV %.3f (%.3f)" % (
    MODEL["on"].mean(), OBS["on"].mean(), MODEL["off"].mean(), OBS["off"].mean(), MODEL["pv"].mean(), OBS["pv"].mean()))
log("coherence of unit onshore CF (model / observed): " + ", ".join(f"{b} {COH_M[b]:.3f} / {COH_R[b]:.3f}" for b in BANDS))
with open(f"{D}/blocks_{TAG}.pkl", "wb") as f:
    pickle.dump({"res": RES, "N": N, "units land": LAND, "units offshore": OFF, "weights land": W_land, "weights offshore": W_off,
                 "lat": LAT_U, "lon": LON_U, "eof explained": EXPL, "var p": {s: VAR[s]["p"] for s in SEASONS},
                 "ar orders median": {s: float(np.median(AR[s]["orders"])) for s in SEASONS}, "pool days": len(POOL),
                 "model": MODEL, "observed": OBS, "coherence model": COH_M, "coherence observed": COH_R,
                 "years": YEARS, "seconds": time.time() - T0}, f)
for v in VARS:
    fn = MM[v].filename
    del MM[v]
    os.remove(fn)
log(f"saved {D}/blocks_{TAG}.pkl; memmaps removed")
