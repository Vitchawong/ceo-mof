# =============================================================================
# CEO-MOF — group ablation ladder (R1 amplitude | R2 differencing | R3 moments | R4 all)
#
# Runtime design (why this is fast):
#   1. Features are computed ONCE per record in O(n) (Savitzky–Golay + rolling
#      raw moments via cumulative filters), then sliced per window.
#   2. Per window, squared-distance matrices are built once per feature GROUP
#      (x | diff | mom); the 4 rungs are sums of these (A, A+B, A+C, A+B+C),
#      so no distance work is repeated across rungs.
#   3. MOF is fully vectorised: one row-wise sort + tie-aware rank, no Python
#      loops over points. Exact (ties counted with <=, as in the MOF definition).
#   4. All 4 rungs are scored in ONE pass over each window (load/filter once).
#   5. GPU (Colab T4) is used automatically if available: the 4 rungs are
#      sorted in a single batched call. Otherwise records run in parallel on CPU.
#   6. Results are cached per record -> safe to resume after a Colab disconnect.
#
# Colab:  !pip install -q wfdb neurokit2 joblib      (torch is preinstalled)
#         Runtime > Change runtime type > T4 GPU   (strongly recommended)
# =============================================================================
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker process

import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

# ----------------------------------------------------------------------------- CONFIG
DATA_DIR = Path("data")
OUT_DIR  = Path("results_ladder")

WINDOW_SEC  = 10.0   # <-- SET TO YOUR MOW-ECG WINDOW W (not stated in the thesis)
OVERLAP_SEC = 1.0    # thesis 3.3: overlap = 1 s
CTX_MS      = 100    # context window for local moments (~ one QRS width)
DER_MS      = 20     # Savitzky–Golay window for derivatives
TOL_SEC     = 0.15   # R-peak matching tolerance (AAMI EC57: 150 ms)

# Eq. (63) is garbled in the PDF ("scores_min - scores_min").  The threshold below is
# relative (works for every rung).  If your MOW-ECG code uses a different rule,
# replace mof_threshold() so that rung 1 reproduces the thesis numbers.
MOF_THR_FRAC = 0.2

# Global hyperparameters (thesis 5.1.1 / 4.2).  Use GLOBAL values for the ablation:
# the per-record values in Tables 1-2 were tuned for rung 1 and would bias the ladder.
PARAMS = {"mitdb": dict(irp=0.35, mri=0.35), "qtdb": dict(irp=0.40, mri=0.40)}

MITDB_EXCLUDE = {"104", "107", "217", "118", "207"}       # thesis 4.1
BEAT_SYMBOLS  = set("NLRBAaJSVrFejnE/fQ?")
QT_ANN        = "q1c"   # QT-DB reference: manual q1c (sparse) - match your thesis protocol

QUICK = True            # True: a few hard + easy records for development
QUICK_RECORDS = {"mitdb": ["100", "105", "108", "203", "228", "233"],
                 "qtdb":  ["sel100", "sel38", "sel308", "sel42", "sele0121"]}

# Feature columns:  0 x | 1 d1 | 2 d2 | 3 d3 | 4 mean | 5 var | 6 skew | 7 kurt
FEATURE_NAMES = ["x", "d1_velocity", "d2_acceleration", "d3_jerk",
                 "mean", "var", "skew", "kurt"]
GROUPS = {"x": [0], "diff": [1, 2, 3], "mom": [4, 5, 6, 7]}
RUNGS  = {"R1_amp":  ("x",),
          "R2_diff": ("x", "diff"),
          "R3_mom":  ("x", "mom"),
          "R4_all":  ("x", "diff", "mom")}

# ----------------------------------------------------------------------------- BACKEND
try:
    import torch
    USE_GPU = torch.cuda.is_available()
except Exception:          # torch missing or broken -> CPU path
    torch, USE_GPU = None, False
DEVICE = "cuda" if USE_GPU else "cpu"


# ============================================================================= FEATURES
def odd(n, lo):
    n = max(lo, int(n))
    return n if n % 2 else n + 1


def ceo_features(x, fs):
    """All 8 features for a whole record in O(n). Returns (n, 8) float32."""
    x = np.asarray(x, np.float64)
    sg = odd(DER_MS * fs / 1000, 7)
    k  = odd(CTX_MS * fs / 1000, 5)
    d1 = savgol_filter(x, sg, 3, deriv=1)             # X(1)  velocity
    d2 = savgol_filter(x, sg, 3, deriv=2)             # X(2)  acceleration
    d3 = savgol_filter(x, sg, 4, deriv=3)             # X(3)  third-order difference
    m1, m2, m3, m4 = (uniform_filter1d(x ** p, k, mode="nearest") for p in (1, 2, 3, 4))
    eps  = 1e-8 * np.var(x) + 1e-12
    var  = np.maximum(m2 - m1 ** 2, eps)                         # E[(X-mu)^2]
    skew = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / var ** 1.5         # E[(X-mu)^3]/s^3
    kurt = (m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4) / var ** 2
    return np.column_stack([x, d1, d2, d3, m1, var, skew, kurt]).astype(np.float32)


def robust_z(F):
    """Per-window robust z-score per column (median / MAD)."""
    med = np.median(F, axis=0)
    mad = np.median(np.abs(F - med), axis=0)
    return ((F - med) / (1.4826 * mad + 1e-6)).astype(np.float32)


# ============================================================================= MOF
def mof_numpy(D):
    """Exact MOF from a (W,W) squared-distance matrix.
    R[a,b] = #{o : d(a,o) <= d(a,b)};  massRatio_x(y) = R[y,x] / R[x,y];
    MOF(x) = variance over y != x  (Eqs. 54-57)."""
    W = D.shape[0]
    order = np.argsort(D, axis=1)
    S = np.take_along_axis(D, order, axis=1)
    end = np.ones(S.shape, bool)
    end[:, :-1] = S[:, 1:] != S[:, :-1]
    idx = np.where(end, np.arange(W, dtype=np.int32), np.int32(W))
    last = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]   # last index of tie group
    R = np.empty((W, W), np.float32)
    np.put_along_axis(R, order, (last + 1).astype(np.float32), axis=1)
    M = R.T / R
    n = W - 1
    s1 = M.sum(1, dtype=np.float64) - 1.0          # remove y = x (ratio 1)
    s2 = (M * M).sum(1, dtype=np.float64) - 1.0
    mu = s1 / n
    return np.maximum(s2 / n - mu * mu, 0.0)


def mof_torch(D):
    """Same as mof_numpy, batched: D is (B,W,W) float32 tensor -> (B,W) numpy."""
    B, W, _ = D.shape
    S, order = torch.sort(D, dim=-1)
    end = torch.ones_like(S, dtype=torch.bool)
    end[..., :-1] = S[..., 1:] != S[..., :-1]
    ar = torch.arange(W, device=D.device).expand(B, W, W)
    idx = torch.where(end, ar, torch.full((B, W, W), W, device=D.device, dtype=ar.dtype))
    last = torch.flip(torch.cummin(torch.flip(idx, [-1]), dim=-1).values, [-1])
    R = torch.empty_like(S)
    R.scatter_(-1, order, (last + 1).to(S.dtype))
    M = R.transpose(-1, -2) / R
    n = W - 1
    s1 = M.sum(-1, dtype=torch.float64) - 1.0
    s2 = (M * M).sum(-1, dtype=torch.float64) - 1.0
    mu = s1 / n
    return torch.clamp(s2 / n - mu * mu, min=0.0).cpu().numpy()


def sqdist_cols(Z, cols, lib=np):
    """Exact squared Euclidean distances using only `cols` (no Gram trick -> ties stay exact)."""
    W = Z.shape[0]
    D = lib.zeros((W, W), dtype=lib.float32) if lib is np else \
        torch.zeros((W, W), dtype=torch.float32, device=Z.device)
    for c in cols:
        f = Z[:, c]
        diff = f[:, None] - f[None, :]
        D += diff * diff
    return D


def rung_scores(Fw):
    """MOF scores for all rungs on one window. Returns {rung: (W,) array}."""
    Z = robust_z(Fw)
    if USE_GPU:
        Zt = torch.from_numpy(Z).to(DEVICE)
        G = {g: sqdist_cols(Zt, c, torch) for g, c in GROUPS.items()}
        D = torch.stack([sum(G[g] for g in gs) for gs in RUNGS.values()])
        S = mof_torch(D)
        return {r: S[i] for i, r in enumerate(RUNGS)}
    G = {g: sqdist_cols(Z, c) for g, c in GROUPS.items()}
    return {r: mof_numpy(sum(G[g] for g in gs)) for r, gs in RUNGS.items()}


# ============================================================================= DETECTION
# Re-implemented from thesis 3.4. If you have the original MOW-ECG code, drop its
# detect/post-process in here and keep everything else unchanged.
def mof_threshold(s):
    return s.min() + MOF_THR_FRAC * (s.max() - s.min())


def detect_window(score, sig, fs, irp, mri):
    s = score.astype(np.float64).copy()
    a = np.abs((sig - sig.mean()) / (sig.std() + 1e-12))
    thr = mof_threshold(s)
    half = max(1, int(round(irp * fs / 2)))
    peaks = []
    while True:
        i = int(np.argmax(s))
        if not np.isfinite(s[i]) or s[i] < thr:
            break
        lo, hi = max(0, i - half), min(len(s), i + half + 1)
        p = lo + int(np.argmax(a[lo:hi]))                 # morphological validation
        peaks.append(p)
        lo, hi = max(0, p - half), min(len(s), p + half + 1)
        s[lo:hi] = -np.inf                                 # refractory nulling
        a[lo:hi] = 0.0
    if len(peaks) < 2:
        return np.array(sorted(peaks), int)
    peaks = np.sort(peaks)                                 # post-processing (3.4.3)
    lb = min(mri * fs, 0.5 * np.median(np.diff(peaks)))
    keep = [peaks[0]]
    for p in peaks[1:]:
        if p - keep[-1] >= lb:
            keep.append(p)
    return np.array(keep, int)


def window_starts(n, W, O):
    if n <= W:
        return [0]
    hop = W - O
    st = list(range(0, n - W + 1, hop))
    if st[-1] + W < n:
        st.append(n - W)
    return st


# ============================================================================= EVALUATION
def match(ref, det, tol):
    i = j = tp = 0
    while i < len(ref) and j < len(det):
        d = det[j] - ref[i]
        if abs(d) <= tol:
            tp += 1; i += 1; j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    fp, fn = len(det) - tp, len(ref) - tp
    se  = 100 * tp / max(tp + fn, 1)
    ppv = 100 * tp / max(tp + fp, 1)
    der = 100 * (fp + fn) / max(tp + fn, 1)
    f1  = 100 * 2 * tp / max(2 * tp + fp + fn, 1)
    return dict(TP=tp, FP=fp, FN=fn, Se=se, PPV=ppv, DER=der, F1=f1)


# ============================================================================= DATA
def download(db):
    import wfdb
    d = DATA_DIR / db
    if not d.exists():
        ann = ["atr"] if db == "mitdb" else [QT_ANN]
        wfdb.dl_database(db, str(d), annotators=ann)
    return d


def record_list(db):
    d = download(db)
    recs = sorted(p.stem for p in d.glob("*.hea"))
    if db == "mitdb":
        recs = [r for r in recs if r not in MITDB_EXCLUDE]
    if QUICK:
        recs = [r for r in recs if r in QUICK_RECORDS[db]]
    return recs


def load(db, rec):
    import wfdb, neurokit2 as nk
    path = str(DATA_DIR / db / rec)
    r = wfdb.rdrecord(path, channels=[0])
    fs = r.fs
    x = nk.ecg_clean(r.p_signal[:, 0], sampling_rate=fs, method="neurokit")  # Butterworth + 50 Hz
    a = wfdb.rdann(path, "atr" if db == "mitdb" else QT_ANN)
    ref = np.array([s for s, sym in zip(a.sample, a.symbol) if sym in BEAT_SYMBOLS], int)
    return x, fs, ref


# ============================================================================= RUN
def run_record(db, rec):
    out = OUT_DIR / f"{db}_{rec}.json"
    if out.exists():
        return json.loads(out.read_text())
    t0 = time.time()
    x, fs, ref = load(db, rec)
    F = ceo_features(x, fs)
    W, O = int(WINDOW_SEC * fs), int(OVERLAP_SEC * fs)
    irp, mri = PARAMS[db]["irp"], PARAMS[db]["mri"]
    final = {r: [] for r in RUNGS}
    for s0 in window_starts(len(x), W, O):
        sl = slice(s0, s0 + W)
        scores = rung_scores(F[sl])
        for r, sc in scores.items():
            for p in detect_window(sc, x[sl], fs, irp, mri) + s0:
                acc = final[r]
                if not acc or p - acc[-1] >= mri * fs:     # cross-window check (3.4.3)
                    acc.append(int(p))
    tol = int(TOL_SEC * fs)
    rows = []
    for r, det in final.items():
        det = np.array(det, int)
        if db == "qtdb":                                    # sparse reference: annotated span only
            det = det[(det >= ref.min() - tol) & (det <= ref.max() + tol)]
        rows.append(dict(db=db, record=rec, rung=r, **match(ref, det, tol)))
    res = dict(rows=rows, seconds=time.time() - t0)
    OUT_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps(res))
    return res


def run_ladder(dbs=("mitdb", "qtdb")):
    jobs = [(db, rec) for db in dbs for rec in record_list(db)]
    print(f"{len(jobs)} records | backend: {'GPU' if USE_GPU else 'CPU x' + str(os.cpu_count())}")
    t0 = time.time()
    if USE_GPU:
        results = []
        for k, (db, rec) in enumerate(jobs, 1):
            results.append(run_record(db, rec))
            print(f"  [{k}/{len(jobs)}] {db}/{rec}  {results[-1]['seconds']:.1f}s")
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1, verbose=10)(delayed(run_record)(db, r) for db, r in jobs)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    return summarize(results)


def summarize(results):
    df = pd.DataFrame([row for res in results for row in res["rows"]])
    g = df.groupby(["db", "rung"])[["TP", "FP", "FN"]].sum()
    g["Se"]  = 100 * g.TP / (g.TP + g.FN)
    g["PPV"] = 100 * g.TP / (g.TP + g.FP)
    g["DER"] = 100 * (g.FP + g.FN) / (g.TP + g.FN)
    g["F1"]  = 100 * 2 * g.TP / (2 * g.TP + g.FP + g.FN)
    per_record = df.pivot_table(index=["db", "record"], columns="rung", values="F1")
    return g.round(2), per_record.round(2), df


# ============================================================================= SELF-TEST
def mof_naive(X):
    """Direct O(W^3) implementation of Eqs. 53-57 for verification."""
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    W = len(X)
    out = np.zeros(W)
    for x in range(W):
        r = [(D[y] <= D[y, x]).sum() / (D[x] <= D[x, y]).sum() for y in range(W) if y != x]
        out[x] = np.var(r)
    return out


def self_test():
    rng = np.random.default_rng(0)
    X = np.round(rng.normal(size=(300, 3)) * 2).astype(np.float32)   # many ties on purpose
    D = sqdist_cols(X, [0, 1, 2])
    ref = mof_naive(X)
    assert np.allclose(mof_numpy(D), ref, rtol=1e-4, atol=1e-6), "numpy MOF mismatch"
    if torch is not None:
        Dt = torch.from_numpy(D)[None].to(DEVICE)
        assert np.allclose(mof_torch(Dt)[0], ref, rtol=1e-4, atol=1e-6), "torch MOF mismatch"
    print("MOF self-test passed (exact match with direct Eq. 53-57, ties included)")


if __name__ == "__main__":
    self_test()
    totals, per_record, raw = run_ladder()
    print(totals)
    print(per_record)
    totals.to_csv(OUT_DIR / "ladder_totals.csv")
    per_record.to_csv(OUT_DIR / "ladder_per_record_F1.csv")
