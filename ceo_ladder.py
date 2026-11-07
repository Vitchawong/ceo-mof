"""
CEO-MOF group ablation ladder on top of the ORIGINAL MOW-ECG pipeline.

    R1_amp   [x]                       (must reproduce mow_ecg_clean.mow_ecg exactly)
    R2_diff  x + differences  (cols 1-3)
    R3_mom   x + moments      (cols 4-7)
    R4_all   all 8 columns

Feature columns: 0 x | 1 X(1) velocity | 2 X(2) acceleration | 3 X(3) third difference
                 4 local mean | 5 local variance | 6 local skewness | 7 local kurtosis

The only thing that changes between rungs is the input to MOF.
Preprocessing, windows, candidate search, filtering, merging and evaluation all come from
mow_ecg_clean.py, which is NOT modified. For each rung, mc.mow_ecg() itself runs with
pymof's MOF swapped (inside this process only) for a stand-in that hands back
precomputed scores for that rung. The stand-in checks that every window it gets is
exactly the expected slice of the filtered signal.

Speed:
  * features are computed once per record, on the same filtered signal pymof sees
  * per window, squared distances are built once per feature group (x | diff | mom)
    and summed for each rung
  * vectorised MOF, bit-identical to pymof (same tie rule, same float operations)
  * all rungs are scored in one pass per window; records run in parallel (joblib)
  * scores and per-record results are cached, so a run can be resumed

Usage (PowerShell: put lists in quotes)
  python ceo_ladder.py --verify                      # MOF tests + R1 == mow_ecg on 6 records
  python ceo_ladder.py                               # quick subset, default features
  python ceo_ladder.py --deriv gradient              # variant a
  python ceo_ladder.py --ctx-ms 30                   # variant b
  python ceo_ladder.py --scale minmax                # variant c
  python ceo_ladder.py --quick-variants              # default + all variants above, quick subset
  python ceo_ladder.py --records all                 # full MIT-BIH (48 records)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker process

import argparse, contextlib, io, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mow_ecg_clean as mc

# ----------------------------------------------------------------------------- CONFIG
DATA_DIR = HERE / "mitdb"
OUT_DIR = HERE / "results_ladder"

ALL_RECORDS = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112",
               "113", "114", "115", "116", "117", "118", "119", "121", "122", "123", "124", "200",
               "201", "202", "203", "205", "207", "208", "209", "210", "212", "213", "214", "215",
               "217", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234"]
QUICK_RECORDS = ["100", "105", "108", "203", "228", "233"]
EXCLUDE = {"thesis": {"104", "107", "118", "207", "217"},      # primary
           "aami": {"102", "104", "107", "217"}}               # paced records (ANSI/AAMI EC57)
BEATS = mc.BEAT_SYMBOLS

GLOBAL = (0.6, 0.3)                                            # IRP, MRI for all rungs
IRPS = np.round(np.arange(0.25, 0.701, 0.05), 2)               # same 80-setting grid as
MRIS = np.round(np.arange(0.20, 0.551, 0.05), 2)               # compare_representations.py
TOL_MS = 50

FEATURE_NAMES = ["x", "d1", "d2", "d3", "mean", "var", "skew", "kurt"]
# Every column has its own squared-distance matrix P[c]. A group's matrix is the sum of its
# columns' matrices in the listed order; a rung's matrix is the sum of its groups in order.
# (This gives exactly the same floats as the earlier per-group sqdist, so cached R2-R4 stay valid.)
GROUPS = {"x": [0], "diff": [1, 2, 3], "mom": [4, 5, 6, 7],
          "vel": [1], "acc": [2], "jerk": [3], "velacc": [1, 2],
          "mean": [4], "var": [5], "skew": [6], "kurt": [7],
          "meanvar": [4, 5], "meanvarskew": [4, 5, 6]}
RUNG_DEFS = {"R1_amp": ("x",), "R2_diff": ("x", "diff"), "R3_mom": ("x", "mom"),
             "R4_all": ("x", "diff", "mom"),
             # differencing sub-ladder
             "A_vel": ("x", "vel"), "A_acc": ("x", "acc"), "A_jerk": ("x", "jerk"),
             "A_vel_acc": ("x", "velacc"),
             # moments sub-ladder
             "A_mean": ("x", "mean"), "A_var": ("x", "var"), "A_skew": ("x", "skew"),
             "A_kurt": ("x", "kurt"), "A_mean_var": ("x", "meanvar"),
             "A_mean_var_skew": ("x", "meanvarskew"),
             # used only by --check-subdiff / --check-submom
             "A_only_scaled": ("x",), "A_vel_acc_jerk_seq": ("x", "vel", "acc", "jerk"),
             "A_mom_seq": ("x", "mean", "var", "skew", "kurt")}
LADDERS = {"main": ["R1_amp", "R2_diff", "R3_mom", "R4_all"],
           "subdiff": ["R1_amp", "A_vel", "A_acc", "A_jerk", "A_vel_acc", "R2_diff"],
           "submom": ["R1_amp", "A_mean", "A_var", "A_skew", "A_kurt", "A_mean_var",
                      "A_mean_var_skew", "R3_mom"]}
# moment rungs depend on context length and scaling only (not on the derivative filter)
MOM_RUNGS = {"A_mean", "A_var", "A_skew", "A_kurt", "A_mean_var", "A_mean_var_skew", "A_mom_seq"}
RUNGS = {r: RUNG_DEFS[r] for r in LADDERS["main"]}
DEFAULT_CFG = dict(deriv="savgol", sg_ms=20, ctx_ms=100, scale="robust")


def rung_key(rung, cfg):
    """Cache key: only the settings that can change this rung's scores."""
    d = cfg["deriv"] + (f"{cfg['sg_ms']:g}" if cfg["deriv"] == "savgol" else "")
    c, sc = f"ctx{cfg['ctx_ms']:g}", cfg["scale"]
    if rung == "R1_amp":
        return "R1_amp"
    if rung in ("R2_diff", "R3_mom", "R4_all"):
        return {"R2_diff": f"R2_{d}_{sc}", "R3_mom": f"R3_{c}_{sc}", "R4_all": f"R4_{d}_{c}_{sc}"}[rung]
    if rung in MOM_RUNGS:
        return f"{rung}_{c}_{sc}"                          # moments sub-ladder
    return f"{rung}_{d}_{sc}"                              # differencing sub-ladder


def cfg_tag(cfg):
    d = cfg["deriv"] + (f"{cfg['sg_ms']:g}" if cfg["deriv"] == "savgol" else "")
    return f"{d}_ctx{cfg['ctx_ms']:g}_{cfg['scale']}"


# ----------------------------------------------------------------------------- DATA
def load(rec):
    """MLII lead (lead 0 if absent), full record, raw ADC values; beat-only reference."""
    path = str(DATA_DIR / rec)
    r = wfdb.rdrecord(path)
    a = wfdb.rdann(path, "atr")
    lead = r.sig_name.index("MLII") if "MLII" in r.sig_name else 0
    sig = r.adc()[:, lead].astype(float)
    ref = np.array([s for s, c in zip(a.sample, a.symbol) if c in BEATS])
    return sig, ref, r.fs


def mow_windows(n, fs):
    """The windows mc.mow_ecg visits, in order (copied loop bounds; checked at replay)."""
    window = 3 * fs
    stride = window - 1 * fs
    overlap = window - stride
    out = []
    for start in range(-overlap, n, stride):
        lo, hi = max(0, start), min(n, start + window)
        if hi > lo:
            out.append((lo, hi))
    return out


# ----------------------------------------------------------------------------- FEATURES
def odd(n, lo):
    n = max(lo, int(n))
    return n if n % 2 else n + 1


def local_moments(x, k, chunk=65536):
    """Centred moments over k samples (edges padded with the end value):
    mean, variance, skewness, kurtosis (non-excess). Exact two-pass per position."""
    h = k // 2
    xp = np.pad(x, h, mode="edge")
    out = np.empty((len(x), 4))
    for s in range(0, len(x), chunk):
        e = min(len(x), s + chunk)
        V = np.lib.stride_tricks.sliding_window_view(xp[s:e + 2 * h], k)
        mu = V.mean(1)
        C = V - mu[:, None]
        m2 = (C ** 2).mean(1)
        m3 = (C ** 3).mean(1)
        m4 = (C ** 4).mean(1)
        ok = m2 > 1e-12
        safe = np.where(ok, m2, 1.0)
        out[s:e, 0] = mu
        out[s:e, 1] = m2
        out[s:e, 2] = np.where(ok, m3 / safe ** 1.5, 0.0)
        out[s:e, 3] = np.where(ok, m4 / safe ** 2, 0.0)
    return out


def ceo_features(x, fs, cfg):
    """All 8 columns for a whole filtered record. Returns (n, 8) float64."""
    x = np.asarray(x, np.float64)
    if cfg["deriv"] == "savgol":
        sg = odd(cfg["sg_ms"] * fs / 1000, 7)
        d1 = savgol_filter(x, sg, 3, deriv=1)
        d2 = savgol_filter(x, sg, 3, deriv=2)
        d3 = savgol_filter(x, sg, 4, deriv=3)
    elif cfg["deriv"] == "gradient":
        d1 = np.gradient(x)
        d2 = np.gradient(d1)
        d3 = np.gradient(d2)
    else:
        raise ValueError(cfg["deriv"])
    mom = local_moments(x, odd(cfg["ctx_ms"] * fs / 1000, 3))
    return np.column_stack([x, d1, d2, d3, mom])


def scale_window(F, how):
    if how == "robust":
        med = np.median(F, axis=0)
        mad = np.median(np.abs(F - med), axis=0)
        return (F - med) / (1.4826 * mad + 1e-6)
    if how == "minmax":
        lo, rng = F.min(0), np.ptp(F, axis=0)
        return np.where(rng > 0, (F - lo) / np.where(rng > 0, rng, 1.0), 0.0)
    raise ValueError(how)


# ----------------------------------------------------------------------------- MOF
def mof_from_dist(D):
    """MOF from a (W, W) distance matrix (any monotone function of Euclidean distance).
    N[a, b] = #{o : d(a,o) <= d(a,b)};  score(x) = var over y != x of N[y,x] / N[x,y].
    Same integer counts, float division and np.var order as pymof -> bit-identical."""
    W = D.shape[0]
    order = np.argsort(D, axis=1)
    S = np.take_along_axis(D, order, axis=1)
    end = np.ones(S.shape, bool)
    end[:, :-1] = S[:, 1:] != S[:, :-1]
    idx = np.where(end, np.arange(W), W)
    last = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]      # last position of tie group
    N = np.empty((W, W), np.float64)
    np.put_along_axis(N, order, (last + 1).astype(np.float64), axis=1)
    M = N.T / N
    return M[~np.eye(W, dtype=bool)].reshape(W, W - 1).var(axis=1)


def sqdist(Z, cols):
    D = np.zeros((Z.shape[0], Z.shape[0]))
    for c in cols:
        f = Z[:, c]
        diff = f[:, None] - f[None, :]
        D += diff * diff
    return D


def column_dist(Z, c):
    """Squared distance matrix of one feature column."""
    f = Z[:, c]
    diff = f[:, None] - f[None, :]
    return diff * diff


def group_dist(P, cols):
    """Sum of per-column matrices in order (same floats as sqdist(Z, cols))."""
    D = P[cols[0]].copy()
    for c in cols[1:]:
        D += P[c]
    return D


def rung_dist(P, groups):
    """Sum of group matrices in order (same floats as the earlier sum(G[g] for g in gs))."""
    D = group_dist(P, GROUPS[groups[0]])
    for g in groups[1:]:
        D += group_dist(P, GROUPS[g])
    return D


def rung_scores(win, Fw, cfg, rungs):
    """MOF scores for the requested rungs on one window.
    R1 uses |x_a - x_b| of the unscaled window: exactly what pymof sees, so R1 equals
    MOW-ECG by construction (MOF depends only on distance ranks)."""
    out = {}
    if "R1_amp" in rungs:
        out["R1_amp"] = mof_from_dist(np.abs(win[:, None] - win[None, :]))
    rest = [r for r in rungs if r != "R1_amp"]
    if rest:
        Z = scale_window(Fw, cfg["scale"])
        cols = sorted({c for r in rest for g in RUNG_DEFS[r] for c in GROUPS[g]})
        P = {c: column_dist(Z, c) for c in cols}              # one matrix per feature
        for r in rest:
            out[r] = mof_from_dist(rung_dist(P, RUNG_DEFS[r]))
    return out


# ----------------------------------------------------------------------------- STAND-IN
class _Replay:
    """Stand-in for pymof.MOF inside mc.mow_ecg: returns precomputed scores, window by window."""
    queue = None

    def fit(self, X, window=None):
        lo, hi, sc, ref = _Replay.queue.pop(0)
        assert X.shape == (hi - lo, 1) and np.array_equal(X[:, 0], ref[lo:hi]), "window mismatch"
        self.decision_scores_ = sc
        return self


@contextlib.contextmanager
def replayed(filtered, windows, scores, signal):
    """Run mc.mow_ecg with MOF replaced by the given scores. mc.preprocess is also
    memoised for this signal (same function, called once instead of once per setting)."""
    orig_mof, orig_pre = mc.MOF, mc.preprocess

    def pre(sig, fs):
        assert sig is signal, "unexpected signal"
        return filtered.copy()

    _Replay.queue = [(lo, hi, s, filtered) for (lo, hi), s in zip(windows, scores)]
    mc.MOF, mc.preprocess = _Replay, pre
    try:
        yield
        assert not _Replay.queue, "not all windows used"
    finally:
        mc.MOF, mc.preprocess = orig_mof, orig_pre
        _Replay.queue = None


def detect_with_scores(sig, filtered, windows, scores, fs, irp, mri):
    with replayed(filtered, windows, scores, sig):
        return mc.mow_ecg(sig, fs, irp=irp, mri=mri)


# ----------------------------------------------------------------------------- METRICS
def metrics(tp, fp, fn):
    tp, fp, fn = np.asarray(tp, float), np.asarray(fp, float), np.asarray(fn, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(Se=100 * tp / (tp + fn), PPV=100 * tp / (tp + fp),
                    DER=100 * (fp + fn) / (tp + fn),              # thesis Eq. 66
                    DER_TP=100 * (fp + fn) / tp,                   # old compare_representations
                    F1=100 * 2 * tp / (2 * tp + fp + fn))


# ----------------------------------------------------------------------------- PER RECORD
def score_file(rung, cfg, rec):
    return OUT_DIR / "scores" / rung_key(rung, cfg) / f"{rec}.npz"


def record_scores(rec, cfg, sig=None, fs=None, filtered=None, timing=None, rungs=None):
    """Scores of the requested rungs for every window. Cached per rung and per the
    settings that affect that rung, so unchanged rungs are never recomputed."""
    rungs = list(rungs or RUNGS)
    need = [r for r in rungs if not score_file(r, cfg, rec).exists()]
    if need:
        if filtered is None:
            sig, _, fs = load(rec)
            filtered = mc.preprocess(sig, fs)
        windows = mow_windows(len(filtered), fs)
        F = ceo_features(filtered, fs, cfg) if any(r != "R1_amp" for r in need) else None
        acc = {r: [] for r in need}
        t0 = time.time()
        for lo, hi in windows:
            sc = rung_scores(filtered[lo:hi], None if F is None else F[lo:hi], cfg, need)
            for r in need:
                acc[r].append(sc[r])
        if timing is not None:
            timing["mof_seconds"] = time.time() - t0
        for r in need:
            f = score_file(r, cfg, rec)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, scores=np.concatenate(acc[r]))
    return {r: np.load(score_file(r, cfg, rec))["scores"] for r in rungs}


def split(flat, windows):
    cuts = np.cumsum([hi - lo for lo, hi in windows])[:-1]
    return np.split(flat, cuts)


def grid_file(rung, cfg, rec):
    return OUT_DIR / "grid" / rung_key(rung, cfg) / f"{rec}.csv"


def run_record(rec, cfg, rungs=None):
    """All 80 IRP/MRI settings for every rung of one record. Cached per rung."""
    rungs = list(rungs or RUNGS)
    t0 = time.time()
    todo = [r for r in rungs if not grid_file(r, cfg, rec).exists()]
    t = dict(record=rec, mof_seconds=0.0, detect_grid_s=0.0, rungs_computed=",".join(todo) or "-")
    if todo:
        sig, ref, fs = load(rec)
        filtered = mc.preprocess(sig, fs)
        windows = mow_windows(len(filtered), fs)
        timing = {}
        flat = record_scores(rec, cfg, sig, fs, filtered, timing, rungs=todo)
        t["mof_seconds"] = timing.get("mof_seconds", 0.0)
        t1 = time.time()
        for rung in todo:
            scores = split(flat[rung], windows)
            rows = []
            for irp in IRPS:
                for mri in MRIS:
                    det = detect_with_scores(sig, filtered, windows, scores, fs, float(irp), float(mri))
                    m = mc.evaluate(ref, det, fs, TOL_MS)
                    rows.append(dict(record=rec, rung=rung, irp=irp, mri=mri,
                                     TP=m["TP"], FP=m["FP"], FN=m["FN"]))
            f = grid_file(rung, cfg, rec)
            f.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(f, index=False)
        t["detect_grid_s"] = time.time() - t1
    t["total_s"] = time.time() - t0
    df = pd.concat([pd.read_csv(grid_file(r, cfg, rec), dtype={"record": str}) for r in rungs],
                   ignore_index=True)
    return df, t


# ----------------------------------------------------------------------------- SUMMARY
def summarise(df, records, cfg, exclusion, rungs=None):
    """Gross totals per rung at GLOBAL, and at each rung's own best global setting."""
    keep = [r for r in records if r not in EXCLUDE[exclusion]]
    d = df[df.record.isin(keep)]
    g = d.groupby(["rung", "irp", "mri"])[["TP", "FP", "FN"]].sum().reset_index()
    g = g.assign(**metrics(g.TP, g.FP, g.FN))
    at_global = g[np.isclose(g.irp, GLOBAL[0]) & np.isclose(g.mri, GLOBAL[1])].set_index("rung")
    best = g.loc[g.groupby("rung").F1.idxmax()].set_index("rung")
    order = list(rungs or RUNGS)
    cols = ["irp", "mri", "TP", "FP", "FN", "Se", "PPV", "DER", "DER_TP", "F1"]

    def per_record(sel):
        x = d.merge(sel.reset_index()[["rung", "irp", "mri"]], on=["rung", "irp", "mri"])
        x = x.assign(**metrics(x.TP, x.FP, x.FN))
        return x.pivot(index="record", columns="rung", values="F1")[order]

    return dict(n=len(keep), global_=at_global.loc[order, cols], best=best.loc[order, cols],
                pr_global=per_record(at_global), pr_best=per_record(best))


def print_summary(S, title):
    fmt = lambda t: t.to_string(float_format=lambda v: f"{v:.2f}")
    print(f"\n=== {title}: {S['n']} records ===")
    print(f"-- all rungs at IRP {GLOBAL[0]} / MRI {GLOBAL[1]} (gross totals) --")
    print(fmt(S["global_"]))
    print("-- each rung at its own best global setting (max gross F1 on these records) --")
    print(fmt(S["best"]))
    print(f"-- per-record F1 at {GLOBAL[0]}/{GLOBAL[1]} --")
    print(fmt(S["pr_global"]))
    print("-- per-record F1 at each rung's own best global setting --")
    print(fmt(S["pr_best"]))


def run_config(records, cfg, workers, rungs=None, folder=None):
    from joblib import Parallel, delayed
    tag = cfg_tag(cfg)
    print(f"\n##### config {tag}: {len(records)} records, {workers} workers", flush=True)
    t0 = time.time()
    gen = Parallel(n_jobs=workers, return_as="generator_unordered")(
        delayed(run_record)(r, cfg, rungs) for r in records)
    res = []
    for k, (df_r, t) in enumerate(gen, 1):
        res.append((df_r, t))
        print(f"  [{k}/{len(records)}] {t['record']} done in {t['total_s']:.0f} s "
              f"(MOF {t['mof_seconds']:.0f} s, grid {t['detect_grid_s']:.0f} s, computed: {t['rungs_computed']})",
              flush=True)
    wall = time.time() - t0
    df = pd.concat([r[0] for r in res], ignore_index=True)
    print(f"config wall time {wall:.0f} s", flush=True)
    folder = Path(folder or OUT_DIR / tag)
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / "grid_all_records.csv", index=False)
    for exc in ("thesis", "aami"):
        S = summarise(df, records, cfg, exc, rungs)
        print_summary(S, f"{tag} | exclusion {exc}")
        S["global_"].to_csv(folder / f"totals_global_{exc}.csv")
        S["best"].to_csv(folder / f"totals_bestglobal_{exc}.csv")
        S["pr_global"].to_csv(folder / f"per_record_F1_global_{exc}.csv")
        S["pr_best"].to_csv(folder / f"per_record_F1_bestglobal_{exc}.csv")
    return df, wall


# ----------------------------------------------------------------------------- VERIFY
def _pymof(X):
    from pymof import MOF
    with contextlib.redirect_stdout(io.StringIO()):       # pymof prints a line per call
        m = MOF()
        m.fit(X, len(X))
    return np.asarray(m.decision_scores_, float)


def verify_mof():
    from scipy.spatial.distance import cdist
    print("1) vectorised MOF vs pymof")
    rng = np.random.default_rng(0)
    for name, X in [("random 1-D, heavy ties", np.round(rng.normal(size=(400, 1)) * 3)),
                    ("random 3-D, heavy ties", np.round(rng.normal(size=(300, 3)) * 2)),
                    ("random 8-D continuous", rng.normal(size=(300, 8)))]:
        a = _pymof(X)
        b = mof_from_dist(cdist(X, X))
        c = mof_from_dist(sqdist(X, range(X.shape[1])))
        print(f"   {name:26s} max|diff| cdist {np.abs(a - b).max():.3g}   squared {np.abs(a - c).max():.3g}")
    sig, _, fs = load("100")
    f = mc.preprocess(sig, fs)
    F = ceo_features(f, fs, DEFAULT_CFG)
    wins = mow_windows(len(f), fs)
    picks = [wins[i] for i in (0, 1, 2, len(wins) // 3, len(wins) // 2, len(wins) - 2, len(wins) - 1)]
    worst = {r: 0.0 for r in RUNGS}
    bit = True
    for lo, hi in picks:
        ours = rung_scores(f[lo:hi], F[lo:hi], DEFAULT_CFG, list(RUNGS))
        Z = scale_window(F[lo:hi], DEFAULT_CFG["scale"])
        for r, gs in RUNGS.items():
            X = f[lo:hi].reshape(-1, 1) if r == "R1_amp" else Z[:, [c for g in gs for c in GROUPS[g]]]
            ref = _pymof(X)
            worst[r] = max(worst[r], np.abs(ours[r] - ref).max())
            bit &= r != "R1_amp" or np.array_equal(ours[r], ref)
    print(f"   record 100, {len(picks)} windows (incl. first/last short ones): max|diff| per rung "
          + ", ".join(f"{r} {v:.3g}" for r, v in worst.items()))
    print(f"   R1 bit-identical to pymof: {bit}")


def _orig_one(rec):
    """mc.mow_ecg with the REAL pymof on the full record, 0.6/0.3."""
    sig, _, fs = load(rec)
    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        det = mc.mow_ecg(sig, fs, irp=GLOBAL[0], mri=GLOBAL[1])
    return rec, det, time.time() - t


def verify_r1(records, workers):
    from joblib import Parallel, delayed
    print(f"2) R1 vs original mc.mow_ecg (real pymof), IRP {GLOBAL[0]} / MRI {GLOBAL[1]}, full records")
    t0 = time.time()
    orig = Parallel(n_jobs=workers)(delayed(_orig_one)(r) for r in records)
    ours = Parallel(n_jobs=workers)(delayed(record_scores)(r, DEFAULT_CFG) for r in records)
    print(f"   (original runs + our scoring: {time.time() - t0:.0f} s wall)")
    ok = True
    for (rec, det_o, secs), flat in zip(orig, ours):
        sig, ref, fs = load(rec)
        filtered = mc.preprocess(sig, fs)
        windows = mow_windows(len(filtered), fs)
        det_n = detect_with_scores(sig, filtered, windows, split(flat["R1_amp"], windows), fs, *GLOBAL)
        same = det_o == det_n
        ok &= same
        m = mc.evaluate(ref, det_o, fs, TOL_MS)
        print(f"   {rec}: original {len(det_o)} peaks ({secs:.0f} s), R1 {len(det_n)} peaks -> "
              f"{'IDENTICAL' if same else 'DIFFERENT'}   (TP {m['TP']} FP {m['FP']} FN {m['FN']})")
    print(f"   all identical: {ok}")
    return ok


def time_one(rec="100"):
    print("3) timing (single core)")
    sig, _, fs = load(rec)
    f = mc.preprocess(sig, fs)
    t = time.time(); F = ceo_features(f, fs, DEFAULT_CFG)
    print(f"   features for whole record {rec}: {time.time() - t:.2f} s")
    lo, hi = 100000, 101080
    w, Fw = f[lo:hi], F[lo:hi]
    _pymof(w.reshape(-1, 1))
    t = time.time(); _pymof(w.reshape(-1, 1)); print(f"   pymof, one window (R1 original): {1000 * (time.time() - t):.0f} ms")
    for r in RUNGS:
        rung_scores(w, Fw, DEFAULT_CFG, [r])
        t = time.time()
        for _ in range(5):
            rung_scores(w, Fw, DEFAULT_CFG, [r])
        print(f"   vectorised, one window, {r} alone: {200 * (time.time() - t):.0f} ms")
    t = time.time()
    for _ in range(5):
        rung_scores(w, Fw, DEFAULT_CFG, list(RUNGS))
    per_win = (time.time() - t) / 5
    nwin = len(mow_windows(len(f), fs))
    print(f"   vectorised, one window, all 4 rungs: {1000 * per_win:.0f} ms  "
          f"-> one record ({nwin} windows) ~ {per_win * nwin:.0f} s for all rungs")


# ----------------------------------------------------------------------------- SUB-LADDER
SUBDIR = OUT_DIR / "subladder_diff"
FOCUS = ["117", "124", "108", "200", "208", "103", "123", "219", "220", "203"]


# (recomputed rung, cached rung it must reproduce)
CHECKS = {"subdiff": (("R2_diff", "R2_diff"), ("A_vel_acc_jerk_seq", "R2_diff"),
                      ("A_only_scaled", "R1_amp")),
          "submom": (("A_mom_seq", "R3_mom"), ("A_only_scaled", "R1_amp"))}


def _check_one(rec, which="subdiff", cfg=DEFAULT_CFG):
    """Recompute rungs with the per-feature code and compare with the cached scores of the
    equivalent rung, window by window. Nothing is written to the cache."""
    sig, ref, fs = load(rec)
    filtered = mc.preprocess(sig, fs)
    windows = mow_windows(len(filtered), fs)
    F = ceo_features(filtered, fs, cfg)
    pairs = CHECKS[which]
    names = [a for a, _ in pairs]
    acc = {r: [] for r in names}
    for lo, hi in windows:
        sc = rung_scores(filtered[lo:hi], F[lo:hi], cfg, names)
        for r in names:
            acc[r].append(sc[r])
    cached = record_scores(rec, cfg, rungs=sorted({b for _, b in pairs}))
    out = dict(record=rec)
    for r, base in pairs:
        new = np.concatenate(acc[r])
        out[f"{r}_bitequal"] = bool(np.array_equal(new, cached[base]))
        out[f"{r}_maxdiff"] = float(np.abs(new - cached[base]).max())
        if not out[f"{r}_bitequal"]:          # do detections still agree on the whole grid?
            a, b = split(new, windows), split(cached[base], windows)
            same = all(detect_with_scores(sig, filtered, windows, a, fs, float(i), float(m)) ==
                       detect_with_scores(sig, filtered, windows, b, fs, float(i), float(m))
                       for i in IRPS for m in MRIS)
            out[f"{r}_grid_detections_equal"] = same
    return out


def check_subdiff(records, workers, which="subdiff", folder=None):
    from joblib import Parallel, delayed
    print(f"Equivalence check ({which}) on {len(records)} full records", flush=True)
    t0 = time.time()
    rows = []
    for k, r in enumerate(Parallel(n_jobs=workers, return_as="generator_unordered")(
            delayed(_check_one)(r, which) for r in records), 1):
        rows.append(r)
        print(f"  [{k}/{len(records)}] {r['record']} checked", flush=True)
    df = pd.DataFrame(rows).sort_values("record")
    print(df.to_string(index=False))
    print(f"check wall time {time.time() - t0:.0f} s")
    folder = Path(folder or SUBDIR)
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / "equivalence_check.csv", index=False)
    return df


def _wilcoxon_rows(pr, pairs, label):
    from scipy.stats import wilcoxon
    rows = []
    for a, b in pairs:
        d = (pr[a] - pr[b]).dropna()
        nz = d[d != 0]
        p = wilcoxon(pr[a], pr[b]).pvalue if len(nz) >= 1 else np.nan
        rows.append(dict(setting=label, comparison=f"{a} vs {b}", records=len(d),
                         wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()),
                         mean_diff=d.mean(), median_diff=d.median(), p_value=p))
    return rows


def subdiff_report(df, records):
    rungs = LADDERS["subdiff"]
    S = summarise(df, records, DEFAULT_CFG, "thesis", rungs)
    SUBDIR.mkdir(parents=True, exist_ok=True)
    fmt = lambda t: t.to_string(float_format=lambda v: f"{v:.2f}")
    cols = ["irp", "mri", "Se", "PPV", "DER", "DER_TP", "F1"]
    L = [f"Differencing sub-ladder, thesis exclusion ({S['n']} records), gross totals", "",
         f"At IRP {GLOBAL[0]} / MRI {GLOBAL[1]}", fmt(S["global_"][cols]), "",
         "At each rung's own best global setting (max gross F1)", fmt(S["best"][cols]), ""]
    pairs = [("A_vel", "R1_amp"), ("A_acc", "R1_amp"), ("A_jerk", "R1_amp"),
             ("A_vel_acc", "A_vel"), ("R2_diff", "A_vel_acc")]
    W = pd.DataFrame(_wilcoxon_rows(S["pr_global"], pairs, f"{GLOBAL[0]}/{GLOBAL[1]}") +
                     _wilcoxon_rows(S["pr_best"], pairs, "own best global"))
    L += ["Wilcoxon signed-rank on per-record F1 (wins/ties/losses for the first rung; "
          "ties dropped by the test)", fmt(W), ""]
    foc_g = S["pr_global"].loc[[r for r in FOCUS if r in S["pr_global"].index]]
    foc_b = S["pr_best"].loc[[r for r in FOCUS if r in S["pr_best"].index]]
    L += [f"Per-record F1 for focus records at {GLOBAL[0]}/{GLOBAL[1]}", fmt(foc_g), "",
          "Per-record F1 for focus records at each rung's own best global setting", fmt(foc_b)]
    text = "\n".join(L)
    print("\n" + text)
    (SUBDIR / "report.txt").write_text(text, encoding="utf-8")
    S["global_"].to_csv(SUBDIR / "totals_global_thesis.csv")
    S["best"].to_csv(SUBDIR / "totals_bestglobal_thesis.csv")
    S["pr_global"].to_csv(SUBDIR / "per_record_F1_global_thesis.csv")
    S["pr_best"].to_csv(SUBDIR / "per_record_F1_bestglobal_thesis.csv")
    W.to_csv(SUBDIR / "wilcoxon_thesis.csv", index=False)
    foc_g.to_csv(SUBDIR / "focus_records_F1_global.csv")
    foc_b.to_csv(SUBDIR / "focus_records_F1_bestglobal.csv")
    df[df.rung.isin(rungs)].to_csv(SUBDIR / "grid_all_records.csv", index=False)


# ----------------------------------------------------------------------------- MOMENTS SUB-LADDER
def submom_dir(cfg):
    return OUT_DIR / f"subladder_mom_ctx{cfg['ctx_ms']:g}"


def submom_focus(n=5):
    """R3's n biggest losses and n biggest gains vs R1 at GLOBAL, from the main ladder."""
    pr = pd.read_csv(OUT_DIR / cfg_tag(DEFAULT_CFG) / "per_record_F1_global_thesis.csv",
                     dtype={"record": str}).set_index("record")
    d = (pr["R3_mom"] - pr["R1_amp"]).sort_values()
    return list(d.index[:n]) + list(d.index[-n:][::-1]), d


def submom_report(df, records, cfg):
    rungs = LADDERS["submom"]
    out = submom_dir(cfg)
    S = summarise(df, records, cfg, "thesis", rungs)
    out.mkdir(parents=True, exist_ok=True)
    fmt = lambda t: t.to_string(float_format=lambda v: f"{v:.2f}")
    cols = ["irp", "mri", "Se", "PPV", "DER", "DER_TP", "F1"]
    L = [f"Moments sub-ladder, context {cfg['ctx_ms']:g} ms, thesis exclusion ({S['n']} records), "
         "gross totals", "",
         f"At IRP {GLOBAL[0]} / MRI {GLOBAL[1]}", fmt(S["global_"][cols]), "",
         "At each rung's own best global setting (max gross F1)", fmt(S["best"][cols]), ""]
    pairs = [("A_mean", "R1_amp"), ("A_var", "R1_amp"), ("A_skew", "R1_amp"), ("A_kurt", "R1_amp"),
             ("A_mean_var", "A_mean"), ("A_mean_var_skew", "A_mean_var"),
             ("R3_mom", "A_mean_var_skew")]
    W = pd.DataFrame(_wilcoxon_rows(S["pr_global"], pairs, f"{GLOBAL[0]}/{GLOBAL[1]}") +
                     _wilcoxon_rows(S["pr_best"], pairs, "own best global"))
    L += ["Wilcoxon signed-rank on per-record F1 (wins/ties/losses for the first rung; "
          "ties dropped by the test)", fmt(W), ""]
    focus, delta = submom_focus()
    L += [f"Focus records (main ladder, R3_mom - R1_amp F1 at {GLOBAL[0]}/{GLOBAL[1]}): "
          + ", ".join(f"{r} {delta[r]:+.2f}" for r in focus), ""]
    foc_g, foc_b = S["pr_global"].loc[focus], S["pr_best"].loc[focus]
    L += [f"Per-record F1 for focus records at {GLOBAL[0]}/{GLOBAL[1]} "
          "(first 5: biggest R3 losses, last 5: biggest R3 gains)", fmt(foc_g), "",
          "Per-record F1 for focus records at each rung's own best global setting", fmt(foc_b)]
    text = "\n".join(L)
    print("\n" + text)
    (out / "report.txt").write_text(text, encoding="utf-8")
    S["global_"].to_csv(out / "totals_global_thesis.csv")
    S["best"].to_csv(out / "totals_bestglobal_thesis.csv")
    S["pr_global"].to_csv(out / "per_record_F1_global_thesis.csv")
    S["pr_best"].to_csv(out / "per_record_F1_bestglobal_thesis.csv")
    W.to_csv(out / "wilcoxon_thesis.csv", index=False)
    foc_g.to_csv(out / "focus_records_F1_global.csv")
    foc_b.to_csv(out / "focus_records_F1_bestglobal.csv")
    df[df.rung.isin(rungs)].to_csv(out / "grid_all_records.csv", index=False)


# ----------------------------------------------------------------------------- MAIN
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="quick", help='"quick", "all", or e.g. "100,203"')
    ap.add_argument("--deriv", default=DEFAULT_CFG["deriv"], choices=["savgol", "gradient"])
    ap.add_argument("--sg-ms", type=float, default=DEFAULT_CFG["sg_ms"])
    ap.add_argument("--ctx-ms", type=float, default=DEFAULT_CFG["ctx_ms"])
    ap.add_argument("--scale", default=DEFAULT_CFG["scale"], choices=["robust", "minmax"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--quick-variants", action="store_true")
    ap.add_argument("--ladder", default="main", choices=list(LADDERS))
    ap.add_argument("--check-subdiff", action="store_true",
                    help="check new per-feature distances reproduce cached R2 and R1")
    ap.add_argument("--check-submom", action="store_true",
                    help="check sequential moment distances reproduce cached R3 and R1")
    args = ap.parse_args()

    recs = {"quick": QUICK_RECORDS, "all": ALL_RECORDS}.get(args.records, args.records.split(","))
    if args.check_subdiff:
        check_subdiff(recs, args.workers)
        sys.exit()
    if args.check_submom:
        check_subdiff(recs, args.workers, "submom", submom_dir(DEFAULT_CFG))
        sys.exit()
    if args.ladder == "submom":
        mcfg = dict(DEFAULT_CFG, ctx_ms=args.ctx_ms)
        t0 = time.time()
        df, _ = run_config(recs, mcfg, args.workers, LADDERS["submom"], submom_dir(mcfg))
        submom_report(df, recs, mcfg)
        print(f"\nTOTAL wall time {time.time() - t0:.0f} s")
        sys.exit()
    if args.ladder == "subdiff":
        t0 = time.time()
        df, _ = run_config(recs, DEFAULT_CFG, args.workers, LADDERS["subdiff"], SUBDIR)
        subdiff_report(df, recs)
        print(f"\nTOTAL wall time {time.time() - t0:.0f} s")
        sys.exit()
    if args.verify:
        verify_mof()
        verify_r1(QUICK_RECORDS, args.workers)
        time_one()
        sys.exit()

    cfg = dict(deriv=args.deriv, sg_ms=args.sg_ms, ctx_ms=args.ctx_ms, scale=args.scale)
    cfgs = [cfg]
    if args.quick_variants:
        if set(recs) - set(QUICK_RECORDS):
            sys.exit("--quick-variants is for the quick subset only")
        cfgs = [DEFAULT_CFG,
                dict(DEFAULT_CFG, deriv="gradient"),
                dict(DEFAULT_CFG, ctx_ms=30), dict(DEFAULT_CFG, ctx_ms=50),
                dict(DEFAULT_CFG, scale="minmax")]
    t0 = time.time()
    for c in cfgs:
        run_config(recs, c, args.workers)
    print(f"\nTOTAL wall time {time.time() - t0:.0f} s")
