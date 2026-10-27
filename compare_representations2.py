"""
Context enrichment only: does describing each point with more than its height help MOW-ECG?

Compares these inputs to MOF (everything else in MOW-ECG is unchanged):
    1D   height                                              (current MOW-ECG)
    2D   time index, height                                  (labmate's baseline)
    3D   height, height above recent level, steepness         (designed for ECG)
    4D   3D + stability of speed (std of recent speed)
    5D   4D + instant acceleration |second difference|
    6D   time index, height, recent level and spread, recent speed and spread (labmate)
    8D   6D + recent acceleration and its spread                (labmate)
1D -> 3D -> 4D -> 5D adds one ECG feature at a time; 2D, 6D and 8D follow the labmate's method.

For every record and every representation, all 80 IRP/MRI settings are run and scored
against the labels. No Track B (no warm-up, no Q).

Setup (Colab):
    pip install wfdb pymof neurokit2 pandas scipy
    # mow_ecg_clean.py must be in the same folder as this file
    python compare_representations.py --data-dir /content/mit-bih-arrhythmia-database-1.0.0

Options:
    --reps 1D,3D,6D            which representations (default: 1D,2D,3D,4D,5D,6D)
                               In PowerShell, put lists in quotes: --reps "1D,3D,6D"
    --records 100,200,203      which records (default: all 48)
    --workers 2                records processed in parallel (default: number of CPUs)
    --history-ms 50            history length for the context features
    --height-weight 2          Idea 1: multiply the scaled height column by this weight
                               (default 1 = all columns count equally)
    --centred                  Idea 2: context features use a window centred on each point
                               (25 ms before and 25 ms after) instead of the 50 ms before it
Results of a variant are labelled with its settings, e.g. 3D_w2_c, and saved next to the
original results without overwriting them (summary_w2_c.csv, ...). 1D is unaffected by both
options, so its cached scores are reused.
    --cache cache              folder for cached scores and results (point it at Google
                               Drive to keep it between Colab sessions)

Runtime: about 2 minutes per representation per 30-minute record. Re-running the same
command skips anything already computed, so it resumes after a disconnect.

Outputs (in the --cache folder):
    summary.txt                  the main results and statistical tests, as text
    summary.csv                  averages per representation (all records and hard records)
    wilcoxon_tests.csv           each representation vs 1D: win/tie/loss, mean difference, p-value
    per_record.csv               one row per record and representation
    grid_<record>_<rep>.csv      all 80 settings for one record and representation
"""

import argparse, contextlib, io, json, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import wfdb
from pymof import MOF
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mow_ecg_clean as mc

ALL_RECORDS = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112",
               "113", "114", "115", "116", "117", "118", "119", "121", "122", "123", "124", "200",
               "201", "202", "203", "205", "207", "208", "209", "210", "212", "213", "214", "215",
               "217", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234"]
HARD = ["122", "200", "203", "228", "233"]
PACED = ["102", "104", "107", "217"]   # left out of evaluations by the ANSI/AAMI EC57 standard
BEATS = set("NLRBAaJSVrFejnE/fQ?")
IRPS = np.round(np.arange(0.25, 0.701, 0.05), 2)   # 10 values, as in thesis 5.1.1
MRIS = np.round(np.arange(0.20, 0.551, 0.05), 2)   # 8 values
FIXED = {"global_0.35": (0.35, 0.35),               # thesis global setting
         "notebook_0.6_0.3": (0.60, 0.30)}          # values hard-coded in the notebook


# ------------------------------------------------------------------------------ data

def load(data_dir, rec):
    """MLII lead when available (fixes record 114), and beat-only reference labels."""
    path = os.path.join(data_dir, rec)
    r = wfdb.rdrecord(path)
    a = wfdb.rdann(path, "atr")
    lead = r.sig_name.index("MLII") if "MLII" in r.sig_name else 0
    sig = r.adc()[:, lead].astype(float)
    ref = np.array([s for s, c in zip(a.sample, a.symbol) if c in BEATS])
    return sig, ref, r.fs


# -------------------------------------------------------------------------- features

def scale01(c):
    c = np.asarray(c, float)
    rng = c.max() - c.min()
    return (c - c.min()) / rng if rng > 0 else np.zeros_like(c)


def past(c, stat, h, centred=False):
    """Mean or standard deviation over the history of each sample.
    Default: the previous h samples (not including the current one).
    centred=True: h samples centred on the current one (h/2 before, h/2 after)."""
    if centred:
        s = pd.Series(c).rolling(h + 1, center=True, min_periods=1)
    else:
        s = pd.Series(c).shift(1).rolling(h, min_periods=1)
    out = getattr(s, stat)().to_numpy()
    return np.nan_to_num(out, nan=c[0] if stat == "mean" else 0.0)


def features(win, rep, h, height_weight=1.0, centred=False):
    """Turn one window of the filtered signal into the input table for MOF.
    height_weight multiplies the scaled height column (Idea 1);
    centred makes the history features look both ways (Idea 2)."""
    if rep == "1D":
        return win.reshape(-1, 1)
    P = lambda c, stat: past(c, stat, h, centred)
    i = np.arange(len(win))
    dx = np.diff(win, prepend=win[0])          # first difference: speed
    d2 = np.diff(dx, prepend=dx[0])            # second difference: acceleration
    six = [i, win, P(win, "mean"), P(win, "std"), P(dx, "mean"), P(dx, "std")]
    cols = {
        "2D": [i, win],
        "3D": [win, win - P(win, "mean"), np.abs(dx)],
        "4D": [win, win - P(win, "mean"), np.abs(dx), P(dx, "std")],
        "5D": [win, win - P(win, "mean"), np.abs(dx), P(dx, "std"), np.abs(d2)],
        "6D": six,
        "8D": six + [P(d2, "mean"), P(d2, "std")],
    }[rep]
    height_col = 1 if rep in ("2D", "6D", "8D") else 0
    X = np.column_stack([scale01(c) for c in cols])
    X[:, height_col] *= height_weight
    return X


def variant_tag(height_weight, centred):
    """Suffix that identifies a variant, e.g. '_w2_c'. Empty for the original settings."""
    tag = ""
    if height_weight != 1:
        tag += f"_w{height_weight:g}"
    if centred:
        tag += "_c"
    return tag


def label(rep, tag):
    return rep if rep == "1D" else rep + tag


def mof_scores(X):
    with contextlib.redirect_stdout(io.StringIO()):   # pymof prints a line per call
        m = MOF()
        m.fit(X, len(X))
    return np.asarray(m.decision_scores_, float)


# ------------------------------------------------------------------------- detection

def make_windows(n, fs):
    """Same windows as mow_ecg_clean.mow_ecg: 3 s long, a new one every 2 s."""
    win, stride = 3 * fs, 2 * fs
    over = win - stride
    return [(max(0, s), min(n, s + win)) for s in range(-over, n, stride) if min(n, s + win) > max(0, s)]


def detect(filt, windows, scores_list, fs, irp, mri):
    """MOW-ECG's search, filter and merge (identical to mow_ecg_clean.mow_ecg),
    using MOF scores computed beforehand."""
    peaks = []
    mri_s, irp_s = int(mri * fs), int(irp * fs)
    half = int(irp_s / 2)
    for (lo, hi), sc0 in zip(windows, scores_list):
        win = filt[lo:hi]
        s = win - win.mean()
        sc = sc0.copy()
        rounds = int(np.floor(len(s) / irp_s))
        cands = []
        for _ in range(1, rounds):
            top = int(np.argmax(sc))
            st, en = max(0, top - half), min(len(sc), top + half)
            cands.append(int(np.argmax(np.abs(s[st:en]))) + st)
            sc[st:en] = 0
            s[st:en] = 0
        cands = mc.filter_candidates(sorted(cands), fs, mri)
        for c in cands:
            g = c + lo
            if g in peaks or (peaks and g - peaks[-1] < mri_s):
                continue
            peaks.append(g)
    return peaks


def score(det, ref, fs):
    m = mc.evaluate(ref, det, fs)
    der = (m["FP"] + m["FN"]) / m["TP"] if m["TP"] else np.nan
    return dict(Se=m["Sensitivity"], PPV=m["PPV"], DER=der,
                Comp=m["Sensitivity"] + m["PPV"] - der if m["TP"] else np.nan)


# ---------------------------------------------------------------------- per record

def compute_scores(job):
    """Slow part: MOF for every window of one record and representation. Cached."""
    data_dir, rec, rep, cache, h_ms, hw, centred = job
    f = os.path.join(cache, f"scores_{rec}_{label(rep, variant_tag(hw, centred))}.npy")
    if os.path.exists(f):
        return rec, rep, None
    sig, _, fs = load(data_dir, rec)
    filt = mc.preprocess(sig, fs)
    h = int(h_ms / 1000 * fs)
    t = time.time()
    sc = [mof_scores(features(filt[lo:hi], rep, h, hw, centred)) for lo, hi in make_windows(len(filt), fs)]
    np.save(f, np.array(sc, dtype=object), allow_pickle=True)
    return rec, rep, round(time.time() - t)


def evaluate_record(job):
    """Fast part: all 80 settings for every representation of one record. Cached."""
    data_dir, rec, reps, cache, tag = job
    reps = [label(r, tag) for r in reps]
    out = os.path.join(cache, f"result_{rec}{tag}.json")
    if os.path.exists(out):
        rows = json.load(open(out))
        if {r["rep"] for r in rows} >= set(reps):
            return [r for r in rows if r["rep"] in reps]
    sig, ref, fs = load(data_dir, rec)
    filt = mc.preprocess(sig, fs)
    windows = make_windows(len(filt), fs)
    rows = []
    for rep in reps:
        scores = list(np.load(os.path.join(cache, f"scores_{rec}_{rep}.npy"), allow_pickle=True))
        grid = pd.DataFrame([dict(irp=irp, mri=mri, **score(detect(filt, windows, scores, fs, irp, mri), ref, fs))
                             for irp in IRPS for mri in MRIS])
        grid.to_csv(os.path.join(cache, f"grid_{rec}_{rep}.csv"), index=False)
        best = grid.loc[grid.Comp.idxmax()]
        row = dict(record=rec, rep=rep, beats=int(len(ref)),
                   best_irp=best.irp, best_mri=best.mri,
                   best_Se=best.Se, best_PPV=best.PPV, best_DER=best.DER, best_Comp=best.Comp,
                   median_Comp=grid.Comp.median(), worst_Comp=grid.Comp.min(),
                   share_settings_Comp_above_1_9=float((grid.Comp > 1.9).mean()))
        for name, (irp, mri) in FIXED.items():
            g = grid[(grid.irp == irp) & (grid.mri == mri)].iloc[0]
            row.update({f"{name}_Se": g.Se, f"{name}_PPV": g.PPV, f"{name}_DER": g.DER, f"{name}_Comp": g.Comp})
        rows.append(row)
    json.dump(rows, open(out, "w"), default=float)
    return rows


# -------------------------------------------------------------------------- summary

def summarise(df, path, tag=""):
    L = []
    p = L.append
    summary_rows, test_rows = [], []
    folder = os.path.dirname(path)
    pct = lambda x: f"{100 * x:6.2f}%"
    reps = list(df.rep.unique())

    groups = [("ALL RECORDS", "all", df),
              ("NON-PACED RECORDS (without " + ", ".join(PACED) + ")", "non_paced", df[~df.record.isin(PACED)]),
              ("HARD RECORDS: " + ", ".join(HARD), "hard", df[df.record.isin(HARD)])]
    for title, gname, sub in groups:
        if sub.empty:
            continue
        p("=" * 86)
        p(f"{title}   ({sub.record.nunique()} records)")
        p("=" * 86)
        p("Each representation at its OWN BEST setting (does enrichment raise the best result?)")
        p(f"  {'rep':4} {'Se':>9} {'PPV':>9} {'DER':>8} {'Composite':>10}")
        for rep in reps:
            x = sub[sub.rep == rep]
            p(f"  {rep:4} {pct(x.best_Se.mean()):>9} {pct(x.best_PPV.mean()):>9} "
              f"{x.best_DER.mean():8.3f} {x.best_Comp.mean():10.3f}")
        for name, (irp, mri) in FIXED.items():
            p(f"\nAt a FIXED setting {name} (IRP {irp}, MRI {mri}) (is it more forgiving?)")
            p(f"  {'rep':4} {'Se':>9} {'PPV':>9} {'DER':>8} {'Composite':>10}")
            for rep in reps:
                x = sub[sub.rep == rep]
                p(f"  {rep:4} {pct(x[name + '_Se'].mean()):>9} {pct(x[name + '_PPV'].mean()):>9} "
                  f"{x[name + '_DER'].mean():8.3f} {x[name + '_Comp'].mean():10.3f}")
        for rep in reps:
            x = sub[sub.rep == rep]
            r = dict(group=gname, records=sub.record.nunique(), rep=rep,
                     best_Se=x.best_Se.mean(), best_PPV=x.best_PPV.mean(), best_DER=x.best_DER.mean(),
                     best_Comp=x.best_Comp.mean())
            for name in FIXED:
                for m in ["Se", "PPV", "DER", "Comp"]:
                    r[f"{name}_{m}"] = x[f"{name}_{m}"].mean()
            r.update(median_Comp=x.median_Comp.mean(), worst_Comp=x.worst_Comp.mean(),
                     share_settings_Comp_above_1_9=x.share_settings_Comp_above_1_9.mean())
            summary_rows.append(r)
        p("\nAcross ALL 80 settings (robustness: higher = bad settings hurt less)")
        p(f"  {'rep':4} {'median Comp':>12} {'worst Comp':>11} {'share of settings with Comp > 1.9':>36}")
        for rep in reps:
            x = sub[sub.rep == rep]
            p(f"  {rep:4} {x.median_Comp.mean():12.3f} {x.worst_Comp.mean():11.2f} "
              f"{pct(x.share_settings_Comp_above_1_9.mean()):>36}")
        p("")

    p("=" * 86)
    p("PAIRED WILCOXON TESTS vs 1D, record by record (win/tie/loss counts are for the other rep)")
    p("=" * 86)
    if "1D" in reps:
        base = df[df.rep == "1D"].set_index("record")
        for rep in [r for r in reps if r != "1D"]:
            other = df[df.rep == rep].set_index("record")
            for col, label in [("best_Comp", "best setting"), ("global_0.35_Comp", "global 0.35"),
                               ("median_Comp", "median over grid"), ("worst_Comp", "worst setting")]:
                both = pd.concat([other[col], base[col]], axis=1, keys=["a", "b"]).dropna()
                w = int((both.a > both.b).sum()); l = int((both.a < both.b).sum()); t = len(both) - w - l
                pval = wilcoxon(both.a, both.b).pvalue if len(both) >= 5 and (both.a != both.b).any() else np.nan
                pv = f"p = {pval:.4f}" if not np.isnan(pval) else "too few records for a test"
                p(f"  {rep} vs 1D, {label:18}: {w}/{t}/{l}  {pv}")
                test_rows.append(dict(comparison=f"{rep} vs 1D", measure=label, records=len(both),
                                      wins=w, ties=t, losses=l, mean_difference=(both.a - both.b).mean(),
                                      p_value=pval))
    text = "\n".join(L)
    open(path, "w").write(text)
    print(text)
    pd.DataFrame(summary_rows).to_csv(os.path.join(folder, f"summary{tag}.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(folder, f"wilcoxon_tests{tag}.csv"), index=False)


# ----------------------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/content/mit-bih-arrhythmia-database-1.0.0")
    ap.add_argument("--records", default=",".join(ALL_RECORDS))
    ap.add_argument("--reps", default="1D,2D,3D,4D,5D,6D")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--history-ms", type=float, default=50.0)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--height-weight", type=float, default=1.0)
    ap.add_argument("--centred", action="store_true")
    args = ap.parse_args()
    tag = variant_tag(args.height_weight, args.centred)

    os.makedirs(args.cache, exist_ok=True)
    records, reps = args.records.split(","), args.reps.split(",")

    jobs = [(args.data_dir, r, rep, args.cache, args.history_ms, args.height_weight, args.centred)
            for r in records for rep in reps]
    if tag:
        print(f"Variant {tag}: height weight {args.height_weight:g}, centred history {args.centred}", flush=True)
    print(f"Step 1/2: MOF scores ({len(jobs)} record x representation jobs, {args.workers} workers)", flush=True)
    with Pool(args.workers) as pool:
        for i, (rec, rep, secs) in enumerate(pool.imap_unordered(compute_scores, jobs), 1):
            print(f"  [{i}/{len(jobs)}] {rec} {rep}: " + ("cached" if secs is None else f"{secs}s"), flush=True)

    print("Step 2/2: 80 settings per record and representation", flush=True)
    rows = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(evaluate_record, [(args.data_dir, r, reps, args.cache, tag)
                                                                      for r in records]), 1):
            rows += res
            print(f"  [{i}/{len(records)}] {res[0]['record']} done", flush=True)

    df = pd.DataFrame(rows)
    df["record"] = pd.Categorical(df.record, categories=records, ordered=True)
    df["rep"] = pd.Categorical(df.rep, categories=[label(r, tag) for r in reps], ordered=True)
    df = df.sort_values(["rep", "record"])
    df["record"] = df.record.astype(str); df["rep"] = df.rep.astype(str)
    df.to_csv(os.path.join(args.cache, f"per_record{tag}.csv"), index=False)
    summarise(df, os.path.join(args.cache, f"summary{tag}.txt"), tag)
    print("\nResults saved in", os.path.abspath(args.cache))
    for f in ["per_record", "summary", "wilcoxon_tests"]:
        print(f"   {f}{tag}.csv")
    print(f"   summary{tag}.txt")