"""
MOW-ECG R-peak detection: clean version of the notebook (MOW_ECG_v5)

This file contains ONLY the code MOW-ECG actually runs, taken from these
notebook functions, with the logic unchanged:

    nk_preprocess                          -> preprocess()          thesis 3.2
    overlapping_window_MOF_PQRST_v5        -> mow_ecg()             thesis 3.3
    approx_peaks_v5                        -> find_candidates()     thesis 3.4.2
    filter_false_positives_rr_outlier_v5   -> filter_candidates()   thesis 3.4.3
    calculate_detection_metrics            -> evaluate()

What changed compared with the notebook:
  * IRP and MRI are parameters instead of numbers hard-coded inside functions.
    Defaults (0.6 s and 0.3 s) are the values hard-coded in the notebook.
  * The print statements are removed (they slowed down full-length records).
  * P, Q, S, T detection (detect_pqrst_v5) is left out: this file does R-peaks only.
    It does not change which R-peaks are found.
  * load_record() can keep only beat annotations (beats_only=True), so rhythm and
    noise labels are not counted as beats during evaluation. Set it to False to
    reproduce the notebook's evaluation exactly.

Requirements:  pip install wfdb numpy neurokit2 pymof
"""

import numpy as np
import neurokit2 as nk
import wfdb
from pymof import MOF


# MIT-BIH annotation symbols that mark a real beat. Other symbols such as
# '+' (rhythm change), '~' (signal quality change) and '|' are not beats.
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")


# =============================================================================
# 1. Loading a record
# =============================================================================

def load_record(path, samp_from=180, samp_to=4000, beats_only=True):
    """
    Read one MIT-BIH record.

    path       e.g. '/content/mit-bih-arrhythmia-database-1.0.0/100'
    samp_from  first sample to read (notebook: 180)
    samp_to    last sample to read (notebook: 4000, about 10.6 s;
               use None for the whole 30-minute record)
    beats_only keep only annotations that are beats

    Returns  signal (1-D array, first lead, raw recorder values),
             reference (array of annotated beat positions),
             fs (sampling frequency, 360 for MIT-BIH)
    """
    record = wfdb.rdrecord(path, samp_from, samp_to)
    ann = wfdb.rdann(path, "atr", samp_from, samp_to, shift_samps=True)

    signal = record.adc()[:, 0].astype(float)   # first lead only

    reference = np.array(ann.sample)
    if beats_only:
        keep = [i for i, s in enumerate(ann.symbol) if s in BEAT_SYMBOLS]
        reference = reference[keep]

    return signal, reference, ann.fs


# =============================================================================
# 2. Preprocessing (thesis 3.2)
# =============================================================================

def preprocess(signal, fs):
    """
    Filter the whole signal once.

    * Butterworth HIGH-pass at 0.5 Hz, order 5: removes slow baseline drift.
      (Only a low cut-off is given, so nothing above 0.5 Hz is removed.)
    * Powerline filter at 50 Hz: in NeuroKit this is a moving average over
      fs / 50 samples (7 at 360 Hz), applied forwards and backwards.

    Returns a 1-D array the same length as the input.
    """
    signal = nk.as_vector(signal)

    # Fill missing values (NaN) with the previous value, as in the notebook
    if np.isnan(signal).any():
        signal = np.asarray(pd_ffill(signal))

    filtered = nk.signal_filter(signal=signal, sampling_rate=fs,
                                lowcut=0.5, method="butterworth", order=5)
    filtered = nk.signal_filter(signal=filtered, sampling_rate=fs,
                                method="powerline", powerline=50)
    return np.asarray(filtered, dtype=float)


def pd_ffill(signal):
    import pandas as pd
    return pd.Series(signal).ffill().to_numpy()


# =============================================================================
# 3. Candidate search inside one window (thesis 3.4.2)
# =============================================================================

def find_candidates(scores, window_signal, fs, irp=0.6):
    """
    Find R-peak candidates in one window, strongest first.

    scores         MOF score of every sample in the window (1-D)
    window_signal  the window's signal with its mean subtracted (1-D)
    irp            Initial Refractory Period in seconds

    Each round:
      1. take the sample with the highest MOF score
      2. look IRP/2 either side of it (the erased zone is IRP wide, centred)
      3. the R-peak is the largest ABSOLUTE amplitude in that zone
         (works for upward and downward R-peaks)
      4. set scores and signal in the zone to zero, so the next round
         finds a different beat

    Returns the candidates' positions in the window, in time order.
    """
    scores = np.copy(scores)
    sig = np.copy(window_signal)

    irp_samples = int(irp * fs)
    half_zone = int(irp_samples / 2)

    # Number of rounds. The notebook uses range(1, n), so it runs n - 1 times:
    # with a 3 s window and IRP 0.6 s, n = 5 and only 4 candidates are searched.
    n = int(np.floor(len(sig) / irp_samples))

    candidates = []
    for _ in range(1, n):
        top = int(np.argmax(scores))                       # step 1

        start = max(0, top - half_zone)                    # step 2
        end = min(len(scores), top + half_zone)

        zone = sig[start:end]                              # step 3
        r_peak = int(np.argmax(np.abs(zone))) + start
        # The notebook also has "if abs(sig[top]) > max(abs(zone)): use top",
        # but top lies inside the zone, so that can never be true and the
        # largest-amplitude position is always used. Left out here.
        candidates.append(r_peak)

        scores[start:end] = 0                              # step 4
        sig[start:end] = 0

    return sorted(candidates)


# =============================================================================
# 4. Post-processing inside one window (thesis 3.4.3)
# =============================================================================

def filter_candidates(candidates, fs, mri=0.3, factor=0.5):
    """
    Delete candidates that are too close to the previous accepted one.

    limit = min(MRI, factor x median gap between all candidates)

    The first candidate is always kept. Then, in time order, each candidate
    is kept only if it is at least `limit` after the last KEPT candidate.
    When two are too close, the later one is deleted.
    """
    if len(candidates) < 2:
        return list(candidates)

    mri_samples = int(mri * fs)
    median_gap = np.median(np.diff(candidates))      # computed before deleting
    limit = min(mri_samples, median_gap * factor)

    kept = [candidates[0]]
    for c in candidates[1:]:
        if c - kept[-1] >= limit:
            kept.append(c)
    return kept


# =============================================================================
# 5. The full detector: overlapping windows + merging (thesis 3.3)
# =============================================================================

def mow_ecg(signal, fs, irp=0.6, mri=0.3):
    """
    Run MOW-ECG on a whole signal.

    signal  1-D array (one lead)
    fs      sampling frequency
    irp     Initial Refractory Period in seconds (notebook: 0.6)
    mri     Minimum RR Interval in seconds       (notebook: 0.3)

    Returns a list of R-peak positions (sample indices).
    """
    window = 3 * fs                 # 3 s windows
    stride = window - 1 * fs        # a new window every 2 s -> 1 s overlap
    overlap = window - stride
    mri_samples = int(mri * fs)

    filtered = preprocess(signal, fs)
    n = len(filtered)

    r_peaks = []

    # Starting at -overlap makes the first window cover only the first 2 s,
    # so the start of the signal is also seen by two windows.
    for start in range(-overlap, n, stride):
        lo = max(0, start)
        hi = min(n, start + window)
        win = filtered[lo:hi]
        if len(win) == 0:
            continue

        # MOF score for every sample in the window. MOF expects a table of
        # shape (samples, features), so the window is reshaped to a column.
        model = MOF()
        model.fit(win.reshape(-1, 1), len(win))
        scores = model.decision_scores_

        # Subtract the window's mean (the notebook's "Z-score" step; note it
        # does NOT divide by the standard deviation)
        centred = win - np.mean(win)

        candidates = find_candidates(scores, centred, fs, irp)
        candidates = filter_candidates(candidates, fs, mri)

        # Merge with what previous windows found. A peak is added only if it
        # is new AND at least MRI after the last accepted peak (this also
        # rejects anything earlier than the last accepted peak). This is what
        # removes beats found twice in the 1 s overlap.
        for c in candidates:
            pos = c + lo                              # position in full signal
            if pos in r_peaks:
                continue
            if r_peaks and pos - r_peaks[-1] < mri_samples:
                continue
            r_peaks.append(pos)

    return r_peaks


# =============================================================================
# 6. Evaluation (same as calculate_detection_metrics in the notebook)
# =============================================================================

def evaluate(reference, detected, fs, tolerance_ms=50):
    """
    A detection within +/- tolerance of an unmatched reference beat is a true
    positive; other detections are false positives; reference beats left
    unmatched are false negatives (missed beats).
    """
    tol = int(tolerance_ms * fs / 1000)
    reference = np.sort(np.asarray(reference))
    detected = sorted(detected)

    matched = set()
    tp = fp = 0
    for d in detected:
        i0 = np.searchsorted(reference, d - tol, side="left")
        i1 = np.searchsorted(reference, d + tol, side="right")
        hit = next((i for i in range(i0, i1) if i not in matched), None)
        if hit is None:
            fp += 1
        else:
            matched.add(hit)
            tp += 1
    fn = len(reference) - len(matched)

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "Sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "PPV": tp / (tp + fp) if tp + fp else 0.0,
    }


# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":
    path = "/content/mit-bih-arrhythmia-database-1.0.0/100"

    signal, reference, fs = load_record(path, samp_from=180, samp_to=4000)
    detected = mow_ecg(signal, fs, irp=0.6, mri=0.3)

    print("Detected R-peaks:", detected)
    print(evaluate(reference, detected, fs))
