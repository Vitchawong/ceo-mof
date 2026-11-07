# CEO-MOF project — context for a new chat

Paste this whole file at the start of a new conversation. It contains everything an
assistant needs to continue the work without re-deriving it.

---

## 1. Who / what

Master Thesis work (Vitchawong Laohabutr, Chulalongkorn University; advisors Asst. Prof. Arthorn
Luangsodsai and Assoc. Prof. Krung Sinapiromsaran). Topic: **R-peak detection in ECG
using outlier detection**, not machine learning.

**Baseline (already in the thesis): MOW-ECG.** The ECG is cut into overlapping windows.
Inside each window, MOF (Mass-ratio-variance Outlier Factor) scores every sample by how
unusual it is. R-peaks are the tallest spikes, so they get the highest scores. The scores
feed a candidate search, refractory nulling, and post-processing.

**The extension being tested: CEO-MOF** (Context-Enriched Overlapping-window MOF),
proposed at meeting #4 (4 Nov 2026). Instead of scoring each sample by amplitude alone,
MOF scores a feature vector per sample. Working title suggested by A. Arthorn:
"Overlapping Window Anomaly Scoring via Multi-Feature Contextual MOF for Electrocardiogram
R Peak Detection."

**The eight features** (columns 0–7 in the code):

| # | Feature | Family | Meaning |
|---|---|---|---|
| 0 | Amplitude x[t] | — | Original MOW-ECG input |
| 1 | Velocity X(1) | Differencing | Slope |
| 2 | Acceleration X(2) | Differencing | Curvature |
| 3 | Jerk X(3) | Differencing | Third-order difference |
| 4 | Local mean | Moments | Smoothed level |
| 5 | Local variance | Moments | Local energy |
| 6 | Local skewness | Moments | Polarity |
| 7 | Local kurtosis | Moments | Spikiness |

Moments are computed over a ~100 ms context window around each sample (standardised
central moments, not the raw E[X^k] written in the meeting minutes).

---

## 2. Headline result so far

**Equal-weight context features do not improve on amplitude-only MOW-ECG on MIT-BIH.**
This is a clean negative result, and it is trustworthy because R1 (amplitude only) was
verified bit-identical to the original code.

Ordering held in every comparison: **R1 ≥ R2 > R4 > R3.**

---

## 3. Setup (all experiments)

- Data: **MIT-BIH**, all 48 records, MLII lead, full 30-minute records, raw ADC values.
- Detector: the **original `mow_ecg_clean.py` is called unchanged**; only MOF's input
  differs between rungs (a stand-in class returns precomputed scores per window).
- Windows: 3 s, new window every 2 s (1 s overlap). ~904 windows per record.
- Features computed once per record on the filtered signal, then sliced per window.
- Scaling: **robust z per window per feature**, `(v - median) / (1.4826*MAD + 1e-6)`.
  R1 skips scaling (MOF depends only on distance ranks, so this is equivalent).
- Distance: squared Euclidean, one matrix per feature column, summed per rung
  (equal weight — this is the suspected problem).
- Hyperparameters: fixed **IRP 0.6 s / MRI 0.3 s**, plus the full **80-setting grid**
  (IRP 0.25–0.70, MRI 0.20–0.55, step 0.05).
- Scoring: 50 ms matching tolerance; Se, PPV, DER (thesis Eq. 66 = (FP+FN)/(TP+FN)),
  and F1. DER_TP = (FP+FN)/TP is also reported for comparison with older numbers.
- Exclusions applied only at aggregation: thesis {104,107,118,207,217} → 43 records
  (primary); AAMI {102,104,107,217} → 44 records (secondary; same pattern, but R1 and R2 are tied at 0.6/0.3: F1 89.90 vs 89.91).
- Runtime: ~40–110 min per experiment (the moments sub-ladder took ~112 min on 6 workers).
  Everything is cached per record.

---

## 4. Experiments run, in order

### Verification (passed)
- Vectorised MOF vs pymof: **max difference 0** (1-D, 3-D, 8-D, with ties; real windows).
- R1 vs original `mc.mow_ecg`: **identical detections** on 6 records.
- So all differences between rungs come from the features, not from a bug.

### Main group ladder (48 records, 39 min)
R1 = amplitude; R2 = + differences (1–3); R3 = + moments (4–7); R4 = all 8.

| Rung | F1 at 0.6/0.3 | F1 at own best setting |
|---|---|---|
| R1 | 90.29 | **93.99** (0.45/0.40) |
| R2 | 90.27 | 92.38 |
| R3 | 88.69 | 89.48 |
| R4 | 89.25 | 90.28 |

Wilcoxon vs R1 (per-record F1): R2 ties at the fixed setting (p = 0.32) but is worse when
tuned (p = 0.0004); R3 and R4 are worse in both views (p ≈ 0.002 or smaller). R3 lost on 37/43 records.
**Tuning R1 gains more (+3.7 F1) than any feature does.**

### Differencing sub-ladder (46.5 min)
Equivalence check passed on the 6 quick records (max diff 0 vs cached R1 and R2).

| Rung | F1 at 0.6/0.3 | F1 at own best |
|---|---|---|
| R1 | 90.29 | 93.99 |
| A+vel | 90.39 | 93.45 |
| A+acc | 90.37 | 92.78 |
| A+jerk | 90.48 | 92.87 |
| A+vel+acc | 90.70 | 92.90 |
| R2 (all 3) | 90.27 | 92.38 |

- No single slope feature beats R1 (p = 0.83–0.99 at the fixed setting).
- A+vel+acc is highest at the fixed setting but **not significant** (p = 0.88 vs A+vel).
- **Jerk hurts when added last:** R2 < A+vel+acc on 29/43 (p = 0.005; p = 6e-5 when tuned).
- R2's gains need combinations, not single features (e.g. record 124: A+vel 81.6 but
  A+vel+acc 94.5 vs R1 86.7).

### Moments sub-ladder (context 100 ms)
Equivalence check passed on the 6 quick records (max diff 0 vs cached R1 and R3).

| Rung | F1 at 0.6/0.3 | F1 at own best |
|---|---|---|
| R1 | 90.29 | 93.99 |
| A+mean | 90.08 | 92.63 |
| A+var | 89.26 | 90.94 |
| A+skew | 89.86 | 92.59 |
| A+kurt | 89.49 | 91.21 |
| A+mean+var | 89.17 | 91.05 |
| A+mean+var+skew | 89.87 | 91.41 |
| R3 (all 4) | 88.69 | 89.48 |

- **Mean ≈ harmless** (p = 0.26 fixed) — it nearly duplicates amplitude.
- **Skewness and kurtosis clearly hurt** (p = 0.003 and 0.0003 at the fixed setting).
- **Kurtosis is worst added last:** R3 < A+mean+var+skew on 34/43 (p < 1e-4 both views).
- **Variance is the only moment with real upside** on hard records (117 +12.4, 124 +5.6,
  201 +5.5, 202 +5.1) but it costs on clean records and is the worst single moment when
  tuned (−3.25 F1, p < 1e-4).

### Quick-subset feature variants (6 records only, not conclusive)
Plain differences ≈ Savitzky–Golay. Shorter moment context hurts less (30 ms best).
Min-max scaling: R2 89.37 / R3 84.20 / R4 88.08 vs robust 89.01 / 86.35 / 87.54 —
no clear winner, conclusion unchanged. Min-max was never run on all records.

---

## 5. Working hypothesis

**Equal weighting dilutes amplitude.** Amplitude is already a strong R-peak clue; giving
seven other features an equal vote in the distance outvotes it. Supporting evidence: the
helpful features (velocity+acceleration, variance) gain on records where R1 is weak and
lose on records where R1 is near ceiling. **Not yet tested.**

---

## 6. Next steps (in priority order)

1. **Weighted distance (λ)** — the main next experiment.
   `D = D_amp + λ · D_added`, λ ∈ {0, 0.05, 0.1, 0.25, 0.5, 1}.
   Candidate added set: **velocity + acceleration + variance** (drop jerk, skewness,
   kurtosis). λ = 0 must reproduce R1 exactly — use that as the safety check.
   Success = keeping gains on 117, 124, 201, 202 without losses on 103, 219, 220, 221.
2. **Noise robustness** — may change the conclusion. MIT-BIH is mostly clean, and the
   thesis's stated weakness is low SNR. Use the MIT-BIH Noise Stress Test Database, or add
   noise to clean records at several SNRs. If context only helps under noise, that is the
   stronger thesis story.
3. **Unresolved: thesis numbers don't reproduce.** Per-record-tuned R1 gives ~97.3 Se /
   97.6 PPV (by F1 or by the thesis composite rule), or 85.3 / 98.7 (PPV-first rule).
   The thesis headline is 94.48 / 99.67. Cause unknown — candidates: the selection rule,
   the matching tolerance (code uses 50 ms; thesis doesn't state one), the exclusion set,
   or per-record vs gross averaging. **This does not affect rung comparisons** (all rungs
   evaluated identically) but should be explained before presenting.
4. QT-DB as a second database (not yet run; not in the local data folder).
5. If λ and noise both fail: use context only to **confirm or reject candidate peaks after
   MOF**, rather than inside MOF's distance.

---

## 7. Files and tools

Local folder `ohm_codes/` (on the user's laptop; the assistant cannot run these):
- `mow_ecg_clean.py` — original MOW-ECG. **Never modify.**
- `ceo_ladder.py` — the ladder runner (written by Claude Code). Key parts:
  `ceo_features`, `local_moments`, `scale_window`, `mof_from_dist`, `rung_scores`,
  `_Replay` / `replayed` (stand-in for pymof), `run_record`, `summarise`, `subdiff_report`, `submom_report`.
  Flags: `--verify`, `--records quick|all`, `--ladder main|subdiff|submom`,
  `--deriv savgol|gradient`, `--ctx-ms`, `--scale robust|minmax`, `--workers`,
  `--quick-variants`, `--check-subdiff`, `--check-submom`.
- `compare_representations.py`, `compare_representations2.py`, `experiment2.py`,
  `sensitivity_by_setting.py` — earlier work (compare_representations2 tests 1D–8D inputs to
  MOF; experiment2 prints its results from the cache).
- `ceo_mof_ladder.py` — earlier version of the ladder runner; superseded by `ceo_ladder.py`.
- `mitdb/` — the data. `results_ladder/` — all outputs (gitignored).

**How the work is split:** experiments run in **Claude Code** on the laptop (it has the
data and the original code). The chat assistant designs experiments, writes prompts for
Claude Code, interprets results, and maintains the reports. The chat assistant cannot run
the pipeline or download PhysioNet data.

Reports (claude.ai artifacts):
- "CEO-MOF Ablation Experiment: Does Adding Context Help Find Heartbeats?" (main ladder)
- "Sub-Ladder Experiments: Which Slope and Moment Features Matter?" (both sub-ladders)
Both are written at roughly 12th-grade reading level, with a glossary.

---

## 8. How to work with this project

- **Results come as CSV/report.txt uploads** from Claude Code. Read them, don't guess.
- **Report honestly.** Negative results are the finding here; don't dress them up.
- **Treat <1 F1 point as noise** unless a paired Wilcoxon test supports it. With ~10 tests
  at once, use a stricter threshold (~0.005) rather than 0.05.
- **Best-setting p-values are optimistic** (settings chosen on the same records).
- **Always check per-record results**, not just gross totals — averages hide the pattern
  where context helps hard records and hurts clean ones.
- **Mark hypotheses as hypotheses.** The dilution explanation is untested.
- Meetings are roughly weekly; deliverables include a to-do list (`To-do-list.xlsx`) and
  Seminar 2 slides for "A Model-Free AI Framework for Robust PQRST Complex Detection in
  ECG Signals". Proposal defence is planned for April 2027.
