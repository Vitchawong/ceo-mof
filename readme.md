# CEO-MOF: Context-Enriched Overlapping-window MOF for R-peak detection

Vichawong Laohabutr · M.Sc. Computer Science and Information Technology, Chulalongkorn University
Advisors: Asst. Prof. Arthorn Luangsodsai, Assoc. Prof. Krung Sinapiromsaran
Last updated: 27 October 2026

---

## 1. Project in one paragraph

MOW-ECG (Manh Duong Dang's thesis, with the two published papers in JCMDS and IJACSA) detects
R-peaks by treating them as outliers: each 3-second window of ECG is scored with MOF
(Mass-ratio-variance Outlier Factor), and the highest-scoring points are kept as beats.
MOF currently sees only the **height** of each sample. **CEO-MOF** gives MOF extra
**context features** for each sample (recent level, steepness, standard deviation of the
first difference, acceleration), following the idea of the labmate's O-HVOF work. The research
question is whether this makes R-peak detection better or more robust.

---

## 2. Folder contents

```
ohm_codes/
├── README.md                     this file
├── mow_ecg_clean.py              MOW-ECG extracted from the notebook (MOW_ECG_v5), same output
├── compare_representations.py    main experiment: 1D–8D inputs x 80 IRP/MRI settings x records
├── sensitivity_by_setting.py     Se and PPV at the best / median / worst setting (from grid files)
├── mitdb/                        MIT-BIH Arrhythmia Database (48 records: .hea .dat .atr)
└── cache/                        cached MOF scores and all results (CSV)
```

Other scripts from earlier work (not needed for the current experiments):
`run_full_dataset.py` and `test_track_b_context.py` (Track B self-tuning tests),
`show_4_failures_kaggle.py` and `mitbih_failure_examples.py` (plots of failure cases).

### Setup

```
pip install wfdb pymof neurokit2 pandas scipy
```

Python 3.14 works. MIT-BIH was downloaded via Colab from PhysioNet's AWS mirror.

### Running (PowerShell: always put lists in quotes)

```
python compare_representations.py --data-dir mitdb --reps "1D,3D,5D,6D" --workers 4
python compare_representations.py --data-dir mitdb --reps "1D,3D,5D,6D" --workers 4 --height-weight 2
python compare_representations.py --data-dir mitdb --reps "1D,3D,5D,6D" --workers 4 --centred
python sensitivity_by_setting.py --tag _w2 --reps "1D,3D,5D,6D"
```

* About 2 minutes of MOF per representation per 30-minute record. All 48 records with
  4 versions take about 2.5–3 hours on the laptop (Intel Core Ultra 5, 4 workers).
* Everything is cached in `cache/`; re-running the same command resumes after an interruption.
* Variant runs save their own files with a tag (`summary_w2.csv`, `summary_c.csv`, ...).

---

## 3. The representations (inputs to MOF)

Each 3-second window (1,080 samples at 360 Hz) becomes a table with one row per sample.
x = filtered ECG, Δx = first difference, Δ²x = second difference,
μ and σ = mean and standard deviation over the previous 50 ms (18 samples).
Every column except in 1D is min-max scaled to 0–1 inside the window.

| Version | Columns | Origin |
|---|---|---|
| 1D | x | Original MOW-ECG |
| 2D | i (time index), x | Labmate baseline |
| 3D | x, x − μ(x), \|Δx\| | Designed for ECG |
| 4D | 3D + σ(Δx) | 3D + one labmate feature |
| 5D | 4D + \|Δ²x\| | 4D + acceleration |
| 6D | i, x, μ(x), σ(x), μ(Δx), σ(Δx) | Labmate's O-HVOF features |

Everything after MOF (candidate search with IRP, MRI filter, window merging) is unchanged.

---

## 4. Evaluation method

* MIT-BIH, all 48 records, full 30 minutes, MLII lead (V5 for 102 and 104).
* Reference = beat annotations only (rhythm/noise labels such as `+`, `~`, `|` removed).
* A detection within ±50 ms of a labelled beat is correct.
* Every record is run with all **80 IRP/MRI settings** (IRP 0.25–0.70 s, MRI 0.20–0.55 s,
  steps of 0.05 s), as in the thesis grid search.
* Composite = Sensitivity + PPV − DER (perfect = 2; no lower limit).
* For each record: **best** setting (ceiling), **median** setting (typical),
  **worst** setting (floor).
* Averages reported over the **44 non-paced records** (102, 104, 107, 217 excluded,
  following ANSI/AAMI EC57). Wilcoxon signed-rank tests vs 1D use all 48 records.

---

## 5. Results so far (48 records, 1D–6D)

### Averages, 44 non-paced records

| Version | Best composite | Se at best | Median composite | Worst composite |
|---|---|---|---|---|
| 1D | 1.876 | 97.0% | 1.441 | −15.2 |
| 2D | 1.878 | 96.9% | 1.447 | −17.4 |
| 3D | 1.852 | 96.2% | 1.434 | −6.2 |
| 4D | 1.842 | 96.0% | 1.397 | −4.4 |
| 5D | 1.825 | 95.6% | 1.394 | −3.6 |
| 6D | 1.860 | 96.5% | 1.385 | −9.9 |

### Wilcoxon tests vs 1D (48 records; wins/losses for the enriched version)

| Version | Best setting | Median setting | Worst setting |
|---|---|---|---|
| 2D | no difference | no difference | no difference |
| 3D | worse (11/36, p < 0.001) | no difference | better (32/16, p = 0.03) |
| 4D | worse (10/37, p < 0.001) | worse (p = 0.02) | better (33/15, p = 0.001) |
| 5D | worse (6/42, p < 0.001) | worse (p = 0.003) | better (35/13, p < 0.001) |
| 6D | no difference (p = 0.07) | no difference | better (30/18, p = 0.04) |

### Main finding: a floor/ceiling trade-off

* Context enrichment **raises the floor**: bad IRP/MRI settings do far less damage
  (worst composite −15.2 for 1D vs −3.6 for 5D).
* It **lowers the ceiling**: at the best setting, 1D is significantly more accurate
  than 3D, 4D and 5D.
* The effect grows with each ECG feature added (1D → 3D → 4D → 5D).
* Typical (median) settings do not improve.
* 2D (time index) changes nothing; 6D keeps 1D's ceiling and raises the floor somewhat.

Practical meaning: if IRP/MRI can be tuned well, 1D is best; if one fixed setting must
work for unknown patients, enrichment is safer.

### Other observations

* With this evaluation, the thesis's "hard" records do well: 228, 233, 200 and 203 reach
  about 94–99% sensitivity at their best setting (thesis: 54.7%, 60.7%, 62.6%, 81.8%).
* Weakest non-paced records here are 118 and 201 (about 86–87% sensitivity at best),
  then 222, 232, 207, 105, 108. None is discussed in the thesis.
* Early record-100-only results suggested a much bigger benefit for 3D; the full
  dataset shows the effect is smaller and comes with the trade-off above.

---

## 6. Issues found in the original code and thesis (to raise as questions)

| Item | Detail |
|---|---|
| Global setting | IRP = MRI = 0.35 s (thesis best global) gives poor results with the v5 code (record 100: Se 46%, PPV 41%). Probably a different code version. |
| Evaluation length | Notebook evaluates samples 180–4000 (about 10.6 s); thesis beat counts suggest about 1 minute per record. Full records give different results. |
| Non-beat labels | Notebook counts rhythm/noise annotations as beats. |
| Leads | Lead 0 is not MLII for 102 and 104 (V5); 114 has swapped leads. |
| Powerline filter | 50 Hz, but MIT-BIH has 60 Hz interference. |
| Candidate count | Search rounds per window = floor(3 s / IRP) − 1 (only 4 at IRP 0.6 s). Likely cause of the sensitivity drop above IRP 0.55 s in the thesis grid figure. |
| Eq. 63 | MOF threshold has a typo (score_min − score_min) and is not used in the code. |
| Z-score | Code only subtracts the window mean; it does not divide by the standard deviation. |
| Record 232 / 233 | Extension document says 232, thesis says 233. |

---

## 7. Currently running

Two fixes aimed at keeping the higher floor without losing the ceiling
(all 48 records, versions 1D, 3D, 5D, 6D):

| Run | Option | Idea | Output files |
|---|---|---|---|
| 1 | `--height-weight 2` | Height column counts twice as much as each context feature | `*_w2.csv` |
| 2 | `--centred` | Context features use 25 ms before and after each point instead of 50 ms before | `*_c.csv` |

Success = best setting no longer significantly worse than 1D, while the worst setting
stays significantly better.

---

## 8. Future plan

### Next experiments

1. Analyse Runs 1 and 2; if both help, run them together (`--height-weight 2 --centred`).
2. Try other height weights (e.g. 3) for the most promising version.
3. Further fixes if needed: robust scaling (median/IQR or ranks), combining 1D and
   enriched MOF scores, single-feature tests (1D + one feature) to find which feature
   causes the trade-off.
4. Plot records 118 and 201 to see why every version struggles there.
5. Track B (self-tuning IRP/MRI with a label-free quality score Q and a heart-rate check)
   on top of the best CEO-MOF version, so the ceiling is recovered automatically.
6. Robustness: MIT-BIH Noise Stress Test Database; test a 60 Hz powerline filter for MIT-BIH.
7. Other datasets: QT Database (and PQRST detection), INCART if time allows.

### Timeline (from meeting #3, 20 October 2026)

| Month | Plan | Status / notes |
|---|---|---|
| Oct 2026 | Set plan | Meeting #4 on 28 Oct: To-do list, thesis direction, research plan |
| Nov 2026 | Literature review, seminar slides | Seminar: MOW-ECG papers (JCMDS, IJACSA), O-HVOF, classic detectors |
| Dec 2026 | Datasets, run existing algorithms | MIT-BIH done; add QT DB, Noise Stress Test; baselines (Pan-Tompkins, etc.) |
| Jan 2027 | Implement CEO-MOF | Prototype and 48-record study already done |
| Feb 2027 | Novel algorithm on ECG | Trade-off fixes, Track B, noise robustness |
| Mar 2027 | Draft proposal and slides | |
| Apr 2027 | Proposal defence; draft first paper | |
| May–Jun 2027 | Results and discussion; submit first paper | |
| Jul–Aug 2027 | Thesis chapters 1–5 | |
| Sep–Dec 2027 | Check thesis, update paper, defence, final thesis | |

### Before meeting #4 (28 October 2026)

* Update To-do-list.xlsx (name in English and Thai with contacts, courses, thesis credits,
  English requirement, seminar papers, proposal plan).
* Present the CEO-MOF direction with the 48-record trade-off results.
* Ask about the open questions in Section 6.