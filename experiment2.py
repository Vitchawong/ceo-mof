"""
Show the results of Experiment 2 (height weight 2) next to the original versions.

Reads files already in the cache folder; nothing is recomputed.
    python show_experiment2.py                 # Experiment 2 (tag _w2)
    python show_experiment2.py --tag _c        # the centred run, when it is done
    python show_experiment2.py --cache cache

Prints:
  1. Averages over the 44 non-paced records: 1D, the original versions and the weighted versions
  2. Wilcoxon tests of the weighted versions vs 1D (from wilcoxon_tests<tag>.csv)
  3. Wilcoxon tests of each weighted version vs its original (e.g. 3D_w2 vs 3D)
Saves experiment<tag>_results.csv in the cache folder.
"""
import argparse, os
import pandas as pd
from scipy.stats import wilcoxon

PACED = {"102", "104", "107", "217"}

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache")
ap.add_argument("--tag", default="_w2")
ap.add_argument("--reps", default="3D,5D,6D")
args = ap.parse_args()
tag, reps = args.tag, args.reps.split(",")

orig = pd.read_csv(os.path.join(args.cache, "per_record.csv"), dtype={"record": str})
new = pd.read_csv(os.path.join(args.cache, f"per_record{tag}.csv"), dtype={"record": str})
new = new[new.rep != "1D"]                       # 1D is the same in both runs
both = pd.concat([orig, new], ignore_index=True)

# 1. Averages over non-paced records
order = ["1D"] + [v for r in reps for v in (r, r + tag)]
cols = ["best_Comp", "best_Se", "best_PPV", "median_Comp", "worst_Comp"]
np_ = both[~both.record.isin(PACED)]
avg = np_.groupby("rep")[cols].mean().reindex([o for o in order if o in set(np_.rep)])

print(f"1. Averages over {np_.record.nunique()} non-paced records\n")
print(f"{'version':10} {'best Comp':>10} {'Se at best':>11} {'PPV at best':>12} {'median Comp':>12} {'worst Comp':>11}")
for rep, r in avg.iterrows():
    print(f"{rep:10} {r.best_Comp:10.3f} {100 * r.best_Se:10.1f}% {100 * r.best_PPV:11.1f}% "
          f"{r.median_Comp:12.3f} {r.worst_Comp:11.2f}")

# 2. Wilcoxon tests vs 1D (as saved by compare_representations.py)
f = os.path.join(args.cache, f"wilcoxon_tests{tag}.csv")
if os.path.exists(f):
    print(f"\n2. Wilcoxon tests vs 1D (all records) from {os.path.basename(f)}\n")
    w = pd.read_csv(f)
    for _, r in w.iterrows():
        print(f"  {r.comparison:14} {r.measure:18} wins/ties/losses {r.wins}/{r.ties}/{r.losses}  p = {r.p_value:.4f}")

# 3. Weighted version vs its original version, record by record
print(f"\n3. Each {tag} version vs its original (all records; wins/ties/losses for the {tag} version)\n")
rows = []
for rep in reps:
    a = both[both.rep == rep + tag].set_index("record")
    b = both[both.rep == rep].set_index("record")
    for col, label in [("best_Comp", "best setting"), ("median_Comp", "median setting"), ("worst_Comp", "worst setting")]:
        d = pd.concat([a[col], b[col]], axis=1, keys=["new", "old"]).dropna()
        wins, losses = int((d.new > d.old).sum()), int((d.new < d.old).sum())
        ties = len(d) - wins - losses
        p = wilcoxon(d.new, d.old).pvalue if len(d) >= 5 and (d.new != d.old).any() else float("nan")
        print(f"  {rep + tag:10} vs {rep:3} {label:15} {wins}/{ties}/{losses}  p = {p:.4f}")
        rows.append(dict(comparison=f"{rep + tag} vs {rep}", measure=label, records=len(d),
                         wins=wins, ties=ties, losses=losses, p_value=p))

out = os.path.join(args.cache, f"experiment{tag}_results.csv")
avg.reset_index().assign(table="averages_non_paced").to_csv(out, index=False)
pd.DataFrame(rows).to_csv(out.replace(".csv", "_vs_original.csv"), index=False)
print(f"\nSaved {out} and {os.path.basename(out.replace('.csv', '_vs_original.csv'))}")