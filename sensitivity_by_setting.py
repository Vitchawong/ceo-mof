"""
Sensitivity and PPV at the best, median and worst IRP/MRI setting, for each representation.

Reads the grid_<record>_<rep>.csv files that compare_representations.py already saved
(all 80 settings per record), so nothing is recomputed. Takes a few seconds.

    python sensitivity_by_setting.py                    # original run (1D to 6D)
    python sensitivity_by_setting.py --tag _w2          # a variant run, e.g. height weight 2
    python sensitivity_by_setting.py --cache cache --reps "1D,3D,5D,6D"

"Median setting" = the setting whose composite is the middle one of the 80 (the 40th
when sorted), so its sensitivity and PPV come from one real setting.
Paced records (102, 104, 107, 217) are left out, as in the ANSI/AAMI EC57 standard.
Output: printed table and sensitivity_by_setting<tag>.csv in the cache folder.
"""
import argparse, glob, os, re
import pandas as pd

PACED = {"102", "104", "107", "217"}

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache")
ap.add_argument("--reps", default="1D,2D,3D,4D,5D,6D")
ap.add_argument("--tag", default="", help="variant suffix, e.g. _w2, _c or _w2_c")
args = ap.parse_args()

rows = []
for rep in args.reps.split(","):
    name = rep if rep == "1D" else rep + args.tag
    for f in glob.glob(os.path.join(args.cache, f"grid_*_{name}.csv")):
        rec = re.search(r"grid_(\d+)_", os.path.basename(f)).group(1)
        if rec in PACED:
            continue
        g = pd.read_csv(f).sort_values("Comp").reset_index(drop=True)
        picks = {"worst": g.iloc[0], "median": g.iloc[(len(g) - 1) // 2], "best": g.iloc[-1]}
        row = dict(rep=name, record=rec)
        for k, r in picks.items():
            row.update({f"{k}_Comp": r.Comp, f"{k}_Se": r.Se, f"{k}_PPV": r.PPV})
        rows.append(row)

df = pd.DataFrame(rows)
order = [r if r == "1D" else r + args.tag for r in args.reps.split(",")]
avg = df.groupby("rep").mean(numeric_only=True).reindex(order)
avg["records"] = df.groupby("rep").size().reindex(order)

pct = lambda x: f"{100 * x:5.1f}%"
print(f"Averages over {int(avg.records.max())} non-paced records\n")
print(f"{'rep':10} | {'best Comp':>9} {'Se':>7} {'PPV':>7} | {'median Comp':>11} {'Se':>7} {'PPV':>7} | "
      f"{'worst Comp':>10} {'Se':>7} {'PPV':>7}")
for rep, r in avg.iterrows():
    print(f"{rep:10} | {r.best_Comp:9.3f} {pct(r.best_Se):>7} {pct(r.best_PPV):>7} | "
          f"{r.median_Comp:11.3f} {pct(r.median_Se):>7} {pct(r.median_PPV):>7} | "
          f"{r.worst_Comp:10.2f} {pct(r.worst_Se):>7} {pct(r.worst_PPV):>7}")

out = os.path.join(args.cache, f"sensitivity_by_setting{args.tag}.csv")
avg.to_csv(out)
df.to_csv(os.path.join(args.cache, f"sensitivity_by_setting_per_record{args.tag}.csv"), index=False)
print(f"\nSaved {out}")
