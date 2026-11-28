#!/usr/bin/env python3 -B
"""reg2csv: turn a regression run's .dat and .txt into the csv the plots read.

The .dat (from tree_eg --reg) holds raw MAE per fold:
    data | mae_tree1 mae_tree30 mae_mean | secs | test row ids
The .txt (from lgbm_reg.py) holds the same folds as RELATIVE mae, on
%folds lines, with LightGBM added.  An optional third file (from
tabicl_reg.py) adds TabICL on %folds lines of its own.  This joins them
into one row per (data, fold), with both the raw and the relative
numbers:

    data,fold,mae_tree1,mae_tree30,mae_mean,mae_lgbm,
              rel_tree1,rel_tree30,rel_lgbm,rel_tabicl

rel_X = mae_X / mae_mean, so 1.0 is no better than predicting the mean.
mae_lgbm is recovered as rel_lgbm * mae_mean, since the .txt keeps only
the ratio.  rel_tabicl is empty on the sets TabICL did not run: it is
capped by rows, and the readers treat an empty cell as "not run" rather
than as a score.

usage: python3 reg2csv.py runs/sereg.dat runs/sereg.txt [runs/sereg-tabicl.txt]
       > runs/sereg.csv
"""
import sys, collections

ARMS = ("tree(1)", "tree(30)", "lgbm")
OUT  = ("tree1", "tree30", "lgbm")


def main():
    dat, txt = sys.argv[1], sys.argv[2]

    raw = collections.defaultdict(list)          # set -> [(m1, m30, mmean)...]
    for line in open(dat):
        if "|" not in line or line.startswith("#"): continue
        f, maes, *_ = line.split("|")
        raw[f].append([float(v) for v in maes.split()])

    rel = collections.defaultdict(dict)          # set -> arm -> [per fold]
    for line in open(txt):
        if not line.startswith("%folds"): continue
        _, f, arm, *vals = line.split()
        rel[f][arm] = [float(v) for v in vals]

    tab = collections.defaultdict(list)          # set -> [per fold]
    if len(sys.argv) > 3:
        for line in open(sys.argv[3]):
            if not line.startswith("%folds"): continue
            _, f, _arm, *vals = line.split()
            tab[f] = [float(v) for v in vals]

    print("data,fold,mae_tree1,mae_tree30,mae_mean,mae_lgbm,"
          "rel_tree1,rel_tree30,rel_lgbm,rel_tabicl")
    for f in raw:
        if f not in rel: continue                # lgbm skipped this set
        for i, (m1, m30, mmean) in enumerate(raw[f]):
            r = [rel[f][a][i] for a in ARMS]
            mlgbm = r[2] * mmean                 # the .txt keeps only ratios
            t = f"{tab[f][i]:.4f}" if i < len(tab[f]) else ""
            print(f"{f},{i},{m1:.6g},{m30:.6g},{mmean:.6g},{mlgbm:.6g},"
                  f"{r[0]:.4f},{r[1]:.4f},{r[2]:.4f},{t}")


if __name__ == "__main__": main()
