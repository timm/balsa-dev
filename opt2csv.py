#!/usr/bin/env python3 -B
"""opt2csv: an optimization run's .txt into the csv the plots read.

lgbm_opt.py already prints every arm's per-fold SCORE on %folds
lines -- tree(1), tree(30), random and lgbm alike -- so unlike the
regression pipeline there is nothing to reconstruct here, only to
transpose:

    data,fold,score_tree1,score_tree30,score_random,score_lgbm

Each score is 100 * (1 - (got - lo) / (mu - lo)) for that fold's
labelled winner, against the whole table's min and mean d2h.  100 is
the table's own optimum and HIGHER IS BETTER -- the opposite of the
regression csvs, whose rel_ columns are ratios to beat downwards.

usage: python3 opt2csv.py runs/opt.txt > runs/opt.csv
"""
import csv, sys, collections

ARMS = (("tree(1)", "tree1"), ("tree(30)", "tree30"),
        ("random", "random"), ("lgbm", "lgbm"))


def main():
    got = collections.defaultdict(dict)          # set -> arm -> per fold
    order = []
    for line in open(sys.argv[1]):
        if not line.startswith("%folds"): continue
        _, f, arm, *vals = line.split()
        if f not in got: order.append(f)
        got[f][arm] = [float(v) for v in vals]
    w = csv.writer(sys.stdout)
    w.writerow(["data", "fold"] + ["score_%s" % o for _, o in ARMS])
    for f in order:
        n = max(len(v) for v in got[f].values())
        for k in range(n):
            row = [f, k]
            for arm, _ in ARMS:
                v = got[f].get(arm, [])
                row.append("%.2f" % v[k] if k < len(v) else "")
            w.writerow(row)


if __name__ == "__main__":
    main()
