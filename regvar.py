#!/usr/bin/env python3 -B
"""regvar: where does a regression score move from?  The variance split of
the paper's Table 7, run on rel = MAE / MAE(predict the mean).

Two-way ANOVA with replication, data set crossed with learner, R folds per
cell, reported as eta-squared.  Four shares, summing to 100:

    data     = JR sum_i (ybar_i - yhat)^2  x 100 / S
    learner  = IR sum_j (ybar_j - yhat)^2  x 100 / S
    fold     =    sum    (y_ijr - ybar_ij)^2 x 100 / S
    other    = 100 - data - learner - fold

rel is used, not raw MAE: regression targets carry their own units, so a
split on raw MAE only reports that cal.housing has bigger numbers than
bolts -- it gives the data set 90% and the learner 0.0%.

The learners are balsa(30), LightGBM and TabICL.  balsa(1) is left out,
as in the paper's classification split: a deliberately crippled single
tree would inflate the learner share with a gap no reader is choosing
between.  The mean-predictor is left out too -- it is the divisor, not a
learner, and its rel is 1.0 on every fold, a constant column that would
distort both the learner and the interaction terms.

The decomposition needs a full data-set x learner x fold grid, and TabICL
is capped by rows (see tabicl_reg.py), so every row covers only the sets
TabICL finished.

usage: python3 regvar.py runs/reg56.csv runs/sereg.csv
"""
import csv, os, sys, collections, statistics as st

# balsa(1) is left out, as in the paper's classification split: it is a
# deliberately crippled single tree, and the question the decomposition
# asks is what separates the learners a reader would actually consider.
THREE = ("balsa30", "lgbm")
FOUR  = THREE + ("tabicl",)
ARMS  = THREE


def folds(path):
    "set -> list of rows, in file order."
    D = collections.defaultdict(list)
    for r in csv.DictReader(open(path)): D[r["data"]].append(r)
    return D


def split(D, sets, ARMS=None):
    "eta-squared for data set, learner, fold, interaction.  Sums to 100."
    ARMS = ARMS or THREE
    I, J = len(sets), len(ARMS)
    R = len(D[sets[0]])
    val = lambda s, r, a: float(r[f"rel_{a}"])
    y = [val(s, r, a) for s in sets for r in D[s] for a in ARMS]
    hat = st.fmean(y)
    row = {s: st.fmean([val(s, r, a) for r in D[s] for a in ARMS]) for s in sets}
    col = {a: st.fmean([val(s, r, a) for s in sets for r in D[s]]) for a in ARMS}
    cell = {(s, a): st.fmean([val(s, r, a) for r in D[s]])
            for s in sets for a in ARMS}
    tot = sum((v - hat) ** 2 for v in y)
    d = 100 * J * R * sum((row[s] - hat) ** 2 for s in sets) / tot
    m = 100 * I * R * sum((col[a] - hat) ** 2 for a in ARMS) / tot
    f = 100 * sum((val(s, r, a) - cell[(s, a)]) ** 2
                  for s in sets for r in D[s] for a in ARMS) / tot
    return d, m, f, 100 - d - m - f


def groups(noxse, se):
    "The seven panels of the two figures, in their plotting order."
    MOOT = os.path.expanduser("~/gits/moot/regression")
    n, numeric = {}, {}
    for s in noxse:
        L = [l for l in open(f"{MOOT}/{s}.csv") if l.strip()]
        head = [h.strip() for h in L[0].split(",")]
        n[s] = len(L) - 1
        x = [h for h in head
             if not h.rstrip("+-!").endswith("X") and h[-1] not in "+-!"]
        numeric[s] = all(h[0].isupper() for h in x)
    G = [("numeric",      [s for s in noxse if n[s] < 1000 and numeric[s]]),
         ("mixed",        [s for s in noxse if n[s] < 1000 and not numeric[s]]),
         ("large, mixed", [s for s in noxse if n[s] >= 1000])]
    for d in sorted({s.split("/")[0] for s in se}):
        G.append((d, [s for s in se if s.split("/")[0] == d]))
    return G


def main():
    A, B = folds(sys.argv[1]), folds(sys.argv[2])
    both = {**A, **B}
    G = groups(sorted(A), sorted(B))
    ran = {s for s in both
           if all(r.get("rel_tabicl", "") != "" for r in both[s])}

    def table(arms, keep):
        print(f"{'group':16s}{'sets':>5}{'data':>7}{'learner':>9}{'fold':>7}"
              f"{'other':>7}{'data/learner':>14}")
        def row(name, g):
            g = sorted(s for s in g if s in keep)
            if not g: return
            d, m, f, o = split(both, g, arms)
            print(f"{name:16s}{len(g):5d}{d:7.0f}{m:9.0f}{f:7.0f}{o:7.0f}"
                  f"{d / m:13.0f}x")
        for name, g in G: row(name, g)
        print()
        row("ALL non-SE", A); row("ALL SE", B); row("EVERYTHING", both)

    print(f"=== {', '.join(FOUR)} -- the {len(ran)} sets "
          f"TabICL finished\n")
    table(FOUR, ran)


if __name__ == "__main__": main()
