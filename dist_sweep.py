# vim: set ts=2 sw=2 sts=2 et cc=75:
"""dist_sweep: hunt a better setting for the no-model dist ranker.
Sweeps the three knobs dist has -- the Minkowski p, the best-table
boundary exponent, the best/rest score -- plus a kNN variant that
ranks a test row by the mean d2h of its k nearest labelled rows.
Every variant replays the same folds and pools as active.py, so the
one number per variant is comparable with the published 80.4.

usage: pypy3 dist_sweep.py runs/active-sets.txt
"""
# (c) 2026 Tim Menzies <timm@ieee.org>, MIT license
import os, sys, random
from balsa import the, srand, csv, Tbl, distx, mids
from active import optimize, xval, expand

def mkDist(score):
  "A byDist with a pluggable best/rest score."
  def rank(tb, both, best, rest):
    bm, rm = mids(best), mids(rest)
    return lambda r: score(distx(both, r, bm), distx(both, r, rm))
  return rank

def mkKnn(k):
  "No centroid: mean d2h of the k nearest labelled rows."
  def rank(tb, both, best, rest):
    def key(r):
      ds = sorted((distx(both, r, x), tb.disty(x)) for x in both.rows)
      return sum(y for _, y in ds[:k]) / k
    return key
  return rank

SCORES = {"b-r":     lambda b, r: b - r,
          "b/(b+r)": lambda b, r: b / (b + r + 1E-32),
          "b":       lambda b, r: b}

VARIANTS = (
  [("p=%s %s" % (p, s), p, .5, mkDist(f))
   for p in (1, 2, 4) for s, f in SCORES.items()] +
  [("p=2 b-r bexp=%s" % b, 2, b, mkDist(SCORES["b-r"]))
   for b in (.33, .67)] +
  [("p=2 knn k=%s" % k, 2, .5, mkKnn(k)) for k in (1, 3, 5)])

def run(sets):
  root = os.getenv("MOOTOPT", "~/gits/moot/optimize")
  got = {name: [] for name, *_ in VARIANTS}
  for f in sets:
    tb = Tbl(csv("%s/%s.csv" % (root, f)))
    random.seed(the.seed)
    folds = list(xval(tb.rows))
    for name, p, bexp, rank in VARIANTS:
      the.p, the.bexp = p, bexp
      v = []
      for fi, (train, test) in enumerate(folds):
        random.seed(the.seed + fi)             # same pool as active.py
        v.append(optimize(tb, train, test, rank, True))
      got[name].append(sum(v) / len(v))
    print("#done %s" % f, file=sys.stderr, flush=True)
  for name, *_ in sorted(VARIANTS, key=lambda v: -sum(got[v[0]])):
    mu = sum(got[name]) / len(got[name])
    print("%-20s %5.1f" % (name, mu), flush=True)

if __name__ == "__main__":
  srand(the.seed)
  random.seed(the.seed)
  run(expand(sys.argv[1:]) or ["misc/auto93"])
