# vim: set ts=2 sw=2 sts=2 et cc=75:
"""Active learning for optimization: spend a label budget to find a
good row. Warm-start W random labels, then acquire B-C more by
chasing the best/rest boundary, then spend C labels checking the
top of a held-out test set -- ranked either by the best/rest
centroid distance (no model) or by a forest/tree trained on the
labelled rows. One win per ranker, higher is better.

The leak line: inference (acquisition, the test ranker) sees only
the LABELLED rows (best, rest, both). The disty oracle and the win
grader may use every row's y -- that is the price of a label and
the scale of the score, never a feature.

usage: pypy3 active.py [set ...]      moot/optimize sets > a table
"""
# (c) 2026 Tim Menzies <timm@ieee.org>, MIT license
import os, sys, random
from math import sqrt
from balsa import (the, srand, csv, Tbl, Col, Forest, distx, mids)

the.warm, the.budget, the.check, the.pool = 4, 50, 5, 256

def shuffle(t):
  t = t[:]; random.shuffle(t); return t

# -- acquire -----------------------------------------------------------
def label(both, best, rest, row, y):
  "Pay for one label: the row joins best; best's worst spills to rest."
  both.add(best.add(row))
  while len(best.rows) > int(len(both.rows) ** .5):
    best.add(rest.add(max(best.rows, key=y)), -1)

def acquire(tb, y, warm, budget):
  "Label budget rows of a capped pool, chasing the best centroid."
  pool = shuffle(tb.rows)[:the.pool]
  head = [tb.cols.names]
  both, best, rest = Tbl(head), Tbl(head), Tbl(head)
  for _ in range(warm):
    label(both, best, rest, pool.pop(), y)
  while len(both.rows) < budget and pool:
    bm, rm = mids(best), mids(rest)
    *pool, row = sorted(pool,
                   key=lambda r: distx(both, r, rm) - distx(both, r, bm))
    label(both, best, rest, row, y)
  return both, best, rest

# -- test-set rankers: (tb, both, best, rest) -> a key over test rows --
def byDist(tb, both, best, rest):
  "No model: nearer best's centroid than rest's."
  bm, rm = mids(best), mids(rest)
  return lambda r: distx(both, r, bm) - distx(both, r, rm)

def byForest(tb, both, best, rest):
  "A regression forest on the labelled rows predicts d2h."
  m = Tbl([both.cols.names] + both.rows)
  y = lambda r: m.disty(r)
  the.trees, the.rf = 30, True
  return Forest(m, m.rows, y, Col).predict

def byTree(tb, both, best, rest):
  "One tree instead of thirty."
  m = Tbl([both.cols.names] + both.rows)
  y = lambda r: m.disty(r)
  the.trees, the.rf = 1, False
  return Forest(m, m.rows, y, Col).predict

RANKERS = {"dist": byDist, "forest": byForest, "tree": byTree}

# -- optimize ----------------------------------------------------------
def scorer(tb, y):
  "Grade a row 0..100 against the table's reachable min and mean."
  ys = [y(r) for r in tb.rows]
  lo, mu = min(ys), sum(ys) / len(ys)
  return lambda row: 100 * (1 - (y(row) - lo) / (mu - lo + 1e-32))

def optimize(tb, train, test, rank):
  "Acquire on train, rank test, check the top C, grade the best."
  y = lambda r: tb.disty(r)
  both, best, rest = acquire(tb, y, the.warm, the.budget - the.check)
  checked = sorted(test, key=rank(tb, both, best, rest))[:the.check]
  return scorer(tb, y)(min(checked, key=y))

# -- rig ---------------------------------------------------------------
def xval(rows, m=5, n=2):
  for _ in range(m):
    u = shuffle(rows)
    for kf in range(n):
      yield ([r for j, r in enumerate(u) if j % n != kf],
             [r for j, r in enumerate(u) if j % n == kf])

def run(sets):
  arms = list(RANKERS) + ["rand"]
  print("#%-28s %s" % ("data", " ".join("%7s" % a for a in arms)))
  tot = {a: [] for a in arms}
  for f in sets:
    root = os.getenv("MOOTOPT", "~/gits/moot/optimize")
    tb = Tbl(csv("%s/%s.csv" % (root, f)))
    got = {a: [] for a in arms}
    for train, test in xval(tb.rows):
      for a, rk in RANKERS.items():
        got[a].append(optimize(tb, train, test, rk))
      y = lambda r: tb.disty(r)
      win = scorer(tb, y)
      got["rand"].append(win(min(shuffle(test)[:the.check], key=y)))
    print("%-29s %s" % (f.split("/")[-1],
          " ".join("%7.1f" % (sum(got[a]) / len(got[a])) for a in arms)))
    for a in arms: tot[a].append(sum(got[a]) / len(got[a]))
  print("#%-28s %s" % ("mean",
        " ".join("%7.1f" % (sum(tot[a]) / len(tot[a])) for a in arms)))

if __name__ == "__main__":
  srand(the.seed)
  random.seed(the.seed)
  run(sys.argv[1:] or ["misc/auto93", "config/SS-A", "config/SS-B",
                       "systems/SS-N", "process/pom3a"])
