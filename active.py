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
from balsa import (the, srand, csv, Tbl, Col, Num, Sym, Forest, distx, mids)

the.warm, the.budget, the.check, the.pool = 4, 50, 5, 256

def shuffle(t):
  t = t[:]; random.shuffle(t); return t

# -- acquire -----------------------------------------------------------
def label(both, best, rest, row, y):
  "Pay for one label: the row joins best; best's worst spills to rest."
  both.add(best.add(row))
  while len(best.rows) > int(len(both.rows) ** .5):
    best.add(rest.add(max(best.rows, key=y)), -1)

def acquire(tb, y, warm, budget, smart=True):
  "Label budget pool rows: smart chases the best centroid; else random."
  pool = shuffle(tb.rows)[:the.pool]
  head = [tb.cols.names]
  both, best, rest = Tbl(head), Tbl(head), Tbl(head)
  for _ in range(warm):
    label(both, best, rest, pool.pop(), y)
  while len(both.rows) < budget and pool:
    if smart:
      bm, rm = mids(best), mids(rest)
      pool.sort(key=lambda r: distx(both, r, rm) - distx(both, r, bm))
    label(both, best, rest, pool.pop(), y)     # last = best, or random
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

def byLgbm(tb, both, best, rest):
  "LightGBM regressor on the labelled rows predicts d2h (CPython only)."
  import numpy as np, lightgbm as lgb
  xs = tb.cols.x
  cats = [j for j, c in enumerate(xs) if isinstance(c, Sym)]
  def X(rows):
    return np.array([[r[c.at] if (isinstance(c, Num) or r[c.at] == "?")
                      else (c.labels.index(r[c.at]) if r[c.at] in c.labels
                            else len(c.labels))
                      for c in xs] for r in rows], dtype=float)
  ys = [tb.disty(r) for r in both.rows]
  m = lgb.LGBMRegressor(verbose=-1, n_jobs=1)
  m.fit(X(both.rows), ys, categorical_feature=cats)
  pred = {}
  def key(r):                                  # cache a one-shot batch
    if not pred:
      pred.update(zip(map(id, key.test), m.predict(X(key.test))))
    return pred[id(r)]
  key.test = None                              # optimize fills this in
  return key

def byRand(tb, both, best, rest):
  "Floor: ignore the model, rank the holdout at random."
  import random
  return lambda r: random.random()

RANKERS = {"dist": byDist, "b1": byTree, "b30": byForest,
           "lgbm": byLgbm, "rand": byRand}

# -- optimize ----------------------------------------------------------
def scorer(tb, y):
  "Grade a row 0..100 against the table's reachable min and mean."
  ys = [y(r) for r in tb.rows]
  lo, mu = min(ys), sum(ys) / len(ys)
  return lambda row: 100 * (1 - (y(row) - lo) / (mu - lo + 1e-32))

def optimize(tb, train, test, rank, smart=True):
  "Acquire on train, rank test, check the top C, grade the best."
  y = lambda r: tb.disty(r)
  both, best, rest = acquire(tb, y, the.warm, the.budget - the.check, smart)
  key = rank(tb, both, best, rest)
  if hasattr(key, "test"): key.test = test     # byLgbm batches once
  checked = sorted(test, key=key)[:the.check]
  return scorer(tb, y)(min(checked, key=y))

# -- rig ---------------------------------------------------------------
def xval(rows, m=5, n=2):
  for _ in range(m):
    u = shuffle(rows)
    for kf in range(n):
      yield ([r for j, r in enumerate(u) if j % n != kf],
             [r for j, r in enumerate(u) if j % n == kf])

ARMS = ["%s.%s" % (a, r) for a in "ar" for r in RANKERS]  # 2x5

def run(sets):
  print("#%-22s %s" % ("data", " ".join("%7s" % a for a in ARMS)))
  tot = {a: [] for a in ARMS}
  for f in sets:
    root = os.getenv("MOOTOPT", "~/gits/moot/optimize")
    tb = Tbl(csv("%s/%s.csv" % (root, f)))
    got = {a: [] for a in ARMS}
    random.seed(the.seed)
    folds = list(xval(tb.rows))                 # fixed splits per set
    for fi, (train, test) in enumerate(folds):
      for aq, smart in (("a", True), ("r", False)):
        for rn, rk in RANKERS.items():
          random.seed(the.seed + fi)            # same pool for every arm
          got["%s.%s" % (aq, rn)].append(
            optimize(tb, train, test, rk, smart))
    print("%-23s %s" % (f.split("/")[-1],
          " ".join("%7.1f" % (sum(got[a]) / len(got[a])) for a in ARMS)))
    for a in ARMS: tot[a].append(sum(got[a]) / len(got[a]))
  print("#%-22s %s" % ("mean",
        " ".join("%7.1f" % (sum(tot[a]) / len(tot[a])) for a in ARMS)))

if __name__ == "__main__":
  srand(the.seed)
  random.seed(the.seed)
  run(sys.argv[1:] or ["misc/auto93", "config/SS-A", "config/SS-B",
                       "systems/SS-N", "process/pom3a"])
