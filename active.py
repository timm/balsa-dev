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
import os, sys, time, random
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
  def cell(c, v):
    if v == "?": return float("nan")             # lgbm handles nan natively
    if isinstance(c, Num): return float(v)
    return c.labels.index(v) if v in c.labels else len(c.labels)
  def X(rows):
    return np.array([[cell(c, r[c.at]) for c in xs] for r in rows],
                    dtype=float)
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

def byTabicl(tb, both, best, rest):
  "TabICL in-context regressor on the labelled rows predicts d2h."
  for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(v, "1")
  import numpy as np
  from tabicl import TabICLRegressor
  xs = tb.cols.x
  def cell(c, v):
    if v == "?": return float("nan")
    if isinstance(c, Num): return float(v)
    return c.labels.index(v) if v in c.labels else len(c.labels)
  def X(rows):
    a = np.array([[cell(c, r[c.at]) for c in xs] for r in rows], dtype=float)
    for c in range(a.shape[1]):                  # TabICL takes no NaN
      m = np.nanmedian(a[:, c])
      a[np.isnan(a[:, c]), c] = 0.0 if np.isnan(m) else m
    return a
  Xtr = X(both.rows)
  ys = np.array([tb.disty(r) for r in both.rows])
  m = TabICLRegressor(batch_size=1, device="cpu")
  m.fit(Xtr, ys)
  pred = {}
  def key(r):                                    # cache a one-shot batch
    if not pred:
      pred.update(zip(map(id, key.test), m.predict(X(key.test))))
    return pred[id(r)]
  key.test = None                                # optimize fills this in
  return key

def byRand(tb, both, best, rest):
  "Floor: ignore the model, rank the holdout at random."
  import random
  return lambda r: random.random()

RANKERS = {"dist": byDist, "b1": byTree, "b30": byForest,
           "lgbm": byLgbm, "tabicl": byTabicl, "rand": byRand}

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

# Which rankers to run this pass. ARMS=dist,b1,b30,rand under pypy
# (fast, no lightgbm); ARMS=lgbm under CPython. The fold shuffles are
# the same Mersenne Twister on both, so the two passes align exactly.
WANT = os.getenv("ARMS", ",".join(RANKERS)).split(",")
ACQS = os.getenv("ACQS", "ar")                    # "a" = acquire only
ARMS = ["%s.%s" % (a, r) for a in ACQS for r in RANKERS if r in WANT]

def run(sets):
  for f in sets:
    root = os.getenv("MOOTOPT", "~/gits/moot/optimize")
    tb = Tbl(csv("%s/%s.csv" % (root, f)))
    got = {a: [] for a in ARMS}
    sec = {a: [] for a in ARMS}
    random.seed(the.seed)
    folds = list(xval(tb.rows))                 # fixed splits per set
    for fi, (train, test) in enumerate(folds):
      for aq, smart in (("a", True), ("r", False)):
        if aq not in ACQS: continue
        for rn in WANT:
          random.seed(the.seed + fi)            # same pool for every arm
          t0 = time.process_time()
          got["%s.%s" % (aq, rn)].append(
            optimize(tb, train, test, RANKERS[rn], smart))
          sec["%s.%s" % (aq, rn)].append(time.process_time() - t0)
    for a in ARMS:                              # one line per (set,arm)
      print("%%folds %s %s %s" % (f, a, " ".join("%.1f" % x for x in got[a])),
            flush=True)
      print("%%secs %s %s %s" % (f, a, " ".join("%.4f" % x for x in sec[a])),
            flush=True)

def expand(args):
  "A .txt arg is a file of set names, one per line; else a set name."
  out = []
  for a in args:
    if a.endswith(".txt"):
      out += [l.split()[0] for l in open(a) if l.split()]
    else: out.append(a)
  return out

if __name__ == "__main__":
  srand(the.seed)
  random.seed(the.seed)
  run(expand(sys.argv[1:]) or ["misc/auto93", "config/SS-A"])
