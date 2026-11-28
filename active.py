# vim: set ts=2 sw=2 sts=2 et cc=75:
"""Active learning for optimization: spend a label budget to find a
good row. Warm-start W random labels, then acquire B-C more by
chasing the best/rest boundary, then spend C labels checking the
top of a held-out test set -- ranked by the nearest labelled
neighbour, trees, or a boosted/pretrained regressor fit to the
labelled rows. One win per ranker, higher is better.

The leak line: inference (acquisition, the test ranker) sees only
the LABELLED rows (best, rest, both). The disty oracle and the win
grader may use every row's y -- that is the price of a label and
the scale of the score, never a feature.

usage: pypy3 active.py [set ...]      moot/optimize sets > a table
"""
# (c) 2026 Tim Menzies <timm@ieee.org>, MIT license
import os, sys, time, random
from enuf import (the, srand, csv, Tbl, Col, Num, Sym, Forest,
                   distx, mids)

the.warm, the.budget, the.check, the.pool = 4, 50, 5, 256
the.bexp = .5                                  # best holds n**bexp of both

def shuffle(t):
  t = t[:]; random.shuffle(t); return t

# -- acquire -----------------------------------------------------------
def label(both, best, rest, row, y):
  "Pay for one label: the row joins best; best's worst spills to rest."
  both.add(best.add(row))
  while len(best.rows) > int(len(both.rows) ** the.bexp):
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

# -- test-set rankers: (tb, both, best, rest) -> a key over test rows.
# None of the live rankers reads best/rest, but the slot stays: the
# centroid variants in dist_sweep.py plug in through the same socket.
def byKnn(tb, both, best, rest):
  "No model: the d2h of the nearest labelled row."
  xy = [(x, tb.disty(x)) for x in both.rows]   # label ys once, not per row
  return lambda r: min((distx(both, r, x), y) for x, y in xy)[1]

def byTrees(trees, rf):
  "A forest (or one tree) on the labelled rows predicts d2h."
  def rank(tb, both, best, rest):
    the.trees, the.rf, m = trees, rf, Tbl([both.cols.names] + both.rows)
    return Forest(m, m.rows, lambda r: m.disty(r), Col).predict
  return rank

def matrix(tb, rows, impute=False):
  "Rows as floats: ? is nan (or the column median), syms their code."
  import numpy as np
  def cell(c, v):
    return (float("nan") if v == "?" else
            float(v) if isinstance(c, Num) else
            c.labels.index(v) if v in c.labels else len(c.labels))
  a = np.array([[cell(c, r[c.at]) for c in tb.cols.x] for r in rows],
               dtype=float)
  if impute:                                   # TabICL takes no NaN
    for j in range(a.shape[1]):
      m = np.nanmedian(a[:, j])
      a[np.isnan(a[:, j]), j] = 0.0 if np.isnan(m) else m
  return a

def batch(model, tb, impute=False):
  "A sort key that predicts the whole test set in one call."
  pred = {}
  def key(r):
    if not pred:
      pred.update(zip(map(id, key.test),
                      model.predict(matrix(tb, key.test, impute))))
    return pred[id(r)]
  key.test = None                              # optimize fills this in
  return key

def byModel(fit, impute=False):
  "Fit any regressor to the labelled x,y; batch-rank the holdout."
  def rank(tb, both, best, rest):
    X = matrix(tb, both.rows, impute)
    y = [tb.disty(r) for r in both.rows]
    return batch(fit(tb, X, y), tb, impute)
  return rank

def lgbm(tb, X, y):
  "LightGBM; nan is native, syms are declared (CPython only)."
  import lightgbm as lgb
  m = lgb.LGBMRegressor(verbose=-1, n_jobs=1)
  m.fit(X, y, categorical_feature=[j for j, c in enumerate(tb.cols.x)
                                   if isinstance(c, Sym)])
  return m

def tabicl(tb, X, y):
  "TabICL in-context; pin its threads or it eats the machine."
  for v in ("OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS "
            "NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS "
            "LOKY_MAX_CPU_COUNT").split():
    os.environ.setdefault(v, "1")
  from tabicl import TabICLRegressor
  m = TabICLRegressor(batch_size=1, device="cpu")
  m.fit(X, y)
  return m

def byRand(tb, both, best, rest):
  "Floor: ignore the model, rank the holdout at random."
  return lambda r: random.random()

RANKERS = {"knn": byKnn, "b1": byTrees(1, False),
           "b30": byTrees(30, True), "lgbm": byModel(lgbm),
           "tabicl": byModel(tabicl, True), "rand": byRand}

# -- optimize ----------------------------------------------------------
def scorer(tb, y):
  "Grade a row 0..100 against the table's reachable min and mean."
  ys = [y(r) for r in tb.rows]
  lo, mu = min(ys), sum(ys) / len(ys)
  return lambda row: 100 * (1 - (y(row) - lo) / (mu - lo + 1e-32))

def optimize(tb, train, test, rank, smart=True):
  "Acquire on train, rank test, check the top C, grade the best."
  y, b = (lambda r: tb.disty(r)), the.budget - the.check
  both, best, rest = acquire(tb, y, the.warm, b, smart)
  key = rank(tb, both, best, rest)
  if hasattr(key, "test"): key.test = test     # batch rankers fill once
  checked = sorted(test, key=key)[:the.check]
  return scorer(tb, y)(min(checked, key=y))

# -- rig ---------------------------------------------------------------
def xval(rows, m=5, n=2):
  for _ in range(m):
    u = shuffle(rows)
    for kf in range(n):
      yield ([r for j, r in enumerate(u) if j % n != kf],
             [r for j, r in enumerate(u) if j % n == kf])

# Which arms to run this pass. ARMS=knn,b1,b30,rand under pypy
# (fast, no numpy); ARMS=lgbm,tabicl under CPython. ACQS="a" skips the
# random-acquisition twin. The fold shuffles are the same Mersenne
# Twister everywhere, so separate passes align exactly.
WANT = os.getenv("ARMS", ",".join(RANKERS)).split(",")
ACQS = os.getenv("ACQS", "ar")
ARMS = ["%s.%s" % (a, r) for a in ACQS for r in RANKERS if r in WANT]

def run(sets):
  root = os.getenv("MOOTOPT", "~/gits/moot/optimize")
  for f in sets:
    tb = Tbl(csv("%s/%s.csv" % (root, f)))
    got, sec = {a: [] for a in ARMS}, {a: [] for a in ARMS}
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
    for a in ARMS:                              # two lines per (set,arm)
      for tag, fmt, v in (("folds", "%.1f", got), ("secs", "%.4f", sec)):
        print("%%%s %s %s %s" % (tag, f, a,
              " ".join(fmt % x for x in v[a])), flush=True)

def expand(args):
  "A .txt arg is a file of set names, one per line; else a set name."
  return [s for a in args for s in
          ([l.split()[0] for l in open(a) if l.split()]
           if a.endswith(".txt") else [a])]

if __name__ == "__main__":
  srand(the.seed)
  random.seed(the.seed)
  run(expand(sys.argv[1:]) or ["misc/auto93", "config/SS-A"])
