#!/usr/bin/env python3 -B
"""lgbm_opt: add LightGBM to tree_eg --opt's table, on the same
5x2 folds.  Reads opt.dat (data|scores|secs|test row ids), refits
LGBMRegressor on each fold's train d2h, sorts that fold's test rows
by predicted d2h, LABELS THE BEST 5, and keeps the winner.

Score = 100 * (1 - (got - lo) / (mu - lo)), where lo and mu are the
min and mean d2h of the WHOLE table: 100 = found the table's
optimum, 0 = no better than an average row.

d2h here must match tree.yue's disty digit-for-digit or the arms
are not comparable: a goal column's name ends - (minimize, heaven
0) or + (maximize, heaven 1); each value is a logistic z-score
(1.7, clipped at 3 sd) against that column's mu/sd over every row;
d2h is their euclidean distance to heaven, over known cells only.

usage: python3 lgbm_opt.py runs/opt.dat > runs/opt.txt"""
import os, sys, time, math, warnings
import numpy as np, lightgbm as lgb
warnings.filterwarnings("ignore")

ROOT = os.getenv("MOOTOPT", "~/gits/moot/optimize")


def load(f):
  "x matrix, d2h per row, categorical column indexes"
  rows = [l.strip().split(",") for l in
          open(os.path.expanduser("%s/%s.csv" % (ROOT, f))) if l.strip()]
  head, rows = [h.strip() for h in rows[0]], rows[1:]
  ys = [i for i, h in enumerate(head)
        if h[-1] in "+-!" and not h.rstrip("+-!").endswith("X")]
  xs = [i for i, h in enumerate(head)
        if i not in ys and not h.rstrip("+-!").endswith("X")]

  def nums(i):
    return [np.nan if r[i] == "?" else float(r[i]) for r in rows]

  def col(i):
    if head[i][0].isupper(): return nums(i)
    seen = {}                           # first-seen order, like tree
    out = []
    for r in rows:
      v = r[i]
      if v == "?": out.append(np.nan); continue
      if v not in seen: seen[v] = len(seen)
      out.append(seen[v])
    return out

  d2h = np.zeros(len(rows))
  n = np.zeros(len(rows))
  for i in ys:
    v = np.array(nums(i))
    ok = ~np.isnan(v)
    mu = v[ok].mean()
    sd = v[ok].std(ddof=1) if ok.sum() > 1 else 0.0
    z = np.clip((v - mu) / (sd + 1e-32), -3, 3)
    norm = 1 / (1 + np.exp(-1.7 * z))
    goal = 0.0 if head[i].endswith("-") else 1.0
    d2h[ok] += (norm[ok] - goal) ** 2
    n[ok] += 1
  d2h = np.sqrt(d2h / np.maximum(n, 1))
  cats = [j for j, i in enumerate(xs) if not head[i][0].isupper()]
  return np.array([col(i) for i in xs]).T, d2h, cats


def main():
  lines = open(sys.argv[1]).readlines()
  T = ["tree(1)", "tree(30)", "random", "lgbm"]
  folds = {}
  for line in lines[1:]:
    f, a, s, t = line.strip().split("|")
    folds.setdefault(f, []).append(([float(v) for v in a.split()],
                                    [float(v) for v in s.split()],
                                    [int(v) for v in t.split()]))
  W, means, secs, per = 12, {t: [] for t in T}, dict.fromkeys(T, 0.0), {}
  print("%-34s" % "#data" + "".join("%-*s" % (W, t) for t in T))
  for f, fs in folds.items():
    X, d2h, cats = load(f)
    lo, mu = float(d2h.min()), float(d2h.mean())
    span = mu - lo + 1e-32
    d = {t: [] for t in T}
    for a, s, test in fs:
      for t, v in zip(T[:3], a): d[t].append(v)
      secs["tree(1)"] += s[0]; secs["tree(30)"] += s[1]
      tset  = set(test)
      train = [i for i in range(len(d2h)) if i not in tset]
      t0 = time.time()
      m = lgb.LGBMRegressor(verbose=-1, n_jobs=1)
      m.fit(X[train], d2h[train], categorical_feature=cats)
      got  = m.predict(X[test])
      best = min(d2h[test[i]] for i in np.argsort(got)[:5])
      secs["lgbm"] += time.time() - t0
      d["lgbm"].append(100 * (1 - (best - lo) / span))
    per[f] = d
    cells = []
    for t in T:
      v = sum(d[t]) / len(d[t]); means[t].append(v)
      cells.append("%6.1f" % v)
    print("%-34s" % f + "".join("%-*s" % (W, c) for c in cells), flush=True)
  print("\n%-34s" % "#mean score" + "".join(
        "%6.1f      " % (sum(means[t]) / len(means[t])) for t in T))
  print("%-34s" % "#secs (all data)" + "".join(
        "%6.1f      " % secs[t] for t in T))
  for f, d in per.items():
    for t in T:
      print("%%folds %s %s %s" % (f, t, " ".join("%.2f" % v for v in d[t])))


if __name__ == "__main__":
  main()
