#!/usr/bin/env python3 -B
"""lgbm_reg: add LightGBM regression to balsa_eg --reg's table, on
the same 5x2 folds.  Reads reg60.dat (data|mae1 mae30 maeMean|secs|
test row ids), refits LGBMRegressor per fold on the complement of
the recorded test rows, and prints one table plus %folds lines of
RELATIVE MAE: mae / guess-the-mean mae (0 = perfect, 1 = no better
than the mean, >1 = worse).
usage: python3 lgbm_reg.py runs/reg60.dat > runs/reg60.txt"""
import os, sys, time, warnings
import numpy as np, lightgbm as lgb
warnings.filterwarnings("ignore")

def load(f):
  "x matrix (nums float, syms category codes), target list, cats"
  rows = [l.strip().split(",") for l in open(
          os.path.expanduser("~/gits/moot/regression/%s.csv" % f))
          if l.strip()]
  head, rows = rows[0], rows[1:]
  k  = next(i for i, h in enumerate(head) if h.endswith("!"))
  xs = [i for i, h in enumerate(head)
        if i != k and not h.rstrip("+-!").endswith("X")]
  def col(i):
    vs = [r[i] for r in rows]
    if head[i][0].isupper():
      return [np.nan if v == "?" else float(v) for v in vs]
    codes = {v: j for j, v in enumerate(sorted(set(vs) - {"?"}))}
    return [codes.get(v, np.nan) for v in vs]
  cats = [j for j, i in enumerate(xs) if not head[i][0].isupper()]
  y = [np.nan if r[k] == "?" else float(r[k]) for r in rows]
  return np.array([col(i) for i in xs]).T, np.array(y), cats

def main():
  folds, lines = {}, open(sys.argv[1]).readlines()
  T = ["balsa(1)", "balsa(30)", "mean", "lgbm"]
  for line in lines[1:]:
    f, a, s, t = line.strip().split("|")
    folds.setdefault(f, []).append(([float(v) for v in a.split()],
                                    [float(v) for v in s.split()],
                                    [int(v) for v in t.split()]))
  W, means, secs = 12, {t: [] for t in T}, dict.fromkeys(T, 0.0)
  rel = {}
  print("%-18s" % "#data" + "".join("%-*s" % (W, t) for t in T))
  for f, fs in folds.items():
    X, y, cats = load(f)
    ok = [i for i in range(len(y)) if not np.isnan(y[i])]
    d  = {t: [] for t in T}
    for a, s, test in fs:
      base = a[2]                        # guess-the-mean mae
      d["balsa(1)"].append(a[0] / base)
      d["balsa(30)"].append(a[1] / base)
      d["mean"].append(1.0)
      secs["balsa(1)"] += s[0]; secs["balsa(30)"] += s[1]
      tset  = set(test)
      train = [i for i in ok if i not in tset]
      te    = [i for i in ok if i in tset]
      t0 = time.time()
      m = lgb.LGBMRegressor(verbose=-1, n_jobs=1)
      m.fit(X[train], y[train], categorical_feature=cats)
      got = m.predict(X[te])
      secs["lgbm"] += time.time() - t0
      d["lgbm"].append(float(np.mean(np.abs(y[te] - got))) / base)
    rel[f] = d
    cells = []
    for t in T:
      mu = sum(d[t]) / len(d[t]); means[t].append(mu)
      cells.append("%5.2f" % mu)
    print("%-18s" % f + "".join("%-*s" % (W, c) for c in cells),
          flush=True)
  print("\n%-18s" % "#mean rel-mae" + "".join(
        "%5.2f       " % (sum(means[t]) / len(means[t])) for t in T))
  print("%-18s" % "#secs (all data)" + "".join(
        "%5.1f       " % secs[t] for t in T))
  for f, d in rel.items():
    for t in T:
      print("%%folds %s %s %s" % (f, t,
            " ".join("%.3f" % v for v in d[t])))

if __name__ == "__main__":
  main()
