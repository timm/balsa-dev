#!/usr/bin/env python3 -B
"""active_var: where does a win move come from, in the active-learning
study?  Same two-way ANOVA-with-replication decomposition as
variance.py (data set x learner, 10 folds per cell, eta-squared,
no F and no p), over the acquire arms a reader would consider:
knn, b30, lgbm, tabicl.  dist is retired, b1 is the crippled single
tree, rand is the floor; none of them is a choice anyone defends.

usage: python3 active_var.py runs/active-secs-enuf.txt \\
           runs/active-secs-heavy.txt runs/active-knn.txt
"""
import sys, statistics as st

ARMS = ["a.knn", "a.b30", "a.lgbm", "a.tabicl"]

F = {}
for path in sys.argv[1:]:
  for l in open(path):
    if l.startswith("%folds"):
      _, s, a, *v = l.split()
      F.setdefault(s, {})[a] = [float(x) for x in v]
sets = [s for s in F if all(a in F[s] for a in ARMS)]

def split(ss):
  "eta-squared for data set, learner, fold, interaction.  Sums to 100."
  I, J = len(ss), len(ARMS)
  R = len(F[ss[0]][ARMS[0]])
  allv = [v for f in ss for a in ARMS for v in F[f][a]]
  g = st.fmean(allv)
  row  = {f: st.fmean([v for a in ARMS for v in F[f][a]]) for f in ss}
  col  = {a: st.fmean([v for f in ss for v in F[f][a]]) for a in ARMS}
  cell = {(f, a): st.fmean(F[f][a]) for f in ss for a in ARMS}
  tot  = sum((v - g) ** 2 for v in allv)
  return [100 * x / tot for x in (
    J * R * sum((row[f] - g) ** 2 for f in ss),
    I * R * sum((col[a] - g) ** 2 for a in ARMS),
    sum((v - cell[(f, a)]) ** 2 for f in ss for a in ARMS for v in F[f][a]),
    R * sum((cell[(f, a)] - row[f] - col[a] + g) ** 2
            for f in ss for a in ARMS))]

groups = {}
for s in sets: groups.setdefault(s.split("/")[0], []).append(s)
rows = [(n, len(ss), split(ss)) for n, ss in groups.items() if len(ss) > 1]
rows.sort(key=lambda r: r[2][1])
rows.append(("all", len(sets), split(sets)))
print("%-10s %4s %8s %8s %6s %12s %7s" % (
  "group", "sets", "data set", "learner", "fold", "interaction", "d/l"))
for n, k, v in rows:
  print("%-10s %4d %8.0f %8.0f %6.0f %12.0f %7.0f" % (
    n, k, v[0], v[1], v[2], v[3], v[0] / max(v[1], 1e-9)))
