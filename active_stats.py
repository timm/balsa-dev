# vim: set ts=2 sw=2 sts=2 et cc=75:
"""active_stats: the ANOVA-style table for the active-learning arms.
One row per arm over the 62 optimize sets' per-set mean wins; arms
that the rig's same() test (Cliff's delta + KS, prudence on means)
cannot distinguish share a rank.  Rank 1 is best.

usage: pypy3 active_stats.py runs/*.txt"""
# (c) 2026 Tim Menzies <timm@ieee.org>, MIT license
import sys
from tree_eg import rank, median

F = {}
for path in sys.argv[1:]:
  for l in open(path):
    if l.startswith("%folds"):
      _, s, a, *v = l.split()
      F.setdefault(a, {})[s] = sum(map(float, v)) / len(v)

d = {a: [-x for x in F[a].values()] for a in F}    # negate: rank 1 = best
r = rank(d)
print("%-4s %-9s %7s %7s %7s" % ("rank", "arm", "median", "mean", "sd"))
for a in sorted(d, key=lambda a: median(d[a])):
  v = sorted(F[a].values())
  n = len(v)
  sd = (v[int(.9 * n)] - v[int(.1 * n)]) / 2.56
  print("%-4s %-9s %7.1f %7.1f %7.1f" % (
    r[a], a, median(v), sum(v) / n, sd))
