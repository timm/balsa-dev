#!/usr/bin/env python3 -B
"""acq_plot: does smart acquisition matter to the 1nn optimizer?
One line: per-set win of (acquire x 1nn) minus (random-label x 1nn),
sorted by that gap. Grey column = the two treatments' ten folds are
statistically the same (means within prudence, or Cliff's delta and
a KS test both under their 0.05 thresholds).

usage: python3 acq_plot.py runs/active-knn.txt ens-1nn-acq.png
"""
import sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRUDENCE = 0.01

def bisect(a, x, right=False):
  lo, hi = 0, len(a)
  while lo < hi:
    m = (lo + hi) // 2
    if (a[m] <= x if right else a[m] < x): lo = m + 1
    else: hi = m
  return lo

def cliffs(xs, ys):
  d = sum(bisect(ys, x) - (len(ys) - bisect(ys, x, True)) for x in xs)
  return abs(d) / (len(xs) * len(ys))

def ks(xs, ys):
  n, m = len(xs), len(ys)
  d = max(abs(bisect(xs, v, True) / n - bisect(ys, v, True) / m)
          for v in xs + ys)
  return d / ((n + m) / (n * m)) ** .5

def same(xs, ys):
  xs, ys = sorted(xs), sorted(ys)
  if abs(sum(xs)/len(xs) - sum(ys)/len(ys)) < PRUDENCE: return True
  return cliffs(xs, ys) <= 0.195 and ks(xs, ys) <= 1.36

F = {}
for path in sys.argv[1:-1]:
  for l in open(path):
    if l.startswith("%folds"):
      _, s, a, *v = l.split()
      F[(s, a)] = [float(x) for x in v]
sets = sorted({s for s, a in F if (s, "a.knn") in F and (s, "r.knn") in F})
mu = {k: st.fmean(v) for k, v in F.items()}
gap = {s: mu[(s, "a.knn")] - mu[(s, "r.knn")] for s in sets}
order = sorted(sets, key=gap.get)
xs = np.arange(len(order))

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.axhline(0, color="#999", lw=1)
for arm, c, m, lab in (("a.knn", "#7b3fa0", "P", "1nn(acq)"),
                       ("a.tabicl", "#000000", "D", "tabicl(acq)")):
  d = {s: mu[(s, arm)] - mu[(s, "r.knn")] for s in order}
  ok = {s: not same(F[(s, arm)], F[(s, "r.knn")]) for s in order}
  n = sum(ok.values())
  ax.plot(xs, [d[s] if ok[s] else np.nan for s in order],
          color=c, marker=m, ms=4, lw=1.4, zorder=2,
          label="%-11s %+5.1f  %2d/%d differ" % (
            lab, st.fmean(d.values()), n, len(order)))
ax.set_title("minus random labelling x 1nn, %d sets, sorted by the "
             "1nn gap (points only where folds differ)" % len(order),
             fontsize=11)
ax.legend(fontsize=10, loc="lower center", framealpha=.9,
          prop={"family": "monospace", "size": 10})
ax.grid(axis="y", color="#eee")
ax.set_xlabel("optimize set (sorted by gap)", fontsize=11)
ax.set_ylabel("win difference", fontsize=11)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
fig.set_facecolor("#fcfcfa")
ax.set_facecolor("#fcfcfa")
plt.savefig(sys.argv[-1], dpi=200, bbox_inches="tight")
plt.savefig(sys.argv[-1].rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
print("a.knn %.1f  a.tabicl %.1f  r.knn %.1f" % (
  st.fmean([mu[(s, "a.knn")] for s in sets]),
  st.fmean([mu[(s, "a.tabicl")] for s in sets]),
  st.fmean([mu[(s, "r.knn")] for s in sets])))
