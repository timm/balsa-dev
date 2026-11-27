#!/usr/bin/env python3 -B
"""active_plot: the active-learning figure from runs/active.txt.

One chart, acquire only (chase the best centroid), four learners --
dist, b1, b30, lgbm -- on a shared win scale, fold spread in a
histogram below. Sets sorted by the dist arm so its line is smooth.
Win = 100 is the best row in the table.

usage: python3 active_plot.py runs/active.txt ens-active.png
"""
import sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQ = "a"                                           # acquire only
LEARNERS = [("dist", "#2ca02c", "^"), ("b1", "#e8613c", "v"),
            ("b30", "#d62728", "s"), ("lgbm", "#2a6fdb", "o")]

def spread(v):
  v = sorted(v); n = len(v)
  q = lambda p: v[min(n - 1, int(p * n))]
  return (q(.9) - q(.1)) / 2.56

F = {}
for l in open(sys.argv[1]):
  if l.startswith("%folds"):
    _, s, a, *v = l.split()
    F[(s, a)] = [float(x) for x in v]
sets = list(dict.fromkeys(s for s, _ in F))
mu = {k: st.fmean(v) for k, v in F.items()}
grand = {a: st.fmean([mu[(s, a)] for s in sets if (s, a) in mu])
         for a in {a for _, a in F}}
pivot = "a.dist"                                   # sort key
order = sorted(sets, key=lambda s: mu.get((s, pivot), 0))
xs = np.arange(len(order))

fig = plt.figure(figsize=(11, 6))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=.28)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])
for ln, c, m in LEARNERS:
  a = "%s.%s" % (AQ, ln)
  if a not in grand: continue
  ax.plot(xs, [mu.get((s, a), np.nan) for s in order], color=c,
          marker=m, ms=4, lw=1.4, alpha=.9,
          label="%-5s %4.1f" % (ln, grand[a]))
  bx.hist([spread(F[(s, a)]) for s in sets if (s, a) in F],
          bins=list(np.arange(0, 40, 3)), histtype="step", color=c, lw=1.5)
ax.set_ylim(0, 108); ax.set_xlim(-1, len(order))
ax.axhline(100, color="#999", lw=1, ls=(0, (6, 4)))
ax.set_title("acquire: chase the best centroid", fontsize=13)
ax.legend(fontsize=10, loc="lower right", framealpha=.9, title="learner mean")
ax.grid(axis="y", color="#eee")
ax.set_ylabel("win (100 = optimum)\nmean over 5x2 folds", fontsize=12)
bx.set_xlabel("per-set fold spread", fontsize=10)
bx.set_ylabel("# sets", fontsize=10); bx.set_xlim(0, 39)
for p in (ax, bx):
  for s_ in ("top", "right"): p.spines[s_].set_visible(False)

fig.suptitle("62 optimize sets, acquire only, sorted by dist", fontsize=14)
fig.set_facecolor("#fcfcfa")
for a in fig.axes: a.set_facecolor("#fcfcfa")
plt.savefig(sys.argv[2], dpi=200, bbox_inches="tight")
plt.savefig(sys.argv[2].rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
for a in sorted(grand, key=lambda a: -grand[a]):
  print("%-8s %5.1f" % (a, grand[a]))
