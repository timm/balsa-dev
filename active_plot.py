#!/usr/bin/env python3 -B
"""active_plot: the active-learning figure from runs/active.txt.

Two charts, one per acquisition: left ACQUIRE (chase the best
centroid), right RANDOM, on a shared win scale. Each plots five
ranker lines -- how the C-row holdout check is sorted -- and keeps
its fold spread in its own histogram at the bottom. Sets share one
x order, sorted by the single best-mean arm, so the two panels line
up set-for-set. Win = 100 is the best row in the table.

usage: python3 active_plot.py runs/active.txt ens-active.png
"""
import sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RANKERS = [("dist", "#2ca02c", "^"), ("b1", "#e8613c", "v"),
           ("b30", "#d62728", "s"), ("lgbm", "#2a6fdb", "o"),
           ("rand", "#888888", "x")]
ACQ = [("a", "acquire: chase the best centroid"),
       ("r", "random: label at random")]

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
best = max(grand, key=grand.get)                 # strongest arm, any panel
order = sorted(sets, key=lambda s: mu.get((s, best), 0))
xs = np.arange(len(order))

fig = plt.figure(figsize=(14.3, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=.33, wspace=.12)
for col, (aq, title) in enumerate(ACQ):
  ax = fig.add_subplot(gs[0, col])
  bx = fig.add_subplot(gs[1, col])
  for rn, c, m in RANKERS:
    a = "%s.%s" % (aq, rn)
    if a not in grand: continue
    ax.plot(xs, [mu.get((s, a), np.nan) for s in order], color=c,
            marker=m, ms=4, lw=1.6, label="%-5s %4.1f" % (rn, grand[a]))
    bx.hist([spread(F[(s, a)]) for s in sets if (s, a) in F],
            bins=list(np.arange(0, 40, 3)), histtype="step", color=c, lw=1.5)
  ax.set_ylim(0, 108); ax.set_xlim(-1, len(order))
  ax.axhline(100, color="#999", lw=1, ls=(0, (6, 4)))
  ax.set_title(title, fontsize=13)
  ax.legend(fontsize=9, loc="lower right", framealpha=.9,
            title="ranker mean")
  ax.grid(axis="y", color="#eee")
  bx.set_xlabel("per-set fold spread", fontsize=10)
  bx.set_xlim(0, 39)
  if col == 0:
    ax.set_ylabel("win (100 = optimum)\nmean over 5x2 folds", fontsize=12)
    bx.set_ylabel("# sets", fontsize=10)
  else:
    ax.set_yticklabels([]); bx.set_yticklabels([])
  for p in (ax, bx):
    for s_ in ("top", "right"): p.spines[s_].set_visible(False)

fig.suptitle("%d optimize sets, sorted by the best arm (%s = %.1f)"
             % (len(order), best, grand[best]), fontsize=14, y=.98)
fig.set_facecolor("#fcfcfa")
for a in fig.axes: a.set_facecolor("#fcfcfa")
plt.savefig(sys.argv[2], dpi=200)
plt.savefig(sys.argv[2].rsplit(".", 1)[0] + ".pdf")
for a in sorted(grand, key=lambda a: -grand[a]):
  print("%-8s %5.1f" % (a, grand[a]))
