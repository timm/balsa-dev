#!/usr/bin/env python3 -B
"""active_plot: the active-learning figure from runs/active.txt.

Five charts, one per learner (dist, b1, b30, lgbm, rand). Each plots
two lines -- acquire (chase the best centroid) vs random labelling --
on a shared win scale, with its fold spread in a histogram below.
Sets share one x order, sorted by the median-mean arm, so all five
panels line up set-for-set. Win = 100 is the best row in the table.

usage: python3 active_plot.py runs/active.txt ens-active.png
"""
import sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LEARNERS = ["dist", "b1", "b30", "lgbm", "rand"]
ACQ = [("a", "acquire", "#d62728", "s", "-"),
       ("r", "random",  "#2a6fdb", "o", "--")]

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
ranked = sorted(grand, key=grand.get)             # weakest..strongest
pivot = ranked[len(ranked) // 2]                  # the median-mean arm
order = sorted(sets, key=lambda s: mu.get((s, pivot), 0))
xs = np.arange(len(order))

fig = plt.figure(figsize=(20, 5.4))
gs = fig.add_gridspec(2, 5, height_ratios=[3, 1], hspace=.33, wspace=.1)
for col, ln in enumerate(LEARNERS):
  ax = fig.add_subplot(gs[0, col])
  bx = fig.add_subplot(gs[1, col])
  for aq, lab, c, m, ls in ACQ:
    a = "%s.%s" % (aq, ln)
    if a not in grand: continue
    ax.plot(xs, [mu.get((s, a), np.nan) for s in order], color=c, ls=ls,
            marker=m, ms=3, lw=1.3, alpha=.9,
            label="%-7s %4.1f" % (lab, grand[a]))
    bx.hist([spread(F[(s, a)]) for s in sets if (s, a) in F],
            bins=list(np.arange(0, 40, 3)), histtype="step", color=c, lw=1.5)
  ax.set_ylim(0, 108); ax.set_xlim(-1, len(order))
  ax.axhline(100, color="#999", lw=1, ls=(0, (6, 4)))
  ax.set_title(ln, fontsize=13)
  ax.legend(fontsize=9, loc="lower right", framealpha=.9)
  ax.grid(axis="y", color="#eee")
  bx.set_xlabel("per-set fold spread", fontsize=9)
  bx.set_xlim(0, 39)
  if col == 0:
    ax.set_ylabel("win (100 = optimum)\nmean over 5x2 folds", fontsize=11)
    bx.set_ylabel("# sets", fontsize=9)
  else:
    ax.set_yticklabels([]); bx.set_yticklabels([])
  for p in (ax, bx):
    for s_ in ("top", "right"): p.spines[s_].set_visible(False)

fig.suptitle("62 optimize sets, acquire vs random per learner, "
             "sorted by the median arm %s" % pivot, fontsize=14, y=.99)
fig.set_facecolor("#fcfcfa")
for a in fig.axes: a.set_facecolor("#fcfcfa")
plt.savefig(sys.argv[2], dpi=200, bbox_inches="tight")
plt.savefig(sys.argv[2].rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
for a in sorted(grand, key=lambda a: -grand[a]):
  print("%-8s %5.1f" % (a, grand[a]))
