#!/usr/bin/env python3 -B
"""active_plot: the active-learning figure from runs/active.txt.

Every arm is {a,r}.{dist,b1,b30,lgbm,rand}: acquisition a=chase the
best centroid, r=random; ranker = how the C-row holdout check is
sorted. Win = 100 means the best row in the table; higher is better.

Top: one line per arm, mean win over the 5x2 folds, sets sorted by
the single best-mean arm (so the strongest arm's curve rises). The
mean and the spread do not share an axis: the bottom strip is a
histogram of every arm's per-set fold spread, (p90-p10)/2.56.

usage: python3 active_plot.py runs/active.txt ens-active.png
"""
import sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# arm -> (colour, marker, dashed?)  acquisition sets the colour family,
# ranker the marker; random acquisition is the lighter/dashed twin.
STY = {
  "a.dist": ("#2ca02c", "^", 0), "r.dist": ("#2ca02c", "^", 1),
  "a.b1":   ("#e8613c", "v", 0), "r.b1":   ("#e8613c", "v", 1),
  "a.b30":  ("#d62728", "s", 0), "r.b30":  ("#d62728", "s", 1),
  "a.lgbm": ("#2a6fdb", "o", 0), "r.lgbm": ("#2a6fdb", "o", 1),
  "a.rand": ("#888888", "x", 0), "r.rand": ("#888888", "x", 1)}

def spread(v):
  v = sorted(v); n = len(v)
  q = lambda p: v[min(n - 1, int(p * n))]
  return (q(.9) - q(.1)) / 2.56

F = {}                                          # (set, arm) -> fold wins
for l in open(sys.argv[1]):
  if l.startswith("%folds"):
    _, s, a, *v = l.split()
    F[(s, a)] = [float(x) for x in v]
sets = list(dict.fromkeys(s for s, _ in F))
arms = [a for a in STY if any((s, a) in F for s in sets)]

mu = {(s, a): st.fmean(F[(s, a)]) if (s, a) in F else float("nan")
      for s in sets for a in arms}
grand = {a: st.fmean([mu[(s, a)] for s in sets if mu[(s, a)] == mu[(s, a)]])
         for a in arms}
best = max(grand, key=grand.get)                # the strongest arm
order = sorted(sets, key=lambda s: mu[(s, best)])

fig = plt.figure(figsize=(14.3, 6.6))
gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=.35)
ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

xs = np.arange(len(order))
for a in arms:
  c, m, dash = STY[a]
  ax.plot(xs, [mu[(s, a)] for s in order], color=c, marker=m, ms=4,
          lw=1.6, ls="--" if dash else "-", alpha=.85 if dash else 1,
          label="%-7s %4.1f" % (a, grand[a]))
ax.set_ylim(0, 108); ax.set_xlim(-1, len(order))
ax.axhline(100, color="#999", lw=1, ls=(0, (6, 4)))
ax.set_ylabel("win (100 = table optimum)\nmean over 5x2 folds",
              fontsize=12)
ax.set_title("%d optimize sets, sorted by the best arm (%s = %.1f)"
             % (len(order), best, grand[best]), fontsize=14)
ax.legend(ncol=2, fontsize=8, loc="lower right", framealpha=.9,
          title="arm   mean")
ax.grid(axis="y", color="#eee")
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)

# bottom: the spread, as its own histogram -- one per arm, overlaid
for a in arms:
  c, _, dash = STY[a]
  sds = [spread(F[(s, a)]) for s in sets if (s, a) in F]
  bx.hist(sds, bins=list(np.arange(0, 40, 2.5)), histtype="step",
          color=c, lw=1.4, ls="--" if dash else "-", alpha=.8)
bx.set_xlabel("per-set fold spread (p90-p10)/2.56", fontsize=11)
bx.set_ylabel("# sets", fontsize=11)
bx.set_title("spread of each arm (lower = steadier)", fontsize=11)
for s_ in ("top", "right"): bx.spines[s_].set_visible(False)

fig.set_facecolor("#fcfcfa")
for a in (ax, bx): a.set_facecolor("#fcfcfa")
plt.savefig(sys.argv[2], dpi=200)
plt.savefig(sys.argv[2].rsplit(".", 1)[0] + ".pdf")
print("best arm:", best, "%.1f" % grand[best])
for a in sorted(arms, key=lambda a: -grand[a]):
  print("%-8s mean %5.1f" % (a, grand[a]))
