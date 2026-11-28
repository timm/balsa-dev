#!/usr/bin/env python3 -B
"""active_enuf_plot: the LHS-style panel for the three enuf-family
rankers only -- dist (centroid), b1 (one tree), b30 (forest) -- on the
optimize sets.  Means sorted by b30, each banded by a dashed fold-sd
curve, legend by mean, with total runtime as labelled log bars below.

usage: python3 active_enuf_plot.py runs/active-secs-enuf.txt ens-active-enuf.png
"""
import sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LEARNERS = [("knn", "#7b3fa0", "P"), ("b1", "#2ca02c", "v"),
            ("b30", "#e8613c", "s")]

def spread(v):
  v = sorted(v); n = len(v)
  q = lambda p: v[min(n - 1, int(p * n))]
  return (q(.9) - q(.1)) / 2.56

F, S = {}, {}
for path in sys.argv[1:-1]:
  for l in open(path):
    if l.startswith("%folds"):
      _, s, a, *v = l.split(); F[(s, a)] = [float(x) for x in v]
    elif l.startswith("%secs"):
      _, s, a, *v = l.split(); S[(s, a)] = [float(x) for x in v]
sets = list(dict.fromkeys(s for s, _ in F))
mu = {k: st.fmean(v) for k, v in F.items()}
sd = {k: spread(v) for k, v in F.items()}
grand = {a: st.fmean([mu[(s, a)] for s in sets if (s, a) in mu])
         for a in {a for _, a in F}}

fig = plt.figure(figsize=(11, 4.8))
gs = fig.add_gridspec(1, 2, width_ratios=[.55, 2.6], wspace=.3)
hx = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1])
order = sorted(sets, key=lambda s: mu.get((s, "a.knn"), 0))
xs = np.arange(len(order))
bymean = sorted(LEARNERS, key=lambda t: -grand.get("a." + t[0], 0))
totsec = {}
for ln, c, m in bymean:
  a = "a." + ln
  if a not in grand: continue
  nm = {"knn": "1nn"}.get(ln, ln)
  ax.plot(xs, [mu.get((s, a), np.nan) for s in order], color=c, marker=m,
          ms=4, lw=1.4, alpha=.9, label="%-5s %3.0f" % (nm, grand[a]))
  ax.plot(xs, [sd.get((s, a), 0) for s in order], color=c, lw=1,
          ls=(0, (4, 2)), alpha=.7)
  totsec[ln] = sum(v for s in sets if (s, a) in S for v in S[(s, a)])
bars = [(ln, c, totsec[ln]) for ln, c, _ in bymean if totsec.get(ln)]
hx.bar(range(len(bars)), [t for _, _, t in bars],
       color=[c for _, c, _ in bars], width=.6)
for i, (ln, c, t) in enumerate(bars):
  hx.text(i, t, "{:,}s".format(round(t)), ha="center", va="bottom",
          fontsize=10, fontweight="bold")
hx.set_xticks(range(len(bars)))
hx.set_xticklabels([{"knn": "1nn"}.get(b[0], b[0]) for b in bars])
hx.set_yscale("log")
hx.set_title("total secs over all\nfolds (log, 1 thread)", fontsize=10)
hx.grid(axis="y", color="#eee"); hx.margins(y=.3)
ax.set_ylim(0, 112); ax.set_xlim(-1, len(order))
ax.axhline(100, color="#999", lw=1, ls=(0, (6, 4)))
ax.set_title("acquire, three enuf rankers (mean + dashed fold sd), sorted by 1nn",
             fontsize=12)
ax.legend(fontsize=11, loc="lower right", bbox_to_anchor=(.995, .12),
          framealpha=.9, title="ranker mean",
          prop={"family": "monospace", "size": 11})
ax.set_yticks([20, 40, 60, 80, 100])
ax.tick_params(axis="y", length=0)
for v_ in range(10, 101, 10):
  ax.axhline(v_, color="#bbb", lw=.8, ls=(0, (4, 4)), zorder=0)
ax.set_ylabel("win (100 = optimum)\nmean over 5x2 folds", fontsize=12)
ax.set_xlabel("optimize set (sorted)", fontsize=11)
for p in (ax, hx):
  for s_ in ("top", "right"): p.spines[s_].set_visible(False)
fig.set_facecolor("#fcfcfa")
for a in fig.axes: a.set_facecolor("#fcfcfa")
plt.savefig(sys.argv[-1], dpi=200, bbox_inches="tight")
plt.savefig(sys.argv[-1].rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
for a in sorted(grand, key=lambda a: -grand[a]):
  print("%-8s %5.1f" % (a, grand[a]))
