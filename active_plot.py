#!/usr/bin/env python3 -B
"""active_plot: the active-learning figure from runs/active.txt.

Left: acquire only (chase the best centroid), four learners -- dist,
b1, b30, lgbm -- each line banded by its fold sd, sorted by b30.
Right: acquire-b30 vs random-b30, sets sorted by their gap, with a
grey column wherever the two are statistically the same (Cliff's
delta and a KS test both small, or means within prudence).
Win = 100 is the best row in the table.

usage: python3 active_plot.py runs/active.txt ens-active.png
"""
import sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQ = "a"                                           # acquire only
NAME = {"knn": "1nn"}                              # data key -> shown name
LEARNERS = [("knn", "#7b3fa0", "P"), ("b1", "#2ca02c", "v"),
            ("b30", "#e8613c", "s"), ("lgbm", "#2a6fdb", "o"),
            ("tabicl", "#000000", "D")]
PRUDENCE = 0.01

def spread(v):
  v = sorted(v); n = len(v)
  q = lambda p: v[min(n - 1, int(p * n))]
  return (q(.9) - q(.1)) / 2.56

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

F, S = {}, {}                                      # wins, seconds
for path in sys.argv[1:-1]:                        # all inputs but the png
  for l in open(path):
    if l.startswith("%folds"):
      _, s, a, *v = l.split()
      F[(s, a)] = [float(x) for x in v]
    elif l.startswith("%secs"):
      _, s, a, *v = l.split()
      S[(s, a)] = [float(x) for x in v]
sets = list(dict.fromkeys(s for s, _ in F))
mu = {k: st.fmean(v) for k, v in F.items()}
sd = {k: spread(v) for k, v in F.items()}
grand = {a: st.fmean([mu[(s, a)] for s in sets if (s, a) in mu])
         for a in {a for _, a in F}}

fig = plt.figure(figsize=(20, 4.8))
gs = fig.add_gridspec(1, 2, width_ratios=[.55, 3.2], wspace=.18)

# -- secs bars at far left; win lines centre; delta right ---------------
hx = fig.add_subplot(gs[0])                        # runtime bars
ax = fig.add_subplot(gs[1])
order = sorted(sets, key=lambda s: mu.get((s, "a.knn"), 0))
xs = np.arange(len(order))
bymean = sorted(LEARNERS, key=lambda t: -grand.get("a." + t[0], 0))
totsec = {}
for ln, c, m in bymean:
  a = "%s.%s" % (AQ, ln)
  if a not in grand: continue
  ax.plot(xs, [mu.get((s, a), np.nan) for s in order], color=c, marker=m,
          ms=4, lw=1.4, alpha=.9, label="%-6s %3.0f" % (NAME.get(ln, ln), grand[a]))
  ax.plot(xs, [sd.get((s, a), 0) for s in order], color=c, lw=1,
          ls=(0, (4, 2)), alpha=.7)                # sd curve, low on chart
  totsec[ln] = sum(v for s in sets if (s, a) in S for v in S[(s, a)])
# runtime: one horizontal bar per learner, total secs, log scale
bars = sorted(((ln, c, totsec[ln]) for ln, c, _ in bymean
               if totsec.get(ln)), key=lambda b: b[2])
hx.bar(range(len(bars)), [t for _, _, t in bars],
       color=[c for _, c, _ in bars], width=.7)
for i, (ln, c, t) in enumerate(bars):
  hx.text(i, t, "{:,}s".format(round(t)), ha="center", va="bottom",
          fontsize=10, fontweight="bold")
hx.set_xticks(range(len(bars)))
hx.set_xticklabels([NAME.get(b[0], b[0]) for b in bars], fontsize=9)
hx.margins(y=.3)
ax.set_ylim(0, 112); ax.set_xlim(-1, len(order))
ax.axhline(100, color="#999", lw=1, ls=(0, (6, 4)))
ax.set_title("acquire, five learners (mean + dashed fold sd), sorted by 1nn",
             fontsize=12)
ax.legend(fontsize=10, loc="lower right", bbox_to_anchor=(.995, .12),
          framealpha=.9, title="learner mean",
          prop={"family": "monospace", "size": 10})
ax.set_yticks([20, 40, 60, 80, 100])
ax.tick_params(axis="y", length=0)
for v_ in range(10, 101, 10):
  ax.axhline(v_, color="#bbb", lw=.8, ls=(0, (4, 4)), zorder=0)
ax.set_ylabel("win (100 = optimum)\nmean over 5x2 folds", fontsize=12)
ax.set_xlabel("optimize set (sorted)", fontsize=11)
hx.set_yscale("log")
hx.set_title("total secs over all\nfolds (log, 1 thread)", fontsize=10)
hx.grid(axis="y", color="#eee")

for p in (ax, hx):
  for s_ in ("top", "right"): p.spines[s_].set_visible(False)
fig.set_facecolor("#fcfcfa")
for a in fig.axes: a.set_facecolor("#fcfcfa")
plt.savefig(sys.argv[-1], dpi=200, bbox_inches="tight")
plt.savefig(sys.argv[-1].rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
for a in sorted(grand, key=lambda a: -grand[a]):
  print("%-8s %5.1f" % (a, grand[a]))
