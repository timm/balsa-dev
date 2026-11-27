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
LEARNERS = [("dist", "#2ca02c", "^"), ("b1", "#e8613c", "v"),
            ("b30", "#d62728", "s"), ("lgbm", "#2a6fdb", "o"),
            ("tabicl", "#7b3fa0", "D")]
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

F = {}
for path in sys.argv[1:-1]:                        # all inputs but the png
  for l in open(path):
    if l.startswith("%folds"):
      _, s, a, *v = l.split()
      F[(s, a)] = [float(x) for x in v]
sets = list(dict.fromkeys(s for s, _ in F))
mu = {k: st.fmean(v) for k, v in F.items()}
sd = {k: spread(v) for k, v in F.items()}
grand = {a: st.fmean([mu[(s, a)] for s in sets if (s, a) in mu])
         for a in {a for _, a in F}}

fig = plt.figure(figsize=(21, 4.2))
gs = fig.add_gridspec(1, 3, wspace=.12)

# -- left: five acquire learners, sd-banded, sorted by tabicl -----------
ax = fig.add_subplot(gs[0])
order = sorted(sets, key=lambda s: mu.get((s, "a.tabicl"), 0))
xs = np.arange(len(order))
for ln, c, m in LEARNERS:
  a = "%s.%s" % (AQ, ln)
  if a not in grand: continue
  ax.plot(xs, [mu.get((s, a), np.nan) for s in order], color=c, marker=m,
          ms=4, lw=1.4, alpha=.9, label="%-5s %4.1f" % (ln, grand[a]))
  ax.plot(xs, [sd.get((s, a), 0) for s in order], color=c, lw=1,
          ls=(0, (4, 2)), alpha=.7)                # sd curve, low on chart
ax.set_ylim(0, 112); ax.set_xlim(-1, len(order))
ax.axhline(100, color="#999", lw=1, ls=(0, (6, 4)))
ax.set_title("acquire, five learners (mean + dashed fold sd), sorted by tabicl",
             fontsize=12)
ax.legend(fontsize=10, loc="lower right", bbox_to_anchor=(.995, .12),
          framealpha=.9, title="learner mean")
ax.grid(axis="y", color="#eee")
ax.set_ylabel("win (100 = optimum)\nmean over 5x2 folds", fontsize=12)
ax.set_xlabel("optimize set (sorted)", fontsize=11)

# -- middle & right: one learner minus random, contiguous + grey ties ---
def deltapanel(gsi, arm, c, m, lab):
  "Full delta line (arm - r.b30), own sort, grey span where tied."
  px = fig.add_subplot(gsi)
  d = {s: mu.get((s, arm), 0) - mu.get((s, "r.b30"), 0) for s in sets}
  o = sorted(sets, key=d.get)
  x = np.arange(len(o))
  nsame = 0
  for i, s in enumerate(o):
    if (s, arm) in F and (s, "r.b30") in F and same(F[(s, arm)], F[(s, "r.b30")]):
      px.axvspan(i - .5, i + .5, color="#d9d9d9", lw=0, zorder=0)
      nsame += 1
  px.axhline(0, color="#999", lw=1)
  px.plot(x, [d[s] for s in o], color=c, marker=m, ms=4, lw=1.4, alpha=.9,
          zorder=2, label="%s - random  %+.1f" % (lab, grand[arm] - grand["r.b30"]))
  px.set_xlim(-1, len(o))
  px.set_title("%s acquire minus random, sorted "
               "(grey = same, %d/%d)" % (lab, nsame, len(o)), fontsize=12)
  px.legend(fontsize=10, loc="upper left", framealpha=.9)
  px.grid(axis="y", color="#eee")
  px.set_xlabel("optimize set (sorted by %s acquire - random)" % lab,
                fontsize=11)
  return px

bx = deltapanel(gs[1], "a.b30",    "#d62728", "s", "b30")
cx = deltapanel(gs[2], "a.tabicl", "#7b3fa0", "D", "tabicl")
bx.set_ylabel("win difference (acquire - random)", fontsize=12)
ymin = min(bx.get_ylim()[0], cx.get_ylim()[0])     # share y across mid+right
ymax = max(bx.get_ylim()[1], cx.get_ylim()[1])
bx.set_ylim(ymin, ymax); cx.set_ylim(ymin, ymax)
cx.set_yticklabels([])

for p in (ax, bx, cx):
  for s_ in ("top", "right"): p.spines[s_].set_visible(False)
fig.suptitle("62 optimize sets", fontsize=14)
fig.set_facecolor("#fcfcfa")
for a in fig.axes: a.set_facecolor("#fcfcfa")
plt.savefig(sys.argv[-1], dpi=200, bbox_inches="tight")
plt.savefig(sys.argv[-1].rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
for a in sorted(grand, key=lambda a: -grand[a]):
  print("%-8s %5.1f" % (a, grand[a]))
