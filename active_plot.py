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
            ("b30", "#d62728", "s"), ("lgbm", "#2a6fdb", "o")]
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
for l in open(sys.argv[1]):
  if l.startswith("%folds"):
    _, s, a, *v = l.split()
    F[(s, a)] = [float(x) for x in v]
sets = list(dict.fromkeys(s for s, _ in F))
mu = {k: st.fmean(v) for k, v in F.items()}
sd = {k: spread(v) for k, v in F.items()}
grand = {a: st.fmean([mu[(s, a)] for s in sets if (s, a) in mu])
         for a in {a for _, a in F}}

fig = plt.figure(figsize=(16, 4.2))
gs = fig.add_gridspec(1, 2, wspace=.1)

# -- left: four acquire learners, sd-banded, sorted by b30 --------------
ax = fig.add_subplot(gs[0])
order = sorted(sets, key=lambda s: mu.get((s, "a.b30"), 0))
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
ax.set_title("acquire, four learners (mean + dashed fold sd), sorted by b30",
             fontsize=12)
ax.legend(fontsize=10, loc="lower right", bbox_to_anchor=(.995, .12),
          framealpha=.9, title="learner mean")
ax.grid(axis="y", color="#eee")
ax.set_ylabel("win (100 = optimum)\nmean over 5x2 folds", fontsize=12)
ax.set_xlabel("optimize set (sorted)", fontsize=11)

# -- right: acquire-b30 vs random-b30, sorted by gap, grey where same ---
bx = fig.add_subplot(gs[1])
gap = {s: mu.get((s, "a.b30"), 0) - mu.get((s, "r.b30"), 0) for s in sets}
order2 = sorted(sets, key=gap.get)
x2 = np.arange(len(order2))
nsame = 0
for i, s in enumerate(order2):
  if (s, "a.b30") in F and (s, "r.b30") in F \
     and same(F[(s, "a.b30")], F[(s, "r.b30")]):
    bx.axvspan(i - .5, i + .5, color="#d9d9d9", lw=0, zorder=0)
    nsame += 1
bx.axhline(0, color="#999", lw=1)
bx.plot(x2, [gap[s] for s in order2], color="#333", marker="o", ms=3,
        lw=1.4, zorder=2, label="acquire - random  %+.1f" % (
        grand["a.b30"] - grand["r.b30"]))
bx.set_xlim(-1, len(order2))
bx.set_title("b30: acquire minus random, sorted "
             "(grey = statistically same, %d/%d)" % (nsame, len(order2)),
             fontsize=12)
bx.legend(fontsize=10, loc="upper left", framealpha=.9)
bx.grid(axis="y", color="#eee")
bx.set_xlabel("optimize set (sorted by acquire - random)", fontsize=11)
bx.set_ylabel("win difference (acquire - random)", fontsize=12)

for p in (ax, bx):
  for s_ in ("top", "right"): p.spines[s_].set_visible(False)
fig.suptitle("62 optimize sets", fontsize=14)
fig.set_facecolor("#fcfcfa")
for a in fig.axes: a.set_facecolor("#fcfcfa")
plt.savefig(sys.argv[2], dpi=200, bbox_inches="tight")
plt.savefig(sys.argv[2].rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
print("b30 acquire vs random: same on %d of %d sets" % (nsame, len(sets)))
for a in sorted(grand, key=lambda a: -grand[a]):
  print("%-8s %5.1f" % (a, grand[a]))
