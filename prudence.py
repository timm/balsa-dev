#!/usr/bin/env python3 -B
"""prudence: percent of data sets each arm is ranked best on, as the
prudence floor rises from 1 to 10 AUC points.

Prudence is balsa_eg.yue's floor on an effect: two arms whose mean AUCs
differ by less than it are never called different, whatever cohen, cliffs
and ks say.  At 1 point the ranking is almost all signal; by 10 points
almost everything ties, and the curves say how fast that happens.

The statistics below are a line-for-line port of balsa_eg.yue, NOT a stats
library, so this chart and the rig agree by construction.  One wrinkle: the
yue `rank` sorts ascending, so its rank 1 is the SMALLEST median.  AUC is
better when bigger, so we rank on -median, as the python rigs do.

usage: python3 prudence.py runs/nose39-yue.txt runs/se61-yue.txt prudence.png
"""
import bisect as B, math, sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS, CLIFF, KS = 0.35, 0.195, 1.36      # balsa_eg.yue's thresholds
PRU = range(1, 11)
# name, key in the .txt, colour and marker: the same ink as the AUC figures
ARMS = [("balsa(1)",  "1:q16",  "#2ca02c", "^"),
        ("lgbm(100)", "lgbm",   "#2a6fdb", "o"),
        ("balsa(30)", "n30",    "#e8613c", "s"),
        ("TabICL",    "tabicl", "#000000", "D")]


def bis(vs, v, le=False):
  return B.bisect_right(vs, v) if le else B.bisect_left(vs, v)


def num(xs):                            # adds: n, mean, sd
  n = len(xs)
  mu = sum(xs) / n
  m2 = sum((x - mu) ** 2 for x in xs)
  return n, mu, (math.sqrt(m2 / (n - 1)) if n > 1 else 0)


def cohen(a, b):
  (n1, mu1, sd1), (n2, mu2, sd2) = a, b
  sp = math.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) /
                 max(1, n1 + n2 - 2))
  return abs(mu1 - mu2) / (sp + 1e-32)


def cliffs(xs, ys):
  d = sum(bis(ys, x) - (len(ys) - bis(ys, x, True)) for x in xs)
  return abs(d) / (len(xs) * len(ys))


def ks(xs, ys):
  n, m, d = len(xs), len(ys), 0
  for v in (xs, ys):
    for x in v:
      d = max(d, abs(bis(xs, x, True) / n - bis(ys, x, True) / m))
  return d / math.sqrt((n + m) / (n * m))


def same(xs, ys, prudence):
  xs, ys = sorted(xs), sorted(ys)
  a, b = num(xs), num(ys)
  if abs(a[1] - b[1]) < prudence: return True       # too small to care about
  return cohen(a, b) <= EPS and cliffs(xs, ys) <= CLIFF and ks(xs, ys) <= KS


def rank(d, prudence):                  # [(name, folds)...] -> {name: rank}
  d = sorted(d, key=lambda kv: -st.median(kv[1]))
  r, pool, out = 1, None, {}
  for name, nums in d:
    if pool is not None and not same(pool, nums, prudence): r, pool = r + 1, None
    pool = (pool or []) + list(nums)
    out[name] = r
  return out


def folds(path):
  "Every %folds line: set -> arm -> the 10 fold AUCs."
  out = {}
  for line in open(path):
    if not line.startswith("%folds"): continue
    _, f, arm, *vals = line.split()
    out.setdefault(f, {})[arm] = [float(v) for v in vals]
  return out


def main():
  sets = []
  for path in sys.argv[1:-1]:
    sets += list(folds(path).values())
  pcts = {a: [] for _, a, _, _ in ARMS}
  for p in PRU:
    for s in sets:
      rk = rank([(a, s[a]) for _, a, _, _ in ARMS], p)
      for _, a, _, _ in ARMS:
        if len(pcts[a]) < p: pcts[a].append(0)
        pcts[a][p - 1] += 100 * (rk[a] == 1) / len(sets)
  # drawn at 2x a one-column figure, so a 12pt label prints at 6pt
  f, ax = plt.subplots(figsize=(6.6, 4.0))
  for name, a, c, m in ARMS:
    ax.plot(list(PRU), pcts[a], color=c, marker=m, lw=2.2, ms=7, label=name)
  ax.set_xlabel("prudence: smallest gap we call a difference (AUC points)",
                fontsize=12)
  ax.set_ylabel(f"% of {len(sets)} data sets ranked best", fontsize=12)
  ax.set_xticks(list(PRU)); ax.set_ylim(-3, 103); ax.set_yticks(range(0, 101, 20))
  ax.tick_params(labelsize=12)
  ax.grid(axis="y", color="#ddd")
  for s in ("top", "right"): ax.spines[s].set_visible(False)
  ax.legend(fontsize=11, frameon=False, loc="center right", handlelength=2.4)
  f.set_facecolor("#fcfcfa"); ax.set_facecolor("#fcfcfa")
  f.tight_layout(pad=.3)
  f.savefig(sys.argv[-1], dpi=200)
  f.savefig(sys.argv[-1].rsplit(".", 1)[0] + ".pdf")
  for name, a, _, _ in ARMS:
    print(f"{name:10s}" + "".join(f"{v:6.1f}" for v in pcts[a]))


if __name__ == "__main__": main()
