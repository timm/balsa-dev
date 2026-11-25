#!/usr/bin/env python3 -B
"""prudence: percent of data sets each arm is ranked best on, as the
prudence floor rises from 1 to 10 AUC points.

Prudence is balsa_eg.yue's floor on an effect: two arms whose mean AUCs
differ by less than it are never called different, whatever cohen, cliffs
and ks say.  Here that floor is measured in standard deviations, not in AUC
points, because a point is worth different amounts on different data sets:
the per-set fold sd runs from 0.31 to 14.18 points, a 45-fold spread, so one
absolute threshold is strict on a quiet set and vacuous on a noisy one.

k runs .1 to 1.  Below .1 the floor does nothing (k=0, the three tests
alone, gives 0/9/13/93 against k=.1's 0/10/13/94).  Above 1 it does too
much: one fold sd is already 2.3 AUC points on the median set, and a floor
of four sds would be discarding nine-point gaps, which is not prudence but
indifference.  The range is also Cohen's: .2 small, .5 medium, .8 large.

WHICH sd matters.  Dividing the gap by the POOLED sd of the two arms being
compared gives Cohen's d exactly (checked: the two agree to 0.0), so that
choice would merely restate the cohen test already inside `same`.  The
denominator here is instead a property of the DATA SET: the mean over arms
of that set's fold spread, (p90-p10)/2.56, which is the dashed line in the
AUC figures.  So x=2 reads "a gap under two of this set's own fold sds".

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
PRU = [i / 10 for i in range(1, 11)]    # prudence = k * sd, k = .1 .. 1
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


def spread(v):
  "The sd the AUC figures draw: a robust (p90-p10)/2.56 over the folds."
  v = sorted(v); n = len(v)
  q = lambda p: v[min(n - 1, int(p * n))]
  return (q(.9) - q(.1)) / 2.56


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
  sd = [max(1e-9, st.fmean([spread(s[a]) for _, a, _, _ in ARMS])) for s in sets]
  pcts = {a: [] for _, a, _, _ in ARMS}
  for p in PRU:
    got = {a: 0 for _, a, _, _ in ARMS}
    for s, v in zip(sets, sd):
      rk = rank([(a, s[a]) for _, a, _, _ in ARMS], p * v)   # p sds, this set
      for _, a, _, _ in ARMS:
        got[a] += rk[a] == 1
    for _, a, _, _ in ARMS: pcts[a].append(100 * got[a] / len(sets))
  print(f"per-set fold sd (AUC points): median "
        f"{st.median(sd):.2f}, range {min(sd):.2f}-{max(sd):.2f}")
  # drawn at 2x a one-column figure, so a 12pt label prints at 6pt
  f, ax = plt.subplots(figsize=(6.6, 4.0))
  for name, a, c, m in ARMS:
    ax.plot(PRU, pcts[a], color=c, marker=m, lw=2.2, ms=7, label=name)
  ax.set_xlabel("prudence: smallest gap we call a difference,\n"
                "as a fraction $k$ of that data set's own fold sd", fontsize=12)
  ax.set_ylabel(f"% of {len(sets)} data sets ranked best", fontsize=12)
  ax.set_xticks([.1,.2,.3,.4,.5,.6,.7,.8,.9,1]); ax.set_ylim(-3, 103); ax.set_yticks(range(0, 101, 20))
  ax.tick_params(labelsize=12)
  ax.grid(axis="y", color="#ddd")
  for s in ("top", "right"): ax.spines[s].set_visible(False)
  ax.legend(fontsize=11, frameon=False, loc="center left",
            bbox_to_anchor=(.03, .62), handlelength=2.4)
  f.set_facecolor("#fcfcfa"); ax.set_facecolor("#fcfcfa")
  f.tight_layout(pad=.3)
  f.savefig(sys.argv[-1], dpi=200)
  f.savefig(sys.argv[-1].rsplit(".", 1)[0] + ".pdf")
  for name, a, _, _ in ARMS:
    print(f"{name:10s}" + "".join(f"{v:6.1f}" for v in pcts[a]))


if __name__ == "__main__": main()
