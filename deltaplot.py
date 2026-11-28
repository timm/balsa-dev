#!/usr/bin/env python3 -B
"""deltaplot: one line, the difference between two treatments.

Two curves lying on top of each other is a weak way to show "no
difference" -- the eye cannot tell a gap of zero from a gap it cannot
resolve.  This plots the gap itself: one point per data set, sorted,
with zero drawn as a line.  Points are grey where the rig's own test
cannot separate the two treatments on that set (Cliff's delta at most
0.195 and a KS test at most its 0.05 critical value, or means within
`prudence`), and coloured where it can.

Grey is therefore "the test says these are the same here", and the
height of a coloured point is how much difference a reader would be
buying.

usage: deltaplot.py OUT.png A B runs/active*.txt
       ... where A and B are arm keys, e.g. a.knn r.knn
"""
import sys, math, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRUDENCE, CLIFF, KSCRIT = 0.01, 0.195, 1.36
FS = dict(tick=14, lab=15, mono=13)
NICE = {"a.knn": "acquire, then 1nn", "r.knn": "random, then 1nn",
        "a.b30": "acquire, then tree(30)", "r.b30": "random, then tree(30)",
        "a.tabicl": "acquire, then TabICL", "a.rand": "random rows, random pick"}


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
    "The rig's own rule, from sec:code's listing."
    xs, ys = sorted(xs), sorted(ys)
    if abs(st.fmean(xs) - st.fmean(ys)) < PRUDENCE: return True
    return cliffs(xs, ys) <= CLIFF and ks(xs, ys) <= KSCRIT


def main():
    out, A, B, paths = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    F = {}
    for p in paths:
        for l in open(p):
            if l.startswith("%folds"):
                _, s, a, *v = l.split()
                F.setdefault(s, {})[a] = [float(x) for x in v]
    sets = [s for s in F if A in F[s] and B in F[s]]
    rows = sorted(((st.fmean(F[s][A]) - st.fmean(F[s][B]), same(F[s][A], F[s][B]))
                   for s in sets), key=lambda r: r[0])
    d = [r[0] for r in rows]
    tied = [r[1] for r in rows]
    xs = np.arange(len(d))

    fig = plt.figure(figsize=(14.3, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[20, 1, 7], wspace=0)
    ax, kx = fig.add_subplot(gs[0]), fig.add_subplot(gs[2])
    # a full-height band on every set the test cannot separate: the eye
    # reads the grey as a column, not as a marker it has to compare
    gx = [x for x, t in zip(xs, tied) if t]
    cx = [x for x, t in zip(xs, tied) if not t]
    for x in gx:
        ax.axvspan(x - .5, x + .5, color="#dfe2e6", lw=0, zorder=0)
    ax.axhline(0, color="#444", lw=1.4, zorder=2)
    ax.plot(xs, d, color="#8a8f98", lw=1.6, zorder=3)
    ax.plot(xs, d, "o", color="#e8613c", ms=5, zorder=4)
    # the key sits off the plot: a grey swatch inside it was invisible,
    # landing on the grey bands it was there to explain
    kx.set_xlim(0, 10); kx.set_ylim(0, 10)
    kx.set_xticks([]); kx.set_yticks([])
    kx.set_facecolor("#fcfcfa")
    for s_ in kx.spines.values(): s_.set_visible(False)
    kx.add_patch(plt.Rectangle((.3, 6.4), 1.5, 1.0, color="#dfe2e6", lw=0))
    kx.text(2.3, 6.9, f"no difference the test\ncan see  ({len(gx)} sets)",
            ha="left", va="center", fontsize=FS["mono"])
    kx.plot(1.05, 4.6, "o", color="#e8613c", ms=6)
    kx.text(2.3, 4.6, f"one data set\n({len(d)} in all, {len(cx)} of them\n"
            "outside a grey band)", ha="left", va="center",
            fontsize=FS["mono"])
    ax.set_xlim(-1, len(d)); ax.set_xticks([])
    ax.set_xlabel(f"{len(d)} data sets, ordered by the difference",
                  fontsize=FS["lab"])
    ax.set_ylabel(f"{NICE.get(A, A)}\nminus {NICE.get(B, B)}",
                  fontsize=FS["lab"] - 2)
    ax.tick_params(axis="y", labelsize=FS["tick"])
    ax.grid(axis="y", color="#ddd")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    fig.set_facecolor("#fcfcfa"); ax.set_facecolor("#fcfcfa")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=.02)
    plt.savefig(out.rsplit(".", 1)[0] + ".pdf",
                bbox_inches="tight", pad_inches=.02)
    print(f"{len(d)} sets: {len(gx)} tied, {len(cx)} separated;"
          f" median {st.median(d):+.2f}, mean {st.fmean(d):+.2f}")
    print(f"  {A} ahead on {sum(1 for v in d if v>0)},"
          f" {B} ahead on {sum(1 for v in d if v<0)}")


if __name__ == "__main__": main()
