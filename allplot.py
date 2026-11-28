#!/usr/bin/env python3 -B
"""allplot: the whole corpus on one axis, in one group.

Figures 2-5 split their sets into panels -- numeric, symbolic, mixed,
large, issues, smells, defects, static, config, process, systems -- which
is four crowded figures to read before the headline arrives.  This draws
the same numbers with the panels taken out: every data set on one axis,
ordered by LightGBM, so the shape of the comparison is one glance.

Per-set names are dropped with the panels.  A hundred rotated labels at
this width are unreadable anyway, and the point of this figure is the
shape of the four curves, not which set sits at x=57.  Figures 2-5 keep
the names for anyone who wants them.

Two modes, because the two corpora are scored differently:

  cls   AUC, 0..100, higher is better.  Reads the %folds lines of the
        run tables written by tabicl3.py.
  reg   rel = MAE/MAE(predict the mean), where SMALL is good, so the
        axis is inverted and higher still means better.  Reads the csvs
        written by reg2csv.py.
  abl   the acquisition ablation: smart acquire against random select,
        with the ranker held fixed at 1nn and the same budget.  The one
        comparison that asks whether the "active" in active learning is
        doing any work.
  act   active learning: the win score of the row the optimizer chose,
        where 100 is the best row in the table.  Higher is better.
        Reads the %folds and %secs lines of runs/active*.txt.

usage: allplot.py cls OUT.png runs/nose39-yue.txt runs/se61-yue.txt
       allplot.py reg OUT.png runs/reg56.csv runs/sereg.csv
       allplot.py act OUT.png runs/active*.txt
"""
import csv, os, sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# label, colour, marker -- the same ink as every other figure here
INK = {"enuf(1)":  ("#2ca02c", "^"), "lgbm(100)": ("#2a6fdb", "o"),
       "enuf(30)": ("#e8613c", "s"), "TabICL":    ("#000000", "D"),
       "random":    ("#8a8f98", "v"), "1nn": ("#7b3fa0", "P"),
       "acquire+1nn": ("#e8613c", "s"), "random+1nn": ("#8a8f98", "v"),
       "acq:enuf(1)":   ("#2ca02c", "^"), "acq:lgbm(100)": ("#2a6fdb", "o"),
       "acq:enuf(30)":  ("#e8613c", "s"), "acq:1nn":       ("#7b3fa0", "P"),
       "acq:TabICL":    ("#000000", "D"), "rand:1nn":      ("#8a8f98", "v")}
ORDER = ["enuf(1)", "lgbm(100)", "enuf(30)", "TabICL"]
# `random` -- the same budget spent with no model -- is the floor these
# are judged against, and sec:active reports it; it is kept out of the
# figure, where six lines were two too many to follow.
# the optimizer study names its treatments by how the 45 training rows
# were chosen as well as by what ranked the test set, because the first
# half of that pair turns out to matter less than anyone expects
ACTORDER = ["acq:lgbm(100)", "acq:enuf(1)", "acq:enuf(30)", "acq:1nn",
            "acq:TabICL", "rand:1nn"]
ABLORDER = ["random+1nn", "acquire+1nn"]
FS = dict(tick=14, lab=15, mono=13, bar=15)


def spread(v):
    "The sd the other figures draw: a robust (p90-p10)/2.56 over folds."
    v = sorted(v); n = len(v)
    q = lambda p: v[min(n - 1, int(p * n))]
    return (q(.9) - q(.1)) / 2.56


def read_cls(paths):
    "set -> treatment -> folds, plus total seconds per treatment."
    key = {"1:q16": "enuf(1)", "n30": "enuf(30)",
           "lgbm": "lgbm(100)", "tabicl": "TabICL"}
    F, secs = {}, dict.fromkeys(ORDER, 0.0)
    for p in paths:
        L = open(p).read().splitlines()
        head = [key.get(h, h) for h in L[0].split()[1:]]
        for line in L:
            if line.startswith("%folds"):
                _, f, t, *v = line.split()
                F.setdefault(f, {})[key.get(t, t)] = [float(x) for x in v]
            elif line.startswith("#secs"):
                for h, x in zip(head, line.split()[3:]):
                    if h in secs: secs[h] += float(x)
    return F, secs


def read_act(paths, abl=False):
    """The acquire-only arms of the active-learning run.  Its tables carry
    their own per-fold seconds, so the runtime bars here are measured on
    exactly the folds the curves are drawn from."""
    key = ({"a.knn": "acquire+1nn", "r.knn": "random+1nn"} if abl else
           {"a.b1": "acq:enuf(1)", "a.b30": "acq:enuf(30)",
            "a.lgbm": "acq:lgbm(100)", "a.tabicl": "acq:TabICL",
            "a.knn": "acq:1nn", "r.knn": "rand:1nn"})
    F, secs = {}, dict.fromkeys(ABLORDER if abl else ACTORDER, 0.0)
    for p in paths:
        for l in open(p):
            if l.startswith("%folds"):
                _, s_, a, *v = l.split()
                if a in key: F.setdefault(s_, {})[key[a]] = [float(x) for x in v]
            elif l.startswith("%secs"):
                _, s_, a, *v = l.split()
                if a in key: secs[key[a]] += sum(float(x) for x in v)
    return F, secs


def read_reg(paths):
    key = {"enuf1": "enuf(1)", "enuf30": "enuf(30)",
           "lgbm": "lgbm(100)", "tabicl": "TabICL"}
    F, secs = {}, dict.fromkeys(ORDER, 0.0)
    for p in paths:
        for r in csv.DictReader(open(p)):
            for a, nm in key.items():
                v = r.get(f"rel_{a}", "")
                if v != "": F.setdefault(r["data"], {}).setdefault(nm, []).append(float(v))
        stem = p.rsplit(".", 1)[0]
        if os.path.exists(stem + ".txt"):
            head = [key.get(h, h) for h in open(stem + ".txt").readline().split()[1:]]
            for line in open(stem + ".txt"):
                if line.startswith("#secs"):
                    for h, x in zip(head, line.split()[3:]):
                        if h in secs: secs[h] += float(x)
        if os.path.exists(stem + "-tabicl.txt"):
            for line in open(stem + "-tabicl.txt"):
                if line.startswith("#secs"):
                    secs["TabICL"] += float(line.split()[-1])
    return F, secs


def main():
    mode, out, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    if mode == "abl":
        F, secs = read_act(paths, abl=True)
    else:
        F, secs = {"cls": read_cls, "reg": read_reg,
                   "act": read_act}[mode](paths)
    global ORDER, VARARMS
    if mode == "abl":
        ORDER = ABLORDER
        VARARMS = tuple(ABLORDER)
    elif mode == "act":
        ORDER = ACTORDER
        # `random` is the budget spent without a model: it is the baseline
        # the others are judged against, not a learner being compared
        VARARMS = ("acq:1nn", "acq:lgbm(100)", "acq:enuf(30)", "acq:TabICL")
    nan = float("nan")
    mu = {(s, a): (st.fmean(F[s][a]) if a in F[s] else nan)
          for s in F for a in ORDER}
    sd = {(s, a): (spread(F[s][a]) if a in F[s] else nan)
          for s in F for a in ORDER}
    # One treatment sets the order, so its curve is monotone and the
    # others are read against it.  If that treatment missed a set (TabICL
    # is capped by rows in regression), those sets sort last, among
    # themselves by the fallback -- a NaN key would order nothing.
    SORT = os.getenv("SORTBY", "acquire+1nn" if mode == "abl" else
                     "acq:1nn" if mode == "act" else "enuf(30)")
    FALL = "lgbm(100)"
    hi = mode != "reg"                     # only rel is small-is-good
    def key(s):
        v = mu[(s, SORT)]
        if v == v: return (0, -v if hi else v)
        w = mu[(s, FALL)]
        return (1, -w if hi else w)
    sets = sorted(F, key=key)
    xs = np.arange(len(sets))

    fig = plt.figure(figsize=(14.3, 3.9))
    gs = fig.add_gridspec(1, 5, wspace=0,
                          width_ratios=[.72 * len(ORDER), 2.3, 20, 1.0, 10.5])
    bx, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[2])
    vx = fig.add_subplot(gs[4])

    for a in ORDER:
        c, m = INK[a]
        ms = 5 if len(sets) < 80 else 3.5
        ax.plot(xs, [mu[(s, a)] for s in sets], color=c, marker=m, lw=1.8, ms=ms)
        ax.plot(xs, [sd[(s, a)] for s in sets], color=c, ls="--", lw=1.4,
                marker=m, ms=ms * .6)

    if mode in ("cls", "act", "abl"):
        ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
        ax.set_yticks(range(10, 100, 20), minor=True)
        ax.set_ylabel("AUC" if mode == "cls" else "win",
                      fontsize=FS["lab"])
    else:
        ax.set_ylim(1.45, 0)                  # inverted: up is still better
        ax.set_yticks([0, .25, .5, .75, 1, 1.25])
        ax.set_yticks([.125, .375, .625, .875, 1.125], minor=True)
        ax.axhline(1.0, color="#999", lw=1.4, ls=(0, (6, 4)))
        ax.set_ylabel("err = MAE/mean(MAE)", fontsize=FS["lab"])
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="major", color="#ddd")
    ax.grid(axis="y", which="minor", color="#b0b0b0", ls=(0, (4, 4)), lw=1)
    ax.tick_params(axis="y", labelsize=FS["tick"])
    ax.set_xlim(-1, len(sets))
    ax.set_xticks([])
    miss = sum(1 for s in sets if mu[(s, SORT)] != mu[(s, SORT)])
    ax.set_xlabel(f"{len(sets)} data sets, ordered by {SORT}'s mean"
                  + (f" ({miss} it did not run come last)" if miss else ""),
                  fontsize=FS["lab"])
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)

    bt = sorted((secs[a], a, INK[a][0]) for a in ORDER if secs.get(a))
    bx.bar(range(len(bt)), [b[0] for b in bt], color=[b[2] for b in bt],
           width=.84)
    bx.set_yscale("log")
    # rotated labels need headroom above the tallest bar, and this is
    # a log axis, so the room has to be bought in decades
    bx.set_ylim(min(b[0] for b in bt) / 3, max(b[0] for b in bt) * 90)
    for i, b in enumerate(bt):
        bx.text(i, b[0] * 1.35, f"{b[0]:,.0f}s" if b[0] >= 100
                else f"{b[0]:.1f}s", ha="center", va="bottom",
                rotation=90, weight="bold", fontsize=FS["bar"] - 6)
    bx.set_xticks(range(len(bt)))
    # four names side by side in a narrow panel collide; stand them up
    bx.set_xticklabels([b[1] for b in bt], rotation=90, ha="center",
                       va="top", fontsize=FS["bar"] - 4)
    bx.set_title("total secs, log scale\n(summed over folds,\n1 thread each)",
                 fontsize=FS["lab"] - 4)
    bx.set_yticks([]); bx.set_facecolor("#fcfcfa")
    bx.set_xlim(-.55, len(bt) - .45)
    for s_ in ("top", "right", "left"): bx.spines[s_].set_visible(False)

    # right: the whole key, so nothing sits on top of the curves.
    # name, its solid line and mean, its dashed sd line, and what the
    # step up to it costs.  NO marks a treatment nothing would buy:
    # something else already scores as well in no more time.
    #
    # A "point" here is the table's point, not the raw unit: for reg the
    # score is a ratio, and one point is 0.01 of it.  Without the x100
    # this panel priced a whole unit of rel and read 100 times the table.
    PT = 100 if mode == "reg" else 1
    # averaged over the sets EVERY treatment ran, not over each one's own
    # set list.  TabICL is capped by rows, so scoring it on 69 sets and
    # the rest on 79 made this panel disagree with Table 7 by a third.
    common = [s_ for s_ in sets
              if all(mu[(s_, a)] == mu[(s_, a)] for a in ORDER)]
    tv = [(a, st.fmean([mu[(s_, a)] for s_ in common]),
           st.fmean([sd[(s_, a)] for s_ in common]), secs.get(a, 0))
          for a in ORDER if secs.get(a)]
    tv.sort(key=lambda r: -r[1] if mode == "reg" else r[1])
    DP = 2 if mode == "reg" else 0
    # a wider y range with the rows unchanged pulls them together:
    # the text is fixed in points, the axis is not
    # column stops follow the longest name: "acq:lgbm(100)" is half
    # again as wide as "lgbm(100)" and ran straight into its own swatch
    NW = 3.3 * max(len(n) for n, *_ in tv)
    X1, X2 = NW + 4, NW + 36          # solid swatch, and the mean
    X3, X4 = NW + 42, NW + 88         # dashed swatch, and secs/unit
    vx.set_xlim(0, X4 + 4); vx.set_ylim(-1.9, len(tv) + .55)
    vx.set_xticks([]); vx.set_yticks([])
    mono2 = dict(family="monospace", fontsize=FS["mono"] - 1)
    hd = dict(family="monospace", fontsize=FS["mono"] - 3, color="#555")
    UNIT = {"cls": "AUC", "reg": "err"}.get(mode, "win")
    vx.text(X2, len(tv) - .55, "mean", ha="right", va="center", **hd)
    vx.text(X3 + 5, len(tv) - .55, "sd", ha="center", va="center", **hd)
    vx.text(X4, len(tv) - .55, f"secs/{UNIT}", ha="right", va="center", **hd)
    # every row priced the same way: its total seconds over the points it
    # gains on the worst treatment here.  Stepping from the row above
    # instead would go negative wherever a cheaper treatment scores
    # better, which happens, and a negative price reads as a mistake.
    base = tv[0][1]
    for k, (n, v, sv, c) in enumerate(tv):
        y = len(tv) - 1 - k
        d = (base - v if mode == "reg" else v - base) * PT
        cell = "--" if d <= 0 else ("<1" if c / d < 1 else f"{c/d:,.0f}")
        col, mk = INK[n]
        vx.text(0, y, n, ha="left", va="center", color=col, **mono2)
        vx.plot([X1, X1 + 10], [y, y], color=col, lw=2.0)
        vx.plot(X1 + 5, y, mk, color=col, ms=5)
        vx.text(X2, y, f"{v:.{DP}f}", ha="right", va="center", **mono2)
        vx.plot([X3, X3 + 10], [y, y], color=col, lw=1.8, ls="--")
        vx.text(X4, y, cell, ha="right", va="center", **mono2)
    vx.set_title(f"solid = mean, dashed = sd\n({len(common)} sets all ran)",
                 fontsize=FS["lab"] - 4)
    vx.set_facecolor("#fcfcfa")
    for s_ in vx.spines.values(): s_.set_visible(False)

    fig.set_facecolor("#fcfcfa"); ax.set_facecolor("#fcfcfa")
    plt.tight_layout()
    # park the y label against its own axis, clear of the runtime bars
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    w    = ax.yaxis.label.get_window_extent(rend).width
    bars = bx.get_tightbbox(rend).x1
    tick = min(t.get_window_extent(rend).x0 for t in ax.get_yticklabels())
    ax.yaxis.set_label_coords(ax.transAxes.inverted().transform(
        (max(bars + 8 + w / 2, tick + 7 - w / 2), 0))[0], .5)
    plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=.02)
    plt.savefig(out.rsplit(".", 1)[0] + ".pdf",
                bbox_inches="tight", pad_inches=.02)
    for a in ORDER:
        v = [mu[(s, a)] for s in sets if mu[(s, a)] == mu[(s, a)]]
        print(f"{a:10s} mean {st.fmean(v):6.2f}  ({len(v)} sets)"
              f"  {secs.get(a, 0):,.0f}s")
    for n, v, sv, c in tv:
        print(f"  {n:14s} {v:6.2f} {c:9,.0f}s")


if __name__ == "__main__": main()
