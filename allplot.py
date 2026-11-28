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
       "acquire+1nn": ("#e8613c", "s"), "random+1nn": ("#8a8f98", "v")}
ORDER = ["enuf(1)", "lgbm(100)", "enuf(30)", "TabICL"]
# `random` -- the same budget spent with no model -- is the floor these
# are judged against, and sec:active reports it; it is kept out of the
# figure, where six lines were two too many to follow.
ACTORDER = ["1nn", "enuf(1)", "lgbm(100)", "enuf(30)", "TabICL"]
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
           {"a.b1": "enuf(1)", "a.b30": "enuf(30)", "a.lgbm": "lgbm(100)",
            "a.tabicl": "TabICL", "a.knn": "1nn"})
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
        VARARMS = ("1nn", "lgbm(100)", "enuf(30)", "TabICL")
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
                     "1nn" if mode == "act" else "enuf(30)")
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
                          width_ratios=[.72 * len(ORDER), 2.3, 20, 1.0, 6.2])
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
        ax.set_ylabel("MAE / MAE(mean)", fontsize=FS["lab"])
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

    # Back inside the plot, bottom left, lifted clear of the x axis.
    # Underneath the frame it was legible but far from the lines it
    # names; in here the eye goes straight from swatch to curve.
    tx = ax.inset_axes((.02, .05, .4, .4))
    LEGEND_IN = (2.15, .20 * len(ORDER) + .03)
    tx.set_xlim(0, 28); tx.set_ylim(0, len(ORDER) + .6)
    tx.patch.set_alpha(0); tx.set_xticks([]); tx.set_yticks([])
    for s_ in tx.spines.values(): s_.set_visible(False)
    mono = dict(family="monospace", fontsize=FS["mono"] - 3,
                bbox=dict(fc="#fcfcfa", ec="none", pad=1))
    # The mean is worth carrying here -- it is the one number a reader
    # wants per treatment.  The sd is not: it is the dashed line, which
    # the eye can already see, and a second column of digits crowds the
    # names.  One precision for the column, set by the score.
    DP = 2 if mode == "reg" else 0
    avg, num = {}, {}
    for a in ORDER:
        ok = [s_ for s_ in sets if mu[(s_, a)] == mu[(s_, a)]]
        num[a] = st.fmean([mu[(s_, a)] for s_ in ok])
        avg[a] = f"{num[a]:.{DP}f}"
    # best first, which in reg means smallest: the legend then reads as a
    # ranking rather than as whatever order the treatments were declared
    rank = sorted(ORDER, key=lambda a: -num[a] if mode != "reg" else num[a])
    for k, a in enumerate(rank):
        c, m = INK[a]
        y = len(ORDER) - .4 - k
        tx.text(0, y, a, ha="left", va="center", **mono)
        tx.plot([11.0, 14.2], [y, y], color=c, lw=2.0)
        tx.plot(12.6, y, m, color=c, ms=5)
        tx.text(19.4, y, avg[a], ha="right", va="center", **mono)
        tx.text(20.8, y, "sd", ha="left", va="center", **mono)
        tx.plot([24.6, 27.6], [y, y], color=c, lw=1.8, ls="--")

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

    # right: what the next step up costs.  Treatments in score order,
    # each priced by the extra seconds over the one below it; NO marks a
    # treatment nothing would buy, because something else already scores
    # at least as well in no more time.
    tv = [(a, st.fmean([mu[(s_, a)] for s_ in sets
                        if mu[(s_, a)] == mu[(s_, a)]]), secs.get(a, 0))
          for a in ORDER if secs.get(a)]
    tv.sort(key=lambda r: -r[1] if mode == "reg" else r[1])
    dom = {n for n, v, c in tv
           if any((v2 >= v if mode != "reg" else v2 <= v) and c2 <= c
                  and (v2, c2) != (v, c) for _, v2, c2 in tv)}
    vx.set_xlim(0, 10.6); vx.set_ylim(-.6, len(tv) - .4)
    vx.set_xticks([]); vx.set_yticks([])
    vx.set_title("cost of the step up\n(secs per extra point)",
                 fontsize=FS["lab"] - 4)
    mono2 = dict(family="monospace", fontsize=FS["mono"] - 1)
    prev = None
    for k, (n, v, c) in enumerate(tv):
        y = len(tv) - 1 - k
        if n in dom:
            cell = "NO"
        elif prev is None:
            cell = "--"
        else:
            d, dc = (v - prev[0]) * (-1 if mode == "reg" else 1), c - prev[1]
            cell = "<1" if dc / d < 1 else f"{dc/d:,.0f}"
        if n not in dom: prev = (v, c)
        vx.text(0, y, n, ha="left", va="center", color=INK[n][0], **mono2)
        vx.text(10.4, y, cell, ha="right", va="center",
                weight="bold" if n in dom else "normal", **mono2)
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
    # inset_axes keeps a locator that ignores set_position; drop it, then
    # the numbers below are figure fractions.  Lifted a little off the
    # axis floor so the box does not sit on the x spine.
    _bb = ax.get_position()
    _lh = LEGEND_IN[1] / fig.get_figheight()
    tx.set_axes_locator(None)
    # In reg the axis is inverted, so the lazy-learner line at 1.0 sits
    # LOW on screen; clear it rather than straddle it.
    _y0 = .35 if mode == "reg" else .20
    tx.set_position([_bb.x0 + .006, _bb.y0 + _y0 * _bb.height,
                     LEGEND_IN[0] / fig.get_figwidth(), _lh])
    plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=.02)
    plt.savefig(out.rsplit(".", 1)[0] + ".pdf",
                bbox_inches="tight", pad_inches=.02)
    for a in ORDER:
        v = [mu[(s, a)] for s in sets if mu[(s, a)] == mu[(s, a)]]
        print(f"{a:10s} mean {st.fmean(v):6.2f}  ({len(v)} sets)"
              f"  {secs.get(a, 0):,.0f}s")
    for n, v, c in tv:
        print(f"  {n:12s} {v:6.2f} {c:9,.0f}s"
              f"{'   DOMINATED' if n in dom else ''}")


if __name__ == "__main__": main()
