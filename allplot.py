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

usage: allplot.py cls OUT.png runs/nose39-yue.txt runs/se61-yue.txt
       allplot.py reg OUT.png runs/reg56.csv runs/sereg.csv
"""
import csv, os, sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# label, colour, marker -- the same ink as every other figure here
INK = {"balsa(1)":  ("#2ca02c", "^"), "lgbm(100)": ("#2a6fdb", "o"),
       "balsa(30)": ("#e8613c", "s"), "TabICL":    ("#000000", "D")}
ORDER = ["balsa(1)", "lgbm(100)", "balsa(30)", "TabICL"]
FS = dict(tick=14, lab=15, mono=13, bar=15)


def spread(v):
    "The sd the other figures draw: a robust (p90-p10)/2.56 over folds."
    v = sorted(v); n = len(v)
    q = lambda p: v[min(n - 1, int(p * n))]
    return (q(.9) - q(.1)) / 2.56


def read_cls(paths):
    "set -> treatment -> folds, plus total seconds per treatment."
    key = {"1:q16": "balsa(1)", "n30": "balsa(30)",
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


def read_reg(paths):
    key = {"balsa1": "balsa(1)", "balsa30": "balsa(30)",
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
    F, secs = (read_cls if mode == "cls" else read_reg)(paths)
    nan = float("nan")
    mu = {(s, a): (st.fmean(F[s][a]) if a in F[s] else nan)
          for s in F for a in ORDER}
    sd = {(s, a): (spread(F[s][a]) if a in F[s] else nan)
          for s in F for a in ORDER}
    sets = sorted(F, key=lambda s: mu[(s, "lgbm(100)")],
                  reverse=(mode == "cls"))     # worst LightGBM first
    xs = np.arange(len(sets))

    fig = plt.figure(figsize=(14.3, 3.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[3.4, 2.3, 20], wspace=0)
    bx, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[2])

    for a in ORDER:
        c, m = INK[a]
        ms = 5 if len(sets) < 80 else 3.5
        ax.plot(xs, [mu[(s, a)] for s in sets], color=c, marker=m, lw=1.8, ms=ms)
        ax.plot(xs, [sd[(s, a)] for s in sets], color=c, ls="--", lw=1.4,
                marker=m, ms=ms * .6)

    if mode == "cls":
        ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
        ax.set_yticks(range(10, 100, 20), minor=True)
        ax.set_ylabel("AUC", fontsize=FS["lab"])
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
    ax.set_xlabel(f"{len(sets)} data sets, ordered by LightGBM's mean",
                  fontsize=FS["lab"])
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)

    # legend: name, its solid line (the mean), its dashed line (the sd)
    tx = ax.inset_axes((.02, .05, .4, .4))
    LEGEND_IN = (3.20, 1.05)
    tx.set_xlim(0, 30); tx.set_ylim(0, 4.6)
    tx.patch.set_alpha(0); tx.set_xticks([]); tx.set_yticks([])
    for s_ in tx.spines.values(): s_.set_visible(False)
    mono = dict(family="monospace", fontsize=FS["mono"],
                bbox=dict(fc="#fcfcfa", ec="none", pad=1))
    # one precision for the whole table, set by the score: whole AUC
    # points, two decimals for a ratio.  Ragged columns of significant
    # figures (0.07 beside 0.075) are harder to read than rounded ones.
    DP = 0 if mode == "cls" else 2
    col = lambda vals: [f"{v:.{DP}f}" for v in vals]
    _m, _s = {}, {}
    for a in ORDER:
        ok = [s_ for s_ in sets if mu[(s_, a)] == mu[(s_, a)]]
        _m[a] = st.fmean([mu[(s_, a)] for s_ in ok])
        _s[a] = st.fmean([sd[(s_, a)] for s_ in ok])
    _mt = dict(zip(ORDER, col([_m[a] for a in ORDER])))
    _st = dict(zip(ORDER, col([_s[a] for a in ORDER])))
    for k, a in enumerate(ORDER):
        c, m = INK[a]
        y = 3.6 - 1.0 * k
        tx.text(0, y, a, ha="left", va="center", **mono)
        tx.plot([10.4, 13.6], [y, y], color=c, lw=2.2)
        tx.plot(12.0, y, m, color=c, ms=6)
        tx.text(18.2, y, _mt[a], ha="right", va="center", **mono)
        tx.text(19.8, y, "sd", ha="left", va="center", **mono)
        tx.plot([23.0, 26.0], [y, y], color=c, lw=2, ls="--")
        tx.text(30.0, y, _st[a], ha="right", va="center", **mono)

    bt = sorted((secs[a], a, INK[a][0]) for a in ORDER if secs.get(a))
    bx.bar(range(len(bt)), [b[0] for b in bt], color=[b[2] for b in bt],
           width=.84)
    bx.set_yscale("log")
    bx.set_ylim(min(b[0] for b in bt) / 2, max(b[0] for b in bt) * 4)
    for i, b in enumerate(bt):
        bx.text(i, b[0] * 1.25, f"{b[0]:,.0f}s" if b[0] >= 10
                else f"{b[0]:.1f}s", ha="center", va="bottom",
                weight="bold", fontsize=FS["bar"] - 1)
    bx.set_xticks(range(len(bt)))
    # four names side by side in a narrow panel collide; stand them up
    bx.set_xticklabels([b[1] for b in bt], rotation=90, ha="center",
                       va="top", fontsize=FS["bar"] - 4)
    bx.set_title("total secs, log scale\n(summed over folds,\n1 thread each)",
                 fontsize=FS["lab"] - 4)
    bx.set_yticks([]); bx.set_facecolor("#fcfcfa")
    bx.set_xlim(-.55, len(bt) - .45)
    for s_ in ("top", "right", "left"): bx.spines[s_].set_visible(False)

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
    _bb = ax.get_position()
    tx.set_axes_locator(None)
    tx.set_position([_bb.x0 + .006, _bb.y0 + .02,
                     LEGEND_IN[0] / fig.get_figwidth(),
                     LEGEND_IN[1] / fig.get_figheight()])
    plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=.02)
    plt.savefig(out.rsplit(".", 1)[0] + ".pdf",
                bbox_inches="tight", pad_inches=.02)
    for a in ORDER:
        v = [mu[(s, a)] for s in sets if mu[(s, a)] == mu[(s, a)]]
        print(f"{a:10s} mean {st.fmean(v):6.2f}  ({len(v)} sets)"
              f"  {secs.get(a, 0):,.0f}s")


if __name__ == "__main__": main()
