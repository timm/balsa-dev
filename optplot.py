#!/usr/bin/env python3 -B
"""optplot: the ens-tree-opt figure, for optimization.

Each arm fits a model on the train half, sorts the test half by its
PREDICTED d2h, pays for the true d2h of the best 5, and keeps the winner.
y is that winner scored against the whole table:

    100 * (1 - (got - lo) / (mu - lo))

so 100 = the table's own optimum, 0 = no better than an average row, and
HIGHER IS BETTER.  `random` is the same budget spent without a model: five
test rows drawn at random.  It is the line every learner must clear.

One panel per moot/optimize folder.  Solid: the mean over the ten 5x2
folds.  Dashed: the spread of those folds, as (p90-p10)/2.56.  Left: total
run time, log scale, summed over folds.

usage: python3 optplot.py runs/opt.csv ens-tree-opt.png
"""
import csv, os, sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MOOT = os.path.expanduser(os.getenv("MOOTOPT", "~/gits/moot/optimize"))
# name in the csv, label, colour, marker -- the same ink as the AUC figures
ARMS = [("tree1",  "tree(1)",  "#2ca02c", "^"),
        ("lgbm",    "lgbm(100)", "#2a6fdb", "o"),
        ("tree30", "tree(30)", "#e8613c", "s"),
        ("random",  "random",    "#888888", "x")]
FS = dict(tick=14, lab=14, head=17, mono=14, bar=15)
MIN_PANEL = 6      # folders thinner than this pool into "other"


def spread(v):
    "The sd the AUC figures draw: a robust (p90-p10)/2.56 over the folds."
    v = sorted(v); n = len(v)
    q = lambda p: v[min(n - 1, int(p * n))]
    return (q(.9) - q(.1)) / 2.56


def vals(D, s, a):
    "The folds of one treatment on one set, or [] if it did not run it."
    k = f"score_{a}"
    return [float(r[k]) for r in D[s] if r.get(k, "") != ""]


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    D = {}
    for r in rows: D.setdefault(r["data"], []).append(r)
    sets = sorted(D)
    # rows, and whether every x column is numeric
    n, numeric = {}, {}
    for s in sets:
        L = [l for l in open(f"{MOOT}/{s}.csv") if l.strip()]
        head = [h.strip() for h in L[0].split(",")]
        n[s] = len(L) - 1
        x = [h for h in head
             if not h.rstrip("+-!").endswith("X") and h[-1] not in "+-!"]
        numeric[s] = all(h[0].isupper() for h in x)
    minrows = int(os.getenv("MINROWS", "0"))   # drop the tiny sets
    sets = [s for s in sets if n[s] >= minrows]
    keep = os.getenv("FOLDERS")              # SE vs the rest: name the
    if keep:                                 # folders this figure wants
        want = set(keep.split(","))
        sets = [s for s in sets if s.split("/")[0] in want]
    if any("/" in s for s in sets):          # moot/optimize: group by
        dirs = {}                            # folder, but a panel needs
        for s in sets:                       # width to carry a header,
            dirs.setdefault(s.split("/")[0], []).append(s)   # so the
        thin = [s for d, v in dirs.items() if len(v) < MIN_PANEL          
                for s in v]                  # thin ones pool as "other"
        G = [(d, v) for d, v in sorted(dirs.items()) if len(v) >= MIN_PANEL]
        if thin: G.append((os.getenv("TITLE", "other"), sorted(thin)))
    else:
        G = [("numeric\nattributes", [s for s in sets if n[s] < 1000 and numeric[s]]),
             ("mixed\nattributes",   [s for s in sets if n[s] < 1000 and not numeric[s]]),
             ("large, mixed",        [s for s in sets if n[s] >= 1000])]

    # nan where a treatment did not run a set, so its line simply breaks
    # there instead of dropping to zero or dragging the mean around
    nan = float("nan")
    mu = {(s, a): (st.fmean(v) if (v := vals(D, s, a)) else nan)
          for s in sets for a, *_ in ARMS}
    sd = {(s, a): (spread(v) if (v := vals(D, s, a)) else nan)
          for s in sets for a, *_ in ARMS}
    # timings live in the sibling .txt, on its "#secs (all data)" line
    secs = {}
    txt = sys.argv[1].rsplit(".", 1)[0] + ".txt"
    if os.path.exists(txt):
        head = open(txt).readline().split()[1:]
        for line in open(txt):
            if line.startswith("#secs"):
                v = line.split()[3:]
                secs = {("tree1" if "(1)" in h else
                         "tree30" if "(30)" in h else h): float(x)
                        for h, x in zip(head, v)}
    tt = sys.argv[1].rsplit(".", 1)[0] + "-tabicl.txt"
    if os.path.exists(tt):
        for line in open(tt):
            if line.startswith("#secs"): secs["tabicl"] = float(line.split()[-1])

    wide = 14.3 if len(sets) < 70 else 20.0
    XT   = 10 if len(sets) < 70 else 6
    fig = plt.figure(figsize=(wide, 6.6 if len(sets) < 70 else 8.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[3.4, 3.6, 20], wspace=0)
    bx, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[2])

    x0, ticks, labs, H = 0, [], [], []
    for gi, (g, fs) in enumerate(G):
        fs = sorted(fs, key=lambda s: mu[(s, "lgbm")])   # best lgbm last
        xs = np.arange(x0, x0 + len(fs))
        for a, lab, c, m in ARMS:
            ax.plot(xs, [mu[(s, a)] for s in fs], color=c, marker=m, lw=2, ms=6)
            ax.plot(xs, [sd[(s, a)] for s in fs], color=c, ls="--", lw=1.8,
                    marker=m, ms=3.5)
            # a treatment with no score anywhere (TabICL on a csv built
            # before it ran) earns no legend row rather than a crash
            if gi == 0 and (ok := [s for s in sets
                                   if mu[(s, a)] == mu[(s, a)]]):
                H.append((lab, c, m, st.fmean([mu[(s, a)] for s in ok]),
                          st.fmean([sd[(s, a)] for s in ok]), len(ok)))
        ax.text(xs.mean(), 170, g, ha="center", va="bottom",
                fontsize=FS["head"] - 3)
        ticks += list(xs); labs += [f.split("/")[-1] for f in fs]
        x0 += len(fs) + 3
        if gi < len(G) - 1: ax.axvline(x0 - 2, color="#555", lw=1)

    ax.axhline(100, color="#999", lw=1.4, ls=(0, (6, 4)))
    ax.text(x0 - 1.6, 101, "the table's own optimum", ha="right",
            va="bottom", fontsize=FS["tick"] - 1, color="#666")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs, rotation=90, ha="center", va="top",
                       fontsize=XT)
    ax.set_ylim(0, 168); ax.set_xlim(-.9, x0 - 2.1)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([10, 30, 50, 70, 90, 110], minor=True)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="major", color="#ddd")
    ax.grid(axis="y", which="minor", color="#b0b0b0", ls=(0, (4, 4)), lw=1)
    ax.set_ylabel("score: 100 = the table's optimum\n"
                  "higher is better, 5x2 folds", fontsize=FS["lab"])
    ax.tick_params(axis="y", labelsize=FS["tick"])
    ax.spines["top"].set_visible(False)

    # top left, and wholly inside the first panel.  Legend text is fixed in
    # points while the panel is only as wide as its share of the sets, so the
    # layout has to earn its width: one swatch, not two.  The sd column says
    # it is the dashed series by underlining its heading in dashes.
    gbig = max(range(len(G)), key=lambda i: len(G[i][1]))
    x0s = 0                              # left edge of widest panel
    for g, fs in G[:gbig]: x0s += len(fs) + 1
    left = (x0s + .9) / (x0 - 1.2)
    wide = len(G[gbig][1]) / (x0 - 1.2)
    tx = ax.inset_axes((left + .012, .70, wide - .025, .28))
    tx.set_xlim(0, 24); tx.set_ylim(0, 5.7)
    tx.patch.set_alpha(0); tx.set_xticks([]); tx.set_yticks([])
    for s_ in tx.spines.values(): s_.set_visible(False)
    mono = dict(family="monospace", fontsize=FS["mono"] - 2,
                bbox=dict(fc="#fcfcfa", ec="none", pad=1))
    tx.text(17.2, 5.2, "mean", ha="right", va="center", **mono)
    tx.text(23.8, 5.2, "sd", ha="right", va="center", **mono)
    tx.plot([21.4, 23.8], [4.6, 4.6], color="#555", lw=1.4, ls=(0, (3, 2)))
    for k, (lab, c, m, a, b, _n) in enumerate(H):
        y = 4.0 - 1.1 * k
        tx.text(0, y, lab, ha="left", va="center", **mono)
        tx.plot([9.4, 11.4], [y, y], color=c, lw=2.2)
        tx.plot(10.4, y, m, color=c, ms=6)
        tx.text(17.2, y, f"{a:.1f}", ha="right", va="center", **mono)
        tx.text(23.8, y, f"{b:.1f}", ha="right", va="center", **mono)

    if keep: secs = {}                   # the .txt totals are corpus-
    if sum(secs.get(a, 0) for a, *_ in ARMS) > 0:   # wide, not this cut
        bt = sorted((secs[a], lab, c) for a, lab, c, _ in ARMS if a in secs)
        bx.bar(range(len(bt)), [b[0] for b in bt],
               color=[b[2] for b in bt], width=.84)
        bx.set_yscale("log")
        bx.set_ylim(min(b[0] for b in bt) / 2, max(b[0] for b in bt) * 4)
        for i, b in enumerate(bt):
            bx.text(i, b[0] * 1.25, f"{b[0]:,.0f}s", ha="center", va="bottom",
                    weight="bold", fontsize=FS["bar"] - 1)
        bx.set_xticks(range(len(bt)))
        bx.set_xticklabels([b[1] for b in bt], rotation=90, va="top",
                           fontsize=FS["bar"] - 1)
        bx.set_title("total secs, log scale\n(all folds, 1 thread)",
                     fontsize=FS["head"] - 5)
    else:
        bx.text(.5, .5, "timings:\nsee the\nwhole-corpus\nfigure",
                ha="center", va="center",
                fontsize=FS["head"] - 3, color="#888", transform=bx.transAxes)
        bx.set_xticks([])
    bx.set_yticks([]); bx.set_facecolor("#fcfcfa")
    for s_ in ("top", "right", "left"): bx.spines[s_].set_visible(False)

    fig.set_facecolor("#fcfcfa"); ax.set_facecolor("#fcfcfa")
    plt.tight_layout()
    # Matplotlib parks the y label just left of the tick labels, which on
    # this layout lands its first line on top of the runtime bars.  No
    # fixed padding is right for every corpus -- the bar panel's width
    # follows the set count -- so measure once the layout has settled and
    # butt the label's LEFT EDGE up against the bars.  Centring it in the
    # gap is not enough: the label is two lines wide, so its own left half
    # still reaches back over the bars.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    w    = ax.yaxis.label.get_window_extent(rend).width
    bars = bx.get_tightbbox(rend).x1
    tick = min(t.get_window_extent(rend).x0 for t in ax.get_yticklabels())
    x = min(bars + 40 + w / 2, tick - 12 - w / 2)      # ticks win a tie
    ax.yaxis.set_label_coords(
        ax.transAxes.inverted().transform((x, 0))[0], .5)
    plt.savefig(sys.argv[2], dpi=200)
    plt.savefig(sys.argv[2].rsplit(".", 1)[0] + ".pdf")
    for lab, c, m, a, b, n in H:
        print(f"{lab:10s} mean {a:.1f}  sd {b:.1f}  ({n} sets)")
    for g, fs in G: print(f"{g.replace(chr(10),' '):22s} {len(fs)} sets")


if __name__ == "__main__": main()
