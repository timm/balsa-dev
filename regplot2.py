#!/usr/bin/env python3 -B
"""regplot2: the ens-balsa-nose figure, for regression.

y is 1 - MAE/MAE(predict the mean): the share of the baseline's error that
a learner removes.  1.0 is perfect, 0.0 is no better than the mean, and
negative is worse than it.  Plotted this way round so HIGHER IS BETTER, as
in the AUC figures, and -- the reason for it -- so the solid means sit at
the top while the dashed spreads sit near zero.  On MAE/MAE the two ran
together along the bottom and could not be told apart.

Three groups, as far as this corpus allows the classification split:
numeric (all x columns numeric, < 1000 rows), mixed (at least one symbolic
column, < 1000 rows), and large (1000 rows or more).  Only `sensory` and
`servo` have no numeric column at all, so there is no symbolic group; both
sit in mixed.

Solid: the mean over the ten 5x2 folds.  Dashed: the spread of those folds,
as (p90-p10)/2.56.  Left: total run time, log scale, summed over folds.

usage: python3 regplot2.py runs/reg60.csv ens-balsa-reg.png
"""
import csv, os, sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MOOT = os.path.expanduser(os.getenv("MOOTREG", "~/gits/moot/regression"))
# name in the csv, label, colour, marker -- the same ink as the AUC figures
ARMS = [("balsa1",  "balsa(1)",  "#2ca02c", "^"),
        ("lgbm",    "lgbm(100)", "#2a6fdb", "o"),
        ("balsa30", "balsa(30)", "#e8613c", "s")]
FS = dict(tick=14, lab=14, head=17, mono=14, bar=15)


def spread(v):
    "The sd the AUC figures draw: a robust (p90-p10)/2.56 over the folds."
    v = sorted(v); n = len(v)
    q = lambda p: v[min(n - 1, int(p * n))]
    return (q(.9) - q(.1)) / 2.56


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
    if any("/" in s for s in sets):          # moot/optimize: group by folder
        G = [(d, [s for s in sets if s.split("/")[0] == d])
             for d in sorted({s.split("/")[0] for s in sets})]
    else:
        G = [("numeric\nattributes", [s for s in sets if n[s] < 1000 and numeric[s]]),
             ("mixed\nattributes",   [s for s in sets if n[s] < 1000 and not numeric[s]]),
             ("large, mixed",        [s for s in sets if n[s] >= 1000])]

    mu = {(s, a): st.fmean([1 - float(r[f"rel_{a}"]) for r in D[s]])
          for s in sets for a, *_ in ARMS}
    sd = {(s, a): spread([1 - float(r[f"rel_{a}"]) for r in D[s]])
          for s in sets for a, *_ in ARMS}
    # timings live in the sibling .txt, on its "#secs (all data)" line
    secs = {}
    txt = sys.argv[1].rsplit(".", 1)[0] + ".txt"
    if os.path.exists(txt):
        head = open(txt).readline().split()[1:]
        for line in open(txt):
            if line.startswith("#secs"):
                v = line.split()[3:]
                secs = {("balsa1" if "(1)" in h else
                         "balsa30" if "(30)" in h else h): float(x)
                        for h, x in zip(head, v)}

    fig = plt.figure(figsize=(14.3, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[3.4, 1.6, 20], wspace=0)
    bx, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[2])

    x0, ticks, labs, H = 0, [], [], []
    for gi, (g, fs) in enumerate(G):
        fs = sorted(fs, key=lambda s: -mu[(s, "lgbm")])  # best lgbm first
        xs = np.arange(x0, x0 + len(fs))
        for a, lab, c, m in ARMS:
            ax.plot(xs, [mu[(s, a)] for s in fs], color=c, marker=m, lw=2, ms=6)
            ax.plot(xs, [sd[(s, a)] for s in fs], color=c, ls="--", lw=1.8,
                    marker=m, ms=3.5)
            if gi == 0:
                H.append((lab, c, m,
                          st.fmean([mu[(s, a)] for s in sets]),
                          st.fmean([sd[(s, a)] for s in sets])))
        ax.text(xs.mean(), 1.44, g, ha="center", va="bottom", fontsize=FS["head"])
        ticks += list(xs); labs += [f.split("/")[-1] for f in fs]
        x0 += len(fs) + 3
        if gi < len(G) - 1: ax.axvline(x0 - 2, color="#555", lw=1)

    ax.axhline(0.0, color="#999", lw=1.4, ls=(0, (6, 4)))
    ax.text(-.5, .03, "no better than predicting the mean", ha="left",
            va="bottom", fontsize=FS["tick"] - 1, color="#666")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs, rotation=60, ha="right", rotation_mode="anchor",
                       fontsize=10)
    ax.set_ylim(-.5, 1.42); ax.set_xlim(-.9, x0 - 2.1)
    ax.set_yticks([-.5, -.25, 0, .25, .5, .75, 1])
    ax.set_yticks([-.375, -.125, .125, .375, .625, .875], minor=True)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="major", color="#ddd")
    ax.grid(axis="y", which="minor", color="#b0b0b0", ls=(0, (4, 4)), lw=1)
    ax.set_ylabel("1 - MAE/MAE(predict the mean)\nhigher is better, 5x2 folds",
                  fontsize=FS["lab"], labelpad=6)
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
    tx = ax.inset_axes((left + .012, .74, wide - .025, .22))
    tx.set_xlim(0, 24); tx.set_ylim(0, 4.6)
    tx.patch.set_alpha(0); tx.set_xticks([]); tx.set_yticks([])
    for s_ in tx.spines.values(): s_.set_visible(False)
    mono = dict(family="monospace", fontsize=FS["mono"] - 2,
                bbox=dict(fc="#fcfcfa", ec="none", pad=1))
    tx.text(17.2, 4.1, "mean", ha="right", va="center", **mono)
    tx.text(23.8, 4.1, "sd", ha="right", va="center", **mono)
    tx.plot([21.4, 23.8], [3.5, 3.5], color="#555", lw=1.4, ls=(0, (3, 2)))
    for k, (lab, c, m, a, b) in enumerate(H):
        y = 2.9 - 1.1 * k
        tx.text(0, y, lab, ha="left", va="center", **mono)
        tx.plot([9.4, 11.4], [y, y], color=c, lw=2.2)
        tx.plot(10.4, y, m, color=c, ms=6)
        tx.text(17.2, y, f"{a:.2f}", ha="right", va="center", **mono)
        tx.text(23.8, y, f"{b:.2f}", ha="right", va="center", **mono)

    if sum(secs.get(a, 0) for a, *_ in ARMS) > 0:
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
        bx.set_title("total secs, log scale\n(summed over folds,\n1 thread each)",
                     fontsize=FS["head"] - 5)
    else:
        bx.text(.5, .5, "no timings\nin this csv", ha="center", va="center",
                fontsize=FS["head"] - 3, color="#888", transform=bx.transAxes)
        bx.set_xticks([])
    bx.set_yticks([]); bx.set_facecolor("#fcfcfa")
    for s_ in ("top", "right", "left"): bx.spines[s_].set_visible(False)

    fig.set_facecolor("#fcfcfa"); ax.set_facecolor("#fcfcfa")
    plt.tight_layout()
    plt.savefig(sys.argv[2], dpi=200)
    plt.savefig(sys.argv[2].rsplit(".", 1)[0] + ".pdf")
    for lab, c, m, a, b in H: print(f"{lab:10s} mean {a:.2f}  sd {b:.2f}")
    for g, fs in G: print(f"{g.replace(chr(10),' '):22s} {len(fs)} sets")


if __name__ == "__main__": main()
