#!/usr/bin/env python3 -B
"""regplot: ens-balsa-reg.png from lgbm_reg.py output.  Per set:
mean RELATIVE MAE over 5x2 folds (mae / guess-the-mean mae; 0 =
perfect, 1 = the mean, dashed line), solid lines, sets sorted by
lgbm; total secs as log bars on the left.  No TabICL at this stage.
Usage: python3 regplot.py reg60.txt ens-balsa-reg.png (+ .pdf)"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
L = open(sys.argv[1]).read().splitlines()
sec = [float(v) for v in
       [l for l in L if l.startswith("#secs")][0].split()[3:]]
F = {}
for l in L:
  if l.startswith("%folds"):
    _, f, t, *v = l.split(); F[(f, t)] = [float(x) for x in v]
TK = ["balsa(1)", "balsa(30)", "lgbm"]
sets = sorted({f for f, _ in F},
              key=lambda f: np.mean(F[(f, "lgbm")]))
d = {f: [float(np.mean(F[(f, t)])) for t in TK] for f in sets}
T = [("balsa(1)", 0, "#2ca02c", "^"), ("balsa(30)", 1, "#e8613c", "s"),
     ("lgbm(100)", 2, "#2a6fdb", "o")]
FS = dict(tick=14, lab=14, head=17, mono=14, bar=15)
fig = plt.figure(figsize=(14.3, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[2.6, 1.2, 20], wspace=0)
bx = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[2])
xs = np.arange(len(sets))
H = []
for name, j, c, m in T:
  ax.plot(xs, [d[f][j] for f in sets], color=c, marker=m, lw=2, ms=6)
  H.append((name, c, m, np.mean([d[f][j] for f in sets])))
ax.axhline(1, color="#555", lw=1.4, ls="--")
ax.set_xticks(xs)
ax.set_xticklabels(sets, rotation=60, ha="right",
                   rotation_mode="anchor", fontsize=10)
ax.set_ylim(0, 1.4); ax.set_xlim(-.7, len(sets) - .3)
ax.grid(axis="y", color="#ddd")
ax.set_ylabel("MAE / guess-the-mean MAE, 5x2 folds",
              fontsize=FS["lab"])
ax.tick_params(axis="y", labelsize=FS["tick"])
ax.text(len(sets) / 2, 1.45,
        "%d regression sets (moot/regression)" % len(sets),
        ha="center", va="bottom", fontsize=FS["head"])
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
tx = ax.inset_axes([.02, .66, .36, .30])
tx.set_xlim(0, 30); tx.set_ylim(0, 4)
tx.patch.set_alpha(0); tx.set_xticks([]); tx.set_yticks([])
for s_ in tx.spines.values(): s_.set_visible(False)
mono = dict(family="monospace", fontsize=FS["mono"],
            bbox=dict(fc="#fcfcfa", ec="none", pad=1))
tx.text(29.5, 3.5, "mean", ha="right", va="center", **mono)
for i, (name, c, m, mu) in enumerate(H):
  y = 2.6 - 1.0 * i
  tx.text(0, y, name, ha="left", va="center", **mono)
  tx.plot([12.6, 15.2], [y, y], color=c, lw=2.2)
  tx.plot(13.9, y, m, color=c, ms=6)
  tx.text(29.5, y, f"{mu:.2f}", ha="right", va="center", **mono)
bt = sorted([(sec[0], "balsa\n(1)", "#2ca02c"),
             (sec[1], "balsa\n(30)", "#e8613c"),
             (sec[3], "lgbm\n(100)", "#2a6fdb")])
bx.bar(range(3), [b[0] for b in bt], color=[b[2] for b in bt],
       width=.84)
bx.set_yscale("log")
bx.set_ylim(min(b[0] for b in bt) / 3, max(b[0] for b in bt) * 6)
for i, b in enumerate(bt):
  bx.text(i, b[0] * 1.25,
          f"{b[0]:,.0f}s" if b[0] >= 10 else f"{b[0]:.1f}s",
          ha="center", va="bottom", weight="bold",
          fontsize=FS["bar"] - 1)
bx.set_xticks(range(3))
bx.set_xticklabels([b[1] for b in bt], fontsize=FS["bar"] - 2)
bx.set_title("total secs,\nlog scale", fontsize=FS["head"] - 5)
bx.set_yticks([]); bx.set_facecolor("#fcfcfa")
bx.set_xlim(-.55, 2.55)
for s_ in ("top", "right", "left"): bx.spines[s_].set_visible(False)
fig.set_facecolor("#fcfcfa"); ax.set_facecolor("#fcfcfa")
plt.tight_layout()
plt.savefig(sys.argv[2], dpi=200)
plt.savefig(sys.argv[2].rsplit(".", 1)[0] + ".pdf")
