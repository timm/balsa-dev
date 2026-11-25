#!/usr/bin/env python3 -B
"""variance: where does an AUC move come from?  One stacked bar per group of
data sets, each summing to 100%: the data set, the learner, the fold, and
the interaction between learner and data set.

This is a two-way ANOVA with replication (data set x learner, R folds per
cell), reported as eta-squared.  balsa(1), the deliberately crippled single
tree, is left out: the question is what separates learners anyone would use.

Bars are sorted by the learner share, so the groups that trouble the claim
sit at the bottom.

usage: python3 variance.py runs/nose39-yue.txt runs/se61-yue.txt variance.png
"""
import sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["lgbm", "n30", "tabicl"]        # the three a reader would consider
# group name -> how to recognise its data sets in the two run files
NOSE = [("numeric", 0, 10), ("symbolic", 10, 20), ("mixed", 20, 30),
        ("large", 30, 39)]
SE   = [("issues", "se/issues/"), ("smells", "se/smells/"),
        ("defects", "se/defect"), ("static", "se/static")]
# the four shares, and the ink for each
PARTS = [("data set",     "#2a6fdb"),
         ("learner",      "#e8613c"),
         ("fold",         "#999999"),
         ("interaction",  "#111111")]


def folds(path):
  "Every %folds line: set -> arm -> its R fold AUCs, in file order."
  out = {}
  for line in open(path):
    if not line.startswith("%folds"): continue
    _, f, arm, *vals = line.split()
    out.setdefault(f, {})[arm] = [float(v) for v in vals]
  return out


def split(sets, src):
  "eta-squared for data set, learner, fold and interaction.  Sums to 100."
  I, J = len(sets), len(ARMS)
  R = len(src[sets[0]][ARMS[0]])
  allv = [v for f in sets for a in ARMS for v in src[f][a]]
  g = st.fmean(allv)
  row  = {f: st.fmean([v for a in ARMS for v in src[f][a]]) for f in sets}
  col  = {a: st.fmean([v for f in sets for v in src[f][a]]) for a in ARMS}
  cell = {(f, a): st.fmean(src[f][a]) for f in sets for a in ARMS}
  tot  = sum((v - g) ** 2 for v in allv)
  return [100 * x / tot for x in (
    J * R * sum((row[f] - g) ** 2 for f in sets),
    I * R * sum((col[a] - g) ** 2 for a in ARMS),
    sum((v - cell[(f, a)]) ** 2 for f in sets for a in ARMS for v in src[f][a]),
    R * sum((cell[(f, a)] - row[f] - col[a] + g) ** 2
            for f in sets for a in ARMS))]


def main():
  nose, se = folds(sys.argv[1]), folds(sys.argv[2])
  order = list(nose)
  groups  = [(n, order[a:b], nose) for n, a, b in NOSE]
  groups += [(n, [f for f in se if f.startswith(p)], se) for n, p in SE]
  rows = [(f"{n} ({len(s)})", split(s, src)) for n, s, src in groups]
  rows.sort(key=lambda r: -r[1][1])      # most learner-effect at the top
  allsets = {f: src[f] for _, s, src in groups for f in s}
  rows.append((f"ALL ({len(allsets)})", split(list(allsets), allsets)))

  # drawn at 2x a one-column figure, so a 12pt label prints at 6pt
  f, ax = plt.subplots(figsize=(6.6, 4.2))
  y = list(range(len(rows)))[::-1]
  left = [0] * len(rows)
  for k, (name, colour) in enumerate(PARTS):
    w = [r[1][k] for r in rows]
    ax.barh(y, w, left=left, color=colour, height=.68, label=name)
    for yy, ww, ll in zip(y, w, left):
      if ww >= 7:                        # only label a slice with room
        ax.text(ll + ww / 2, yy, f"{ww:.0f}", ha="center", va="center",
                color="white", fontsize=11, weight="bold")
    left = [a + b for a, b in zip(left, w)]
  ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=12)
  ax.set_xlim(0, 100); ax.set_xticks(range(0, 101, 20))
  ax.set_xlabel("% of the variance in AUC within that group", fontsize=12)
  ax.tick_params(axis="x", labelsize=12)
  for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
  ax.axhline(y[-1] + .5, color="#555", lw=1)      # ALL sits below a rule
  ax.legend(ncol=4, fontsize=11, frameon=False,
            loc="lower center", bbox_to_anchor=(.5, 1.0))
  f.set_facecolor("#fcfcfa"); ax.set_facecolor("#fcfcfa")
  f.tight_layout(pad=.3)
  f.savefig(sys.argv[-1], dpi=200)
  f.savefig(sys.argv[-1].rsplit(".", 1)[0] + ".pdf")
  for name, v in rows:
    print(f"{name:16s}" + "".join(f"{x:6.1f}" for x in v) + f"   sum {sum(v):5.1f}")


if __name__ == "__main__": main()
