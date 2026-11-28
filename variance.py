#!/usr/bin/env python3 -B
"""variance: where does an AUC move come from?  Splits the variance inside
each group of data sets four ways -- the data set, the learner, the fold,
and the interaction between learner and data set -- each group summing to
100%.  Prints the LaTeX rows the paper's table is built from, so that table
cannot drift from these numbers; --png also draws them as stacked bars.

This is a two-way ANOVA with replication (data set x learner, R folds per
cell), reported as eta-squared.  tree(1), the deliberately crippled single
tree, is left out: the question is what separates learners anyone would use.

Bars are sorted by the learner share, so the groups that trouble the claim
sit at the bottom.

usage: python3 variance.py runs/nose39-yue.txt runs/se61-yue.txt
       python3 variance.py runs/nose39-yue.txt runs/se61-yue.txt --png F.png
"""
import sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["lgbm", "n30", "tabicl"]        # the three a reader would consider
# TabICL did not finish ten folds on these two (its per-fold timeout fired),
# which would leave the design unbalanced -- and an unbalanced two-way layout
# does not decompose orthogonally, so the shares would not sum to 100.
PARTIAL = {"bank-marketing", "adult"}
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


def tex(rows):
  "The table body, sorted so the learner column runs low to high."
  def line(name, n, v):
    out = [v[0], v[2], v[3], v[1]]            # data, fold, other, learner
    hi  = max(out)
    cs  = " & ".join((r"\cellcolor{gray!25}" + f"{x:.0f}") if x == hi
                     else f"{x:.0f}" for x in out)
    return f"{name} & {n} & {cs} & {v[0] / v[1]:.0f} " + r"\\"
  for name, n, v in rows[:-1]: print(line(name, n, v))
  print(r"\cmidrule{1-7}")
  print(line(*rows[-1]))


def main():
  nose, se = folds(sys.argv[1]), folds(sys.argv[2])
  order   = [f for f in nose if f not in PARTIAL]
  groups  = [(n, order[a:b], nose) for n, a, b in NOSE]
  groups += [(n, [f for f in se if f.startswith(p)], se) for n, p in SE]
  rows = [(n, len(s), split(s, src)) for n, s, src in groups]
  rows.sort(key=lambda r: r[2][1])       # least learner-effect first
  allsets = {f: src[f] for _, s, src in groups for f in s}
  rows.append(("all", len(allsets), split(list(allsets), allsets)))
  if "--png" not in sys.argv:
    return tex(rows)
  rows = [(f"{n} ({c})", v) for n, c, v in rows][::-1]

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
