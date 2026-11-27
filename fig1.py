#!/usr/bin/env python3 -B
"""fig1: the paper's opening figure, generated rather than typed.

Three area-proportional panels, side by side and bottom aligned:

    (a) lines of code          this code    against LightGBM's C++ core
    (b) secs, classification   balsa(30)    against TabICL
    (c) secs, regression       balsa(30)    against TabICL
    (d) secs, optimization     balsa(30)    against TabICL

Area, not side, carries the number: side = S*sqrt(value/biggest).  In (b)
and (c) the two cheap learners land on the same speck -- 34s against 36s,
97s against 30s -- which is the panel's point, so one square is drawn and
both numbers are named beside it.

The counts and the timings are read from the source and from runs/*.txt.
This figure used to be hand-written TikZ with its numbers typed in, and
they went stale twice: the line count said 245 when the file held 247,
and it still said balsa.yue after the learner was ported to Python.

LOC rule: non-blank and not a # comment -- the same rule the Yue count
used, and the rule Table 2 states.  Docstrings are NOT stripped: balsa.py's
is the Options help the program prints for -h, so it is doing a job, and
stripping it would cut 14 lines here and 9 in the rig for no good reason.
Raw wc -l would instead ADD 104 lines of blanks and comments, which would
flatter us against a LightGBM number counted the other way.

LightGBM's number is from its own repo at 7810ece, src/ plus
include/LightGBM, C++ only, no cuda; there is no checkout here to recount,
so it stays a constant.

usage: python3 fig1.py > fig1.tex
"""
import os, re, sys

Here = os.path.dirname(os.path.abspath(__file__))
Lgbm = 36328
S    = 2.55          # cm, the WIDTH of each panel's biggest box
Tall = 1.00          # ... and its height, as a multiple of that width
Gap  = 0.55          # cm between panels, left to right
VGap = 0.95          # ... and between the two rows


# The optimizer of sec:active is a handful of functions inside a file
# that also holds four other rankers and the rig.  These are it: spend a
# label, chase the best/rest boundary, rank the test set by its nearest
# labelled neighbour, grade the row you kept.  Everything they lean on
# (Tbl, distx, mids, disty) is in balsa.py, already counted as the
# learner, so nothing here is double counted.
OPT = ("shuffle", "label", "acquire", "byKnn", "scorer", "optimize")


def loc_defs(path, names):
    "Non-blank, non-comment lines of just these top-level definitions."
    import ast
    src = open(os.path.join(Here, path)).read()
    lines = src.splitlines()
    want = set(names)
    n = 0
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in want:
            n += sum(1 for l in lines[node.lineno - 1:node.end_lineno]
                     if not re.match(r"^\s*(#.*)?$", l))
        # the knobs the optimizer reads: `the.warm, the.budget, ... = ...`
        elif isinstance(node, ast.Assign) and "the." in ast.unparse(node.targets[0]):
            n += 1
    return n


def loc(path):
    """Non-blank, non-comment lines -- the same rule the Yue count used.
    Docstrings are left in: they are the learner's own help text, printed
    in the paper, and sec:code quotes this number too."""
    return sum(1 for l in open(os.path.join(Here, path))
               if not re.match(r"^\s*(#.*)?$", l))


def secs(*paths):
    "treatment -> seconds, summed over the given run tables."
    out = {}
    for p in paths:
        p = os.path.join(Here, p)
        if not os.path.exists(p): continue
        head = open(p).readline().split()[1:]
        for line in open(p):
            if not line.startswith("#secs"): continue
            for h, v in zip(head, line.split()[3:]):
                out[h] = out.get(h, 0) + float(v)
    return out


def act_secs(*paths):
    "treatment -> seconds, from the active-learning tables' %secs lines."
    out = {}
    for p in paths:
        p = os.path.join(Here, p)
        if not os.path.exists(p): continue
        for l in open(p):
            if l.startswith("%secs"):
                _, _s, a, *v = l.split()
                out[a] = out.get(a, 0) + sum(float(x) for x in v)
    return out


def tabicl(*paths):
    t = 0.0
    for p in paths:
        p = os.path.join(Here, p)
        if not os.path.exists(p): continue
        for line in open(p):
            if line.startswith("#secs"): t += float(line.split()[-1])
    return t


def panel(x, y, tag, title, big, small, unit="", note=None):
    """One panel: the biggest box, then each smaller one nested in its
    bottom-left corner, with a leader line out to its label.

    All three panels are the same width, and every box is the same shape,
    so AREA still carries the number: a box holding a fraction r of the
    biggest is drawn at sqrt(r) of both its width and its height."""
    H = S * Tall
    bn, bv, bc = big
    o = [f"  \\begin{{scope}}[xshift={x:.2f}cm,yshift={y:.2f}cm]",
         f"    \\node[anchor=south west,font=\\scriptsize\\bfseries]"
         f" at (0,{H + .06:.2f}) {{{tag}. {title}}};",
         f"    \\filldraw[fill={bc},draw={bc}] (0,0) rectangle ({S},{H:.2f});",
         f"    \\node[anchor=north east,align=right,text=white,"
         f"font=\\scriptsize\\bfseries] at ({S - .08:.2f},{H - .08:.2f})"
         f" {{{bn}\\\\{bv:,.0f}{unit}}};"]
    for k, (nm, v, c) in enumerate(small):
        r = (v / bv) ** .5
        side, tall = S * r, H * r
        dx = sum(S * (u / bv) ** .5 + .06 for _, u, _ in small[:k])
        o.append(f"    \\fill[{c}] ({dx:.3f},0) rectangle "
                 f"++({side:.3f},{tall:.3f});")
        y = H * .62 - .38 * k
        o.append(f"    \\node[anchor=west,text=white,inner sep=1pt,align=left,"
                 f"text width={S - .35:.2f}cm,font=\\scriptsize] (l{tag}{k})"
                 f" at (.62,{y:.2f}) {{{nm}}};")
        o.append(f"    \\draw[->,white,line width=.4pt,shorten >=2.5pt]"
                 f" (l{tag}{k}.west) to[out=180,in=60]"
                 f" ({dx + side * .6:.3f},{tall * .9:.3f});")
    if note:
        o.append(f"    \\node[anchor=south west,text=white,inner sep=1pt,"
                 f"font=\\scriptsize\\itshape] at (.1,.08) {{{note}}};")
    o.append("  \\end{scope}")
    return "\n".join(o)


def main():
    learner, harness = loc("balsa.py"), loc("balsa_eg.py")
    opt = loc_defs("active.py", OPT)
    cls = secs("runs/nose39-yue.txt", "runs/se61-yue.txt")
    reg = secs("runs/reg56.txt", "runs/sereg.txt")
    ct  = cls.get("tabicl", 0)
    rt  = tabicl("runs/reg56-tabicl.txt", "runs/sereg-tabicl.txt")
    act = act_secs("runs/active-secs-balsa.txt", "runs/active-secs-heavy.txt")
    ab, at = act.get("a.b30", 0), act.get("a.tabicl", 0)
    cb, cl = cls.get("n30", 0), cls.get("lgbm", 0)
    rb, rl = reg.get("balsa(30)", 0), reg.get("lgbm", 0)

    # The counts go in their own file, loaded in the preamble: the
    # abstract quotes them too, and it comes long before this figure.
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(
            "% GENERATED by balsa-dev/fig1.py -- do not edit here.\n"
            f"\\newcommand{{\\LearnerLOC}}{{{learner}}}\n"
            f"\\newcommand{{\\HarnessLOC}}{{{harness}}}\n"
            f"\\newcommand{{\\TotalLOC}}{{{learner + harness}}}\n")
    print(f"""% GENERATED by balsa-dev/fig1.py -- do not edit here.
% Area $\\propto$ value; side = S*sqrt(value/biggest).
\\begin{{figure*}}[p]
  \\centering
  \\begin{{tikzpicture}}[x=1cm,y=1cm,font=\\scriptsize]
    % Exact series colours, the same ink as the lines in Figs 2-5.
    \\definecolor{{lgbmC}}{{HTML}}{{2A6FDB}}
    \\definecolor{{balsaC}}{{HTML}}{{E8613C}}
    \\definecolor{{tabiclC}}{{HTML}}{{000000}}""")
    print(panel(0, 0, "a", "lines of code",
                ("LightGBM", Lgbm, "lgbmC"),
                [(f"learner, {learner}", learner, "balsaC"),
                 (f"optimizer, {opt}", opt, "balsaC!75!white"),
                 (f"harness, {harness}", harness, "balsaC!45!white")]))
    print(panel(S + Gap, 0, "b", "secs, classification",
                ("TabICL", ct, "tabiclC"),
                [(f"this learner, {cb:,.0f}s", cb, "balsaC")], unit="s"))
    print(panel(2 * (S + Gap), 0, "c", "secs, regression",
                ("TabICL", rt, "tabiclC"),
                [(f"this learner, {rb:,.0f}s", rb, "balsaC")], unit="s"))
    print(panel(3 * (S + Gap), 0, "d", "secs, optimization",
                ("TabICL", at, "tabiclC"),
                [(f"this optimizer, {ab:,.0f}s", ab, "balsaC")],
                unit="s"))
    print(f"""  \\end{{tikzpicture}}
  \\caption{{Area $\\propto$ size: every box is the same shape, so one
  holding a tenth of another is drawn at $\\sqrt{{1/10}}$ of its side.
  (a) lines of code, counted from the source. (b)--(d) total run time,
  one thread each: TabICL costs {ct / max(cl, 1):,.0f}, {rt / max(rl, 1):,.0f}
  and {at / max(ab, 1):,.0f} times LightGBM on the three studies. On all
  three, our code and LightGBM are too close to tell apart at this scale,
  so only ours is drawn.}}
  \\label{{fig:loc}}
\\end{{figure*}}""", file=sys.stdout)

    print(f"% learner {learner}  harness {harness}  lgbm {Lgbm}\n"
          f"% optimizer {opt} (acquire + 1nn, out of {loc('active.py')} in active.py)\n"
          f"% cls: balsa(30) {cb:.1f}  lgbm {cl:.1f}  tabicl {ct:,.1f}\n"
          f"% reg: balsa(30) {rb:.1f}  lgbm {rl:.1f}  tabicl {rt:,.1f}",
          file=sys.stderr)


if __name__ == "__main__": main()
