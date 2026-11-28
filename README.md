# tree-dev : where the mess lives

Development home of [tree](https://github.com/timm/tree), a
218-line tabular learner. This repo holds everything the rock does
not ship: the Lisp twin (`*.lisp`, originally `bald`), the LightGBM
and TabICL hooks (`lgbm.py`, `tabicl3.py`), the experiment rigs
(`tree_eg.yue`, `ens30.lisp`, `replay1.lisp`), every score
(`runs/`), the charts and the chart code (`plot.py`, `plot3.py`).

    make            help
    make sh         work shell: vi = nvim --clean, repo aliases
    make yue        rebuild both .lua from the .yue sources
    make package    copy tree.yue into ../tree and rebuild there
    make charts     redraw bald.png, bald30.png from runs/

Scores in `runs/` are per-fold `data|aucs|secs|test row ids`; the
row ids let any other arm (LightGBM, TabICL, a port) be replayed on
the exact same 5x2 folds. See `tabicl3.py` and `tree_eg.yue
--replay`.

(c) 2026 Tim Menzies <timm@ieee.org>, MIT license.
