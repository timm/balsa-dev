# TODO

Found during the balsa -> enuf rename (2026-11-27), left alone so that
commit stayed a pure rename.

- [ ] **xcolor option clash in the paper.** `~/gh/jssclose` has
  "Option clash for package xcolor" (main.log line ~512), and it
  predates the rename. Overleaf tolerates it; `make pdf` there (latexmk
  `-halt-on-error`) stops on it. Without halt-on-error there are 64
  `!` errors, all downstream of it.
- [ ] **regression.tex is stale.** `make regtex` would change "264
  lines" to "293 lines" (the learner has grown) and reorder the
  ten-set list (`fried` moves before `mv`). Re-run `make regtex` and
  check the diff.
- [ ] **enuf(30) seconds disagree.** main.tex (line ~387) says
  enuf(30) runs the 39 general sets in 32 s, but the nose figure
  (`ens-enuf-nose.pdf`) shows 17 s and `fig1.py` prints 33.6 s. Pick
  one measure, then fix the text or the figure.
